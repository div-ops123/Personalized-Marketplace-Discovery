"""Spark aggregation logic for the two historized daily feature tables.

snapshot_date D aggregates all events strictly before day D's start (i.e.
through day D-1) -- this makes the downstream point-in-time join
(data-flow.md: MAX(snapshot_date) < T) leak-proof by construction, since a
day-D snapshot never contains a day-D event no matter what time within day
D the join's T falls on. Callers are responsible for pre-filtering inputs
to timestamp < snapshot_date (see db_io.read_events_before) -- this module
only aggregates what it's given.
"""

import datetime

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from common.constants import ATTRIBUTION_WINDOW_HOURS

TOP_N = 3

_USER_FEATURE_COLUMNS = [
    "snapshot_date",
    "user_id",
    "preferred_brands",
    "avg_purchase_price",
    "historical_category_affinity",
]
_CANDIDATE_FEATURE_COLUMNS = [
    "snapshot_date",
    "candidate_id",
    "recommendation_ctr",
    "recommendation_cvr",
    "recommendation_impressions",
]


def _top_n_column(df: SparkDataFrame, group_col: str, value_col: str, n: int) -> SparkDataFrame:
    """Up to n most-frequent value_col values per group_col group.

    Ties broken alphabetically by value_col for determinism. Returned
    array is ordered by frequency descending.

    Returns:
        SparkDataFrame: group_col, top_values (array<string>).
    """
    counts = df.groupBy(group_col, value_col).agg(F.count("*").alias("cnt"))
    window = Window.partitionBy(group_col).orderBy(F.col("cnt").desc(), F.col(value_col).asc())
    ranked = counts.withColumn("rank", F.row_number().over(window)).filter(F.col("rank") <= n)
    grouped = ranked.groupBy(group_col).agg(
        F.sort_array(F.collect_list(F.struct("rank", value_col))).alias("ranked")
    )
    return grouped.select(group_col, F.col(f"ranked.{value_col}").alias("top_values"))


def compute_user_daily_features(
    spark: SparkSession,
    purchases_df: pd.DataFrame,
    items_df: pd.DataFrame,
    snapshot_date: datetime.date,
) -> pd.DataFrame:
    """Computes one User Daily Features row per user with purchase history.

    Args:
        spark: An active SparkSession.
        purchases_df: Purchase Event rows already filtered to
            purchase_time < snapshot_date (see db_io.read_events_before).
        items_df: item_id, category, brand, price columns (current-state
            item catalog join -- see db_io.read_item_catalog).
        snapshot_date: The day this snapshot represents.

    Returns:
        pd.DataFrame: snapshot_date, user_id, preferred_brands,
            avg_purchase_price, historical_category_affinity -- one row per
            user with at least one purchase before snapshot_date.
    """
    if purchases_df.empty:
        return pd.DataFrame(columns=_USER_FEATURE_COLUMNS)

    joined = spark.createDataFrame(purchases_df).join(spark.createDataFrame(items_df), on="item_id", how="inner")

    avg_price = joined.groupBy("user_id").agg(F.avg("price").alias("avg_purchase_price"))
    preferred_brands = _top_n_column(joined, "user_id", "brand", TOP_N).withColumnRenamed(
        "top_values", "preferred_brands"
    )
    category_affinity = _top_n_column(joined, "user_id", "category", TOP_N).withColumnRenamed(
        "top_values", "historical_category_affinity"
    )

    result = avg_price.join(preferred_brands, on="user_id").join(category_affinity, on="user_id")
    result = result.withColumn("snapshot_date", F.lit(snapshot_date))
    result = result.select(*_USER_FEATURE_COLUMNS).toPandas()
    # With Arrow-based conversion enabled, toPandas() represents Spark
    # array<string> columns as numpy.ndarray per row -- psycopg2 can't
    # adapt that to a Postgres ARRAY column, only a plain Python list.
    result["preferred_brands"] = result["preferred_brands"].apply(list)
    result["historical_category_affinity"] = result["historical_category_affinity"].apply(list)
    return result


def _attributed_purchase_counts(
    spark: SparkSession, clicks: SparkDataFrame, purchases_df: pd.DataFrame
) -> SparkDataFrame:
    """Per-candidate count of purchases attributed to a click within the window.

    A purchase is attributed if a click on the same item by the same user
    exists within ATTRIBUTION_WINDOW_HOURS before it (data-flow.md's
    Attribution section). Deduplicated per purchase (order_id), not per
    click, so one purchase never counts twice even if it matches multiple
    qualifying clicks.
    """
    schema = "candidate_item_id string, attributed_purchases long"
    if purchases_df.empty:
        return spark.createDataFrame([], schema=schema)

    p = spark.createDataFrame(purchases_df).alias("p")
    c = clicks.alias("c")
    window_expr = F.expr(f"INTERVAL {ATTRIBUTION_WINDOW_HOURS} HOURS")
    joined = p.join(
        c,
        (F.col("p.user_id") == F.col("c.user_id"))
        & (F.col("p.item_id") == F.col("c.candidate_item_id"))
        & (F.col("c.click_time") <= F.col("p.purchase_time"))
        & (F.col("c.click_time") >= F.col("p.purchase_time") - window_expr),
        how="inner",
    )
    attributed = joined.select(
        F.col("p.order_id").alias("order_id"), F.col("p.item_id").alias("candidate_item_id")
    ).dropDuplicates(["order_id"])
    return attributed.groupBy("candidate_item_id").agg(F.count("*").alias("attributed_purchases"))


def compute_candidate_daily_features(
    spark: SparkSession,
    impressions_df: pd.DataFrame,
    clicks_df: pd.DataFrame,
    purchases_df: pd.DataFrame,
    snapshot_date: datetime.date,
) -> pd.DataFrame:
    """Computes one Candidate Daily Features row per candidate with impression history.

    Args:
        spark: An active SparkSession.
        impressions_df: Impression Event rows already filtered to
            timestamp < snapshot_date.
        clicks_df: Click Event rows already filtered to click_time < snapshot_date.
        purchases_df: Purchase Event rows already filtered to
            purchase_time < snapshot_date.
        snapshot_date: The day this snapshot represents.

    Returns:
        pd.DataFrame: snapshot_date, candidate_id, recommendation_ctr,
            recommendation_cvr, recommendation_impressions -- one row per
            candidate with at least one impression before snapshot_date.
    """
    if impressions_df.empty:
        return pd.DataFrame(columns=_CANDIDATE_FEATURE_COLUMNS)

    impressions = spark.createDataFrame(impressions_df)
    impression_counts = impressions.groupBy("candidate_item_id").agg(
        F.count("*").alias("recommendation_impressions")
    )

    if clicks_df.empty:
        click_counts = spark.createDataFrame([], schema="candidate_item_id string, clicks long")
        attributed_counts = spark.createDataFrame(
            [], schema="candidate_item_id string, attributed_purchases long"
        )
    else:
        clicks = spark.createDataFrame(clicks_df)
        click_counts = clicks.groupBy("candidate_item_id").agg(F.count("*").alias("clicks"))
        attributed_counts = _attributed_purchase_counts(spark, clicks, purchases_df)

    result = (
        impression_counts.join(click_counts, on="candidate_item_id", how="left")
        .join(attributed_counts, on="candidate_item_id", how="left")
        .fillna({"clicks": 0, "attributed_purchases": 0})
    )
    result = result.withColumn(
        "recommendation_ctr", F.col("clicks") / F.col("recommendation_impressions")
    ).withColumn(
        "recommendation_cvr",
        F.when(F.col("clicks") > 0, F.col("attributed_purchases") / F.col("clicks")).otherwise(F.lit(0.0)),
    )
    result = result.withColumn("snapshot_date", F.lit(snapshot_date)).withColumnRenamed(
        "candidate_item_id", "candidate_id"
    )
    return result.select(*_CANDIDATE_FEATURE_COLUMNS).toPandas()
