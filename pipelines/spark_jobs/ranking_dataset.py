"""Spark transform for the Ranking Dataset Builder (data-flow.md's
"Two Independent Dataset Builders" -> Ranking Dataset Builder).

One row per impression (already scoped to the 20 exposed items). label = 1
if the impression's own click led to a purchase within the attribution
window (data-flow.md's Attribution section), else 0 -- covers both
clicked-but-not-purchased and shown-but-not-clicked. This attribution is
computed per-impression (via recommendation_impression_id), unlike
daily_features.py's _attributed_purchase_counts, which aggregates
attributed purchases per candidate across every anchor it's ever been
shown under -- the two answer different questions and are intentionally
separate, though both use the same ATTRIBUTION_WINDOW_HOURS constant so
they can't drift apart.
"""

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from common.constants import ATTRIBUTION_WINDOW_HOURS

_OUTPUT_COLUMNS = [
    "recommendation_impression_id",
    "candidate_item_id",
    "user_id",
    "anchor_item_id",
    "timestamp",
    "device",
    "country",
    "position",
    "retrieval_similarity_score",
    "user_preferred_brands",
    "user_avg_purchase_price",
    "user_historical_category_affinity",
    "anchor_category",
    "anchor_subcategory",
    "anchor_brand",
    "anchor_price",
    "candidate_category",
    "candidate_subcategory",
    "candidate_brand",
    "candidate_price",
    "candidate_recommendation_ctr",
    "candidate_recommendation_cvr",
    "candidate_recommendation_impressions",
    "same_brand",
    "same_category",
    "same_subcategory",
    "price_ratio",
    "price_diff",
    "label",
]

# Public: dataset_db_io's chunked writer needs this to fix up array<...>
# columns the same way per-chunk that build_ranking_dataset does for its
# single-shot toPandas() below.
NULLABLE_ARRAY_COLUMNS = ["user_preferred_brands", "user_historical_category_affinity"]


def _label_attributed_purchases(
    spark: SparkSession, impressions: SparkDataFrame, clicks_df: pd.DataFrame, purchases_df: pd.DataFrame
) -> SparkDataFrame:
    """Adds a 0/1 label column: was this impression's candidate purchased via its click?

    A purchase is attributed if it happened within ATTRIBUTION_WINDOW_HOURS
    after the click on this exact impression (same recommendation_impression_id
    + candidate_item_id) by the same user on the same item.
    """
    if clicks_df.empty or purchases_df.empty:
        return impressions.withColumn("label", F.lit(0))

    clicks = spark.createDataFrame(clicks_df).select(
        "recommendation_impression_id", "candidate_item_id", "click_time"
    )
    purchases = spark.createDataFrame(purchases_df).select("user_id", "item_id", "purchase_time")

    with_click = impressions.join(clicks, on=["recommendation_impression_id", "candidate_item_id"], how="left")

    window_expr = F.expr(f"INTERVAL {ATTRIBUTION_WINDOW_HOURS} HOURS")
    b = with_click.alias("b")
    p = purchases.alias("p")
    attributed_keys = (
        b.join(
            p,
            (F.col("b.user_id") == F.col("p.user_id"))
            & (F.col("b.candidate_item_id") == F.col("p.item_id"))
            & F.col("b.click_time").isNotNull()
            & (F.col("p.purchase_time") >= F.col("b.click_time"))
            & (F.col("p.purchase_time") <= F.col("b.click_time") + window_expr),
            how="left_semi",
        )
        .select("recommendation_impression_id", "candidate_item_id")
        .distinct()
        .withColumn("label", F.lit(1))
    )

    labeled = with_click.join(
        attributed_keys, on=["recommendation_impression_id", "candidate_item_id"], how="left"
    ).fillna({"label": 0})
    return labeled.drop("click_time")


def _asof_join(
    base: SparkDataFrame,
    snapshot: SparkDataFrame,
    base_key: str,
    snapshot_key: str,
    feature_columns: list[str],
    prefix: str,
) -> SparkDataFrame:
    """Left-joins base to the latest snapshot row strictly before base's timestamp.

    See data-flow.md's Point-in-Time Correctness section: MAX(snapshot_date)
    < T, strict inequality. The date filter lives inside the join condition
    (not a post-join .filter()) so a base row with no qualifying snapshot
    row is preserved with null features, instead of being dropped.

    Args:
        base: Must have base_key, "recommendation_impression_id",
            "candidate_item_id", and "timestamp" columns.
        snapshot: A historized snapshot table with a "snapshot_date" column.
        base_key: Join column name on base.
        snapshot_key: Join column name on snapshot.
        feature_columns: snapshot columns to bring over, renamed
            "{prefix}_{column}".
        prefix: "user" or "candidate".

    Returns:
        SparkDataFrame: base's columns plus the prefixed feature columns
            (null where no qualifying snapshot exists).
    """
    snap = snapshot.select(snapshot_key, "snapshot_date", *feature_columns)
    for column in feature_columns:
        snap = snap.withColumnRenamed(column, f"{prefix}_{column}")

    b = base.alias("b")
    s = snap.alias("s")
    join_cond = (F.col(f"b.{base_key}") == F.col(f"s.{snapshot_key}")) & (
        F.col("s.snapshot_date") < F.col("b.timestamp")
    )
    joined = b.join(s, join_cond, how="left")

    window = Window.partitionBy("recommendation_impression_id", "candidate_item_id").orderBy(
        F.col("snapshot_date").desc_nulls_last()
    )
    joined = joined.withColumn("rn", F.row_number().over(window)).filter(F.col("rn") == 1)

    prefixed = [f"s.{prefix}_{column}" for column in feature_columns]
    return joined.select("b.*", *prefixed)


def _item_features(spark: SparkSession, items_df: pd.DataFrame, prefix: str) -> SparkDataFrame:
    """Item catalog category/subcategory/brand/price, columns prefixed for a join role."""
    df = spark.createDataFrame(items_df.drop(columns=["tags", "image_embedding", "text_embedding"]))
    for column in ["category", "subcategory", "brand", "price"]:
        df = df.withColumnRenamed(column, f"{prefix}_{column}")
    return df


def build_ranking_spark_df(
    spark: SparkSession,
    impressions_df: pd.DataFrame,
    clicks_df: pd.DataFrame,
    purchases_df: pd.DataFrame,
    items_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    candidate_features_df: pd.DataFrame,
) -> SparkDataFrame | None:
    """Builds the full Ranking Dataset as a Spark DataFrame (no driver collect).

    Same transform as build_ranking_dataset, but stops short of toPandas()
    -- run_dataset_builders.py streams this back to the driver in bounded
    chunks via dataset_db_io's chunked writer instead of collecting the
    whole result at once.

    Args:
        spark: An active SparkSession.
        impressions_df: The full impression_events log (see
            dataset_db_io.read_all_impressions).
        clicks_df: The full click_events log.
        purchases_df: The full purchase_events log.
        items_df: The full item catalog (see dataset_db_io.read_full_item_catalog).
        user_features_df: The full User Daily Features history (see
            dataset_db_io.read_user_daily_features).
        candidate_features_df: The full Candidate Daily Features history (see
            dataset_db_io.read_candidate_daily_features).

    Returns:
        SparkDataFrame | None: One row per impression -- user/anchor/
            candidate features, cross features, label. User and candidate
            historical features are null where no snapshot exists yet
            before that impression (new users / a candidate's first-ever
            impression). None if there are no impressions yet.
    """
    if impressions_df.empty:
        return None

    impressions = spark.createDataFrame(impressions_df)
    labeled = _label_attributed_purchases(spark, impressions, clicks_df, purchases_df)

    if user_features_df.empty:
        with_user = labeled.withColumn("user_preferred_brands", F.lit(None).cast("array<string>"))
        with_user = with_user.withColumn("user_avg_purchase_price", F.lit(None).cast("double"))
        with_user = with_user.withColumn("user_historical_category_affinity", F.lit(None).cast("array<string>"))
    else:
        user_features = spark.createDataFrame(user_features_df)
        with_user = _asof_join(
            labeled,
            user_features,
            base_key="user_id",
            snapshot_key="user_id",
            feature_columns=["preferred_brands", "avg_purchase_price", "historical_category_affinity"],
            prefix="user",
        )

    if candidate_features_df.empty:
        with_candidate = with_user.withColumn("candidate_recommendation_ctr", F.lit(None).cast("double"))
        with_candidate = with_candidate.withColumn("candidate_recommendation_cvr", F.lit(None).cast("double"))
        with_candidate = with_candidate.withColumn(
            "candidate_recommendation_impressions", F.lit(None).cast("int")
        )
    else:
        candidate_features = spark.createDataFrame(candidate_features_df)
        with_candidate = _asof_join(
            with_user,
            candidate_features,
            base_key="candidate_item_id",
            snapshot_key="candidate_id",
            feature_columns=["recommendation_ctr", "recommendation_cvr", "recommendation_impressions"],
            prefix="candidate",
        )

    anchor_items = _item_features(spark, items_df, "anchor")
    candidate_items = _item_features(spark, items_df, "candidate")

    with_items = (
        with_candidate.join(anchor_items, with_candidate["anchor_item_id"] == anchor_items["item_id"], "inner")
        .drop(anchor_items["item_id"])
        .join(
            candidate_items,
            with_candidate["candidate_item_id"] == candidate_items["item_id"],
            "inner",
        )
        .drop(candidate_items["item_id"])
    )

    with_cross = (
        with_items.withColumn("same_brand", (F.col("anchor_brand") == F.col("candidate_brand")).cast("int"))
        .withColumn("same_category", (F.col("anchor_category") == F.col("candidate_category")).cast("int"))
        .withColumn(
            "same_subcategory", (F.col("anchor_subcategory") == F.col("candidate_subcategory")).cast("int")
        )
        .withColumn("price_ratio", F.col("candidate_price") / F.col("anchor_price"))
        .withColumn("price_diff", F.abs(F.col("candidate_price") - F.col("anchor_price")))
    )

    return with_cross.select(*_OUTPUT_COLUMNS)


def build_ranking_dataset(
    spark: SparkSession,
    impressions_df: pd.DataFrame,
    clicks_df: pd.DataFrame,
    purchases_df: pd.DataFrame,
    items_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    candidate_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Builds the full Ranking Dataset from raw events, the item catalog, and daily features.

    Args:
        spark: An active SparkSession.
        impressions_df: The full impression_events log (see
            dataset_db_io.read_all_impressions).
        clicks_df: The full click_events log.
        purchases_df: The full purchase_events log.
        items_df: The full item catalog (see dataset_db_io.read_full_item_catalog).
        user_features_df: The full User Daily Features history (see
            dataset_db_io.read_user_daily_features).
        candidate_features_df: The full Candidate Daily Features history (see
            dataset_db_io.read_candidate_daily_features).

    Returns:
        pd.DataFrame: One row per impression -- user/anchor/candidate
            features, cross features, label. User and candidate historical
            features are null where no snapshot exists yet before that
            impression (new users / a candidate's first-ever impression).
            Empty (with the right columns) if there are no impressions yet.
    """
    spark_df = build_ranking_spark_df(
        spark, impressions_df, clicks_df, purchases_df, items_df, user_features_df, candidate_features_df
    )
    if spark_df is None:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    result_df = spark_df.toPandas()
    for column in NULLABLE_ARRAY_COLUMNS:
        result_df[column] = result_df[column].apply(lambda v: list(v) if v is not None else None)
    return result_df
