"""Spark transform for the Retrieval Dataset Builder (data-flow.md's
"Two Independent Dataset Builders" -> Retrieval Dataset Builder).

One row per (recommendation_impression_id, candidate_item_id) -- the same
grain as impression_events itself. label = 1 if the candidate was clicked,
else 0 (exposure-derived negative). In-batch negatives are not materialized
here -- left to the training loop, per build-phases.md step 14.
"""

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common.taxonomy import price_tier

_OUTPUT_COLUMNS = [
    "recommendation_impression_id",
    "candidate_item_id",
    "anchor_item_id",
    "anchor_category",
    "anchor_subcategory",
    "anchor_brand",
    "anchor_price_tier",
    "anchor_tags",
    "anchor_image_embedding",
    "anchor_text_embedding",
    "candidate_category",
    "candidate_subcategory",
    "candidate_brand",
    "candidate_price_tier",
    "candidate_tags",
    "candidate_image_embedding",
    "candidate_text_embedding",
    "label",
]

# Public: dataset_db_io's chunked writer needs this to fix up array<...>
# columns the same way per-chunk that build_retrieval_dataset does for its
# single-shot toPandas() below.
ARRAY_COLUMNS = [
    "anchor_tags",
    "anchor_image_embedding",
    "anchor_text_embedding",
    "candidate_tags",
    "candidate_image_embedding",
    "candidate_text_embedding",
]


def _item_features(spark: SparkSession, items_df: pd.DataFrame, prefix: str) -> SparkDataFrame:
    """Item catalog rows with price_tier added and columns prefixed for a join role.

    Args:
        spark: An active SparkSession.
        items_df: item_id, category, subcategory, brand, price, tags,
            image_embedding, text_embedding columns (see
            dataset_db_io.read_full_item_catalog).
        prefix: "anchor" or "candidate" -- every non-join column is
            renamed "{prefix}_{column}".

    Returns:
        SparkDataFrame: item_id (unprefixed, used as the join key) plus
            prefixed category/subcategory/brand/price_tier/tags/
            image_embedding/text_embedding columns.
    """
    with_tier = items_df.copy()
    with_tier["price_tier"] = [price_tier(p, c) for p, c in zip(with_tier["price"], with_tier["category"])]
    with_tier = with_tier.drop(columns=["price"])

    df = spark.createDataFrame(with_tier)
    for column in ["category", "subcategory", "brand", "price_tier", "tags", "image_embedding", "text_embedding"]:
        df = df.withColumnRenamed(column, f"{prefix}_{column}")
    return df


def build_retrieval_spark_df(
    spark: SparkSession,
    impressions_df: pd.DataFrame,
    clicks_df: pd.DataFrame,
    items_df: pd.DataFrame,
) -> SparkDataFrame | None:
    """Builds the full Retrieval Dataset as a Spark DataFrame (no driver collect).

    Same transform as build_retrieval_dataset, but stops short of
    toPandas() -- run_dataset_builders.py streams this back to the driver
    in bounded chunks via dataset_db_io's chunked writer instead of
    collecting the whole (multi-GB, once real event volume is involved)
    result at once.

    Args:
        spark: An active SparkSession.
        impressions_df: The full impression_events log (see
            dataset_db_io.read_all_impressions) -- not filtered to a time
            window; this builder rebuilds the whole dataset each run.
        clicks_df: The full click_events log.
        items_df: The full item catalog, including embeddings (see
            dataset_db_io.read_full_item_catalog).

    Returns:
        SparkDataFrame | None: One row per (recommendation_impression_id,
            candidate_item_id) -- anchor_features, candidate_features,
            label. None if there are no impressions yet.
    """
    if impressions_df.empty:
        return None

    impressions = spark.createDataFrame(impressions_df)
    clicks = spark.createDataFrame(clicks_df) if not clicks_df.empty else None

    if clicks is None:
        labeled = impressions.withColumn("label", F.lit(0))
    else:
        click_keys = clicks.select("recommendation_impression_id", "candidate_item_id").withColumn(
            "clicked", F.lit(1)
        )
        labeled = impressions.join(
            click_keys, on=["recommendation_impression_id", "candidate_item_id"], how="left"
        ).withColumn("label", F.when(F.col("clicked") == 1, 1).otherwise(0))

    anchor_items = _item_features(spark, items_df, "anchor")
    candidate_items = _item_features(spark, items_df, "candidate")

    result = (
        labeled.join(
            anchor_items, labeled["anchor_item_id"] == anchor_items["item_id"], how="inner"
        )
        .drop(anchor_items["item_id"])
        .join(candidate_items, labeled["candidate_item_id"] == candidate_items["item_id"], how="inner")
        .drop(candidate_items["item_id"])
    )

    return result.select(*_OUTPUT_COLUMNS)


def build_retrieval_dataset(
    spark: SparkSession,
    impressions_df: pd.DataFrame,
    clicks_df: pd.DataFrame,
    items_df: pd.DataFrame,
) -> pd.DataFrame:
    """Builds the full Retrieval Dataset from raw events and the item catalog.

    Args:
        spark: An active SparkSession.
        impressions_df: The full impression_events log (see
            dataset_db_io.read_all_impressions) -- not filtered to a time
            window; this builder rebuilds the whole dataset each run.
        clicks_df: The full click_events log.
        items_df: The full item catalog, including embeddings (see
            dataset_db_io.read_full_item_catalog).

    Returns:
        pd.DataFrame: One row per (recommendation_impression_id,
            candidate_item_id) -- anchor_features, candidate_features,
            label. Empty (with the right columns) if there are no
            impressions yet.
    """
    spark_df = build_retrieval_spark_df(spark, impressions_df, clicks_df, items_df)
    if spark_df is None:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    result_df = spark_df.toPandas()
    # With Arrow-based conversion enabled, toPandas() represents Spark
    # array<...> columns as numpy.ndarray per row -- psycopg2 can't adapt
    # that to a Postgres ARRAY column, only a plain Python list.
    for column in ARRAY_COLUMNS:
        result_df[column] = result_df[column].apply(lambda v: list(v) if v is not None else None)
    return result_df
