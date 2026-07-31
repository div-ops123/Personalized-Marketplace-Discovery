"""Reads for the training scripts -- thin pd.read_sql wrappers over the two
tables pipelines/spark_jobs/run_dataset_builders.py already builds, plus
one raw-event lookup needed only because retrieval_training_examples has
no timestamp column of its own (see retrieval_train.py).
"""

import logging

import pandas as pd
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def read_retrieval_training_examples(engine: Engine) -> pd.DataFrame:
    """Reads the full retrieval_training_examples table.

    Returns:
        pd.DataFrame: All rows, per pipelines/spark_jobs/dataset_schema.py's
            retrieval_training_examples_table columns.
    """
    return pd.read_sql("SELECT * FROM retrieval_training_examples", engine)


def read_ranking_training_examples(engine: Engine) -> pd.DataFrame:
    """Reads the full ranking_training_examples table.

    Returns:
        pd.DataFrame: All rows, per pipelines/spark_jobs/dataset_schema.py's
            ranking_training_examples_table columns.
    """
    return pd.read_sql("SELECT * FROM ranking_training_examples", engine)


def read_impression_timestamps(engine: Engine) -> pd.DataFrame:
    """Reads one timestamp per impression, for splits.py's temporal_split.

    retrieval_training_examples has no timestamp column (it's a pure
    item-pair table -- see dataset_schema.py) -- this is joined back in
    purely to decide train/val/test membership, then dropped again before
    feature encoding.

    Returns:
        pd.DataFrame: recommendation_impression_id, timestamp.
    """
    return pd.read_sql(
        "SELECT DISTINCT recommendation_impression_id, timestamp FROM impression_events",
        engine,
    )


def read_user_daily_features(engine: Engine) -> pd.DataFrame:
    """Reads the full, historized user_daily_features table.

    Returns every snapshot_date's rows -- callers wanting only the latest
    snapshot per user (e.g. serving/export_feature_snapshot.py) must dedupe
    themselves; training never reads this table directly (the offline
    dataset builder already point-in-time-joined it into
    ranking_training_examples).

    Returns:
        pd.DataFrame: snapshot_date, user_id, preferred_brands,
            avg_purchase_price, historical_category_affinity columns,
            per pipelines/spark_jobs/features_schema.py.
    """
    return pd.read_sql(
        "SELECT snapshot_date, user_id, preferred_brands, avg_purchase_price, "
        "historical_category_affinity FROM user_daily_features",
        engine,
    )


def read_candidate_daily_features(engine: Engine) -> pd.DataFrame:
    """Reads the full, historized candidate_daily_features table.

    Returns every snapshot_date's rows -- see read_user_daily_features's
    docstring for why dedup-to-latest is left to the caller.

    Returns:
        pd.DataFrame: snapshot_date, candidate_id, recommendation_ctr,
            recommendation_cvr, recommendation_impressions columns, per
            pipelines/spark_jobs/features_schema.py.
    """
    return pd.read_sql(
        "SELECT snapshot_date, candidate_id, recommendation_ctr, recommendation_cvr, "
        "recommendation_impressions FROM candidate_daily_features",
        engine,
    )


def read_full_item_catalog(engine: Engine) -> pd.DataFrame:
    """Reads the full item catalog -- the retrieval eval gallery.

    Mirrors pipelines/spark_jobs/dataset_db_io.py's read_full_item_catalog
    exactly (same columns needed to encode every catalog item through the
    Siamese encoder for Recall@K).

    Returns:
        pd.DataFrame: item_id, category, subcategory, brand, price, tags,
            image_embedding, text_embedding columns.
    """
    return pd.read_sql(
        "SELECT item_id, category, subcategory, brand, price, tags, "
        "image_embedding, text_embedding FROM item_catalog",
        engine,
    )
