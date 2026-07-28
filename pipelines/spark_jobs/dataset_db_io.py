"""Side-effecting reads/writes for the two dataset builders.

Unlike db_io.py's read_events_before (scoped to events before a single
day's cutoff, for the daily batch job), these readers pull the *entire*
history each run -- the dataset builders rebuild their whole output table
from scratch every time they're invoked, not incrementally.
"""

import logging

import pandas as pd
from sqlalchemy.engine import Engine

from pipelines.spark_jobs.dataset_schema import (
    dataset_metadata,
    ranking_training_examples_table,
    retrieval_training_examples_table,
)

logger = logging.getLogger(__name__)


def read_all_impressions(engine: Engine) -> pd.DataFrame:
    """Reads the full impression_events log.

    Returns:
        pd.DataFrame: All impression_events rows.
    """
    return pd.read_sql("SELECT * FROM impression_events", engine)


def read_all_clicks(engine: Engine) -> pd.DataFrame:
    """Reads the full click_events log.

    Returns:
        pd.DataFrame: All click_events rows.
    """
    return pd.read_sql("SELECT * FROM click_events", engine)


def read_all_purchases(engine: Engine) -> pd.DataFrame:
    """Reads the full purchase_events log.

    Returns:
        pd.DataFrame: All purchase_events rows.
    """
    return pd.read_sql("SELECT * FROM purchase_events", engine)


def read_full_item_catalog(engine: Engine) -> pd.DataFrame:
    """Reads the full item catalog, including embeddings.

    Unlike db_io.read_item_catalog (which only pulls category/brand/price
    for the daily feature job), the dataset builders also need
    subcategory/tags/image_embedding/text_embedding to assemble the
    anchor/candidate feature vectors documented in data-flow.md.

    Returns:
        pd.DataFrame: item_id, category, subcategory, brand, price, tags,
            image_embedding, text_embedding columns.
    """
    return pd.read_sql(
        "SELECT item_id, category, subcategory, brand, price, tags, "
        "image_embedding, text_embedding FROM item_catalog",
        engine,
    )


def read_user_daily_features(engine: Engine) -> pd.DataFrame:
    """Reads the full history of User Daily Features snapshots.

    Returns:
        pd.DataFrame: All user_daily_features rows.
    """
    return pd.read_sql("SELECT * FROM user_daily_features", engine)


def read_candidate_daily_features(engine: Engine) -> pd.DataFrame:
    """Reads the full history of Candidate Daily Features snapshots.

    Returns:
        pd.DataFrame: All candidate_daily_features rows.
    """
    return pd.read_sql("SELECT * FROM candidate_daily_features", engine)


def create_dataset_tables(engine: Engine) -> None:
    """Creates the two dataset tables if they don't already exist.

    Args:
        engine: A SQLAlchemy engine for the target warehouse.
    """
    dataset_metadata.create_all(engine)
    logger.debug("Ensured retrieval_training_examples and ranking_training_examples tables exist.")


def write_retrieval_dataset(df: pd.DataFrame, engine: Engine) -> None:
    """Replaces the full retrieval_training_examples table (truncate-then-insert).

    Not historized like the daily feature tables -- each run rebuilds the
    whole dataset from current history, so a full replace (rather than a
    per-day delete) is what makes a rerun idempotent.

    Args:
        df: Rows matching retrieval_training_examples_table's schema.
        engine: A SQLAlchemy engine for the target warehouse.
    """
    table_name = retrieval_training_examples_table.name
    with engine.begin() as conn:
        conn.execute(retrieval_training_examples_table.delete())
        if not df.empty:
            df.to_sql(table_name, conn, if_exists="append", index=False, method="multi")
    logger.info("Wrote %d rows to %s.", len(df), table_name)


def write_ranking_dataset(df: pd.DataFrame, engine: Engine) -> None:
    """Replaces the full ranking_training_examples table (truncate-then-insert).

    See write_retrieval_dataset's docstring -- same idempotency rationale.

    Args:
        df: Rows matching ranking_training_examples_table's schema.
        engine: A SQLAlchemy engine for the target warehouse.
    """
    table_name = ranking_training_examples_table.name
    with engine.begin() as conn:
        conn.execute(ranking_training_examples_table.delete())
        if not df.empty:
            df.to_sql(table_name, conn, if_exists="append", index=False, method="multi")
    logger.info("Wrote %d rows to %s.", len(df), table_name)
