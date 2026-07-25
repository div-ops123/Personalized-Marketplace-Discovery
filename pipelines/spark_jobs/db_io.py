"""Side-effecting reads/writes for the daily feature aggregation job."""

import datetime
import logging

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from pipelines.spark_jobs.features_schema import (
    candidate_daily_features_table,
    features_metadata,
    user_daily_features_table,
)

logger = logging.getLogger(__name__)


def read_events_before(
    engine: Engine, table_name: str, timestamp_column: str, cutoff: datetime.date
) -> pd.DataFrame:
    """Reads all rows from a raw event table strictly before a cutoff.

    Args:
        engine: A SQLAlchemy engine for the target warehouse.
        table_name: The raw event table to read (table_name/timestamp_column
            are internal constants, never user input -- safe to interpolate).
        timestamp_column: The table's event-time column.
        cutoff: Rows with timestamp_column >= cutoff are excluded.

    Returns:
        pd.DataFrame: Matching rows.
    """
    query = f"SELECT * FROM {table_name} WHERE {timestamp_column} < %(cutoff)s"
    return pd.read_sql(query, engine, params={"cutoff": cutoff})


def read_item_catalog(engine: Engine) -> pd.DataFrame:
    """Reads item_id/category/brand/price -- all this job needs from the catalog.

    Args:
        engine: A SQLAlchemy engine for the target warehouse.

    Returns:
        pd.DataFrame: item_id, category, brand, price columns.
    """
    return pd.read_sql("SELECT item_id, category, brand, price FROM item_catalog", engine)


def create_feature_tables(engine: Engine) -> None:
    """Creates the daily feature tables if they don't already exist.

    Args:
        engine: A SQLAlchemy engine for the target warehouse.
    """
    features_metadata.create_all(engine)
    logger.debug("Ensured user_daily_features and candidate_daily_features tables exist.")


def write_user_features(df: pd.DataFrame, snapshot_date: datetime.date, engine: Engine) -> None:
    """Replaces one day's User Daily Features rows (delete-then-insert).

    Deleting the day's existing rows before inserting makes a rerun of the
    same snapshot_date (Airflow retry, or `airflow tasks clear`) idempotent
    instead of hitting the (snapshot_date, user_id) primary key on a second
    insert. Delete and insert run in one transaction.

    Args:
        df: Rows matching user_daily_features_table's schema, all with the
            same snapshot_date.
        snapshot_date: The day being (re)written.
        engine: A SQLAlchemy engine for the target warehouse.
    """
    table_name = user_daily_features_table.name
    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {table_name} WHERE snapshot_date = :snapshot_date"),
            {"snapshot_date": snapshot_date},
        )
        if not df.empty:
            df.to_sql(table_name, conn, if_exists="append", index=False, method="multi")
    logger.info("Wrote %d rows to %s for snapshot_date=%s.", len(df), table_name, snapshot_date)


def write_candidate_features(df: pd.DataFrame, snapshot_date: datetime.date, engine: Engine) -> None:
    """Replaces one day's Candidate Daily Features rows (delete-then-insert).

    See write_user_features's docstring -- same idempotency rationale.

    Args:
        df: Rows matching candidate_daily_features_table's schema, all with
            the same snapshot_date.
        snapshot_date: The day being (re)written.
        engine: A SQLAlchemy engine for the target warehouse.
    """
    table_name = candidate_daily_features_table.name
    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {table_name} WHERE snapshot_date = :snapshot_date"),
            {"snapshot_date": snapshot_date},
        )
        if not df.empty:
            df.to_sql(table_name, conn, if_exists="append", index=False, method="multi")
    logger.info("Wrote %d rows to %s for snapshot_date=%s.", len(df), table_name, snapshot_date)
