"""Side-effecting reads/writes for the two dataset builders.

Unlike db_io.py's read_events_before (scoped to events before a single
day's cutoff, for the daily batch job), these readers pull the *entire*
history each run -- the dataset builders rebuild their whole output table
from scratch every time they're invoked, not incrementally.
"""

import logging

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from sqlalchemy import Table, text
from sqlalchemy.engine import Connection, Engine

from pipelines.spark_jobs.dataset_schema import (
    dataset_metadata,
    ranking_training_examples_table,
    retrieval_training_examples_table,
)
from pipelines.spark_jobs.ranking_dataset import NULLABLE_ARRAY_COLUMNS
from pipelines.spark_jobs.retrieval_dataset import ARRAY_COLUMNS

logger = logging.getLogger(__name__)

# Rows per chunk streamed from Spark to Postgres. Bounds peak driver memory
# to roughly chunk_size x row width instead of the full dataset (1.6M+ rows
# x two 384-dim embedding columns for retrieval, several GB) -- see the OOM
# history on the old single toPandas() + to_sql() path.
_DEFAULT_CHUNK_SIZE = 50_000

# Rows per individual INSERT statement (to_sql's own chunksize, separate
# from _DEFAULT_CHUNK_SIZE above). method="multi" builds one multi-row
# VALUES clause per to_sql call with no internal sub-batching unless told
# to -- at chunk_size=50_000 rows x ~17-28 columns, that's 850K+ bound
# parameters in a single statement, well past Postgres's hard 65,535
# parameter-per-query limit. 1,000 rows keeps every table's statement
# comfortably under that regardless of column count.
_SQL_INSERT_CHUNKSIZE = 1_000


def read_all_impressions(engine: Engine, days: int | None = None) -> pd.DataFrame:
    """Reads the impression_events log, optionally capped to the most recent N days.

    Everything downstream (clicks, purchases, item catalog) is left-joined
    onto impressions (see retrieval_dataset.py / ranking_dataset.py), so
    capping impressions alone shrinks the whole build proportionally --
    this is the intended local-dev memory lever, not a data-quality
    concern. The cutoff is relative to the data's own most recent
    timestamp, not wall-clock now(), since backfilled/simulated event
    timestamps aren't tied to the real current date.

    Args:
        engine: A SQLAlchemy engine for the target warehouse.
        days: If set, only rows within this many days of the most recent
            impression timestamp are returned. None reads full history.

    Returns:
        pd.DataFrame: impression_events rows (all, or the last `days`).
    """
    if days is None:
        return pd.read_sql("SELECT * FROM impression_events", engine)
    query = text(
        "SELECT * FROM impression_events "
        "WHERE timestamp >= (SELECT MAX(timestamp) FROM impression_events) - (INTERVAL '1 day' * :days)"
    )
    return pd.read_sql(query, engine, params={"days": days})


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


def _write_chunk(
    rows: list[dict], columns: list[str], array_columns: list[str], table_name: str, conn: Connection
) -> int:
    """Writes one batch of already-collected rows to `table_name` and returns its row count."""
    chunk_df = pd.DataFrame(rows, columns=columns)
    # toLocalIterator() rows carry array<...> columns as plain Python lists
    # already (unlike Arrow-based toPandas(), which yields numpy.ndarray --
    # psycopg2 can't adapt that to a Postgres ARRAY column) -- normalized
    # defensively in case that ever changes.
    for column in array_columns:
        chunk_df[column] = chunk_df[column].apply(lambda v: list(v) if v is not None else None)
    chunk_df.to_sql(
        table_name, conn, if_exists="append", index=False, method="multi", chunksize=_SQL_INSERT_CHUNKSIZE
    )
    return len(chunk_df)


def _write_spark_dataset(
    spark_df: SparkDataFrame | None,
    engine: Engine,
    table: Table,
    array_columns: list[str],
    chunk_size: int,
) -> int:
    """Truncate-then-chunked-insert: replaces `table`'s full contents from `spark_df`.

    Streams `spark_df` back to the driver one partition at a time
    (toLocalIterator, not toPandas()/collect()) and writes it to Postgres
    chunk_size rows at a time, so peak driver memory is bounded by a
    single chunk rather than the whole dataset. The delete and every
    chunk insert share one transaction, preserving the same
    truncate-then-insert idempotency as before -- a rerun still replaces
    the table atomically, just via many smaller inserts instead of one.

    Args:
        spark_df: Rows matching `table`'s schema, or None if there's
            nothing to write (the table is still truncated).
        engine: A SQLAlchemy engine for the target warehouse.
        table: The destination table.
        array_columns: Columns needing the array<...> -> list fix-up.
        chunk_size: Rows per collected batch.

    Returns:
        int: Total rows written.
    """
    rows_written = 0
    with engine.begin() as conn:
        conn.execute(table.delete())
        if spark_df is not None:
            columns = spark_df.columns
            batch: list[dict] = []
            for row in spark_df.toLocalIterator(prefetchPartitions=True):
                batch.append(row.asDict())
                if len(batch) >= chunk_size:
                    rows_written += _write_chunk(batch, columns, array_columns, table.name, conn)
                    batch = []
            if batch:
                rows_written += _write_chunk(batch, columns, array_columns, table.name, conn)
    logger.info("Wrote %d rows to %s.", rows_written, table.name)
    return rows_written


def write_retrieval_dataset(
    spark_df: SparkDataFrame | None, engine: Engine, chunk_size: int = _DEFAULT_CHUNK_SIZE
) -> int:
    """Replaces the full retrieval_training_examples table (truncate-then-chunked-insert).

    Not historized like the daily feature tables -- each run rebuilds the
    whole dataset from current history, so a full replace (rather than a
    per-day delete) is what makes a rerun idempotent.

    Args:
        spark_df: Rows matching retrieval_training_examples_table's schema
            (see retrieval_dataset.build_retrieval_spark_df), or None.
        engine: A SQLAlchemy engine for the target warehouse.
        chunk_size: Rows per collected batch.

    Returns:
        int: Total rows written.
    """
    return _write_spark_dataset(spark_df, engine, retrieval_training_examples_table, ARRAY_COLUMNS, chunk_size)


def write_ranking_dataset(
    spark_df: SparkDataFrame | None, engine: Engine, chunk_size: int = _DEFAULT_CHUNK_SIZE
) -> int:
    """Replaces the full ranking_training_examples table (truncate-then-chunked-insert).

    See write_retrieval_dataset's docstring -- same idempotency rationale.

    Args:
        spark_df: Rows matching ranking_training_examples_table's schema
            (see ranking_dataset.build_ranking_spark_df), or None.
        engine: A SQLAlchemy engine for the target warehouse.
        chunk_size: Rows per collected batch.

    Returns:
        int: Total rows written.
    """
    return _write_spark_dataset(spark_df, engine, ranking_training_examples_table, NULLABLE_ARRAY_COLUMNS, chunk_size)
