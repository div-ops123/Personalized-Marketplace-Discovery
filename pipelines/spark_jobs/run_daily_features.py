"""Entrypoint: compute one day's User Daily Features and Candidate Daily
Features from raw events, validate, and write to the warehouse.

Run standalone (bypassing Airflow) with the docker-compose Postgres
warehouse already up, after data_gen/generate_events.py has populated the
raw event tables:

    WAREHOUSE_BACKEND=postgres uv run python pipelines/spark_jobs/run_daily_features.py --snapshot-date 2025-02-01

Airflow (pipelines/dags/daily_feature_pipeline_dag.py) shells out to this
same CLI, inside the isolated pipeline-venv (see infra/airflow/Dockerfile)
-- Airflow's own environment doesn't have pandas/pyspark installed.
"""

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

from common.warehouse import get_engine
from pipelines.spark_jobs.daily_features import compute_candidate_daily_features, compute_user_daily_features
from pipelines.spark_jobs.db_io import (
    create_feature_tables,
    read_events_before,
    read_item_catalog,
    write_candidate_features,
    write_user_features,
)
from pipelines.spark_jobs.spark_session import get_spark_session
from pipelines.spark_jobs.validation import validate_candidate_features, validate_user_features

logger = logging.getLogger(__name__)


class FeatureValidationError(Exception):
    """Raised when a day's computed features fail validation -- the write never runs."""


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for a single day's feature computation.

    Returns:
        argparse.Namespace: Parsed snapshot_date.
    """
    parser = argparse.ArgumentParser(
        description="Compute one day's User/Candidate Daily Features from raw events."
    )
    parser.add_argument("--snapshot-date", type=datetime.date.fromisoformat, required=True)
    return parser.parse_args()


def main(snapshot_date: datetime.date) -> None:
    """Computes, validates, and writes one day's Daily Features.

    Args:
        snapshot_date: Aggregates all raw events strictly before this day's
            start (through snapshot_date - 1); see daily_features.py's
            module docstring for why this makes the downstream
            point-in-time join leak-proof.

    Raises:
        FeatureValidationError: If either table's rows fail validation.
            The write never runs in that case.
    """
    start = time.perf_counter()
    cutoff = datetime.datetime.combine(snapshot_date, datetime.time.min)
    logger.info("Starting daily feature computation: snapshot_date=%s", snapshot_date)

    engine = get_engine()
    impressions_df = read_events_before(engine, "impression_events", "timestamp", cutoff)
    clicks_df = read_events_before(engine, "click_events", "click_time", cutoff)
    purchases_df = read_events_before(engine, "purchase_events", "purchase_time", cutoff)
    items_df = read_item_catalog(engine)
    logger.info(
        "Loaded raw events before %s: impressions=%d clicks=%d purchases=%d",
        snapshot_date,
        len(impressions_df),
        len(clicks_df),
        len(purchases_df),
    )

    expected_active_users = purchases_df["user_id"].nunique() if not purchases_df.empty else 0
    expected_active_candidates = (
        impressions_df["candidate_item_id"].nunique() if not impressions_df.empty else 0
    )

    spark = get_spark_session()
    try:
        user_features = compute_user_daily_features(spark, purchases_df, items_df, snapshot_date)
        candidate_features = compute_candidate_daily_features(
            spark, impressions_df, clicks_df, purchases_df, snapshot_date
        )
    finally:
        spark.stop()

    failures = validate_user_features(user_features, expected_active_users) + validate_candidate_features(
        candidate_features, expected_active_candidates
    )
    if failures:
        for failure in failures:
            logger.error("Validation failure: %s", failure)
        raise FeatureValidationError(
            f"{len(failures)} validation check(s) failed for snapshot_date={snapshot_date}; not writing."
        )

    create_feature_tables(engine)
    write_user_features(user_features, snapshot_date, engine)
    write_candidate_features(candidate_features, snapshot_date, engine)

    elapsed = time.perf_counter() - start
    logger.info(
        "Daily feature computation complete in %.1fs. user_rows=%d candidate_rows=%d",
        elapsed,
        len(user_features),
        len(candidate_features),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(override=True)
    args = parse_args()
    main(args.snapshot_date)
