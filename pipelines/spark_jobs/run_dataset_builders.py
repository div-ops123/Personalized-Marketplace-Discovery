"""Entrypoint: build the Retrieval and/or Ranking training datasets from
raw events, the item catalog, and the daily feature tables, and write them
to the warehouse.

Unlike run_daily_features.py, this isn't a per-day backfill -- each run
rebuilds the whole dataset table from current history (see
dataset_db_io.py's truncate-then-insert write). No Airflow DAG wraps this
yet (build-phases.md's Phase 4 doesn't call for one); run manually before
a training run, with the docker-compose Postgres warehouse already up and
after run_daily_features.py has been backfilled:

    WAREHOUSE_BACKEND=postgres python pipelines/spark_jobs/run_dataset_builders.py --builder both

For local-PC runs against the full backfilled history, --days caps the
impression history to the most recent N days (see dataset_db_io.py's
read_all_impressions) so the join/collect phases fit in a single local
Spark driver's heap:

    WAREHOUSE_BACKEND=postgres python pipelines/spark_jobs/run_dataset_builders.py --builder both --days 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

from common.warehouse import get_engine
from pipelines.spark_jobs.dataset_db_io import (
    create_dataset_tables,
    read_all_clicks,
    read_all_impressions,
    read_all_purchases,
    read_candidate_daily_features,
    read_full_item_catalog,
    read_user_daily_features,
    write_ranking_dataset,
    write_retrieval_dataset,
)
from pipelines.spark_jobs.ranking_dataset import build_ranking_spark_df
from pipelines.spark_jobs.retrieval_dataset import build_retrieval_spark_df
from pipelines.spark_jobs.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for a dataset builder run.

    Returns:
        argparse.Namespace: Parsed builder selection.
    """
    parser = argparse.ArgumentParser(
        description="Build the Retrieval and/or Ranking training datasets from raw events."
    )
    parser.add_argument("--builder", choices=["retrieval", "ranking", "both"], default="both")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=(
            "Cap impression history to the most recent N days (relative to the data's own "
            "most recent timestamp), shrinking the whole build proportionally. Omit for the "
            "full history (the default)."
        ),
    )
    return parser.parse_args()


def main(builder: str, days: int | None = None) -> None:
    """Builds, and writes, the selected training dataset(s).

    Args:
        builder: "retrieval", "ranking", or "both".
        days: If set, caps impression history to the most recent N days.
    """
    start = time.perf_counter()
    engine = get_engine()
    create_dataset_tables(engine)

    impressions_df = read_all_impressions(engine, days=days)
    clicks_df = read_all_clicks(engine)
    items_df = read_full_item_catalog(engine)
    logger.info(
        "Loaded raw events: impressions=%d clicks=%d items=%d (days=%s)",
        len(impressions_df),
        len(clicks_df),
        len(items_df),
        days if days is not None else "all",
    )

    spark = get_spark_session(app_name="dataset_builders")
    try:
        if builder in ("retrieval", "both"):
            retrieval_spark_df = build_retrieval_spark_df(spark, impressions_df, clicks_df, items_df)
            retrieval_rows = write_retrieval_dataset(retrieval_spark_df, engine)
            logger.info("Retrieval dataset: %d rows.", retrieval_rows)

        if builder in ("ranking", "both"):
            purchases_df = read_all_purchases(engine)
            user_features_df = read_user_daily_features(engine)
            candidate_features_df = read_candidate_daily_features(engine)
            ranking_spark_df = build_ranking_spark_df(
                spark,
                impressions_df,
                clicks_df,
                purchases_df,
                items_df,
                user_features_df,
                candidate_features_df,
            )
            ranking_rows = write_ranking_dataset(ranking_spark_df, engine)
            logger.info("Ranking dataset: %d rows.", ranking_rows)
    finally:
        spark.stop()

    elapsed = time.perf_counter() - start
    logger.info("Dataset build complete in %.1fs.", elapsed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # override=False (the default): see run_daily_features.py's docstring --
    # inside a container, docker-compose's injected env vars must win over
    # the bind-mounted .env's host-only values.
    load_dotenv()
    args = parse_args()
    main(args.builder, args.days)
