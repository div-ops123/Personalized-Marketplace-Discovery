"""Entrypoint: export the latest user/candidate feature snapshot and the
item catalog to local JSON files (docs/build-phases.md Phase 6), so the
Ranking Service and API Gateway never need Postgres live at request time.

Requires the `training` optional-dependency group (`uv sync --extra
training`) and the docker-compose Postgres warehouse populated by
pipelines/spark_jobs/run_daily_features.py and run_dataset_builders.py:

    WAREHOUSE_BACKEND=postgres python serving/export_feature_snapshot.py

No online store (Redis) exists in this repo -- user_daily_features and
candidate_daily_features are historized (one row per entity per
snapshot_date, see pipelines/spark_jobs/features_schema.py). Serving
always wants the freshest row per key, so this script dedupes to the
max-snapshot_date row per user_id/candidate_id -- simpler than
training's point-in-time "< impression_timestamp" join, since there's no
historical request timestamp to join against at serving time.

Writes user_features.json, candidate_features.json, item_catalog.json to
--output-dir (default serving/features/).
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from common.warehouse import get_engine
from serving.constants import DEFAULT_FEATURES_DIR
from training.db_io import read_candidate_daily_features, read_full_item_catalog, read_user_daily_features

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for a feature-snapshot export run.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Export the latest feature snapshot + item catalog to local JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    return parser.parse_args()


def latest_snapshot(df, key_column: str):
    """Dedupes a historized features DataFrame to its latest row per key.

    Args:
        df: A user_daily_features or candidate_daily_features DataFrame,
            with a snapshot_date column.
        key_column: "user_id" or "candidate_id".

    Returns:
        pd.DataFrame: One row per key_column value -- the max-snapshot_date
            row, indexed by key_column's original column (not reset).
    """
    return df.sort_values("snapshot_date").groupby(key_column, as_index=False).tail(1)


def build_user_features(user_df) -> dict:
    """Builds the user_id -> feature dict written to user_features.json.

    Args:
        user_df: latest_snapshot(read_user_daily_features(engine), "user_id").

    Returns:
        dict: user_id -> {avg_purchase_price, preferred_brands,
            historical_category_affinity}.
    """
    return {
        row.user_id: {
            "avg_purchase_price": row.avg_purchase_price,
            "preferred_brands": list(row.preferred_brands),
            "historical_category_affinity": list(row.historical_category_affinity),
        }
        for row in user_df.itertuples()
    }


def build_candidate_features(candidate_df) -> dict:
    """Builds the candidate_id -> feature dict written to candidate_features.json.

    Args:
        candidate_df: latest_snapshot(read_candidate_daily_features(engine), "candidate_id").

    Returns:
        dict: candidate_id -> {recommendation_ctr, recommendation_cvr,
            recommendation_impressions}.
    """
    return {
        row.candidate_id: {
            "recommendation_ctr": row.recommendation_ctr,
            "recommendation_cvr": row.recommendation_cvr,
            "recommendation_impressions": row.recommendation_impressions,
        }
        for row in candidate_df.itertuples()
    }


def build_item_catalog(catalog_df) -> dict:
    """Builds the item_id -> metadata dict written to item_catalog.json.

    Only category/subcategory/brand/price are kept -- tags/image_embedding/
    text_embedding are retrieval-encoder inputs, not needed by the Ranking
    Service (categorical + price features only) or the API Gateway
    (response enrichment: category/brand/price, no images -- see
    docs/build-phases.md's catalog generation step, image left null).

    Args:
        catalog_df: training.db_io.read_full_item_catalog(engine)'s output.

    Returns:
        dict: item_id -> {category, subcategory, brand, price}.
    """
    return {
        row.item_id: {
            "category": row.category,
            "subcategory": row.subcategory,
            "brand": row.brand,
            "price": row.price,
        }
        for row in catalog_df.itertuples()
    }


def main(args: argparse.Namespace) -> None:
    """Exports the feature snapshot and item catalog end to end.

    Args:
        args: Parsed CLI arguments from parse_args().
    """
    start = time.perf_counter()

    engine = get_engine()
    user_df = latest_snapshot(read_user_daily_features(engine), "user_id")
    candidate_df = latest_snapshot(read_candidate_daily_features(engine), "candidate_id")
    catalog_df = read_full_item_catalog(engine)
    logger.info(
        "Loaded snapshot: %d users, %d candidates, %d catalog items",
        len(user_df),
        len(candidate_df),
        len(catalog_df),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "user_features.json").write_text(json.dumps(build_user_features(user_df)))
    (args.output_dir / "candidate_features.json").write_text(json.dumps(build_candidate_features(candidate_df)))
    (args.output_dir / "item_catalog.json").write_text(json.dumps(build_item_catalog(catalog_df)))
    metadata = {
        "user_count": len(user_df),
        "candidate_count": len(candidate_df),
        "item_count": len(catalog_df),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    elapsed = time.perf_counter() - start
    logger.info("Feature snapshot exported in %.1fs -> %s", elapsed, args.output_dir)


if __name__ == "__main__":
    # force=True: see serving/build_retrieval_index.py's __main__ block --
    # other imports may already attach a root logger handler.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    # override=False (the default): see run_dataset_builders.py's docstring --
    # container-injected env vars must win over the bind-mounted .env.
    load_dotenv()
    main(parse_args())
