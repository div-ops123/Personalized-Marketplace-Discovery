"""Entrypoint: generate and load the item catalog and user population.

Run with the docker-compose Postgres warehouse already up:

    WAREHOUSE_BACKEND=postgres python data_gen/generate_reference_data.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import COLD_START_FRACTION, GLOBAL_SEED, N_ITEMS, N_USERS
from db_writer import create_reference_tables, write_item_catalog, write_users
from item_catalog import build_item_catalog, reserved_cold_start_item_ids
from text_embeddings import embed_descriptions, load_text_encoder
from user_population import build_users

from common.warehouse import get_engine

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for reference data generation.

    Returns:
        argparse.Namespace: Parsed n_items, n_users, and seed.
    """
    parser = argparse.ArgumentParser(
        description="Generate the item catalog and user population reference data."
    )
    parser.add_argument("--n-items", type=int, default=N_ITEMS)
    parser.add_argument("--n-users", type=int, default=N_USERS)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return parser.parse_args()


def main(n_items: int, n_users: int, seed: int) -> None:
    """Generates and loads the Item Catalog and user population.

    Args:
        n_items: Number of catalog items to generate.
        n_users: Number of users to generate.
        seed: The dataset-wide reproducibility seed.
    """
    start = time.perf_counter()
    logger.info(
        "Starting reference data generation: n_items=%d n_users=%d seed=%d", n_items, n_users, seed
    )

    logger.info("Building item catalog...")
    items_df = build_item_catalog(n_items, seed)
    cold_start_ids = reserved_cold_start_item_ids(
        list(items_df["item_id"]), COLD_START_FRACTION, seed
    )
    logger.info(
        "Built %d items, reserved %d for cold-start.", len(items_df), len(cold_start_ids)
    )

    logger.info("Embedding item descriptions...")
    encoder = load_text_encoder()
    embeddings = embed_descriptions(list(items_df["description"]), encoder)
    # psycopg2 can't adapt a raw numpy.ndarray to a Postgres ARRAY column —
    # each row needs to be a plain Python list.
    items_df["text_embedding"] = [vector.tolist() for vector in embeddings]
    logger.info("Embedded %d item descriptions.", len(items_df))

    logger.info("Building user population...")
    users_df = build_users(n_users, seed)
    logger.info("Built %d users.", len(users_df))

    logger.info("Writing reference data to the warehouse...")
    engine = get_engine()
    create_reference_tables(engine)
    write_item_catalog(items_df, engine)
    write_users(users_df, engine)

    elapsed = time.perf_counter() - start
    logger.info("Reference data generation complete in %.1fs.", elapsed)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs one line per Hugging Face Hub HTTP request (model download
    # checks) — noise, not a pipeline milestone.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # override=True: an env var left over from an earlier shell session
    # (e.g. a stale POSTGRES_PORT) should never win over the current .env.
    load_dotenv(override=True)
    args = parse_args()
    main(args.n_items, args.n_users, args.seed)
