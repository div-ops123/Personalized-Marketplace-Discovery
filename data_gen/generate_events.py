"""Entrypoint: simulate and load the raw impression/click/purchase event log.

Run with the docker-compose Postgres warehouse already up, after
generate_reference_data.py has populated item_catalog and users:

    WAREHOUSE_BACKEND=postgres python data_gen/generate_events.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from candidate_selection import build_exclusion_mask, build_item_embedding_matrix, score_candidates
from config import (
    CANDIDATE_POOL_SIZE,
    COLD_START_FRACTION,
    GLOBAL_SEED,
    SIMULATION_START_DATE,
    SIMULATION_WINDOW_DAYS,
)
from db_writer import create_event_tables, write_clicks, write_impressions, write_purchases
from event_simulator import (
    build_item_lookup,
    pick_anchor_item,
    sample_daily_view_count,
    sample_device,
    sample_country,
    sample_view_timestamp,
    simulate_attributed_purchases,
    simulate_clicks,
    simulate_impressions,
    simulate_organic_purchases,
)
from item_catalog import reserved_cold_start_item_ids
from seeding import rng_for_entity
from user_population import sample_user_affinity

from common.warehouse import get_engine

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for event simulation.

    Returns:
        argparse.Namespace: Parsed window_days and seed.
    """
    parser = argparse.ArgumentParser(description="Simulate the raw impression/click/purchase event log.")
    parser.add_argument("--window-days", type=int, default=SIMULATION_WINDOW_DAYS)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return parser.parse_args()


def load_reference_data(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the item catalog and user population back from the warehouse.

    Ordered by id so cold-start reservation reproduces the exact same
    subset Phase 1 selected (numpy's rng.choice depends on element order).

    Args:
        engine: A SQLAlchemy engine for the target warehouse.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (items_df, users_df).
    """
    items_df = pd.read_sql("SELECT * FROM item_catalog ORDER BY item_id", engine)
    users_df = pd.read_sql("SELECT * FROM users ORDER BY user_id", engine)
    return items_df, users_df


def simulate_window(
    items_df: pd.DataFrame,
    user_ids: list[str],
    cold_start_ids: set[str],
    window_days: int,
    seed: int,
    engine,
) -> dict:
    """Simulates and writes the full event window, one day at a time.

    Args:
        items_df: Full item catalog (including reserved cold-start items —
            needed so the embedding matrix has every candidate available;
            exclusion is enforced by score_candidates/pick_anchor_item).
        user_ids: All user ids in the population.
        cold_start_ids: Item ids that must never appear as anchor or
            candidate.
        window_days: Number of days to simulate.
        seed: The dataset-wide reproducibility seed.
        engine: A SQLAlchemy engine for the target warehouse.

    Returns:
        dict: Running totals (impressions, clicks, purchases).
    """
    eligible_items_df = items_df[~items_df["item_id"].isin(cold_start_ids)].reset_index(drop=True)
    item_ids, embedding_matrix = build_item_embedding_matrix(items_df)
    exclusion_mask = build_exclusion_mask(item_ids, cold_start_ids)
    item_category, item_brand = build_item_lookup(items_df)
    # Affinity is a fixed per-user latent trait (keyed only on user_id +
    # seed, not day) -- compute once rather than once per (user, day).
    user_affinities = {user_id: sample_user_affinity(user_id, seed) for user_id in user_ids}

    totals = {"impressions": 0, "clicks": 0, "purchases": 0}

    for day_offset in range(window_days):
        day = SIMULATION_START_DATE + pd.Timedelta(days=day_offset)
        day_impressions, day_clicks, day_purchases = [], [], []

        for user_id in user_ids:
            user_affinity = user_affinities[user_id]
            day_rng = rng_for_entity(f"{user_id}-{day.isoformat()}", seed)
            n_views = sample_daily_view_count(day_rng)

            for view_index in range(n_views):
                view_rng = rng_for_entity(f"{user_id}-{day.isoformat()}-{view_index}", seed)
                anchor_id = pick_anchor_item(eligible_items_df, user_affinity, view_rng)
                candidates = score_candidates(
                    anchor_id, item_ids, embedding_matrix, exclusion_mask, view_rng, CANDIDATE_POOL_SIZE
                )
                timestamp = sample_view_timestamp(day, view_rng)
                impression_id = f"impr_{user_id}_{day.isoformat()}_{view_index:03d}"
                device = sample_device(view_rng)
                country = sample_country(view_rng)

                impressions = simulate_impressions(
                    anchor_id, candidates, timestamp, user_id, impression_id, device, country
                )
                clicks = simulate_clicks(impressions, item_category, item_brand, user_affinity, view_rng)
                purchases = simulate_attributed_purchases(
                    clicks, item_category, item_brand, user_affinity, view_rng, seed
                )

                day_impressions.append(impressions)
                day_clicks.append(clicks)
                day_purchases.append(purchases)

            organic = simulate_organic_purchases(user_id, day, eligible_items_df, user_affinity, day_rng, seed)
            day_purchases.append(organic)

        day_impressions_df = pd.concat(day_impressions, ignore_index=True) if day_impressions else pd.DataFrame()
        day_clicks_df = pd.concat(day_clicks, ignore_index=True) if day_clicks else pd.DataFrame()
        day_purchases_df = pd.concat(day_purchases, ignore_index=True) if day_purchases else pd.DataFrame()

        write_impressions(day_impressions_df, engine)
        write_clicks(day_clicks_df, engine)
        write_purchases(day_purchases_df, engine)

        totals["impressions"] += len(day_impressions_df)
        totals["clicks"] += len(day_clicks_df)
        totals["purchases"] += len(day_purchases_df)

        if (day_offset + 1) % 10 == 0 or day_offset == window_days - 1:
            logger.info(
                "Day %d/%d (%s) done. Running totals: impressions=%d clicks=%d purchases=%d",
                day_offset + 1,
                window_days,
                day.isoformat(),
                totals["impressions"],
                totals["clicks"],
                totals["purchases"],
            )

    return totals


def main(window_days: int, seed: int) -> None:
    """Simulates and loads the raw event log.

    Args:
        window_days: Number of days to simulate.
        seed: The dataset-wide reproducibility seed.
    """
    start = time.perf_counter()
    logger.info("Starting event simulation: window_days=%d seed=%d", window_days, seed)

    engine = get_engine()
    items_df, users_df = load_reference_data(engine)
    logger.info("Loaded %d items and %d users from the warehouse.", len(items_df), len(users_df))

    cold_start_ids = reserved_cold_start_item_ids(list(items_df["item_id"]), COLD_START_FRACTION, seed)
    logger.info("Excluding %d reserved cold-start items from simulation.", len(cold_start_ids))

    create_event_tables(engine)
    totals = simulate_window(
        items_df, users_df["user_id"].tolist(), cold_start_ids, window_days, seed, engine
    )

    elapsed = time.perf_counter() - start
    ctr = totals["clicks"] / totals["impressions"] if totals["impressions"] else 0.0
    logger.info(
        "Event simulation complete in %.1fs. impressions=%d clicks=%d purchases=%d ctr=%.3f",
        elapsed,
        totals["impressions"],
        totals["clicks"],
        totals["purchases"],
        ctr,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(override=True)
    args = parse_args()
    main(args.window_days, args.seed)
