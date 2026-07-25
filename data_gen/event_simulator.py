"""Pure event-simulation logic: anchor selection, impressions, clicks, purchases.

Personalization (the viewing user's hidden affinity) drives anchor choice,
click probability, and purchase probability -- never candidate ranking,
which is pure item-item content similarity (see candidate_selection.py).
This mirrors the production architecture's split between a
personalization-free Stage 1 retrieval and a personalized Stage 2 ranking.
"""

import numpy as np
import pandas as pd

from config import (
    AFFINITY_MULTIPLIER_CLIP,
    ATTRIBUTION_WINDOW_HOURS,
    BASE_CLICK_RATE,
    CLICK_TO_PURCHASE_BASE_RATE,
    COUNTRY_DISTRIBUTION,
    DEVICE_DISTRIBUTION,
    ORGANIC_PURCHASE_RATE,
    PDP_VIEWS_PER_USER_PER_DAY_MEAN,
    TOP_K_IMPRESSIONS,
)
from seeding import stable_seed
from taxonomy import ALL_BRANDS, CATEGORIES

_CATEGORY_INDEX = {category: i for i, category in enumerate(CATEGORIES)}
_BRAND_INDEX = {brand: i for i, brand in enumerate(ALL_BRANDS)}

PURCHASE_COLUMNS = ["purchase_time", "user_id", "item_id", "order_id"]


def build_item_lookup(items_df: pd.DataFrame) -> tuple[dict, dict]:
    """Builds item_id -> category and item_id -> brand lookup dicts.

    Args:
        items_df: Item catalog rows with item_id, category, brand columns.

    Returns:
        tuple[dict, dict]: (item_category, item_brand) lookups.
    """
    item_category = dict(zip(items_df["item_id"], items_df["category"]))
    item_brand = dict(zip(items_df["item_id"], items_df["brand"]))
    return item_category, item_brand


def _affinity_weighted_choice(
    item_ids: np.ndarray,
    categories: np.ndarray,
    brands: np.ndarray,
    user_affinity: dict,
    rng: np.random.Generator,
) -> str:
    """Picks one item id, weighted by the user's category/brand affinity."""
    cat_scores = user_affinity["category_affinity"][[_CATEGORY_INDEX[c] for c in categories]]
    brand_scores = user_affinity["brand_affinity"][[_BRAND_INDEX[b] for b in brands]]
    weights = 0.7 * cat_scores + 0.3 * brand_scores
    weights = weights + rng.normal(0.0, 1e-3, size=len(weights))
    weights = np.clip(weights, 1e-9, None)
    weights = weights / weights.sum()
    return rng.choice(item_ids, p=weights)


def affinity_multiplier(user_affinity: dict, category: str, brand: str) -> float:
    """Converts a user's Dirichlet affinity into a click/purchase-rate multiplier.

    A user with exactly average affinity for a category/brand gets a
    multiplier of ~1.0 (Dirichlet mean is 1/n, so the raw value is rescaled
    by n before clipping) -- above-average affinity scales the rate up,
    below-average scales it down.

    Args:
        user_affinity: Output of user_population.sample_user_affinity.
        category: The candidate item's category.
        brand: The candidate item's brand.

    Returns:
        float: A multiplier clipped to AFFINITY_MULTIPLIER_CLIP.
    """
    cat_score = user_affinity["category_affinity"][_CATEGORY_INDEX[category]] * len(CATEGORIES)
    brand_score = user_affinity["brand_affinity"][_BRAND_INDEX[brand]] * len(ALL_BRANDS)
    multiplier = 0.7 * cat_score + 0.3 * brand_score
    lo, hi = AFFINITY_MULTIPLIER_CLIP
    return float(np.clip(multiplier, lo, hi))


def position_decay(position: int) -> float:
    """Decays click likelihood by impression position (1-indexed)."""
    return 1.0 / np.log2(position + 2)


def click_probability(position: int, category: str, brand: str, user_affinity: dict) -> float:
    """Computes click probability for one impression.

    Args:
        position: 1-indexed impression slot.
        category: The candidate item's category.
        brand: The candidate item's brand.
        user_affinity: Output of user_population.sample_user_affinity.

    Returns:
        float: Probability in [0, 1].
    """
    prob = BASE_CLICK_RATE * position_decay(position) * affinity_multiplier(user_affinity, category, brand)
    return float(np.clip(prob, 0.0, 1.0))


def sample_daily_view_count(rng: np.random.Generator) -> int:
    """Samples how many PDP views a single user generates on a single day."""
    return int(rng.poisson(PDP_VIEWS_PER_USER_PER_DAY_MEAN))


def sample_view_timestamp(day: pd.Timestamp, rng: np.random.Generator) -> pd.Timestamp:
    """Samples a random timestamp within the given day."""
    return pd.Timestamp(day) + pd.Timedelta(seconds=int(rng.integers(0, 86400)))


def sample_device(rng: np.random.Generator) -> str:
    """Samples a device from DEVICE_DISTRIBUTION."""
    devices = list(DEVICE_DISTRIBUTION)
    return rng.choice(devices, p=[DEVICE_DISTRIBUTION[d] for d in devices])


def sample_country(rng: np.random.Generator) -> str:
    """Samples a country from COUNTRY_DISTRIBUTION."""
    countries = list(COUNTRY_DISTRIBUTION)
    return rng.choice(countries, p=[COUNTRY_DISTRIBUTION[c] for c in countries])


def pick_anchor_item(items_df: pd.DataFrame, user_affinity: dict, rng: np.random.Generator) -> str:
    """Picks the anchor item a user views, weighted by their hidden affinity.

    Args:
        items_df: Eligible item catalog rows (cold-start items already
            excluded by the caller) with item_id, category, brand columns.
        user_affinity: Output of user_population.sample_user_affinity.
        rng: A numpy random Generator.

    Returns:
        str: The chosen anchor item_id.
    """
    return _affinity_weighted_choice(
        items_df["item_id"].to_numpy(),
        items_df["category"].to_numpy(),
        items_df["brand"].to_numpy(),
        user_affinity,
        rng,
    )


def simulate_impressions(
    anchor_id: str,
    candidates: list[tuple[str, float]],
    timestamp: pd.Timestamp,
    user_id: str,
    recommendation_impression_id: str,
    device: str,
    country: str,
) -> pd.DataFrame:
    """Builds the top-K impression rows for one anchor view.

    Args:
        anchor_id: The viewed anchor item_id.
        candidates: (item_id, similarity_score) pairs, ranked descending
            (see candidate_selection.score_candidates).
        timestamp: When the PDP view occurred.
        user_id: The viewing user's id.
        recommendation_impression_id: Id shared by all rows from this view.
        device: Sampled device string.
        country: Sampled country string.

    Returns:
        pd.DataFrame: Up to TOP_K_IMPRESSIONS rows matching the Impression
            Event schema.
    """
    top = candidates[:TOP_K_IMPRESSIONS]
    rows = [
        {
            "timestamp": timestamp,
            "user_id": user_id,
            "anchor_item_id": anchor_id,
            "recommendation_impression_id": recommendation_impression_id,
            "candidate_item_id": item_id,
            "position": position,
            "retrieval_similarity_score": score,
            "device": device,
            "country": country,
        }
        for position, (item_id, score) in enumerate(top, start=1)
    ]
    columns = [
        "timestamp",
        "user_id",
        "anchor_item_id",
        "recommendation_impression_id",
        "candidate_item_id",
        "position",
        "retrieval_similarity_score",
        "device",
        "country",
    ]
    return pd.DataFrame(rows, columns=columns)


def simulate_clicks(
    impressions_df: pd.DataFrame,
    item_category: dict,
    item_brand: dict,
    user_affinity: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Rolls a click for each impression, probability decaying with position.

    Args:
        impressions_df: Rows from simulate_impressions (one anchor view).
        item_category: item_id -> category lookup (see build_item_lookup).
        item_brand: item_id -> brand lookup (see build_item_lookup).
        user_affinity: Output of user_population.sample_user_affinity.
        rng: A numpy random Generator.

    Returns:
        pd.DataFrame: 0 or more Click Event rows.
    """
    rows = []
    for row in impressions_df.itertuples():
        category = item_category[row.candidate_item_id]
        brand = item_brand[row.candidate_item_id]
        prob = click_probability(row.position, category, brand, user_affinity)
        if rng.random() < prob:
            click_time = row.timestamp + pd.Timedelta(seconds=int(rng.integers(0, 600)))
            rows.append(
                {
                    "click_time": click_time,
                    "user_id": row.user_id,
                    "candidate_item_id": row.candidate_item_id,
                    "recommendation_impression_id": row.recommendation_impression_id,
                }
            )
    columns = ["click_time", "user_id", "candidate_item_id", "recommendation_impression_id"]
    return pd.DataFrame(rows, columns=columns)


def _make_order_id(user_id: str, item_id: str, purchase_time: pd.Timestamp, global_seed: int) -> str:
    """Builds a deterministic, reproducible order id."""
    key = f"{user_id}-{item_id}-{purchase_time.isoformat()}"
    return f"order_{stable_seed(key, global_seed):010d}"


def simulate_attributed_purchases(
    clicks_df: pd.DataFrame,
    item_category: dict,
    item_brand: dict,
    user_affinity: dict,
    rng: np.random.Generator,
    global_seed: int,
) -> pd.DataFrame:
    """Rolls a within-window purchase for each click.

    Purchase timestamps may land after the simulation window's last day --
    a click near the window's end must still be able to convert within
    ATTRIBUTION_WINDOW_HOURS.

    Args:
        clicks_df: Rows from simulate_clicks.
        item_category: item_id -> category lookup.
        item_brand: item_id -> brand lookup.
        user_affinity: Output of user_population.sample_user_affinity.
        rng: A numpy random Generator.
        global_seed: The dataset-wide reproducibility seed.

    Returns:
        pd.DataFrame: 0 or more Purchase Event rows.
    """
    rows = []
    for row in clicks_df.itertuples():
        category = item_category[row.candidate_item_id]
        brand = item_brand[row.candidate_item_id]
        prob = CLICK_TO_PURCHASE_BASE_RATE * affinity_multiplier(user_affinity, category, brand)
        prob = float(np.clip(prob, 0.0, 1.0))
        if rng.random() < prob:
            hours = rng.uniform(0.0, ATTRIBUTION_WINDOW_HOURS)
            purchase_time = row.click_time + pd.Timedelta(hours=hours)
            rows.append(
                {
                    "purchase_time": purchase_time,
                    "user_id": row.user_id,
                    "item_id": row.candidate_item_id,
                    "order_id": _make_order_id(row.user_id, row.candidate_item_id, purchase_time, global_seed),
                }
            )
    return pd.DataFrame(rows, columns=PURCHASE_COLUMNS)


def simulate_organic_purchases(
    user_id: str,
    day: pd.Timestamp,
    items_df: pd.DataFrame,
    user_affinity: dict,
    rng: np.random.Generator,
    global_seed: int,
) -> pd.DataFrame:
    """Rolls at most one no-preceding-click purchase for a user on a day.

    Args:
        user_id: The purchasing user's id.
        day: The simulated calendar day.
        items_df: Eligible item catalog rows (cold-start items already
            excluded) with item_id, category, brand columns.
        user_affinity: Output of user_population.sample_user_affinity.
        rng: A numpy random Generator.
        global_seed: The dataset-wide reproducibility seed.

    Returns:
        pd.DataFrame: 0 or 1 Purchase Event rows.
    """
    if rng.random() >= ORGANIC_PURCHASE_RATE:
        return pd.DataFrame(columns=PURCHASE_COLUMNS)

    item_id = _affinity_weighted_choice(
        items_df["item_id"].to_numpy(),
        items_df["category"].to_numpy(),
        items_df["brand"].to_numpy(),
        user_affinity,
        rng,
    )
    purchase_time = pd.Timestamp(day) + pd.Timedelta(seconds=int(rng.integers(0, 86400)))
    row = {
        "purchase_time": purchase_time,
        "user_id": user_id,
        "item_id": item_id,
        "order_id": _make_order_id(user_id, item_id, purchase_time, global_seed),
    }
    return pd.DataFrame([row], columns=PURCHASE_COLUMNS)
