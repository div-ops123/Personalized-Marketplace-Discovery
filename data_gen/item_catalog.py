"""Item Catalog generation.

Every item gets a fully-populated catalog row regardless of whether it
will ever appear in a simulated event (see reserved_cold_start_item_ids) —
that's what lets content-based retrieval serve brand-new items with zero
interaction history. Cold-start-ness is not represented in this table at
all; it's enforced later, in the event simulator, by simply never
generating events for the reserved item ids.
"""

import numpy as np
import pandas as pd
from faker import Faker

from seeding import rng_for_entity, stable_seed
from taxonomy import sample_brand, sample_category, sample_price, sample_subcategory, sample_tags


def build_item(item_id: str, rng: np.random.Generator, fake: Faker) -> dict:
    """Builds a single item's catalog record.

    Args:
        item_id: The item's unique identifier.
        rng: A numpy random Generator, already seeded for this item.
        fake: A Faker instance used for the free-text description flavor.

    Returns:
        dict: One item_catalog row (image_embedding always None).
    """
    category = sample_category(rng)
    subcategory = sample_subcategory(rng, category)
    brand = sample_brand(rng, category)
    price = sample_price(rng, category)
    tags = sample_tags(rng, category)
    description = f"{brand} {subcategory} — {', '.join(tags)}. {fake.sentence()}"
    return {
        "item_id": item_id,
        "category": category,
        "subcategory": subcategory,
        "brand": brand,
        "price": price,
        "tags": tags,
        "description": description,
        "image_embedding": None,
    }


def build_item_catalog(n_items: int, seed: int) -> pd.DataFrame:
    """Builds the full synthetic Item Catalog.

    Args:
        n_items: Number of items to generate.
        seed: The dataset-wide reproducibility seed.

    Returns:
        pd.DataFrame: One row per item, columns matching build_item's
            keys. Does not include text_embedding — that's computed
            separately once the description text exists (see
            text_embeddings.embed_descriptions).
    """
    if n_items == 0:
        return pd.DataFrame(
            columns=[
                "item_id",
                "category",
                "subcategory",
                "brand",
                "price",
                "tags",
                "description",
                "image_embedding",
            ]
        )

    fake = Faker()
    Faker.seed(seed)
    item_ids = [f"item_{i:06d}" for i in range(n_items)]
    rows = [build_item(item_id, rng_for_entity(item_id, seed), fake) for item_id in item_ids]
    return pd.DataFrame(rows)


def reserved_cold_start_item_ids(item_ids: list[str], fraction: float, seed: int) -> set[str]:
    """Deterministically selects a reserved cold-start subset of item ids.

    Callers (the event simulator) must never generate any
    impression/click/purchase event referencing these item ids, so they
    end up with genuinely zero interaction history downstream.

    Args:
        item_ids: All item ids in the catalog.
        fraction: Fraction of items to reserve (e.g. 0.07 for 7%).
        seed: The dataset-wide reproducibility seed.

    Returns:
        set[str]: The reserved item ids.
    """
    if not item_ids:
        return set()
    rng = np.random.default_rng(stable_seed("cold_start_selection", seed))
    n_reserved = round(len(item_ids) * fraction)
    chosen = rng.choice(item_ids, size=n_reserved, replace=False)
    return set(chosen)
