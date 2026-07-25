"""Synthetic user population and hidden latent affinity.

sample_user_affinity is intentionally never called from this module's own
table-building path and is never written to the warehouse — it exists
only for the event simulator to import later, so simulated
click/purchase behavior can be realistically biased per user without
that bias ever being visible to any model or pipeline code (which would
leak ground truth into training).
"""

import numpy as np
import pandas as pd
from faker import Faker

from seeding import rng_for_entity
from taxonomy import ALL_BRANDS, CATEGORIES


def build_users(n_users: int, seed: int) -> pd.DataFrame:
    """Builds the synthetic user population.

    Args:
        n_users: Number of users to generate.
        seed: The dataset-wide reproducibility seed.

    Returns:
        pd.DataFrame: Columns are exactly user_id and signup_date — no
            affinity or preference data of any kind.
    """
    if n_users == 0:
        return pd.DataFrame(columns=["user_id", "signup_date"])

    fake = Faker()
    Faker.seed(seed)
    user_ids = [f"user_{i:06d}" for i in range(n_users)]
    signup_dates = [fake.date_between(start_date="-2y", end_date="today") for _ in user_ids]
    return pd.DataFrame({"user_id": user_ids, "signup_date": signup_dates})


def sample_user_affinity(user_id: str, global_seed: int) -> dict:
    """Derives a user's hidden category/brand affinity vector on demand.

    Never persisted to the warehouse and never imported by any
    model/pipeline code — only the event simulator calls this, at
    simulation time, to drive realistic click/purchase behavior.

    Args:
        user_id: The user's identifier.
        global_seed: The dataset-wide reproducibility seed.

    Returns:
        dict: {"category_affinity": np.ndarray of length len(CATEGORIES),
            "brand_affinity": np.ndarray of length len(ALL_BRANDS)}, both
            Dirichlet-distributed (non-negative, sums to 1).

    Raises:
        ValueError: If user_id is empty or blank.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")
    rng = rng_for_entity(user_id, global_seed)
    return {
        "category_affinity": rng.dirichlet(np.ones(len(CATEGORIES))),
        "brand_affinity": rng.dirichlet(np.ones(len(ALL_BRANDS))),
    }
