"""Stable, cross-process deterministic per-entity seeding.

Python's built-in hash() is randomized per-process for strings
(PYTHONHASHSEED), so it cannot be used to derive reproducible seeds
across runs. This module derives seeds via a stable SHA-256 hash instead.
"""

import hashlib

import numpy as np


def stable_seed(entity_id: str, global_seed: int) -> int:
    """Derives a stable, reproducible 32-bit seed from an entity id.

    Args:
        entity_id: A unique identifier (e.g. item_id or user_id).
        global_seed: The dataset-wide reproducibility seed.

    Returns:
        int: A deterministic seed in [0, 2**32).
    """
    digest = hashlib.sha256(f"{entity_id}-{global_seed}".encode()).hexdigest()
    return int(digest, 16) % (2**32)


def rng_for_entity(entity_id: str, global_seed: int) -> np.random.Generator:
    """Builds a numpy Generator seeded deterministically from an entity id.

    Args:
        entity_id: A unique identifier (e.g. item_id or user_id).
        global_seed: The dataset-wide reproducibility seed.

    Returns:
        np.random.Generator: A Generator that is identical across runs
            for the same (entity_id, global_seed) pair.
    """
    return np.random.default_rng(stable_seed(entity_id, global_seed))
