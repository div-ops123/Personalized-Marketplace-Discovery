"""Pure sampling functions over the shared taxonomy (common/taxonomy.py).

TAXONOMY/CATEGORIES/ALL_BRANDS live in common/ because pipelines/ needs
them too (to validate that aggregated features never reference an unknown
category or brand) -- this module re-exports them so existing callers'
`from taxonomy import CATEGORIES, ALL_BRANDS` keep working unchanged, and
adds the generation-specific sampling functions on top.
"""

import numpy as np

from common.taxonomy import ALL_BRANDS, CATEGORIES, TAXONOMY

__all__ = [
    "TAXONOMY",
    "CATEGORIES",
    "ALL_BRANDS",
    "UnknownCategoryError",
    "sample_category",
    "sample_subcategory",
    "sample_brand",
    "sample_price",
    "sample_tags",
]


class UnknownCategoryError(Exception):
    """Raised when a category is not present in the taxonomy."""


def sample_category(rng: np.random.Generator) -> str:
    """Samples a category uniformly from the taxonomy.

    Args:
        rng: A numpy random Generator.

    Returns:
        str: A category name.
    """
    return rng.choice(CATEGORIES)


def sample_subcategory(rng: np.random.Generator, category: str) -> str:
    """Samples a subcategory belonging to the given category.

    Args:
        rng: A numpy random Generator.
        category: A category name present in TAXONOMY.

    Returns:
        str: A subcategory name.

    Raises:
        UnknownCategoryError: If category is not in the taxonomy.
    """
    if category not in TAXONOMY:
        raise UnknownCategoryError(f"Unknown category: {category!r}")
    return rng.choice(TAXONOMY[category]["subcategories"])


def sample_brand(rng: np.random.Generator, category: str) -> str:
    """Samples a brand belonging to the given category.

    Args:
        rng: A numpy random Generator.
        category: A category name present in TAXONOMY.

    Returns:
        str: A brand name.

    Raises:
        UnknownCategoryError: If category is not in the taxonomy.
    """
    if category not in TAXONOMY:
        raise UnknownCategoryError(f"Unknown category: {category!r}")
    return rng.choice(TAXONOMY[category]["brands"])


def sample_price(rng: np.random.Generator, category: str) -> float:
    """Samples a price within the category's price range.

    Uses a lognormal-shaped draw (clipped to the category's range) so
    prices cluster toward the low end with a long tail of premium
    items, mimicking real retail pricing better than a uniform draw.

    Args:
        rng: A numpy random Generator.
        category: A category name present in TAXONOMY.

    Returns:
        float: A price rounded to 2 decimal places.

    Raises:
        UnknownCategoryError: If category is not in the taxonomy.
    """
    if category not in TAXONOMY:
        raise UnknownCategoryError(f"Unknown category: {category!r}")
    low, high = TAXONOMY[category]["price_range"]
    raw = rng.lognormal(mean=0.0, sigma=0.5)
    normalized = min(raw / 4.0, 1.0)
    price = low + normalized * (high - low)
    return round(price, 2)


def sample_tags(rng: np.random.Generator, category: str) -> list[str]:
    """Samples 2-4 tags without replacement from the category's tag pool.

    Args:
        rng: A numpy random Generator.
        category: A category name present in TAXONOMY.

    Returns:
        list[str]: Between 2 and 4 tag strings.

    Raises:
        UnknownCategoryError: If category is not in the taxonomy.
    """
    if category not in TAXONOMY:
        raise UnknownCategoryError(f"Unknown category: {category!r}")
    pool = TAXONOMY[category]["tags"]
    n = int(rng.integers(2, min(4, len(pool)) + 1))
    return list(rng.choice(pool, size=n, replace=False))
