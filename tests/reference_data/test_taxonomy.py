"""Unit tests for taxonomy sampling functions."""

import numpy as np
import pytest

from taxonomy import (
    TAXONOMY,
    UnknownCategoryError,
    sample_brand,
    sample_category,
    sample_price,
    sample_subcategory,
    sample_tags,
)


def test_sample_category_returns_known_category():
    """Sampled category is always a taxonomy key."""
    rng = np.random.default_rng(1)
    assert sample_category(rng) in TAXONOMY


def test_sample_subcategory_belongs_to_category():
    """Sampled subcategory is drawn from its category's own subcategory list."""
    rng = np.random.default_rng(2)
    category = "Footwear"
    subcategory = sample_subcategory(rng, category)
    assert subcategory in TAXONOMY[category]["subcategories"]


def test_sample_brand_belongs_to_category():
    """Sampled brand is drawn from its category's own brand list."""
    rng = np.random.default_rng(3)
    category = "Electronics"
    brand = sample_brand(rng, category)
    assert brand in TAXONOMY[category]["brands"]


def test_sample_price_within_category_range():
    """Sampled price stays within the category's configured range."""
    rng = np.random.default_rng(4)
    category = "Books"
    low, high = TAXONOMY[category]["price_range"]
    for _ in range(50):
        price = sample_price(rng, category)
        assert low <= price <= high


def test_sample_tags_within_pool_and_count():
    """Sampled tags are unique, drawn from the category pool, count in [2, 4]."""
    rng = np.random.default_rng(5)
    category = "Beauty"
    tags = sample_tags(rng, category)
    assert 2 <= len(tags) <= 4
    assert len(tags) == len(set(tags))
    assert set(tags).issubset(set(TAXONOMY[category]["tags"]))


@pytest.mark.parametrize(
    "sampler,args",
    [
        (sample_subcategory, ("unknown",)),
        (sample_brand, ("unknown",)),
        (sample_price, ("unknown",)),
        (sample_tags, ("unknown",)),
    ],
)
def test_unknown_category_raises(sampler, args):
    """An unrecognized category raises a specific, typed error."""
    rng = np.random.default_rng(6)
    with pytest.raises(UnknownCategoryError):
        sampler(rng, *args)
