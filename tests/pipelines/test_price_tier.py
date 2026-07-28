"""Unit tests for common/taxonomy.py's price_tier bucketing."""

import pytest

from common.taxonomy import TAXONOMY, price_tier


def test_price_tier_buckets_within_category_range():
    low, high = TAXONOMY["Footwear"]["price_range"]
    width = (high - low) / 3

    assert price_tier(low, "Footwear") == "budget"
    assert price_tier(low + width - 0.01, "Footwear") == "budget"
    assert price_tier(low + width, "Footwear") == "mid"
    assert price_tier(low + 2 * width - 0.01, "Footwear") == "mid"
    assert price_tier(low + 2 * width, "Footwear") == "premium"
    assert price_tier(high, "Footwear") == "premium"


def test_price_tier_is_relative_to_category_not_absolute():
    """The same dollar price can be a different tier in a different category."""
    # $40 is near the top of Books' range but near the bottom of Footwear's.
    assert price_tier(40.0, "Books") == "premium"
    assert price_tier(40.0, "Footwear") == "budget"


def test_price_tier_unknown_category_raises():
    with pytest.raises(KeyError):
        price_tier(50.0, "Not A Real Category")
