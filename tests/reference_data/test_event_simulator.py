"""Unit tests for raw event simulation: anchors, impressions, clicks, purchases."""

import numpy as np
import pandas as pd
import pytest

import event_simulator
from event_simulator import (
    affinity_multiplier,
    build_item_lookup,
    click_probability,
    pick_anchor_item,
    position_decay,
    simulate_attributed_purchases,
    simulate_clicks,
    simulate_impressions,
    simulate_organic_purchases,
)
from taxonomy import ALL_BRANDS, CATEGORIES


def _uniform_affinity() -> dict:
    return {
        "category_affinity": np.full(len(CATEGORIES), 1.0 / len(CATEGORIES)),
        "brand_affinity": np.full(len(ALL_BRANDS), 1.0 / len(ALL_BRANDS)),
    }


def _items_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [f"item_{i:03d}" for i in range(10)],
            "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(10)],
            "brand": [ALL_BRANDS[i % len(ALL_BRANDS)] for i in range(10)],
        }
    )


def test_position_decay_strictly_decreasing():
    """Later positions get a strictly smaller decay factor."""
    values = [position_decay(p) for p in range(1, 21)]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)


def test_affinity_multiplier_uniform_affinity_is_one():
    """A user with exactly average affinity gets a ~1.0 multiplier."""
    multiplier = affinity_multiplier(_uniform_affinity(), CATEGORIES[0], ALL_BRANDS[0])
    assert multiplier == pytest.approx(1.0)


def test_click_probability_decreases_with_position():
    """At fixed affinity, click probability strictly decreases with position."""
    affinity = _uniform_affinity()
    probs = [click_probability(p, CATEGORIES[0], ALL_BRANDS[0], affinity) for p in range(1, 21)]
    assert probs == sorted(probs, reverse=True)


def test_click_probability_bounded():
    """Click probability is always a valid probability."""
    affinity = _uniform_affinity()
    for p in (1, 10, 20):
        prob = click_probability(p, CATEGORIES[0], ALL_BRANDS[0], affinity)
        assert 0.0 <= prob <= 1.0


def test_pick_anchor_item_only_from_eligible_pool():
    """Never returns an item outside the given (already-filtered) pool."""
    df = _items_df()
    affinity = _uniform_affinity()
    rng = np.random.default_rng(0)
    eligible_ids = set(df["item_id"])
    for _ in range(50):
        anchor = pick_anchor_item(df, affinity, rng)
        assert anchor in eligible_ids


def test_pick_anchor_item_deterministic():
    """Same seed produces the same anchor sequence."""
    df = _items_df()
    affinity = _uniform_affinity()
    picks1 = [pick_anchor_item(df, affinity, np.random.default_rng(3)) for _ in range(5)]
    picks2 = [pick_anchor_item(df, affinity, np.random.default_rng(3)) for _ in range(5)]
    assert picks1 == picks2


def test_simulate_impressions_row_count_and_positions():
    """Builds one row per candidate (capped at TOP_K_IMPRESSIONS), 1-indexed positions."""
    candidates = [(f"cand_{i}", 1.0 - i * 0.01) for i in range(30)]
    timestamp = pd.Timestamp("2025-01-01T12:00:00")
    df = simulate_impressions("anchor_0", candidates, timestamp, "user_0", "impr_1", "mobile", "US")
    assert len(df) == event_simulator.TOP_K_IMPRESSIONS
    assert list(df["position"]) == list(range(1, event_simulator.TOP_K_IMPRESSIONS + 1))
    assert (df["anchor_item_id"] == "anchor_0").all()
    assert (df["recommendation_impression_id"] == "impr_1").all()


def test_simulate_clicks_deterministic():
    """Same seed produces identical click rows."""
    candidates = [(f"cand_{i}", 1.0 - i * 0.01) for i in range(20)]
    timestamp = pd.Timestamp("2025-01-01T12:00:00")
    impressions = simulate_impressions("anchor_0", candidates, timestamp, "user_0", "impr_1", "mobile", "US")
    item_category = {f"cand_{i}": CATEGORIES[i % len(CATEGORIES)] for i in range(20)}
    item_brand = {f"cand_{i}": ALL_BRANDS[i % len(ALL_BRANDS)] for i in range(20)}
    affinity = _uniform_affinity()

    clicks1 = simulate_clicks(impressions, item_category, item_brand, affinity, np.random.default_rng(9))
    clicks2 = simulate_clicks(impressions, item_category, item_brand, affinity, np.random.default_rng(9))
    assert clicks1.equals(clicks2)
    assert set(clicks1["candidate_item_id"]).issubset(set(impressions["candidate_item_id"]))


def test_simulate_attributed_purchases_within_window(monkeypatch):
    """Attributed purchases land within ATTRIBUTION_WINDOW_HOURS of the click."""
    monkeypatch.setattr(event_simulator, "CLICK_TO_PURCHASE_BASE_RATE", 1.0)
    clicks = pd.DataFrame(
        [
            {
                "click_time": pd.Timestamp("2025-01-01T12:00:00"),
                "user_id": "user_0",
                "candidate_item_id": "item_000",
                "recommendation_impression_id": "impr_1",
            }
        ]
    )
    item_category = {"item_000": CATEGORIES[0]}
    item_brand = {"item_000": ALL_BRANDS[0]}
    purchases = simulate_attributed_purchases(
        clicks, item_category, item_brand, _uniform_affinity(), np.random.default_rng(1), global_seed=42
    )
    assert len(purchases) == 1
    row = purchases.iloc[0]
    assert row["purchase_time"] >= clicks.iloc[0]["click_time"]
    assert row["purchase_time"] <= clicks.iloc[0]["click_time"] + pd.Timedelta(
        hours=event_simulator.ATTRIBUTION_WINDOW_HOURS
    )


def test_simulate_attributed_purchases_none_when_rate_zero(monkeypatch):
    """No purchases roll through when the conversion rate is forced to zero."""
    monkeypatch.setattr(event_simulator, "CLICK_TO_PURCHASE_BASE_RATE", 0.0)
    clicks = pd.DataFrame(
        [
            {
                "click_time": pd.Timestamp("2025-01-01T12:00:00"),
                "user_id": "user_0",
                "candidate_item_id": "item_000",
                "recommendation_impression_id": "impr_1",
            }
        ]
    )
    item_category = {"item_000": CATEGORIES[0]}
    item_brand = {"item_000": ALL_BRANDS[0]}
    purchases = simulate_attributed_purchases(
        clicks, item_category, item_brand, _uniform_affinity(), np.random.default_rng(1), global_seed=42
    )
    assert len(purchases) == 0


def test_simulate_organic_purchases_forced_on(monkeypatch):
    """A forced 100% organic rate always produces exactly one purchase row."""
    monkeypatch.setattr(event_simulator, "ORGANIC_PURCHASE_RATE", 1.0)
    df = _items_df()
    day = pd.Timestamp("2025-01-01")
    purchases = simulate_organic_purchases(
        "user_0", day, df, _uniform_affinity(), np.random.default_rng(2), global_seed=42
    )
    assert len(purchases) == 1
    assert purchases.iloc[0]["user_id"] == "user_0"
    assert purchases.iloc[0]["item_id"] in set(df["item_id"])


def test_simulate_organic_purchases_forced_off(monkeypatch):
    """A forced 0% organic rate never produces a purchase row."""
    monkeypatch.setattr(event_simulator, "ORGANIC_PURCHASE_RATE", 0.0)
    df = _items_df()
    day = pd.Timestamp("2025-01-01")
    purchases = simulate_organic_purchases(
        "user_0", day, df, _uniform_affinity(), np.random.default_rng(2), global_seed=42
    )
    assert len(purchases) == 0


def test_purchase_time_can_exceed_attribution_window_start_day(monkeypatch):
    """A click near the window's end can still produce a purchase on a later day."""
    monkeypatch.setattr(event_simulator, "CLICK_TO_PURCHASE_BASE_RATE", 1.0)
    late_click_time = pd.Timestamp("2025-03-31T23:50:00")  # near day 90 of a 90-day window
    clicks = pd.DataFrame(
        [
            {
                "click_time": late_click_time,
                "user_id": "user_0",
                "candidate_item_id": "item_000",
                "recommendation_impression_id": "impr_1",
            }
        ]
    )
    item_category = {"item_000": CATEGORIES[0]}
    item_brand = {"item_000": ALL_BRANDS[0]}
    purchases = simulate_attributed_purchases(
        clicks, item_category, item_brand, _uniform_affinity(), np.random.default_rng(4), global_seed=42
    )
    assert len(purchases) == 1
    assert purchases.iloc[0]["purchase_time"].date() >= late_click_time.date()


def test_build_item_lookup():
    """Builds correct item_id -> category / brand lookups."""
    df = _items_df()
    item_category, item_brand = build_item_lookup(df)
    for item_id, category, brand in zip(df["item_id"], df["category"], df["brand"]):
        assert item_category[item_id] == category
        assert item_brand[item_id] == brand
