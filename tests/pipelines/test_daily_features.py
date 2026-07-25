"""Unit tests for the Spark daily-feature aggregation logic.

Inputs are already pre-filtered to timestamp < snapshot_date by the caller
(see db_io.read_events_before) -- these functions only aggregate what
they're given, so the tests here cover aggregation correctness (top-N
selection, CTR/CVR arithmetic, attribution dedup), not the date-boundary
filter itself (that's a SQL WHERE clause, verified at the integration
level against the live warehouse).
"""

import datetime

import pandas as pd
import pytest

from pipelines.spark_jobs.daily_features import compute_candidate_daily_features, compute_user_daily_features

SNAPSHOT_DATE = datetime.date(2025, 1, 10)


def test_compute_user_daily_features_empty_purchases(spark):
    """No purchase history yet -- empty result with the right columns."""
    result = compute_user_daily_features(spark, pd.DataFrame(), pd.DataFrame(), SNAPSHOT_DATE)
    assert len(result) == 0
    assert list(result.columns) == [
        "snapshot_date",
        "user_id",
        "preferred_brands",
        "avg_purchase_price",
        "historical_category_affinity",
    ]


def test_compute_user_daily_features_basic(spark):
    """Aggregates avg price, preferred brands, and category affinity per user."""
    items_df = pd.DataFrame(
        {
            "item_id": ["item_1", "item_2", "item_3"],
            "category": ["Footwear", "Footwear", "Electronics"],
            "brand": ["Adidas", "Nike", "Sony"],
            "price": [10.0, 20.0, 30.0],
        }
    )
    purchases_df = pd.DataFrame(
        {
            "purchase_time": [
                pd.Timestamp("2025-01-05 10:00:00"),
                pd.Timestamp("2025-01-06 10:00:00"),
                pd.Timestamp("2025-01-05 11:00:00"),
            ],
            "user_id": ["user_a", "user_a", "user_b"],
            "item_id": ["item_1", "item_2", "item_3"],
            "order_id": ["order_1", "order_2", "order_3"],
        }
    )

    result = compute_user_daily_features(spark, purchases_df, items_df, SNAPSHOT_DATE)
    result = result.set_index("user_id")

    assert set(result.index) == {"user_a", "user_b"}
    assert result.loc["user_a", "avg_purchase_price"] == pytest.approx(15.0)
    assert sorted(result.loc["user_a", "preferred_brands"]) == ["Adidas", "Nike"]
    assert result.loc["user_a", "historical_category_affinity"] == ["Footwear"]
    assert result.loc["user_b", "avg_purchase_price"] == pytest.approx(30.0)
    assert result.loc["user_b", "preferred_brands"] == ["Sony"]
    assert (result["snapshot_date"] == SNAPSHOT_DATE).all()


def test_compute_user_daily_features_caps_at_top_n(spark):
    """preferred_brands/historical_category_affinity never exceed 3 entries."""
    items_df = pd.DataFrame(
        {
            "item_id": [f"item_{i}" for i in range(5)],
            "category": [f"cat_{i}" for i in range(5)],
            "brand": [f"brand_{i}" for i in range(5)],
            "price": [10.0] * 5,
        }
    )
    purchases_df = pd.DataFrame(
        {
            "purchase_time": [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(5)],
            "user_id": ["user_a"] * 5,
            "item_id": [f"item_{i}" for i in range(5)],
            "order_id": [f"order_{i}" for i in range(5)],
        }
    )

    result = compute_user_daily_features(spark, purchases_df, items_df, SNAPSHOT_DATE)
    assert len(result.iloc[0]["preferred_brands"]) == 3
    assert len(result.iloc[0]["historical_category_affinity"]) == 3


def test_compute_candidate_daily_features_empty_impressions(spark):
    """No impression history yet -- empty result with the right columns."""
    result = compute_candidate_daily_features(
        spark, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), SNAPSHOT_DATE
    )
    assert len(result) == 0
    assert list(result.columns) == [
        "snapshot_date",
        "candidate_id",
        "recommendation_ctr",
        "recommendation_cvr",
        "recommendation_impressions",
    ]


def test_compute_candidate_daily_features_ctr_cvr(spark):
    """CTR = clicks/impressions, CVR = attributed purchases/clicks."""
    impressions_df = pd.DataFrame(
        {"candidate_item_id": ["cand_1"] * 10 + ["cand_2"] * 5}
    )
    clicks_df = pd.DataFrame(
        {
            "click_time": [pd.Timestamp("2025-01-05 10:00:00")] * 4,
            "user_id": ["user_1", "user_1", "user_2", "user_3"],
            "candidate_item_id": ["cand_1"] * 4,
            "recommendation_impression_id": ["impr_1", "impr_2", "impr_3", "impr_4"],
        }
    )
    purchases_df = pd.DataFrame(
        {
            "purchase_time": [pd.Timestamp("2025-01-05 15:00:00"), pd.Timestamp("2025-01-05 15:00:00")],
            "user_id": ["user_1", "user_9"],
            "item_id": ["cand_1", "cand_1"],
            "order_id": ["order_1", "order_2"],
        }
    )

    result = compute_candidate_daily_features(
        spark, impressions_df, clicks_df, purchases_df, SNAPSHOT_DATE
    ).set_index("candidate_id")

    assert result.loc["cand_1", "recommendation_impressions"] == 10
    assert result.loc["cand_1", "recommendation_ctr"] == pytest.approx(0.4)
    # only user_1's purchase matches a click (within 24h, same user+item);
    # user_9's purchase has no preceding click and isn't attributed.
    assert result.loc["cand_1", "recommendation_cvr"] == pytest.approx(1 / 4)

    assert result.loc["cand_2", "recommendation_impressions"] == 5
    assert result.loc["cand_2", "recommendation_ctr"] == pytest.approx(0.0)
    assert result.loc["cand_2", "recommendation_cvr"] == pytest.approx(0.0)


def test_compute_candidate_daily_features_dedups_purchase_across_multiple_clicks(spark):
    """One purchase matching two qualifying clicks is still counted once."""
    impressions_df = pd.DataFrame({"candidate_item_id": ["cand_1"]})
    clicks_df = pd.DataFrame(
        {
            "click_time": [pd.Timestamp("2025-01-05 09:00:00"), pd.Timestamp("2025-01-05 10:00:00")],
            "user_id": ["user_1", "user_1"],
            "candidate_item_id": ["cand_1", "cand_1"],
            "recommendation_impression_id": ["impr_1", "impr_2"],
        }
    )
    purchases_df = pd.DataFrame(
        {
            "purchase_time": [pd.Timestamp("2025-01-05 15:00:00")],
            "user_id": ["user_1"],
            "item_id": ["cand_1"],
            "order_id": ["order_1"],
        }
    )

    result = compute_candidate_daily_features(
        spark, impressions_df, clicks_df, purchases_df, SNAPSHOT_DATE
    ).set_index("candidate_id")
    # 2 clicks, 1 attributed purchase (deduped) -> cvr = 1/2, not 2/2 or higher
    assert result.loc["cand_1", "recommendation_cvr"] == pytest.approx(0.5)
