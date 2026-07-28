"""Unit tests for the Ranking Dataset Builder's Spark transform."""

import pandas as pd
import pytest

from pipelines.spark_jobs.ranking_dataset import build_ranking_dataset

_ITEMS = pd.DataFrame(
    {
        "item_id": ["item_1", "item_2"],
        "category": ["Footwear", "Footwear"],
        "subcategory": ["Sneakers", "Boots"],
        "brand": ["Nike", "Adidas"],
        "price": [40.0, 200.0],
        "tags": [["casual"], ["waterproof"]],
        "image_embedding": [[0.1, 0.2], [0.3, 0.4]],
        "text_embedding": [[0.5, 0.6], [0.7, 0.8]],
    }
)


def _impressions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recommendation_impression_id": ["impr_1", "impr_2", "impr_3"],
            "candidate_item_id": ["item_2", "item_2", "item_2"],
            "user_id": ["user_1", "user_1", "user_2"],
            "anchor_item_id": ["item_1", "item_1", "item_1"],
            "timestamp": [
                pd.Timestamp("2025-01-10 10:00:00"),
                pd.Timestamp("2025-01-10 11:00:00"),
                pd.Timestamp("2025-01-10 12:00:00"),
            ],
            "device": ["mobile", "mobile", "desktop"],
            "country": ["US", "US", "US"],
            "position": [1, 2, 3],
            "retrieval_similarity_score": [0.9, 0.8, 0.7],
        }
    )


def test_build_ranking_dataset_empty_impressions(spark):
    result = build_ranking_dataset(
        spark, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert len(result) == 0
    assert "label" in result.columns


def test_build_ranking_dataset_attribution_label(spark):
    impressions_df = _impressions()
    clicks_df = pd.DataFrame(
        {
            "recommendation_impression_id": ["impr_1", "impr_3"],
            "candidate_item_id": ["item_2", "item_2"],
            "click_time": [pd.Timestamp("2025-01-10 10:05:00"), pd.Timestamp("2025-01-10 12:05:00")],
            "user_id": ["user_1", "user_2"],
        }
    )
    purchases_df = pd.DataFrame(
        {
            "purchase_time": [
                pd.Timestamp("2025-01-10 10:30:00"),  # within 24h of impr_1's click -> attributed
                pd.Timestamp("2025-01-12 12:10:00"),  # >24h after impr_3's click -> not attributed
            ],
            "user_id": ["user_1", "user_2"],
            "item_id": ["item_2", "item_2"],
            "order_id": ["order_1", "order_2"],
        }
    )

    result = build_ranking_dataset(
        spark, impressions_df, clicks_df, purchases_df, _ITEMS, pd.DataFrame(), pd.DataFrame()
    ).set_index("recommendation_impression_id")

    assert result.loc["impr_1", "label"] == 1  # click + purchase within window
    assert result.loc["impr_2", "label"] == 0  # no click at all
    assert result.loc["impr_3", "label"] == 0  # click, but purchase outside window


def test_build_ranking_dataset_cross_features(spark):
    impressions_df = _impressions().iloc[[0]]  # just impr_1

    result = build_ranking_dataset(
        spark, impressions_df, pd.DataFrame(), pd.DataFrame(), _ITEMS, pd.DataFrame(), pd.DataFrame()
    )
    row = result.iloc[0]

    assert row["same_category"] == 1  # both Footwear
    assert row["same_brand"] == 0  # Nike vs Adidas
    assert row["same_subcategory"] == 0  # Sneakers vs Boots
    assert row["price_ratio"] == pytest.approx(200.0 / 40.0)
    assert row["price_diff"] == pytest.approx(160.0)
    assert row["anchor_price"] == pytest.approx(40.0)
    assert row["candidate_price"] == pytest.approx(200.0)


def test_build_ranking_dataset_point_in_time_user_features(spark):
    impressions_df = _impressions()  # impr_1/impr_2 for user_1 on 2025-01-10, impr_3 for user_2
    user_features_df = pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2025-01-10").date(), pd.Timestamp("2025-01-11").date()],
            "user_id": ["user_1", "user_1"],
            "preferred_brands": [["Nike"], ["Adidas"]],
            "avg_purchase_price": [50.0, 999.0],
            "historical_category_affinity": [["Footwear"], ["Beauty"]],
        }
    )

    result = build_ranking_dataset(
        spark, impressions_df, pd.DataFrame(), pd.DataFrame(), _ITEMS, user_features_df, pd.DataFrame()
    ).set_index("recommendation_impression_id")

    # impressions on 2025-01-10 must use the 2025-01-10 snapshot (built from
    # data strictly before that day), never the later 2025-01-11 one.
    assert result.loc["impr_1", "user_preferred_brands"] == ["Nike"]
    assert result.loc["impr_1", "user_avg_purchase_price"] == pytest.approx(50.0)
    assert result.loc["impr_2", "user_preferred_brands"] == ["Nike"]

    # user_2 has no snapshot at all -- null features, not a dropped row.
    assert result.loc["impr_3", "user_preferred_brands"] is None
    assert pd.isna(result.loc["impr_3", "user_avg_purchase_price"])


def test_build_ranking_dataset_point_in_time_candidate_features(spark):
    impressions_df = _impressions().iloc[[0]]  # impr_1, timestamp 2025-01-10 10:00
    candidate_features_df = pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2025-01-10").date(), pd.Timestamp("2025-01-11").date()],
            "candidate_id": ["item_2", "item_2"],
            "recommendation_ctr": [0.1, 0.9],
            "recommendation_cvr": [0.01, 0.99],
            "recommendation_impressions": [100, 500],
        }
    )

    result = build_ranking_dataset(
        spark, impressions_df, pd.DataFrame(), pd.DataFrame(), _ITEMS, pd.DataFrame(), candidate_features_df
    )
    row = result.iloc[0]

    assert row["candidate_recommendation_ctr"] == pytest.approx(0.1)
    assert row["candidate_recommendation_impressions"] == 100
