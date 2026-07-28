"""Unit tests for the Retrieval Dataset Builder's Spark transform."""

import pandas as pd

from pipelines.spark_jobs.retrieval_dataset import build_retrieval_dataset

_ITEMS = pd.DataFrame(
    {
        "item_id": ["item_1", "item_2", "item_3"],
        "category": ["Footwear", "Footwear", "Footwear"],
        "subcategory": ["Sneakers", "Boots", "Sneakers"],
        "brand": ["Nike", "Adidas", "Nike"],
        "price": [40.0, 200.0, 40.0],
        "tags": [["casual"], ["waterproof"], ["athletic"]],
        "image_embedding": [[0.1, 0.2], [0.3, 0.4], None],
        "text_embedding": [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
    }
)


def test_build_retrieval_dataset_empty_impressions(spark):
    result = build_retrieval_dataset(spark, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert len(result) == 0
    assert "label" in result.columns


def test_build_retrieval_dataset_basic(spark):
    impressions_df = pd.DataFrame(
        {
            "recommendation_impression_id": ["impr_1", "impr_1"],
            "candidate_item_id": ["item_2", "item_3"],
            "anchor_item_id": ["item_1", "item_1"],
        }
    )
    clicks_df = pd.DataFrame(
        {
            "recommendation_impression_id": ["impr_1"],
            "candidate_item_id": ["item_2"],
        }
    )

    result = build_retrieval_dataset(spark, impressions_df, clicks_df, _ITEMS)
    result = result.set_index("candidate_item_id")

    assert result.loc["item_2", "label"] == 1
    assert result.loc["item_3", "label"] == 0

    # anchor is item_1 for both rows: budget tier (price 40 within Footwear's
    # (30, 220) range -> width ~63.3, so 40 < 93.3 is "budget").
    assert result.loc["item_2", "anchor_price_tier"] == "budget"
    assert result.loc["item_2", "candidate_brand"] == "Adidas"
    assert result.loc["item_2", "candidate_price_tier"] == "premium"
    assert result.loc["item_2", "candidate_tags"] == ["waterproof"]
    assert isinstance(result.loc["item_2", "candidate_tags"], list)
    assert result.loc["item_2", "candidate_image_embedding"] == [0.3, 0.4]

    # item_3's image_embedding is None (cold-start-style null in the catalog).
    assert result.loc["item_3", "candidate_image_embedding"] is None


def test_build_retrieval_dataset_no_clicks_all_negative(spark):
    impressions_df = pd.DataFrame(
        {
            "recommendation_impression_id": ["impr_1"],
            "candidate_item_id": ["item_2"],
            "anchor_item_id": ["item_1"],
        }
    )

    result = build_retrieval_dataset(spark, impressions_df, pd.DataFrame(), _ITEMS)
    assert result.iloc[0]["label"] == 0
