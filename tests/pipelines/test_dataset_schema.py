"""Regression guard: dataset builder table definitions match data-flow.md's schema."""

from pipelines.spark_jobs.dataset_schema import (
    ranking_training_examples_table,
    retrieval_training_examples_table,
)


def test_retrieval_training_examples_columns():
    expected = {
        "recommendation_impression_id",
        "candidate_item_id",
        "anchor_item_id",
        "anchor_category",
        "anchor_subcategory",
        "anchor_brand",
        "anchor_price_tier",
        "anchor_tags",
        "anchor_image_embedding",
        "anchor_text_embedding",
        "candidate_category",
        "candidate_subcategory",
        "candidate_brand",
        "candidate_price_tier",
        "candidate_tags",
        "candidate_image_embedding",
        "candidate_text_embedding",
        "label",
    }
    assert set(retrieval_training_examples_table.columns.keys()) == expected


def test_retrieval_training_examples_primary_key():
    assert set(retrieval_training_examples_table.primary_key.columns.keys()) == {
        "recommendation_impression_id",
        "candidate_item_id",
    }


def test_ranking_training_examples_columns():
    expected = {
        "recommendation_impression_id",
        "candidate_item_id",
        "user_id",
        "anchor_item_id",
        "timestamp",
        "device",
        "country",
        "position",
        "retrieval_similarity_score",
        "user_preferred_brands",
        "user_avg_purchase_price",
        "user_historical_category_affinity",
        "anchor_category",
        "anchor_subcategory",
        "anchor_brand",
        "anchor_price",
        "candidate_category",
        "candidate_subcategory",
        "candidate_brand",
        "candidate_price",
        "candidate_recommendation_ctr",
        "candidate_recommendation_cvr",
        "candidate_recommendation_impressions",
        "same_brand",
        "same_category",
        "same_subcategory",
        "price_ratio",
        "price_diff",
        "label",
    }
    assert set(ranking_training_examples_table.columns.keys()) == expected


def test_ranking_training_examples_primary_key():
    assert set(ranking_training_examples_table.primary_key.columns.keys()) == {
        "recommendation_impression_id",
        "candidate_item_id",
    }
