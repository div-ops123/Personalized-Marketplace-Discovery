"""Unit tests for training/ranking_preprocessing.py."""

import pandas as pd

from training.ranking_preprocessing import build_query_groups, encode_ranking_examples
from training.vocab import build_taxonomy_vocabs


def _synthetic_ranking_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recommendation_impression_id": ["impr_b", "impr_a", "impr_b", "impr_a", "impr_c"],
            "candidate_item_id": ["c1", "c2", "c3", "c4", "c5"],
            "device": ["mobile"] * 5,
            "country": ["US"] * 5,
            "position": [1, 1, 2, 2, 1],
            "retrieval_similarity_score": [0.9, 0.8, 0.7, 0.6, 0.5],
            "user_preferred_brands": [["Nike"], None, ["Nike"], None, ["Adidas"]],
            "user_avg_purchase_price": [50.0, None, 50.0, None, 30.0],
            "user_historical_category_affinity": [["Footwear"], None, ["Footwear"], None, None],
            "anchor_category": ["Footwear"] * 5,
            "anchor_subcategory": ["Sneakers"] * 5,
            "anchor_brand": ["Nike"] * 5,
            "anchor_price": [40.0] * 5,
            "candidate_category": ["Footwear"] * 5,
            "candidate_subcategory": ["Boots"] * 5,
            "candidate_brand": ["Adidas"] * 5,
            "candidate_price": [80.0] * 5,
            "candidate_recommendation_ctr": [0.1, None, 0.2, None, 0.3],
            "candidate_recommendation_cvr": [0.01, None, 0.02, None, 0.03],
            "candidate_recommendation_impressions": [100, None, 200, None, 300],
            "same_brand": [0] * 5,
            "same_category": [1] * 5,
            "same_subcategory": [0] * 5,
            "price_ratio": [2.0] * 5,
            "price_diff": [40.0] * 5,
            "label": [1, 0, 0, 1, 0],
        }
    )


def test_build_query_groups_sums_to_row_count_and_matches_manual_boundaries():
    df = _synthetic_ranking_df()
    sorted_df, group_sizes = build_query_groups(df)
    assert sum(group_sizes) == len(df)
    assert group_sizes == [2, 2, 1]  # impr_a: 2, impr_b: 2, impr_c: 1 -- alphabetical after sort
    assert sorted_df["recommendation_impression_id"].tolist() == [
        "impr_a",
        "impr_a",
        "impr_b",
        "impr_b",
        "impr_c",
    ]


def test_encode_ranking_examples_nan_passthrough_for_nullable_columns():
    df = _synthetic_ranking_df()
    vocabs = build_taxonomy_vocabs()
    features, labels = encode_ranking_examples(df, vocabs)

    assert features["user_avg_purchase_price"].isna().tolist() == [False, True, False, True, False]
    assert features["candidate_recommendation_ctr"].isna().tolist() == [False, True, False, True, False]
    assert features["candidate_recommendation_impressions"].isna().tolist() == [False, True, False, True, False]
    assert labels.tolist() == [1, 0, 0, 1, 0]


def test_encode_ranking_examples_user_features_missing_flag():
    df = _synthetic_ranking_df()
    vocabs = build_taxonomy_vocabs()
    features, _ = encode_ranking_examples(df, vocabs)
    assert features["user_features_missing"].tolist() == [0, 1, 0, 1, 0]


def test_encode_ranking_examples_multi_hot_brand_and_category():
    df = _synthetic_ranking_df()
    vocabs = build_taxonomy_vocabs()
    features, _ = encode_ranking_examples(df, vocabs)
    assert features["user_prefers_brand__Nike"].tolist() == [1, 0, 1, 0, 0]
    assert features["user_affinity_category__Footwear"].tolist() == [1, 0, 1, 0, 0]


def test_encode_ranking_examples_single_valued_columns_are_category_dtype():
    df = _synthetic_ranking_df()
    vocabs = build_taxonomy_vocabs()
    features, _ = encode_ranking_examples(df, vocabs)
    for column in ("device", "country", "anchor_category", "candidate_brand"):
        assert str(features[column].dtype) == "category"
