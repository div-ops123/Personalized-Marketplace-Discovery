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


def test_encode_ranking_examples_zero_fills_unknown_user_and_candidate_features():
    df = _synthetic_ranking_df()
    vocabs = build_taxonomy_vocabs()
    features, labels = encode_ranking_examples(df, vocabs)

    # Rows 1 and 3 are unknown (no user snapshot, never-served candidate).
    assert features["user_avg_purchase_price"].tolist() == [50.0, 0.0, 50.0, 0.0, 30.0]
    assert features["candidate_recommendation_ctr"].tolist() == [0.1, 0.0, 0.2, 0.0, 0.3]
    assert features["candidate_recommendation_cvr"].tolist() == [0.01, 0.0, 0.02, 0.0, 0.03]
    assert features["candidate_recommendation_impressions"].tolist() == [100.0, 0.0, 200.0, 0.0, 300.0]
    assert not features.isna().any().any()
    assert labels.tolist() == [1, 0, 0, 1, 0]


def test_encode_ranking_examples_all_none_columns_are_zero_not_object_dtype():
    # One online request for an unknown user and a never-served candidate:
    # every value in these columns is None.
    df = _synthetic_ranking_df().iloc[[1]].copy()
    df["user_avg_purchase_price"] = None
    df["candidate_recommendation_ctr"] = None
    vocabs = build_taxonomy_vocabs()
    features, _ = encode_ranking_examples(df, vocabs)

    assert features["user_avg_purchase_price"].dtype.kind == "f"
    assert features["candidate_recommendation_ctr"].dtype.kind == "f"
    assert features["user_avg_purchase_price"].tolist() == [0.0]
    assert features["candidate_recommendation_ctr"].tolist() == [0.0]


def test_encode_ranking_examples_unknown_user_multi_hot_is_all_zero_and_flag_is_gone():
    df = _synthetic_ranking_df()
    vocabs = build_taxonomy_vocabs()
    features, _ = encode_ranking_examples(df, vocabs)

    assert "user_features_missing" not in features.columns
    multi_hot_columns = [c for c in features.columns if c.startswith(("user_prefers_brand__", "user_affinity_category__"))]
    assert features.loc[1, multi_hot_columns].sum() == 0  # row 1: unknown user


def test_encode_ranking_examples_does_not_fill_not_null_columns():
    # position/prices/cross features are NOT NULL upstream: a NaN there is a
    # bug and must stay visible, not be silently zero-filled.
    df = _synthetic_ranking_df()
    df.loc[0, "candidate_price"] = None
    vocabs = build_taxonomy_vocabs()
    features, _ = encode_ranking_examples(df, vocabs)
    assert features["candidate_price"].isna().tolist() == [True, False, False, False, False]


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
