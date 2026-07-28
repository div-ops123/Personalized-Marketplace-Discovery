"""Feature encoding and query-group construction for the LambdaMART ranker.

Two categorical encoding mechanisms, deliberately: single-valued columns
use LightGBM's native categorical support (pandas `category` dtype);
multi-hot array columns (user_preferred_brands, user_historical_category_affinity)
are expanded into binary columns via the same vocab.py:multi_hot_encode
retrieval uses for tags -- LightGBM cannot accept list-valued cells.
"""

from __future__ import annotations

import pandas as pd

from training.vocab import Vocabulary, multi_hot_encode

_SINGLE_VALUED_CATEGORICALS = [
    "device",
    "country",
    "anchor_category",
    "anchor_subcategory",
    "anchor_brand",
    "candidate_category",
    "candidate_subcategory",
    "candidate_brand",
]

_CONTINUOUS_COLUMNS = [
    "position",
    "retrieval_similarity_score",
    "user_avg_purchase_price",
    "anchor_price",
    "candidate_price",
    "candidate_recommendation_ctr",
    "candidate_recommendation_cvr",
    "candidate_recommendation_impressions",
    "same_brand",
    "same_category",
    "same_subcategory",
    "price_ratio",
    "price_diff",
]


def build_query_groups(
    df: pd.DataFrame, group_col: str = "recommendation_impression_id"
) -> tuple[pd.DataFrame, list[int]]:
    """Sorts by group and returns LightGBM-ready contiguous group sizes.

    Must be applied independently to each split (train/val/test) *after*
    splitting -- computing one global group array before splitting would
    silently corrupt lgb.Dataset's group boundaries across splits.

    Args:
        df: A (single-split) ranking_training_examples DataFrame.
        group_col: The column identifying one query group -- one
            recommendation impression's full candidate set.

    Returns:
        tuple[pd.DataFrame, list[int]]: The DataFrame sorted by group_col
            (stable sort, preserving intra-group order), and the list of
            per-group row counts in that same order, as lgb.Dataset(...,
            group=...) expects.
    """
    sorted_df = df.sort_values(group_col, kind="stable").reset_index(drop=True)
    group_sizes = sorted_df.groupby(group_col, sort=False).size().tolist()
    return sorted_df, group_sizes


def encode_ranking_examples(
    df: pd.DataFrame, vocabs: dict[str, Vocabulary]
) -> tuple[pd.DataFrame, pd.Series]:
    """Encodes a ranking_training_examples slice into LightGBM-ready features.

    NULL handling (explicit, no imputation):
    - user_preferred_brands/user_historical_category_affinity NULL (no
      snapshot yet) -> multi-hot all-zero, plus a user_features_missing
      flag so the model distinguishes "no snapshot" from "known user,
      genuinely zero affinity."
    - user_avg_purchase_price NULL -> passthrough NaN (LightGBM handles
      natively, no imputation).
    - candidate_recommendation_ctr/cvr/impressions NULL (candidate's first
      impression, no snapshot yet) -> passthrough NaN, never synthesized
      as 0 (0 would falsely claim "known to convert at 0%" instead of
      "unknown").

    Args:
        df: A (single-split, already query-grouped) ranking_training_examples
            DataFrame -- typically build_query_groups's sorted output.
        vocabs: Must contain "brand" and "category" Vocabulary instances
            (the same taxonomy vocabs retrieval uses).

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y). Single-valued categorical
            columns in X are pandas `category` dtype -- callers pass
            columns of that dtype to lgb.Dataset's categorical_feature.
    """
    features = pd.DataFrame(index=df.index)

    for column in _SINGLE_VALUED_CATEGORICALS:
        features[column] = df[column].astype("category")

    for column in _CONTINUOUS_COLUMNS:
        features[column] = df[column]

    features["user_features_missing"] = df["user_preferred_brands"].isna().astype(int)

    brand_multi_hot = multi_hot_encode(df["user_preferred_brands"].tolist(), vocabs["brand"])
    for index, brand in enumerate(vocabs["brand"].token_to_index):
        features[f"user_prefers_brand__{brand}"] = brand_multi_hot[:, index]

    category_multi_hot = multi_hot_encode(df["user_historical_category_affinity"].tolist(), vocabs["category"])
    for index, category in enumerate(vocabs["category"].token_to_index):
        features[f"user_affinity_category__{category}"] = category_multi_hot[:, index]

    return features, df["label"]
