"""Unit tests for daily-feature validation thresholds. Pure pandas, no Spark."""

import pandas as pd

from pipelines.spark_jobs.validation import validate_candidate_features, validate_user_features


def _valid_user_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2025-01-10").date()],
            "user_id": ["user_a"],
            "preferred_brands": [["Nike", "Adidas"]],
            "avg_purchase_price": [42.0],
            "historical_category_affinity": [["Footwear"]],
        }
    )


def _valid_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2025-01-10").date()],
            "candidate_id": ["cand_1"],
            "recommendation_ctr": [0.3],
            "recommendation_cvr": [0.1],
            "recommendation_impressions": [10],
        }
    )


def test_validate_user_features_no_active_users_passes_regardless():
    """Before any purchase history exists, an empty result is correct, not a failure."""
    assert validate_user_features(pd.DataFrame(), expected_active_users=0) == []


def test_validate_user_features_empty_but_active_users_expected_fails():
    """Active users existed but the aggregation produced nothing -- a real bug."""
    failures = validate_user_features(pd.DataFrame(), expected_active_users=5)
    assert failures
    assert "row count is 0" in failures[0]


def test_validate_user_features_valid_passes():
    assert validate_user_features(_valid_user_df(), expected_active_users=1) == []


def test_validate_user_features_null_value_fails():
    df = _valid_user_df()
    df.loc[0, "avg_purchase_price"] = None
    failures = validate_user_features(df, expected_active_users=1)
    assert any("null" in f for f in failures)


def test_validate_user_features_non_positive_price_fails():
    df = _valid_user_df()
    df.loc[0, "avg_purchase_price"] = 0.0
    failures = validate_user_features(df, expected_active_users=1)
    assert any("avg_purchase_price" in f for f in failures)


def test_validate_user_features_array_too_long_fails():
    df = _valid_user_df()
    df.at[0, "preferred_brands"] = ["Nike", "Adidas", "Sony", "IKEA"]
    failures = validate_user_features(df, expected_active_users=1)
    assert any("array length" in f for f in failures)


def test_validate_user_features_unknown_brand_fails():
    df = _valid_user_df()
    df.at[0, "preferred_brands"] = ["NotARealBrand"]
    failures = validate_user_features(df, expected_active_users=1)
    assert any("unknown values" in f for f in failures)


def test_validate_user_features_coverage_below_threshold_fails():
    failures = validate_user_features(_valid_user_df(), expected_active_users=100)
    assert any("coverage" in f for f in failures)


def test_validate_candidate_features_no_active_candidates_passes_regardless():
    assert validate_candidate_features(pd.DataFrame(), expected_active_candidates=0) == []


def test_validate_candidate_features_empty_but_active_candidates_expected_fails():
    failures = validate_candidate_features(pd.DataFrame(), expected_active_candidates=5)
    assert failures
    assert "row count is 0" in failures[0]


def test_validate_candidate_features_valid_passes():
    assert validate_candidate_features(_valid_candidate_df(), expected_active_candidates=1) == []


def test_validate_candidate_features_ctr_out_of_range_fails():
    df = _valid_candidate_df()
    df.loc[0, "recommendation_ctr"] = 1.5
    failures = validate_candidate_features(df, expected_active_candidates=1)
    assert any("recommendation_ctr" in f for f in failures)


def test_validate_candidate_features_zero_impressions_fails():
    df = _valid_candidate_df()
    df.loc[0, "recommendation_impressions"] = 0
    failures = validate_candidate_features(df, expected_active_candidates=1)
    assert any("recommendation_impressions" in f for f in failures)


def test_validate_candidate_features_coverage_below_threshold_fails():
    failures = validate_candidate_features(_valid_candidate_df(), expected_active_candidates=100)
    assert any("coverage" in f for f in failures)
