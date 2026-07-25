"""Regression guard: feature table definitions match data-flow.md's schema."""

from pipelines.spark_jobs.features_schema import candidate_daily_features_table, user_daily_features_table


def test_user_daily_features_columns_match_documented_schema():
    """Matches data-flow.md's User Daily Features column list exactly."""
    expected = {
        "snapshot_date",
        "user_id",
        "preferred_brands",
        "avg_purchase_price",
        "historical_category_affinity",
    }
    assert set(user_daily_features_table.columns.keys()) == expected


def test_candidate_daily_features_columns_match_documented_schema():
    """Matches data-flow.md's Candidate Daily Features column list exactly."""
    expected = {
        "snapshot_date",
        "candidate_id",
        "recommendation_ctr",
        "recommendation_cvr",
        "recommendation_impressions",
    }
    assert set(candidate_daily_features_table.columns.keys()) == expected


def test_user_daily_features_primary_key():
    """Historized: primary key is (snapshot_date, user_id), not user_id alone."""
    assert set(user_daily_features_table.primary_key.columns.keys()) == {"snapshot_date", "user_id"}


def test_candidate_daily_features_primary_key():
    """Historized: primary key is (snapshot_date, candidate_id), not candidate_id alone."""
    assert set(candidate_daily_features_table.primary_key.columns.keys()) == {
        "snapshot_date",
        "candidate_id",
    }
