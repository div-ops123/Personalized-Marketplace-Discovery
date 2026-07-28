"""Unit tests for training/splits.py."""

import pandas as pd

from training.splits import temporal_split


def _timestamped_df(days_ago_list: list[int]) -> pd.DataFrame:
    base = pd.Timestamp("2025-04-01")
    return pd.DataFrame(
        {
            "row_id": range(len(days_ago_list)),
            "timestamp": [base - pd.Timedelta(days=d) for d in days_ago_list],
        }
    )


def test_temporal_split_row_counts():
    df = _timestamped_df(list(range(20)))  # days_ago 0..19, day 0 = most recent
    train, val, test = temporal_split(df, "timestamp", val_days=5, test_days=5)
    assert len(train) + len(val) + len(test) == len(df)
    # test_cutoff = max_ts - 5 days lands exactly on the days_ago=5 row (a
    # daily-spaced fixture), and the test split's ">=" comparison includes
    # that boundary row -- so test covers days_ago 0..5 (6 rows), and val
    # (">= val_cutoff, < test_cutoff") covers days_ago 6..10 (5 rows).
    assert len(test) == 6
    assert len(val) == 5


def test_temporal_split_val_and_test_never_overlap():
    df = _timestamped_df(list(range(30)))
    train, val, test = temporal_split(df, "timestamp", val_days=7, test_days=7)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_temporal_split_same_timestamp_rows_never_straddle_a_split():
    # Simulates ranking_training_examples: every row of one impression
    # shares an identical timestamp -- these must always land together,
    # even when that timestamp sits exactly at a split boundary.
    base = pd.Timestamp("2025-04-01")
    boundary_ts = base - pd.Timedelta(days=7)
    df = pd.DataFrame(
        {
            "recommendation_impression_id": ["impr_a"] * 5 + ["impr_b"] * 5,
            "timestamp": [boundary_ts] * 5 + [base] * 5,
        }
    )
    train, val, test = temporal_split(df, "timestamp", val_days=7, test_days=7)
    for group in df["recommendation_impression_id"].unique():
        total = int((df["recommendation_impression_id"] == group).sum())
        per_split = [int((split["recommendation_impression_id"] == group).sum()) for split in (train, val, test)]
        assert sorted(per_split, reverse=True)[0] == total  # entirely in exactly one split
        assert sum(per_split) == total
