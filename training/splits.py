"""Temporal train/val/test split -- the one point in this repo's training
code where leakage would otherwise be easy to introduce.

Everywhere else in this pipeline (daily features, both dataset builders)
already treats point-in-time correctness as a hard invariant via strict
`<` timestamp joins. A random split here would be the one inconsistent
leak point in an otherwise leak-proof pipeline, and wouldn't match how the
model is actually evaluated in production (on data it hasn't seen *in
time*, not just data outside an i.i.d. sample). See docs/build-phases.md
Phase 5 -- no split methodology is documented, this is the chosen default.
"""

import pandas as pd


def temporal_split(
    df: pd.DataFrame, timestamp_col: str, val_days: int, test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into train/val/test by a timestamp column.

    The most recent `test_days` (by the data's own max timestamp) become
    the test split, the `val_days` immediately before that become
    validation, and everything earlier is train. Because the split is a
    pure `<`/`>=` comparison on the exact timestamp value, rows sharing an
    identical timestamp (e.g. every candidate row of one
    recommendation_impression_id in ranking_training_examples) always land
    in the same split -- never straddled.

    Args:
        df: The DataFrame to split.
        timestamp_col: Name of the datetime column to split on.
        val_days: Size of the validation window, in days, immediately
            before the test window.
        test_days: Size of the held-out test window, in days, ending at
            the data's max timestamp.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train, val, test),
            each a view-safe copy preserving the original row order.
    """
    timestamps = pd.to_datetime(df[timestamp_col])
    max_ts = timestamps.max()
    test_cutoff = max_ts - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

    train_df = df[timestamps < val_cutoff]
    val_df = df[(timestamps >= val_cutoff) & (timestamps < test_cutoff)]
    test_df = df[timestamps >= test_cutoff]
    return train_df, val_df, test_df
