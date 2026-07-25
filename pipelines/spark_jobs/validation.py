"""Validation checks for the daily feature tables (LLD.md's DAG 1 spec:
null rates, coverage, value ranges, row counts) with concrete thresholds.

Each validate_* function returns a list of human-readable failure strings;
an empty list means the checks passed. run_daily_features.py raises (and
never writes) if either list is non-empty, matching DAG 1's documented
"fail -> alert on-call, halt pipeline, do not write" branch.
"""

import pandas as pd

MIN_COVERAGE = 0.99
TOP_N = 3

# Must match data-gen/taxonomy.py's CATEGORIES/ALL_BRANDS -- duplicated
# rather than imported because data-gen's hyphenated directory name can't
# be imported as a Python package (see daily_features.py's
# ATTRIBUTION_WINDOW_HOURS for the same tradeoff).
_VALID_CATEGORIES = {
    "Footwear",
    "Electronics",
    "Apparel",
    "Home",
    "Beauty",
    "Sports",
    "Toys",
    "Books",
}
_VALID_BRANDS = {
    "Nike",
    "Adidas",
    "New Balance",
    "Vans",
    "Sony",
    "Anker",
    "JBL",
    "Samsung",
    "Levi's",
    "Uniqlo",
    "Patagonia",
    "Champion",
    "IKEA",
    "Lodge",
    "Philips",
    "Brooklinen",
    "CeraVe",
    "Olaplex",
    "The Ordinary",
    "Dove",
    "Coleman",
    "Trek",
    "Lululemon",
    "REI",
    "LEGO",
    "Hasbro",
    "Mattel",
    "Ravensburger",
    "Penguin",
    "Scholastic",
    "HarperCollins",
    "Vintage",
}


def _check_nulls(df: pd.DataFrame, table_name: str, columns: list[str]) -> list[str]:
    failures = []
    for col in columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            failures.append(f"{table_name}: {null_count} null values in column '{col}'")
    return failures


def _check_array_column(df: pd.DataFrame, table_name: str, col: str, valid_values: set) -> list[str]:
    failures = []
    lengths = df[col].apply(len)
    if ((lengths < 1) | (lengths > TOP_N)).any():
        failures.append(f"{table_name}: '{col}' array length outside [1, {TOP_N}]")
    seen = {value for values in df[col] for value in values}
    invalid = seen - valid_values
    if invalid:
        failures.append(f"{table_name}: unknown values in '{col}': {sorted(invalid)}")
    return failures


def validate_user_features(df: pd.DataFrame, expected_active_users: int) -> list[str]:
    """Validates a day's User Daily Features rows.

    Args:
        df: One day's user_daily_features rows (all the same snapshot_date).
        expected_active_users: Distinct users with a purchase before this
            snapshot_date, computed independently from purchase_events --
            a cross-check against the aggregation itself.

    Returns:
        list[str]: Failure messages; empty means all checks passed.
    """
    table_name = "user_daily_features"
    if expected_active_users == 0:
        # No purchase history exists yet before this snapshot_date at all
        # (true for the first days of a cold-start backfill) -- an empty
        # result is correct, not a failure.
        return []
    if len(df) == 0:
        return [f"{table_name}: row count is 0 but {expected_active_users} active users were expected"]

    failures = _check_nulls(
        df, table_name, ["user_id", "preferred_brands", "avg_purchase_price", "historical_category_affinity"]
    )
    if (df["avg_purchase_price"] <= 0).any():
        failures.append(f"{table_name}: avg_purchase_price has non-positive values")
    failures += _check_array_column(df, table_name, "preferred_brands", _VALID_BRANDS)
    failures += _check_array_column(df, table_name, "historical_category_affinity", _VALID_CATEGORIES)

    if expected_active_users > 0:
        coverage = len(df) / expected_active_users
        if coverage < MIN_COVERAGE:
            failures.append(
                f"{table_name}: coverage {coverage:.3f} below {MIN_COVERAGE} "
                f"({len(df)} rows vs {expected_active_users} expected active users)"
            )
    return failures


def validate_candidate_features(df: pd.DataFrame, expected_active_candidates: int) -> list[str]:
    """Validates a day's Candidate Daily Features rows.

    Args:
        df: One day's candidate_daily_features rows (all the same snapshot_date).
        expected_active_candidates: Distinct candidates with an impression
            before this snapshot_date, computed independently from
            impression_events -- a cross-check against the aggregation itself.

    Returns:
        list[str]: Failure messages; empty means all checks passed.
    """
    table_name = "candidate_daily_features"
    if expected_active_candidates == 0:
        # No impression history exists yet before this snapshot_date at all
        # (true for the first day of a cold-start backfill) -- an empty
        # result is correct, not a failure.
        return []
    if len(df) == 0:
        return [f"{table_name}: row count is 0 but {expected_active_candidates} active candidates were expected"]

    failures = _check_nulls(
        df,
        table_name,
        ["candidate_id", "recommendation_ctr", "recommendation_cvr", "recommendation_impressions"],
    )
    for col in ["recommendation_ctr", "recommendation_cvr"]:
        if ((df[col] < 0.0) | (df[col] > 1.0)).any():
            failures.append(f"{table_name}: '{col}' has values outside [0, 1]")
    if (df["recommendation_impressions"] < 1).any():
        failures.append(f"{table_name}: recommendation_impressions has values < 1")

    if expected_active_candidates > 0:
        coverage = len(df) / expected_active_candidates
        if coverage < MIN_COVERAGE:
            failures.append(
                f"{table_name}: coverage {coverage:.3f} below {MIN_COVERAGE} "
                f"({len(df)} rows vs {expected_active_candidates} expected active candidates)"
            )
    return failures
