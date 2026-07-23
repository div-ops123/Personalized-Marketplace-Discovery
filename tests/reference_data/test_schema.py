"""Regression guard: table definitions match the documented schema."""

from schema import item_catalog_table, users_table


def test_item_catalog_columns_match_documented_schema():
    """item_catalog columns match docs/data-flow.md's Offline Feature Store Shape,
    plus description (build-phases.md step 4) — no undocumented additions."""
    expected = {
        "item_id",
        "category",
        "subcategory",
        "brand",
        "price",
        "tags",
        "description",
        "image_embedding",
        "text_embedding",
    }
    assert set(item_catalog_table.columns.keys()) == expected


def test_users_columns_contain_no_affinity_data():
    """users table is exactly user_id and signup_date — never affinity data."""
    assert set(users_table.columns.keys()) == {"user_id", "signup_date"}


def test_item_catalog_primary_key():
    """item_id is the primary key."""
    assert list(item_catalog_table.primary_key.columns.keys()) == ["item_id"]


def test_users_primary_key():
    """user_id is the primary key."""
    assert list(users_table.primary_key.columns.keys()) == ["user_id"]
