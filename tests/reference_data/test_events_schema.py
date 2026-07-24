"""Regression guard: event table definitions match data-flow.md's schema."""

from events_schema import click_events_table, impression_events_table, purchase_events_table


def test_impression_events_columns_match_documented_schema():
    """Matches data-flow.md's Impression Event column list exactly."""
    expected = {
        "timestamp",
        "user_id",
        "anchor_item_id",
        "recommendation_impression_id",
        "candidate_item_id",
        "position",
        "retrieval_similarity_score",
        "device",
        "country",
    }
    assert set(impression_events_table.columns.keys()) == expected


def test_click_events_columns_match_documented_schema():
    """Matches data-flow.md's Click Event column list exactly."""
    expected = {"click_time", "user_id", "candidate_item_id", "recommendation_impression_id"}
    assert set(click_events_table.columns.keys()) == expected


def test_purchase_events_columns_match_documented_schema():
    """Matches data-flow.md's Purchase Event column list exactly -- no
    recommendation reference at write time (see data-flow.md's Attribution
    section: attribution is decided afterward, never inferred at write time)."""
    expected = {"purchase_time", "user_id", "item_id", "order_id"}
    assert set(purchase_events_table.columns.keys()) == expected


def test_purchase_events_primary_key():
    """order_id is the primary key."""
    assert list(purchase_events_table.primary_key.columns.keys()) == ["order_id"]
