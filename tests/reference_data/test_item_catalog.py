"""Unit tests for Item Catalog generation."""

from item_catalog import build_item_catalog, reserved_cold_start_item_ids


def test_build_item_catalog_empty():
    """n_items=0 returns an empty DataFrame with the expected columns."""
    df = build_item_catalog(0, seed=1)
    assert len(df) == 0
    assert set(df.columns) == {
        "item_id",
        "category",
        "subcategory",
        "brand",
        "price",
        "tags",
        "description",
        "image_embedding",
    }


def test_build_item_catalog_row_count():
    """Generates exactly n_items rows."""
    df = build_item_catalog(50, seed=1)
    assert len(df) == 50


def test_build_item_catalog_image_embedding_always_none():
    """image_embedding is null for every item — no product photos exist."""
    df = build_item_catalog(50, seed=1)
    assert df["image_embedding"].isna().all()


def test_build_item_catalog_deterministic():
    """Same seed produces an identical catalog."""
    df1 = build_item_catalog(30, seed=7)
    df2 = build_item_catalog(30, seed=7)
    assert df1.drop(columns=["tags"]).equals(df2.drop(columns=["tags"]))
    assert list(df1["tags"]) == list(df2["tags"])


def test_build_item_catalog_different_seeds_differ():
    """Different seeds produce a different catalog."""
    df1 = build_item_catalog(30, seed=7)
    df2 = build_item_catalog(30, seed=8)
    assert list(df1["description"]) != list(df2["description"])


def test_reserved_cold_start_item_ids_fraction():
    """Reserves approximately the requested fraction of items."""
    item_ids = [f"item_{i}" for i in range(1000)]
    reserved = reserved_cold_start_item_ids(item_ids, 0.07, seed=1)
    assert len(reserved) == 70
    assert reserved.issubset(set(item_ids))


def test_reserved_cold_start_item_ids_empty():
    """No items means no reservation, no crash."""
    assert reserved_cold_start_item_ids([], 0.07, seed=1) == set()


def test_reserved_cold_start_item_ids_deterministic():
    """Same seed reserves the same subset."""
    item_ids = [f"item_{i}" for i in range(200)]
    reserved1 = reserved_cold_start_item_ids(item_ids, 0.1, seed=3)
    reserved2 = reserved_cold_start_item_ids(item_ids, 0.1, seed=3)
    assert reserved1 == reserved2
