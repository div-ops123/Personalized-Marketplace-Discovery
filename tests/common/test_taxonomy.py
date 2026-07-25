"""Regression guard: common/taxonomy.py is the single source of truth
consumed by both data_gen/ (generation) and pipelines/ (validation).
"""

from common.taxonomy import ALL_BRANDS, CATEGORIES, TAXONOMY


def test_categories_match_taxonomy_keys():
    assert CATEGORIES == list(TAXONOMY.keys())


def test_all_brands_is_sorted_union_of_category_brands():
    expected = sorted({brand for spec in TAXONOMY.values() for brand in spec["brands"]})
    assert ALL_BRANDS == expected


def test_data_gen_taxonomy_reexports_the_same_objects():
    """data_gen/taxonomy.py must not fork its own copy of the taxonomy."""
    from taxonomy import ALL_BRANDS as dg_brands
    from taxonomy import CATEGORIES as dg_categories
    from taxonomy import TAXONOMY as dg_taxonomy

    assert dg_taxonomy is TAXONOMY
    assert dg_categories is CATEGORIES
    assert dg_brands is ALL_BRANDS
