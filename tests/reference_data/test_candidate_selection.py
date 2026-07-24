"""Unit tests for item-item content-similarity candidate selection."""

import numpy as np
import pandas as pd
import pytest

from candidate_selection import build_exclusion_mask, build_item_embedding_matrix, score_candidates


def _items_df(vectors: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"item_id": list(vectors.keys()), "text_embedding": list(vectors.values())}
    )


def test_build_item_embedding_matrix_is_l2_normalized():
    """Every row of the built matrix has unit norm."""
    df = _items_df({"a": [3.0, 4.0], "b": [1.0, 0.0], "c": [0.0, 5.0]})
    item_ids, matrix = build_item_embedding_matrix(df)
    assert list(item_ids) == ["a", "b", "c"]
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0)


def test_build_exclusion_mask():
    """Marks exactly the excluded ids, in item_ids order."""
    item_ids = np.array(["a", "b", "c", "d"])
    mask = build_exclusion_mask(item_ids, {"b", "d"})
    assert list(mask) == [False, True, False, True]


def test_score_candidates_excludes_anchor_itself():
    """The anchor item never appears in its own candidate results."""
    df = _items_df({"a": [1.0, 0.0], "b": [0.9, 0.1], "c": [0.0, 1.0]})
    item_ids, matrix = build_item_embedding_matrix(df)
    mask = build_exclusion_mask(item_ids, set())
    rng = np.random.default_rng(0)
    results = score_candidates("a", item_ids, matrix, mask, rng, pool_size=10)
    assert "a" not in [item_id for item_id, _ in results]


def test_score_candidates_excludes_reserved_ids():
    """Cold-start (or any excluded) item ids never appear as candidates."""
    df = _items_df({"a": [1.0, 0.0], "b": [0.9, 0.1], "c": [0.0, 1.0], "d": [0.8, 0.2]})
    item_ids, matrix = build_item_embedding_matrix(df)
    mask = build_exclusion_mask(item_ids, {"b", "d"})
    rng = np.random.default_rng(0)
    results = score_candidates("a", item_ids, matrix, mask, rng, pool_size=10)
    result_ids = [item_id for item_id, _ in results]
    assert "b" not in result_ids
    assert "d" not in result_ids
    assert result_ids == ["c"]


def test_score_candidates_respects_pool_size():
    """Returns at most pool_size candidates, sorted by score descending."""
    vectors = {f"item_{i}": [np.cos(i), np.sin(i)] for i in range(20)}
    df = _items_df(vectors)
    item_ids, matrix = build_item_embedding_matrix(df)
    mask = build_exclusion_mask(item_ids, set())
    rng = np.random.default_rng(1)
    results = score_candidates("item_0", item_ids, matrix, mask, rng, pool_size=5)
    assert len(results) == 5
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_score_candidates_ranks_similar_item_above_dissimilar():
    """An identical-direction item ranks above an orthogonal one."""
    df = _items_df(
        {
            "anchor": [1.0, 0.0],
            "identical": [1.0, 0.0],
            "orthogonal": [0.0, 1.0],
        }
    )
    item_ids, matrix = build_item_embedding_matrix(df)
    mask = build_exclusion_mask(item_ids, set())
    rng = np.random.default_rng(42)
    results = score_candidates("anchor", item_ids, matrix, mask, rng, pool_size=10)
    scores = dict(results)
    assert scores["identical"] > scores["orthogonal"]


def test_score_candidates_deterministic():
    """Same seed produces identical candidate rankings."""
    vectors = {f"item_{i}": [np.cos(i), np.sin(i), np.cos(2 * i)] for i in range(15)}
    df = _items_df(vectors)
    item_ids, matrix = build_item_embedding_matrix(df)
    mask = build_exclusion_mask(item_ids, set())
    results1 = score_candidates("item_0", item_ids, matrix, mask, np.random.default_rng(5), pool_size=8)
    results2 = score_candidates("item_0", item_ids, matrix, mask, np.random.default_rng(5), pool_size=8)
    assert results1 == results2


def test_score_candidates_no_eligible_items_returns_empty():
    """Excluding every other item leaves no candidates -- no crash."""
    df = _items_df({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    item_ids, matrix = build_item_embedding_matrix(df)
    mask = build_exclusion_mask(item_ids, {"b"})
    rng = np.random.default_rng(0)
    results = score_candidates("a", item_ids, matrix, mask, rng, pool_size=10)
    assert results == []
