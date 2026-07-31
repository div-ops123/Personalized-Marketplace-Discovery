"""Unit tests for serving/retrieval_service/index_store.py -- the
item_id<->position lookup and the anchor-exclusion behavior of
similar_items. Builds a real (tiny) FAISS index via
serving/build_retrieval_index.py's build_hnsw_index and writes it to
tmp_path, exactly like serving/build_retrieval_index.py would, so
IndexStore is exercised against real on-disk artifacts.
"""

import json

import numpy as np
import pytest
import faiss

from serving.build_retrieval_index import build_hnsw_index
from serving.retrieval_service.index_store import AnchorNotFoundError, IndexStore


def _write_index(tmp_path, n: int = 6, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim)).astype(np.float32)
    item_ids = [f"item_{i}" for i in range(n)]

    index = build_hnsw_index(embeddings, m=16, ef_construction=40, ef_search=16)
    faiss.write_index(index, str(tmp_path / "items.faiss"))
    (tmp_path / "item_ids.json").write_text(json.dumps(item_ids))
    return tmp_path, item_ids


def test_item_count_matches_index(tmp_path):
    index_dir, item_ids = _write_index(tmp_path)
    store = IndexStore(index_dir)
    assert store.item_count == len(item_ids)


def test_similar_items_excludes_the_anchor_itself(tmp_path):
    index_dir, item_ids = _write_index(tmp_path)
    store = IndexStore(index_dir)

    for anchor_item_id in item_ids:
        results = store.similar_items(anchor_item_id, k=3)
        result_ids = [item_id for item_id, _ in results]
        assert anchor_item_id not in result_ids
        assert len(results) <= 3


def test_similar_items_descending_by_score(tmp_path):
    index_dir, item_ids = _write_index(tmp_path, n=10)
    store = IndexStore(index_dir)

    results = store.similar_items(item_ids[0], k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_similar_items_unknown_anchor_raises(tmp_path):
    index_dir, _ = _write_index(tmp_path)
    store = IndexStore(index_dir)

    with pytest.raises(AnchorNotFoundError):
        store.similar_items("not_a_real_item", k=5)
