"""Unit tests for serving/build_retrieval_index.py's pure logic: HNSW index
construction and the item_id <-> index-position mapping contract. The
MLflow/Postgres IO (resolve_run_id's search_runs fallback, load_vocabs,
mlflow.pytorch.load_model, main()) is not unit-tested here, matching
training/retrieval_train.py and training/ranking_train.py's existing
precedent of no e2e test for thin IO wrappers over a real server.
"""

import json

import faiss
import numpy as np

from serving.build_retrieval_index import build_hnsw_index, resolve_run_id


def _synthetic_embeddings(n: int = 12, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def test_build_hnsw_index_ntotal_and_metric():
    embeddings = _synthetic_embeddings()
    index = build_hnsw_index(embeddings, m=16, ef_construction=40, ef_search=16)

    assert index.ntotal == len(embeddings)
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT


def test_build_hnsw_index_self_query_returns_itself_as_top_hit():
    embeddings = _synthetic_embeddings()
    index = build_hnsw_index(embeddings, m=16, ef_construction=40, ef_search=16)

    for i in range(len(embeddings)):
        query = np.ascontiguousarray(embeddings[i : i + 1])
        _, top_positions = index.search(query, 1)
        assert top_positions[0][0] == i


def test_build_hnsw_index_does_not_mutate_caller_array():
    embeddings = _synthetic_embeddings()
    original = embeddings.copy()
    build_hnsw_index(embeddings, m=16, ef_construction=40, ef_search=16)
    np.testing.assert_array_equal(embeddings, original)


def test_index_and_item_ids_round_trip(tmp_path):
    # Locks the design contract build_retrieval_index.py relies on: FAISS
    # positions are insertion-order, so a plain item_ids.json list (not an
    # IndexIDMap -- item_ids are strings, IndexIDMap is int64-only) must
    # stay aligned with those positions across a save/load cycle.
    embeddings = _synthetic_embeddings(n=5, dim=4)
    item_ids = [f"item_{i}" for i in range(5)]
    index = build_hnsw_index(embeddings, m=8, ef_construction=40, ef_search=16)

    index_path = tmp_path / "items.faiss"
    ids_path = tmp_path / "item_ids.json"
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(item_ids))

    reloaded_index = faiss.read_index(str(index_path))
    reloaded_item_ids = json.loads(ids_path.read_text())

    assert reloaded_index.ntotal == len(reloaded_item_ids)
    query = np.ascontiguousarray(reloaded_index.reconstruct(2).reshape(1, -1))
    _, top_positions = reloaded_index.search(query, 1)
    assert reloaded_item_ids[top_positions[0][0]] == "item_2"


def test_resolve_run_id_passes_through_explicit_id():
    assert resolve_run_id("abc123") == "abc123"
