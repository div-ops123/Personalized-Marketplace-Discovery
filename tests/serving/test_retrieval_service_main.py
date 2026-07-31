"""FastAPI route tests for serving/retrieval_service/main.py, using
TestClient against a tiny real FAISS index written to tmp_path -- no
Docker needed.
"""

import json

import faiss
import numpy as np
from fastapi.testclient import TestClient

from serving.build_retrieval_index import build_hnsw_index
from serving.retrieval_service.main import create_app


def _write_index(tmp_path, n: int = 6, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim)).astype(np.float32)
    item_ids = [f"item_{i}" for i in range(n)]

    index = build_hnsw_index(embeddings, m=16, ef_construction=40, ef_search=16)
    faiss.write_index(index, str(tmp_path / "items.faiss"))
    (tmp_path / "item_ids.json").write_text(json.dumps(item_ids))
    return item_ids


def test_health_reports_item_count(tmp_path):
    item_ids = _write_index(tmp_path)
    app = create_app(index_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "item_count": len(item_ids)}


def test_similar_items_returns_candidates_excluding_anchor(tmp_path):
    item_ids = _write_index(tmp_path)
    app = create_app(index_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get("/similar-items", params={"anchor_item_id": item_ids[0], "k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["anchor_item_id"] == item_ids[0]
    assert len(body["candidates"]) <= 3
    assert all(c["item_id"] != item_ids[0] for c in body["candidates"])


def test_similar_items_unknown_anchor_returns_404(tmp_path):
    _write_index(tmp_path)
    app = create_app(index_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get("/similar-items", params={"anchor_item_id": "not_real", "k": 3})

    assert response.status_code == 404
