"""FastAPI route tests for serving/api_gateway/main.py. The Retrieval and
Ranking Service calls are intercepted via httpx.MockTransport -- no real
network calls, no Docker needed.
"""

import json

import httpx
from fastapi.testclient import TestClient

from serving.api_gateway.main import create_app

_CATALOG = {
    "cand_1": {"category": "Footwear", "subcategory": "Sneakers", "brand": "Nike", "price": 100.0},
    "cand_2": {"category": "Apparel", "subcategory": "Jackets", "brand": "Adidas", "price": 50.0},
}


def _mock_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/similar-items":
        anchor_item_id = request.url.params["anchor_item_id"]
        if anchor_item_id == "not_real":
            return httpx.Response(404, json={"detail": "unknown anchor"})
        return httpx.Response(
            200,
            json={
                "anchor_item_id": anchor_item_id,
                "candidates": [
                    {"item_id": "cand_1", "retrieval_similarity_score": 0.9},
                    {"item_id": "cand_2", "retrieval_similarity_score": 0.5},
                ],
            },
        )
    if request.url.path == "/rank":
        return httpx.Response(
            200,
            json={
                "anchor_item_id": "anchor_1",
                "ranked": [
                    {"item_id": "cand_1", "rank_score": 2.0},
                    {"item_id": "cand_2", "rank_score": 1.0},
                ],
            },
        )
    return httpx.Response(404)


def _build_app(tmp_path):
    (tmp_path / "item_catalog.json").write_text(json.dumps(_CATALOG))
    mock_client = httpx.Client(transport=httpx.MockTransport(_mock_handler))
    return create_app(
        retrieval_url="http://retrieval-service:8001",
        ranking_url="http://ranking-service:8002",
        features_dir=tmp_path,
        http_client=mock_client,
    )


def test_recommendations_enriches_ranked_results_with_catalog_metadata(tmp_path):
    app = _build_app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/recommendations", params={"anchor_item_id": "anchor_1", "user_id": "user_1"})

    assert response.status_code == 200
    body = response.json()
    assert body["anchor_item_id"] == "anchor_1"
    assert body["recommendations"][0] == {
        "rank": 1,
        "item_id": "cand_1",
        "score": 2.0,
        "category": "Footwear",
        "subcategory": "Sneakers",
        "brand": "Nike",
        "price": 100.0,
    }
    assert body["recommendations"][1]["rank"] == 2
    assert body["recommendations"][1]["item_id"] == "cand_2"


def test_recommendations_respects_k(tmp_path):
    app = _build_app(tmp_path)

    with TestClient(app) as client:
        response = client.get(
            "/recommendations", params={"anchor_item_id": "anchor_1", "user_id": "user_1", "k": 1}
        )

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 1


def test_recommendations_unknown_anchor_returns_404(tmp_path):
    app = _build_app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/recommendations", params={"anchor_item_id": "not_real", "user_id": "user_1"})

    assert response.status_code == 404
