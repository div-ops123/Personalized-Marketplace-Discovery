"""FastAPI route tests for serving/ranking_service/main.py, using
TestClient against a tiny real trained booster + feature snapshot written
to tmp_path -- no Docker needed. Reuses the snapshot fixture from
test_model_store.py rather than duplicating it.
"""

from fastapi.testclient import TestClient

from serving.ranking_service.main import create_app
from tests.serving.test_model_store import _write_snapshot


def test_rank_returns_ranked_candidates(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    app = create_app(ranking_dir=ranking_dir, features_dir=features_dir)

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200

        response = client.post(
            "/rank",
            json={
                "anchor_item_id": "anchor_1",
                "user_id": "user_known",
                "country": "US",
                "device": "mobile",
                "candidates": [
                    {"item_id": "cand_same_brand", "retrieval_similarity_score": 0.9},
                    {"item_id": "cand_diff_brand", "retrieval_similarity_score": 0.5},
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["anchor_item_id"] == "anchor_1"
    ranked_ids = {row["item_id"] for row in body["ranked"]}
    assert ranked_ids == {"cand_same_brand", "cand_diff_brand"}


def test_rank_unknown_anchor_returns_404(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    app = create_app(ranking_dir=ranking_dir, features_dir=features_dir)

    with TestClient(app) as client:
        response = client.post(
            "/rank",
            json={
                "anchor_item_id": "not_real",
                "user_id": "user_known",
                "country": "US",
                "device": "mobile",
                "candidates": [],
            },
        )

    assert response.status_code == 404
