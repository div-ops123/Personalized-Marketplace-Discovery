"""FastAPI Ranking Service (docs/build-phases.md Phase 6) -- scores and
reorders retrieval candidates via the trained LambdaMART ranker. No live
Postgres/MLflow dependency at request time -- only the static model +
feature snapshot mounted read-only from serving/ranking/ and
serving/features/ (see docs/LLD.md:170).

Run standalone (outside docker-compose.serving.yml, for local testing):
    uvicorn serving.ranking_service.main:app --port 8002
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from serving.constants import DEFAULT_FEATURES_DIR, DEFAULT_RANKING_DIR
from serving.ranking_service.model_store import ModelStore, UnknownItemError

logger = logging.getLogger(__name__)


class Candidate(BaseModel):
    item_id: str
    retrieval_similarity_score: float


class RankRequest(BaseModel):
    anchor_item_id: str
    user_id: str
    country: str
    device: str
    candidates: list[Candidate]


def create_app(ranking_dir: Path | None = None, features_dir: Path | None = None) -> FastAPI:
    """Builds the Ranking Service FastAPI app.

    A factory (not a bare module-level app) so tests can point the loaded
    model/feature snapshot at tmp_path fixtures instead of the real
    serving/ranking/ and serving/features/ mounts.

    Args:
        ranking_dir: Directory to load model.txt from. Defaults to the
            RANKING_DIR env var, then DEFAULT_RANKING_DIR.
        features_dir: Directory to load the feature-snapshot JSON files
            from. Defaults to the FEATURES_DIR env var, then
            DEFAULT_FEATURES_DIR.

    Returns:
        FastAPI: The configured app, model/features loaded on startup via
            lifespan.
    """
    resolved_ranking_dir = ranking_dir or Path(os.environ.get("RANKING_DIR", str(DEFAULT_RANKING_DIR)))
    resolved_features_dir = features_dir or Path(os.environ.get("FEATURES_DIR", str(DEFAULT_FEATURES_DIR)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = ModelStore(resolved_ranking_dir, resolved_features_dir)
        logger.info(
            "Loaded ranking model + feature snapshot from %s, %s", resolved_ranking_dir, resolved_features_dir
        )
        yield

    app = FastAPI(title="Ranking Service", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/rank")
    def rank(request: Request, body: RankRequest) -> dict:
        store: ModelStore = request.app.state.store
        try:
            ranked = store.rank(
                body.anchor_item_id,
                body.user_id,
                body.country,
                body.device,
                [candidate.model_dump() for candidate in body.candidates],
            )
        except UnknownItemError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown anchor_item_id: {body.anchor_item_id}") from exc
        return {"anchor_item_id": body.anchor_item_id, "ranked": ranked}

    return app


app = create_app()
