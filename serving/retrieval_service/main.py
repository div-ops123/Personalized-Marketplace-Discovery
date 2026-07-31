"""FastAPI Retrieval Service (docs/build-phases.md Phase 6) -- serves PDP
"similar items" candidates from the precomputed FAISS/HNSW index. No live
model inference and no Postgres/MLflow dependency at request time -- only
the static index mounted read-only from serving/index/ (see
docs/LLD.md:109: item embeddings are precomputed at indexing time, not
recomputed at request time).

Run standalone (outside docker-compose.serving.yml, for local testing):
    uvicorn serving.retrieval_service.main:app --port 8001
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from serving.constants import DEFAULT_INDEX_DIR, DEFAULT_RETRIEVAL_K
from serving.retrieval_service.index_store import AnchorNotFoundError, IndexStore

logger = logging.getLogger(__name__)


def create_app(index_dir: Path | None = None) -> FastAPI:
    """Builds the Retrieval Service FastAPI app.

    A factory (not a bare module-level app) so tests can point the loaded
    index at a tmp_path fixture instead of the real serving/index/ mount.

    Args:
        index_dir: Directory to load items.faiss/item_ids.json from.
            Defaults to the INDEX_DIR env var, then DEFAULT_INDEX_DIR.

    Returns:
        FastAPI: The configured app, index loaded on startup via lifespan.
    """
    resolved_dir = index_dir or Path(os.environ.get("INDEX_DIR", str(DEFAULT_INDEX_DIR)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = IndexStore(resolved_dir)
        logger.info("Loaded FAISS index: %d items from %s", app.state.store.item_count, resolved_dir)
        yield

    app = FastAPI(title="Retrieval Service", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        store: IndexStore = request.app.state.store
        return {"status": "ok", "item_count": store.item_count}

    @app.get("/similar-items")
    def similar_items(request: Request, anchor_item_id: str, k: int = DEFAULT_RETRIEVAL_K) -> dict:
        store: IndexStore = request.app.state.store
        try:
            results = store.similar_items(anchor_item_id, k)
        except AnchorNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown anchor_item_id: {anchor_item_id}") from exc
        return {
            "anchor_item_id": anchor_item_id,
            "candidates": [
                {"item_id": item_id, "retrieval_similarity_score": score} for item_id, score in results
            ],
        }

    return app


app = create_app()
