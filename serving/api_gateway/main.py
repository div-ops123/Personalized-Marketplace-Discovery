"""FastAPI API Gateway -- the single entrypoint a client calls: 
fans out to the Retrieval Service, then the Ranking Service, 
then enriches the top-k with catalog metadata for display 
(category/brand/price -- no images, per the catalog generation
step in docs/build-phases.md leaving image null).

Error handling is a plain 503 passthrough if retrieval/ranking is
unreachable -- not the fuller fallback-cache/circuit-breaker behavior
docs/LLD.md:166 describes for production scale. Deliberate scope trim for
this local demo profile, not a silent gap.

Run standalone (outside docker-compose.serving.yml, for local testing):
    uvicorn serving.api_gateway.main:app --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request

from serving.constants import DEFAULT_FEATURES_DIR, DEFAULT_RESULT_K, DEFAULT_RETRIEVAL_K

logger = logging.getLogger(__name__)


def create_app(
    retrieval_url: str | None = None,
    ranking_url: str | None = None,
    features_dir: Path | None = None,
    http_client: httpx.Client | None = None,
) -> FastAPI:
    """Builds the API Gateway FastAPI app.

    A factory (not a bare module-level app) so tests can point it at a
    tmp_path catalog fixture and inject a mocked HTTP client instead of
    making real network calls to the other two services.

    Args:
        retrieval_url: Base URL of the Retrieval Service. Defaults to the
            RETRIEVAL_SERVICE_URL env var, then http://localhost:8001.
        ranking_url: Base URL of the Ranking Service. Defaults to the
            RANKING_SERVICE_URL env var, then http://localhost:8002.
        features_dir: Directory containing item_catalog.json (for response
            enrichment). Defaults to the FEATURES_DIR env var, then
            DEFAULT_FEATURES_DIR.
        http_client: An httpx.Client to use instead of creating one --
            tests pass one built with httpx.MockTransport. If omitted, a
            real client is created on startup and closed on shutdown; a
            caller-supplied client is never closed here (its lifecycle is
            the caller's, e.g. the test that built it).

    Returns:
        FastAPI: The configured app, catalog + HTTP client set up on
            startup via lifespan.
    """
    resolved_retrieval_url = retrieval_url or os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:8001")
    resolved_ranking_url = ranking_url or os.environ.get("RANKING_SERVICE_URL", "http://localhost:8002")
    resolved_features_dir = features_dir or Path(os.environ.get("FEATURES_DIR", str(DEFAULT_FEATURES_DIR)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.item_catalog = json.loads((resolved_features_dir / "item_catalog.json").read_text())
        owns_client = http_client is None
        app.state.http_client = http_client or httpx.Client(timeout=10.0)
        logger.info(
            "API Gateway ready: retrieval=%s ranking=%s", resolved_retrieval_url, resolved_ranking_url
        )
        yield
        if owns_client:
            app.state.http_client.close()

    app = FastAPI(title="API Gateway", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/recommendations")
    def recommendations(
        request: Request,
        anchor_item_id: str,
        user_id: str,
        country: str = "US",
        device: str = "mobile",
        k: int = DEFAULT_RESULT_K,
    ) -> dict:
        client: httpx.Client = request.app.state.http_client
        catalog: dict = request.app.state.item_catalog

        try:
            retrieval_response = client.get(
                f"{resolved_retrieval_url}/similar-items",
                params={"anchor_item_id": anchor_item_id, "k": DEFAULT_RETRIEVAL_K},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="Retrieval Service unavailable") from exc
        if retrieval_response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Unknown anchor_item_id: {anchor_item_id}")
        if retrieval_response.status_code != 200:
            raise HTTPException(status_code=503, detail="Retrieval Service unavailable")
        candidates = retrieval_response.json()["candidates"]

        try:
            ranking_response = client.post(
                f"{resolved_ranking_url}/rank",
                json={
                    "anchor_item_id": anchor_item_id,
                    "user_id": user_id,
                    "country": country,
                    "device": device,
                    "candidates": candidates,
                },
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="Ranking Service unavailable") from exc
        if ranking_response.status_code != 200:
            raise HTTPException(status_code=503, detail="Ranking Service unavailable")
        ranked = ranking_response.json()["ranked"][:k]

        results = []
        for rank_position, entry in enumerate(ranked, start=1):
            item_id = entry["item_id"]
            metadata = catalog.get(item_id, {})
            results.append(
                {
                    "rank": rank_position,
                    "item_id": item_id,
                    "score": entry["rank_score"],
                    "category": metadata.get("category"),
                    "subcategory": metadata.get("subcategory"),
                    "brand": metadata.get("brand"),
                    "price": metadata.get("price"),
                }
            )

        return {"anchor_item_id": anchor_item_id, "user_id": user_id, "recommendations": results}

    return app


app = create_app()
