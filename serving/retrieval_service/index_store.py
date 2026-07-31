"""Loads the FAISS/HNSW retrieval index once and serves similar-item
lookups by anchor item_id. No live model inference and no Postgres/MLflow
dependency -- see serving/build_retrieval_index.py's docstring: item
embeddings are precomputed at index-build time, so a request is purely an
embedding lookup (index.reconstruct) + ANN search (index.search).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class AnchorNotFoundError(KeyError):
    """Raised when an anchor item_id isn't in the index."""


class IndexStore:
    """A loaded FAISS index plus its item_id<->position mapping."""

    def __init__(self, index_dir: Path):
        """Loads items.faiss and item_ids.json from index_dir.

        Args:
            index_dir: Directory containing items.faiss and item_ids.json,
                as written by serving/build_retrieval_index.py.
        """
        self._index = faiss.read_index(str(index_dir / "items.faiss"))
        self._item_ids: list[str] = json.loads((index_dir / "item_ids.json").read_text())
        # item_ids.json is position->item_id (see build_retrieval_index.py's
        # design contract); no reverse map is persisted anywhere, so build
        # it once here.
        self._position_by_item_id = {item_id: position for position, item_id in enumerate(self._item_ids)}

    @property
    def item_count(self) -> int:
        """int: Number of items in the loaded index."""
        return self._index.ntotal

    def similar_items(self, anchor_item_id: str, k: int) -> list[tuple[str, float]]:
        """Returns the top-k items most similar to anchor_item_id.

        The anchor is always its own nearest neighbor (score 1.0 under
        cosine/inner-product) -- recommending an item as "similar to
        itself" would be a real product bug, so it's excluded from the
        returned results. k+1 is requested from FAISS so excluding it
        still leaves k real candidates.

        Args:
            anchor_item_id: The PDP item being viewed.
            k: Number of neighbors to return (excluding the anchor itself).

        Returns:
            list[tuple[str, float]]: (item_id, similarity_score) pairs,
                descending by score, length <= k.

        Raises:
            AnchorNotFoundError: If anchor_item_id isn't in the index.
        """
        position = self._position_by_item_id.get(anchor_item_id)
        if position is None:
            raise AnchorNotFoundError(anchor_item_id)

        query = np.ascontiguousarray(self._index.reconstruct(position).reshape(1, -1))
        scores, positions = self._index.search(query, k + 1)
        results = [
            (self._item_ids[p], float(s))
            for s, p in zip(scores[0], positions[0])
            if p != -1 and self._item_ids[p] != anchor_item_id
        ]
        return results[:k]
