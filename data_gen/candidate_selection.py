"""Item-item content-similarity candidate selection.

Stands in for what a trained retrieval encoder will eventually learn -- no
retrieval model exists yet when this raw event log is generated, so
candidate ranking and the logged retrieval_similarity_score both reuse the
same proxy: cosine similarity between anchor and candidate text_embedding
vectors (data-schema.md's own documented v0 bootstrap heuristic), plus
noise so ranking isn't a deterministic function of content alone.
"""

import numpy as np
import pandas as pd

from config import SIMILARITY_NOISE_STD


def build_item_embedding_matrix(items_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Builds an L2-normalized embedding matrix for cosine-similarity scoring.

    Args:
        items_df: Item catalog rows with item_id and text_embedding columns.

    Returns:
        tuple[np.ndarray, np.ndarray]: (item_ids, matrix) where matrix has
            shape (n_items, dim), each row L2-normalized so a plain dot
            product against a row equals cosine similarity.
    """
    item_ids = items_df["item_id"].to_numpy()
    matrix = np.stack(items_df["text_embedding"].to_numpy()).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms
    return item_ids, matrix


def build_exclusion_mask(item_ids: np.ndarray, excluded_ids: set[str]) -> np.ndarray:
    """Builds a boolean mask marking permanently-excluded items (e.g. cold-start).

    Meant to be computed once per run and reused across every
    score_candidates call -- excluded_ids (e.g. the reserved cold-start
    set) doesn't change per anchor view, so recomputing a set-membership
    scan on every call is wasted work. Plain Python `in` checks against a
    set are used rather than np.isin, which falls back to a slow
    element-by-element comparison for string arrays.

    Args:
        item_ids: All item ids, aligned with a matrix's rows.
        excluded_ids: Item ids that must never appear as candidates.

    Returns:
        np.ndarray: Boolean mask, shape (len(item_ids),).
    """
    return np.array([item_id in excluded_ids for item_id in item_ids], dtype=bool)


def score_candidates(
    anchor_id: str,
    item_ids: np.ndarray,
    matrix: np.ndarray,
    exclusion_mask: np.ndarray,
    rng: np.random.Generator,
    pool_size: int,
) -> list[tuple[str, float]]:
    """Ranks candidate items by noisy cosine similarity to the anchor item.

    Args:
        anchor_id: The anchor item's id. Always excluded from its own
            candidate pool.
        item_ids: All item ids, aligned with matrix's rows (see
            build_item_embedding_matrix).
        matrix: L2-normalized embedding matrix, shape (n_items, dim).
        exclusion_mask: Boolean mask of permanently-excluded items (see
            build_exclusion_mask), e.g. reserved cold-start items.
        rng: A numpy random Generator.
        pool_size: Number of top-scoring candidates to return.

    Returns:
        list[tuple[str, float]]: (item_id, similarity_score) pairs sorted
            by score descending, length min(pool_size, eligible items).
    """
    anchor_index = np.flatnonzero(item_ids == anchor_id)[0]
    similarities = matrix @ matrix[anchor_index]
    noisy_scores = similarities + rng.normal(0.0, SIMILARITY_NOISE_STD, size=len(item_ids))
    noisy_scores = np.clip(noisy_scores, -1.0, 1.0)

    ineligible = exclusion_mask.copy()
    ineligible[anchor_index] = True
    noisy_scores = np.where(ineligible, -np.inf, noisy_scores)

    k = min(pool_size, len(item_ids) - int(ineligible.sum()))
    if k <= 0:
        return []

    top_indices = np.argpartition(noisy_scores, -k)[-k:]
    top_indices = top_indices[np.argsort(noisy_scores[top_indices])[::-1]]
    return [(item_ids[i], float(noisy_scores[i])) for i in top_indices]
