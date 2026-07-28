"""Standalone NDCG@K -- doesn't depend on a live Booster, so it's testable
without training anything (see tests/training/test_ranking_eval.py).

Per docs/metric-ladder.md, no cross-table join is needed here (unlike
retrieval's Recall@K in retrieval_eval.py) -- ranking_training_examples's
label already *is* "recommendation-attributed purchase," exactly the
documented NDCG@K ground truth.
"""

import numpy as np
import pandas as pd

from training.constants import RANKING_NDCG_K


def _dcg_at_k(gains: np.ndarray, k: int) -> float:
    """Discounted cumulative gain for the top-k of an already-ranked list.

    Args:
        gains: Relevance values, already ordered by descending predicted
            score (index 0 = rank 1).
        k: Cutoff.

    Returns:
        float: DCG@k.
    """
    top = gains[:k]
    discounts = np.log2(np.arange(2, len(top) + 2))
    return float(np.sum(top / discounts))


def ndcg_at_k(
    y_true: pd.Series, y_pred: np.ndarray, groups: list[int], k: int = RANKING_NDCG_K
) -> float:
    """Computes mean NDCG@K across query groups.

    Args:
        y_true: Binary relevance labels, row-aligned with y_pred, ordered
            so each group's rows are contiguous (build_query_groups's
            output shape).
        y_pred: Predicted scores, same order/alignment as y_true.
        groups: Per-group row counts, in the same contiguous order.
        k: Cutoff.

    Returns:
        float: Mean NDCG@K over groups with >=1 positive label (groups
            with none are excluded, mirroring retrieval_eval's same rule).
            NaN if no group has a positive.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    scores = []
    offset = 0
    for size in groups:
        group_true = y_true_arr[offset : offset + size]
        group_pred = y_pred_arr[offset : offset + size]
        offset += size

        if group_true.sum() == 0:
            continue

        predicted_order = np.argsort(-group_pred)
        dcg = _dcg_at_k(group_true[predicted_order], k)
        ideal_order = np.argsort(-group_true)
        idcg = _dcg_at_k(group_true[ideal_order], k)
        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(scores)) if scores else float("nan")
