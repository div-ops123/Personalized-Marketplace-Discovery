"""Unit tests for training/ranking_eval.py's ndcg_at_k."""

import math

import numpy as np

from training.ranking_eval import ndcg_at_k


def test_ndcg_at_k_perfect_ranking_is_one():
    y_true = [0, 0, 1]
    y_pred = [0.1, 0.2, 0.9]  # ranks the positive first
    assert math.isclose(ndcg_at_k(y_true, y_pred, groups=[3], k=3), 1.0, abs_tol=1e-6)


def test_ndcg_at_k_worst_ranking_is_between_zero_and_one():
    y_true = [1, 0, 0]
    y_pred = [0.1, 0.2, 0.9]  # positive ranked last
    ndcg = ndcg_at_k(y_true, y_pred, groups=[3], k=3)
    assert 0 < ndcg < 1.0


def test_ndcg_at_k_group_with_no_positive_is_excluded_from_average():
    y_true = [0, 0, 0, 1]
    y_pred = [0.5, 0.4, 0.3, 0.9]
    # First group (size 3) has no positive -> excluded. Second group
    # (size 1) is trivially perfect -> average should be exactly 1.0, not
    # dragged down by the excluded group.
    ndcg = ndcg_at_k(y_true, y_pred, groups=[3, 1], k=3)
    assert math.isclose(ndcg, 1.0, abs_tol=1e-6)


def test_ndcg_at_k_no_positives_anywhere_is_nan():
    y_true = [0, 0, 0]
    y_pred = [0.1, 0.2, 0.3]
    assert np.isnan(ndcg_at_k(y_true, y_pred, groups=[3], k=3))
