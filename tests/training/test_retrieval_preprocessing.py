"""Unit tests for training/retrieval_preprocessing.py's batch sampler, and
for training/retrieval_model.py's masked-similarity-matrix construction
(the two are tested together since the mask is only meaningful in terms
of what sample_training_batch actually produces)."""

import numpy as np
import pandas as pd
import torch

from training.retrieval_model import contrastive_loss
from training.retrieval_preprocessing import sample_training_batch


def _synthetic_retrieval_df(n_impressions: int = 20, candidates_per_impression: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n_impressions):
        impression_id = f"impr_{i}"
        anchor_id = f"anchor_{i}"
        for c in range(candidates_per_impression):
            rows.append(
                {
                    "recommendation_impression_id": impression_id,
                    "anchor_item_id": anchor_id,
                    "candidate_item_id": f"cand_{i}_{c}",
                    "label": 1 if c == 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def test_sample_training_batch_shapes():
    df = _synthetic_retrieval_df(n_impressions=20, candidates_per_impression=4)
    rng = np.random.default_rng(0)
    batch = sample_training_batch(df, batch_size=8, hard_negatives_per_anchor=2, rng=rng)

    assert len(batch.anchor_rows) == 8
    assert (batch.anchor_rows["label"] == 1).all()
    assert len(batch.hard_negative_rows) == 8
    for hard_negs in batch.hard_negative_rows:
        assert len(hard_negs) <= 2


def test_sample_training_batch_no_duplicate_anchors():
    df = _synthetic_retrieval_df(n_impressions=20, candidates_per_impression=4)
    rng = np.random.default_rng(1)
    batch = sample_training_batch(df, batch_size=8, hard_negatives_per_anchor=1, rng=rng)
    assert batch.anchor_rows["recommendation_impression_id"].nunique() == len(batch.anchor_rows)


def test_sample_training_batch_hard_negatives_belong_to_same_impression():
    df = _synthetic_retrieval_df(n_impressions=20, candidates_per_impression=4)
    rng = np.random.default_rng(2)
    batch = sample_training_batch(df, batch_size=8, hard_negatives_per_anchor=2, rng=rng)
    for anchor_row, hard_negs in zip(batch.anchor_rows.itertuples(), batch.hard_negative_rows):
        if len(hard_negs) > 0:
            assert (hard_negs["recommendation_impression_id"] == anchor_row.recommendation_impression_id).all()
            assert (hard_negs["label"] == 0).all()


def test_contrastive_loss_masks_invalid_hard_negatives():
    torch.manual_seed(0)
    anchor_emb = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    pos_emb = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    hard_neg_emb = torch.nn.functional.normalize(torch.randn(4, 2, 8), dim=-1)
    all_valid = torch.ones(4, 2, dtype=torch.bool)
    all_invalid = torch.zeros(4, 2, dtype=torch.bool)

    loss_with_negs = contrastive_loss(anchor_emb, pos_emb, hard_neg_emb, all_valid)
    loss_all_masked = contrastive_loss(anchor_emb, pos_emb, hard_neg_emb, all_invalid)
    loss_no_hard_neg_arg = contrastive_loss(anchor_emb, pos_emb)

    # Fully-masked hard negatives must contribute nothing to the loss --
    # identical to not passing hard negatives at all.
    assert torch.isclose(loss_all_masked, loss_no_hard_neg_arg, atol=1e-5)
    # Real (valid) hard negatives add extra negative mass to the
    # denominator, so the loss must differ from the fully-masked case.
    assert not torch.isclose(loss_with_negs, loss_all_masked, atol=1e-6)


def test_contrastive_loss_diagonal_targets_are_correct():
    torch.manual_seed(1)
    anchor_emb = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    matching_pos_emb = anchor_emb.clone()  # diagonal is a perfect match
    shuffled_pos_emb = anchor_emb[[1, 2, 3, 0]]  # cyclic shift -- no fixed points

    loss_matching = contrastive_loss(anchor_emb, matching_pos_emb)
    loss_shuffled = contrastive_loss(anchor_emb, shuffled_pos_emb)
    assert loss_matching < loss_shuffled
