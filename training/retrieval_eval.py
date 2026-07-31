"""Offline Recall@K for the retrieval encoder.

Deliberately does NOT reuse retrieval_training_examples's click labels as
ground truth. Per docs/metric-ladder.md: training uses abundant click
signal, but Recall@K is evaluated against the rarer, business-aligned
*purchase* signal -- explicitly documented as a distinct thing. Ground
truth here is ranking_training_examples's recommendation-attributed
purchase label (label==1), reusing the already-tested Phase 4 attribution
join rather than re-deriving it.

The candidate gallery is the *full* item catalog, not just the ~20
candidates logged per impression -- matching how retrieval actually
operates in production (ANN search over the whole index), and the only way
a K=200 cutoff is meaningful (each impression only ever logs ~20
candidates).
"""

import numpy as np
import pandas as pd
from torch import nn

from training.constants import RETRIEVAL_RECALL_K
from training.retrieval_preprocessing import embed_catalog
from training.vocab import Vocabulary


def recall_at_k(
    model: nn.Module,
    eval_positives_df: pd.DataFrame,
    item_catalog_df: pd.DataFrame,
    vocabs: dict[str, Vocabulary],
    image_dim: int,
    k: int = RETRIEVAL_RECALL_K,
) -> float:
    """Computes Recall@K against purchase-labeled ground truth.

    Args:
        model: A trained (or in-training) ItemEncoder.
        eval_positives_df: ranking_training_examples rows for the eval
            window (val or test split), used only for its
            (anchor_item_id, candidate_item_id, label) purchase ground
            truth -- no ranking features are read.
        item_catalog_df: Full item catalog (training/db_io.py's
            read_full_item_catalog) -- the retrieval gallery.
        vocabs: Vocabularies used to encode both anchors and the gallery.
        image_dim: Image embedding dimension.
        k: Cutoff for Recall@K.

    Returns:
        float: Mean Recall@K over anchors with >=1 purchased positive in
            the eval window, or NaN if there are none.
    """
    model.eval()
    positives = eval_positives_df[eval_positives_df["label"] == 1]
    if positives.empty:
        return float("nan")
    purchased_by_anchor = positives.groupby("anchor_item_id")["candidate_item_id"].apply(set).to_dict()

    gallery_embeddings = embed_catalog(model, item_catalog_df, vocabs, image_dim)
    gallery_item_ids = item_catalog_df["item_id"].tolist()

    anchor_ids = list(purchased_by_anchor.keys())
    anchor_catalog = item_catalog_df[item_catalog_df["item_id"].isin(anchor_ids)]
    if anchor_catalog.empty:
        return float("nan")
    anchor_embeddings = embed_catalog(model, anchor_catalog, vocabs, image_dim)
    anchor_item_ids = anchor_catalog["item_id"].tolist()

    similarities = anchor_embeddings @ gallery_embeddings.T
    top_k = similarities.topk(min(k, similarities.shape[1]), dim=1).indices

    recalls = []
    for row_index, anchor_id in enumerate(anchor_item_ids):
        purchased = purchased_by_anchor.get(anchor_id)
        if not purchased:
            continue
        retrieved = {gallery_item_ids[i] for i in top_k[row_index].tolist()}
        recalls.append(len(retrieved & purchased) / len(purchased))

    return float(np.mean(recalls)) if recalls else float("nan")
