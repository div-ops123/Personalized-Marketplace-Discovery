"""Unit tests for training/retrieval_eval.py's recall_at_k."""

import numpy as np
import pandas as pd

from common.taxonomy import CATEGORIES
from training.retrieval_eval import recall_at_k
from training.retrieval_model import ItemEncoder
from training.vocab import build_item_id_vocab, build_taxonomy_vocabs


def _synthetic_catalog(n_items: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n_items):
        category = CATEGORIES[i % len(CATEGORIES)]
        rows.append(
            {
                "item_id": f"item_{i}",
                "category": category,
                "subcategory": "Sneakers" if category == "Footwear" else "Fiction",
                "brand": "Nike" if category == "Footwear" else "Penguin",
                "price": 50.0,
                "tags": ["casual"],
                "image_embedding": None,
                "text_embedding": [0.1] * 8,
            }
        )
    return pd.DataFrame(rows)


def _model_and_vocabs(catalog_df: pd.DataFrame, embedding_dim: int = 16):
    vocabs = build_taxonomy_vocabs()
    vocabs["item_id"] = build_item_id_vocab(catalog_df["item_id"])
    vocab_sizes = {
        "item_id": len(vocabs["item_id"]),
        "category": len(vocabs["category"]),
        "subcategory": len(vocabs["subcategory"]),
        "brand": len(vocabs["brand"]),
        "price_tier": len(vocabs["price_tier"]),
    }
    model = ItemEncoder(
        vocab_sizes, num_tags=len(vocabs["tags"]), text_dim=8, image_dim=8, embedding_dim=embedding_dim
    )
    model.eval()
    return model, vocabs


def test_recall_at_k_full_gallery_cutoff_is_always_one():
    # When k covers the whole gallery, every purchased item is necessarily
    # in the top-k regardless of the (untrained) model's actual ranking --
    # this isolates and verifies the label-grouping/gallery-construction
    # logic from the model's ranking quality.
    catalog_df = _synthetic_catalog(n_items=6)
    model, vocabs = _model_and_vocabs(catalog_df)

    eval_positives_df = pd.DataFrame(
        {
            "anchor_item_id": ["item_0", "item_0", "item_1"],
            "candidate_item_id": ["item_2", "item_3", "item_4"],
            "label": [1, 1, 1],
        }
    )

    recall = recall_at_k(model, eval_positives_df, catalog_df, vocabs, image_dim=8, k=len(catalog_df))
    assert recall == 1.0


def test_recall_at_k_no_positive_labels_is_nan():
    catalog_df = _synthetic_catalog(n_items=6)
    model, vocabs = _model_and_vocabs(catalog_df)
    eval_positives_df = pd.DataFrame(
        {
            "anchor_item_id": ["item_0"],
            "candidate_item_id": ["item_2"],
            "label": [0],
        }
    )
    recall = recall_at_k(model, eval_positives_df, catalog_df, vocabs, image_dim=8, k=5)
    assert np.isnan(recall)
