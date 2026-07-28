"""Feature encoding and batch sampling for the Siamese retrieval encoder.

retrieval_training_examples (pipelines/spark_jobs/dataset_schema.py) is a
pure item-pair table: one row per (anchor, candidate) with a click label.
ItemFeatures groups everything one encoder call needs for a single side of
that pair (anchor or candidate) -- the same encoder is applied to both
sides independently (true weight sharing), so this is exactly the input
shape retrieval_model.ItemEncoder.forward consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from training.vocab import Vocabulary, multi_hot_encode


@dataclass
class ItemFeatures:
    """One encoder call's worth of item features, batched.

    image_embedding is zero-filled wherever image_missing is True --
    ItemEncoder substitutes its own learned "missing image" vector in that
    case (see retrieval_model.py), never a bare zero vector, since zero
    could coincide with a legitimate embedding value.
    """

    item_id: torch.Tensor
    category: torch.Tensor
    subcategory: torch.Tensor
    brand: torch.Tensor
    price_tier: torch.Tensor
    tags: torch.Tensor
    text_embedding: torch.Tensor
    image_embedding: torch.Tensor
    image_missing: torch.Tensor

    def __len__(self) -> int:
        return self.item_id.shape[0]


def infer_embedding_dims(df: pd.DataFrame) -> tuple[int, int]:
    """Infers text/image embedding dimensions from the data itself.

    Never hardcoded -- decouples this code from a specific upstream
    text-embedding model choice (currently all-MiniLM-L6-v2, 384-dim, per
    data_gen/config.py). image_embedding is always None in this repo's
    synthetic data (data_gen/item_catalog.py hardcodes it), so there is no
    real embedding to measure -- fall back to the text dimension so
    concatenation is well-defined regardless.

    Args:
        df: A DataFrame with at least an "anchor_text_embedding" column,
            and optionally "anchor_image_embedding"/"candidate_image_embedding".

    Returns:
        tuple[int, int]: (text_embedding_dim, image_embedding_dim).
    """
    text_dim = len(df["anchor_text_embedding"].iloc[0])
    image_dim = text_dim
    for column in ("anchor_image_embedding", "candidate_image_embedding"):
        if column not in df.columns:
            continue
        non_null = df[column].dropna()
        if len(non_null) > 0:
            image_dim = len(non_null.iloc[0])
            break
    return text_dim, image_dim


def encode_item_side(
    df: pd.DataFrame, prefix: str, vocabs: dict[str, Vocabulary], image_dim: int
) -> ItemFeatures:
    """Encodes one side (anchor or candidate) of a retrieval examples DataFrame.

    Args:
        df: A DataFrame with f"{prefix}_item_id", f"{prefix}_category", etc.
            columns (retrieval_training_examples's anchor_*/candidate_*
            naming, or a caller-renamed catalog DataFrame -- see
            retrieval_eval.py's gallery encoding).
        prefix: "anchor" or "candidate" (or a caller-chosen prefix after
            renaming, e.g. "gallery").
        vocabs: Must contain "item_id", "category", "subcategory", "brand",
            "price_tier", "tags" Vocabulary instances.
        image_dim: Dimension to zero-fill image_embedding to when missing.

    Returns:
        ItemFeatures: Batched tensors for this side.
    """
    n = len(df)
    item_id = torch.tensor(
        [vocabs["item_id"].encode(v) for v in df[f"{prefix}_item_id"]], dtype=torch.long
    )
    category = torch.tensor(
        [vocabs["category"].encode(v) for v in df[f"{prefix}_category"]], dtype=torch.long
    )
    subcategory = torch.tensor(
        [vocabs["subcategory"].encode(v) for v in df[f"{prefix}_subcategory"]], dtype=torch.long
    )
    brand = torch.tensor([vocabs["brand"].encode(v) for v in df[f"{prefix}_brand"]], dtype=torch.long)
    price_tier = torch.tensor(
        [vocabs["price_tier"].encode(v) for v in df[f"{prefix}_price_tier"]], dtype=torch.long
    )
    tags = torch.from_numpy(multi_hot_encode(df[f"{prefix}_tags"].tolist(), vocabs["tags"]))
    text_embedding = torch.tensor(
        np.stack(df[f"{prefix}_text_embedding"].to_numpy()).astype(np.float32), dtype=torch.float32
    )

    image_raw = df[f"{prefix}_image_embedding"].tolist()
    image_missing = torch.tensor([v is None for v in image_raw], dtype=torch.bool)
    image_embedding = torch.zeros((n, image_dim), dtype=torch.float32)
    for row_index, value in enumerate(image_raw):
        if value is not None:
            image_embedding[row_index] = torch.tensor(value, dtype=torch.float32)

    return ItemFeatures(
        item_id=item_id,
        category=category,
        subcategory=subcategory,
        brand=brand,
        price_tier=price_tier,
        tags=tags,
        text_embedding=text_embedding,
        image_embedding=image_embedding,
        image_missing=image_missing,
    )


@dataclass
class RetrievalBatch:
    """A sampled training batch: B positive pairs plus each anchor's own
    (up to H) explicit hard negatives, padded to a fixed (B, H) grid.

    hard_neg_valid marks padding -- an anchor with fewer than H available
    hard negatives (or none at all) has its unused slots masked out rather
    than silently treated as real negatives (see retrieval_model.contrastive_loss).
    """

    anchor_rows: pd.DataFrame
    hard_negative_rows: list[pd.DataFrame]  # length B; each entry 0..H rows
    hard_negatives_per_anchor: int


def sample_training_batch(
    df: pd.DataFrame,
    batch_size: int,
    hard_negatives_per_anchor: int,
    rng: np.random.Generator,
) -> RetrievalBatch:
    """Samples one training batch with in-batch and explicit hard negatives.

    Each step samples `batch_size` positive (label=1) rows as the batch's
    anchors. For each, if the dataset has explicit "shown but not clicked"
    (label=0) rows for the same recommendation_impression_id, up to
    `hard_negatives_per_anchor` of them are sampled as that anchor's own
    hard negatives. Positive rows from *other* anchors in the same batch
    are reused as in-batch negatives implicitly, by the loss function, not
    here -- see data-schema.md's documented in-batch-negative mechanism.

    Args:
        df: The (already split) retrieval_training_examples DataFrame to
            sample from.
        batch_size: Number of positive anchors per batch.
        hard_negatives_per_anchor: Max explicit hard negatives per anchor.
        rng: A numpy random Generator, for reproducibility.

    Returns:
        RetrievalBatch: batch_size anchor rows and their hard negatives.
    """
    positive_rows = df[df["label"] == 1]
    negative_rows = df[df["label"] == 0]
    negative_by_impression = {
        impression_id: group
        for impression_id, group in negative_rows.groupby("recommendation_impression_id")
    }

    n = min(batch_size, len(positive_rows))
    sampled_index = rng.choice(positive_rows.index.to_numpy(), size=n, replace=False)
    anchor_rows = positive_rows.loc[sampled_index].reset_index(drop=True)

    hard_negative_rows: list[pd.DataFrame] = []
    for impression_id in anchor_rows["recommendation_impression_id"]:
        candidates = negative_by_impression.get(impression_id)
        if candidates is None or candidates.empty:
            hard_negative_rows.append(candidates.iloc[:0] if candidates is not None else negative_rows.iloc[:0])
            continue
        take = min(hard_negatives_per_anchor, len(candidates))
        seed = int(rng.integers(0, 2**31 - 1))
        hard_negative_rows.append(candidates.sample(n=take, random_state=seed))

    return RetrievalBatch(
        anchor_rows=anchor_rows,
        hard_negative_rows=hard_negative_rows,
        hard_negatives_per_anchor=hard_negatives_per_anchor,
    )


def encode_hard_negatives(
    batch: RetrievalBatch, vocabs: dict[str, Vocabulary], image_dim: int
) -> tuple[ItemFeatures, torch.Tensor]:
    """Pads each anchor's hard negatives to a fixed (B, H) grid for encoding.

    Anchors with fewer than H real hard negatives (or none) are padded by
    repeating a row -- the padded content is never used for anything,
    since contrastive_loss masks padded slots to -inf via the returned
    valid mask, but every slot must still encode to a valid ItemFeatures
    row for the batch to have a uniform shape.

    Args:
        batch: A RetrievalBatch from sample_training_batch.
        vocabs: Vocabularies for encode_item_side.
        image_dim: Image embedding dimension for encode_item_side.

    Returns:
        tuple[ItemFeatures, torch.Tensor]: Flattened (B*H,) ItemFeatures in
            row-major (anchor, slot) order, and a (B, H) bool valid mask.
    """
    h = batch.hard_negatives_per_anchor
    b = len(batch.anchor_rows)
    valid_mask = torch.zeros((b, h), dtype=torch.bool)
    padded_rows = []
    for anchor_index in range(b):
        real_rows = batch.hard_negative_rows[anchor_index]
        for slot in range(h):
            if slot < len(real_rows):
                padded_rows.append(real_rows.iloc[[slot]])
                valid_mask[anchor_index, slot] = True
            elif len(real_rows) > 0:
                padded_rows.append(real_rows.iloc[[-1]])
            else:
                padded_rows.append(batch.anchor_rows.iloc[[anchor_index]])
    padded_df = pd.concat(padded_rows, ignore_index=True)
    features = encode_item_side(padded_df, "candidate", vocabs, image_dim)
    return features, valid_mask
