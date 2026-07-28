"""The shared-weight Siamese item encoder (docs/data-schema.md Stage 1 --
no User Tower). One ItemEncoder instance is called independently on the
anchor batch and the candidate batch; true weight sharing, not two towers
with weights copied after the fact.
"""

import torch
import torch.nn.functional as F
from torch import nn

from training.constants import RETRIEVAL_EMBEDDING_DIM
from training.retrieval_preprocessing import ItemFeatures

_ITEM_ID_DIM = 64
_CATEGORY_DIM = 16
_SUBCATEGORY_DIM = 16
_BRAND_DIM = 32
_PRICE_TIER_DIM = 8
_HIDDEN_DIM = 256


class ItemEncoder(nn.Module):
    """Encodes one item (anchor or candidate) into an L2-normalized vector."""

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        num_tags: int,
        text_dim: int,
        image_dim: int,
        embedding_dim: int = RETRIEVAL_EMBEDDING_DIM,
    ):
        """Builds the encoder.

        Args:
            vocab_sizes: len() of each Vocabulary, keyed "item_id",
                "category", "subcategory", "brand", "price_tier".
            num_tags: len(vocabs["tags"]) -- the multi-hot tags width.
            text_dim: Text embedding dimension (see infer_embedding_dims).
            image_dim: Image embedding dimension (see infer_embedding_dims).
            embedding_dim: Output embedding dimension.
        """
        super().__init__()
        self.item_id_embedding = nn.Embedding(vocab_sizes["item_id"], _ITEM_ID_DIM)
        self.category_embedding = nn.Embedding(vocab_sizes["category"], _CATEGORY_DIM)
        self.subcategory_embedding = nn.Embedding(vocab_sizes["subcategory"], _SUBCATEGORY_DIM)
        self.brand_embedding = nn.Embedding(vocab_sizes["brand"], _BRAND_DIM)
        self.price_tier_embedding = nn.Embedding(vocab_sizes["price_tier"], _PRICE_TIER_DIM)

        # A learned stand-in for a missing image embedding -- not zeros,
        # since zero could coincide with a legitimate embedding value.
        # image_embedding is always None in this repo's synthetic data
        # (data_gen/item_catalog.py), so this path is exercised on every row
        # today; it's also what keeps the architecture correct once real
        # image embeddings exist.
        self.missing_image_vector = nn.Parameter(torch.randn(image_dim) * 0.01)

        input_dim = (
            _ITEM_ID_DIM + _CATEGORY_DIM + _SUBCATEGORY_DIM + _BRAND_DIM + _PRICE_TIER_DIM
            + num_tags + text_dim + image_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, _HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(_HIDDEN_DIM, embedding_dim),
        )

    def forward(self, features: ItemFeatures) -> torch.Tensor:
        """Encodes a batch of items.

        Args:
            features: Batched item features (either anchors or candidates).

        Returns:
            torch.Tensor: (batch, embedding_dim), L2-normalized.
        """
        missing = features.image_missing.unsqueeze(-1)
        missing_vector = self.missing_image_vector.unsqueeze(0).expand(len(features), -1)
        image_embedding = torch.where(missing, missing_vector, features.image_embedding)

        x = torch.cat(
            [
                self.item_id_embedding(features.item_id),
                self.category_embedding(features.category),
                self.subcategory_embedding(features.subcategory),
                self.brand_embedding(features.brand),
                self.price_tier_embedding(features.price_tier),
                features.tags,
                features.text_embedding,
                image_embedding,
            ],
            dim=-1,
        )
        return F.normalize(self.mlp(x), p=2, dim=-1)


def contrastive_loss(
    anchor_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    hard_neg_emb: torch.Tensor | None = None,
    hard_neg_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch softmax / InfoNCE contrastive loss.

    Positive pairs from other anchors in the same batch are reused as
    in-batch negatives (data-schema.md's documented mechanism) via the
    off-diagonal of anchor_emb @ pos_emb.T. Each anchor's own explicit
    "shown but not clicked" hard negatives are appended as a separate,
    per-anchor block so they never penalize *other* anchors (a hard
    negative for anchor i may be semantically fine for anchor j).

    Args:
        anchor_emb: (B, D) anchor embeddings.
        pos_emb: (B, D) the batch's true-positive candidate embeddings,
            row-aligned with anchor_emb.
        hard_neg_emb: (B, H, D) each anchor's own hard-negative candidate
            embeddings, or None to train with in-batch negatives only.
        hard_neg_valid: (B, H) bool, True where a slot is a real (not
            padded) hard negative. Required if hard_neg_emb is given.

    Returns:
        torch.Tensor: Scalar cross-entropy loss.
    """
    b = anchor_emb.shape[0]
    in_batch_sim = anchor_emb @ pos_emb.T  # (B, B); diagonal = true positive

    if hard_neg_emb is None:
        logits = in_batch_sim
    else:
        h = hard_neg_emb.shape[1]
        own_hard_sim = torch.einsum("bd,bhd->bh", anchor_emb, hard_neg_emb)  # (B, H)
        own_hard_sim = own_hard_sim.masked_fill(~hard_neg_valid, float("-inf"))

        full_hard_sim = torch.full((b, b * h), float("-inf"), device=anchor_emb.device, dtype=anchor_emb.dtype)
        for i in range(b):
            full_hard_sim[i, i * h : (i + 1) * h] = own_hard_sim[i]
        logits = torch.cat([in_batch_sim, full_hard_sim], dim=1)

    targets = torch.arange(b, device=anchor_emb.device)
    return F.cross_entropy(logits, targets)
