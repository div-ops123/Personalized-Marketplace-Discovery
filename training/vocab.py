"""Categorical vocabularies shared by both training scripts.

Taxonomy-backed vocabularies (category/subcategory/brand/price_tier/tags)
are built from common/taxonomy.py's fixed, closed-world constants -- not
re-derived from training data, since the taxonomy is already the source of
truth the rest of the pipeline validates against. item_id has no such
closed-world source and is built from the data itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.taxonomy import ALL_BRANDS, CATEGORIES, PRICE_TIERS, TAXONOMY

_UNK = "<UNK>"


class Vocabulary:
    """A token<->index mapping with index 0 reserved for unknown tokens."""

    def __init__(self, tokens: list[str] | None = None):
        """Builds a vocabulary from a list of tokens.

        Args:
            tokens: Tokens to include, in insertion order (deduplicated).
                Index 0 is always reserved for "<UNK>" regardless of
                whether it appears in tokens.
        """
        self.token_to_index: dict[str, int] = {_UNK: 0}
        for token in tokens or []:
            if token not in self.token_to_index:
                self.token_to_index[token] = len(self.token_to_index)

    def __len__(self) -> int:
        return len(self.token_to_index)

    def encode(self, token: str) -> int:
        """Maps a token to its index, falling back to 0 (<UNK>) if unseen.

        Args:
            token: The token to encode.

        Returns:
            int: The token's index, or 0 if it was never in the vocabulary.
        """
        return self.token_to_index.get(token, 0)

    def to_dict(self) -> dict[str, int]:
        """Returns the token->index mapping, for JSON persistence.

        Returns:
            dict[str, int]: A copy of the internal mapping.
        """
        return dict(self.token_to_index)

    @classmethod
    def from_dict(cls, token_to_index: dict[str, int]) -> "Vocabulary":
        """Reconstructs a Vocabulary from a previously-saved mapping.

        Args:
            token_to_index: A mapping as produced by to_dict().

        Returns:
            Vocabulary: A vocabulary with the exact same token->index mapping.
        """
        vocab = cls()
        vocab.token_to_index = dict(token_to_index)
        return vocab


def build_taxonomy_vocabs() -> dict[str, Vocabulary]:
    """Builds the fixed, closed-world vocabularies derived from the taxonomy.

    Returns:
        dict[str, Vocabulary]: Keyed by "category", "subcategory", "brand",
            "price_tier", "tags".
    """
    subcategories = sorted({sub for spec in TAXONOMY.values() for sub in spec["subcategories"]})
    tags = sorted({tag for spec in TAXONOMY.values() for tag in spec["tags"]})
    return {
        "category": Vocabulary(CATEGORIES),
        "subcategory": Vocabulary(subcategories),
        "brand": Vocabulary(ALL_BRANDS),
        "price_tier": Vocabulary(PRICE_TIERS),
        "tags": Vocabulary(tags),
    }


def build_item_id_vocab(item_ids: pd.Series) -> Vocabulary:
    """Builds a data-derived item_id vocabulary from a training split.

    Unlike the taxonomy vocabs, item IDs have no closed-world source --
    this must be built from whichever items actually appear in the split
    it's called on. An item never seen in training falls back to <UNK> at
    encode time (no learned embedding) -- this is intentionally the
    cold-start-bootstrap path build-phases.md's Phase 5 says not to
    implement in v1.

    Args:
        item_ids: A Series of item IDs (may contain duplicates).

    Returns:
        Vocabulary: Built from the sorted unique item IDs, for determinism.
    """
    return Vocabulary(sorted(item_ids.dropna().unique().tolist()))


def multi_hot_encode(values: list, vocab: Vocabulary) -> np.ndarray:
    """Multi-hot encodes a column of token lists against a vocabulary.

    Args:
        values: One entry per row, each either a list of tokens or None
            (None means "no data" -- e.g. a cold-start user with no
            snapshot yet -- and encodes to an all-zero row, distinct from
            an empty list which means "known to have zero of these").
        vocab: The vocabulary to encode tokens against.

    Returns:
        np.ndarray: Shape (len(values), len(vocab)), dtype float32.
    """
    matrix = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for row_index, tokens in enumerate(values):
        if tokens is None:
            continue
        for token in tokens:
            matrix[row_index, vocab.encode(token)] = 1.0
    return matrix
