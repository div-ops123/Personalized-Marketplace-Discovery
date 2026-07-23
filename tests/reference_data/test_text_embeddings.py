"""Unit tests for text embedding, using a stub encoder (no real model load)."""

import numpy as np
import pytest

from text_embeddings import embed_descriptions


class _StubEncoder:
    """Deterministic stand-in for SentenceTransformer: hashes each string to a fixed-dim vector."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[float(hash(text) % 1000)] * 4 for text in texts])


def test_embed_descriptions_shape():
    """Returns one row per input text, with a consistent dimension."""
    vectors = embed_descriptions(["a shoe", "a book"], _StubEncoder())
    assert vectors.shape == (2, 4)


def test_embed_descriptions_same_text_same_vector():
    """Identical text embeds to an identical vector."""
    vectors = embed_descriptions(["same text", "same text"], _StubEncoder())
    assert (vectors[0] == vectors[1]).all()


def test_embed_descriptions_different_text_different_vector():
    """Different text is very likely to embed to a different vector."""
    vectors = embed_descriptions(["a red sneaker", "a blue novel"], _StubEncoder())
    assert not (vectors[0] == vectors[1]).all()


def test_embed_descriptions_empty_list_raises():
    """No descriptions to embed is an explicit error, not a silent no-op."""
    with pytest.raises(ValueError):
        embed_descriptions([], _StubEncoder())


def test_embed_descriptions_blank_string_raises():
    """A blank description is rejected rather than silently embedded as empty."""
    with pytest.raises(ValueError):
        embed_descriptions(["valid text", "   "], _StubEncoder())
