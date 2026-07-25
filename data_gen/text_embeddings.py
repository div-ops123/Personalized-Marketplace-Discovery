"""Real text embeddings for item descriptions via a pretrained sentence-transformer.

These are genuine embeddings of real (synthetic) text, not fabricated
vectors — required so the cold-start content-similarity heuristic in
docs/data-schema.md (image/text embedding distance) has real semantic
meaning to work with.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import TEXT_EMBEDDING_MODEL_NAME


def load_text_encoder() -> SentenceTransformer:
    """Loads the pretrained sentence-transformer used for description embeddings.

    Returns:
        SentenceTransformer: A ready-to-use encoder.
    """
    return SentenceTransformer(TEXT_EMBEDDING_MODEL_NAME)


def embed_descriptions(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Embeds a batch of item descriptions.

    Args:
        texts: Item description strings, none of which may be empty.
        model: An encoder exposing an `encode(list[str]) -> np.ndarray` method
            (a real SentenceTransformer, or a stub of the same shape in tests).

    Returns:
        np.ndarray: Shape (len(texts), embedding_dim).

    Raises:
        ValueError: If texts is empty or contains an empty/blank string.
    """
    if not texts:
        raise ValueError("texts must not be empty")
    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain empty or blank strings")
    return model.encode(texts)
