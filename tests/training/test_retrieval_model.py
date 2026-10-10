"""Unit tests for training/retrieval_model.py's ItemEncoder."""

import torch

from training.retrieval_model import ItemEncoder
from training.retrieval_preprocessing import ItemFeatures

_VOCAB_SIZES = {"item_id": 5, "category": 3, "subcategory": 3, "brand": 3, "price_tier": 3}


def _tiny_features(n: int, num_tags: int = 5, text_dim: int = 8) -> ItemFeatures:
    return ItemFeatures(
        item_id=torch.randint(0, _VOCAB_SIZES["item_id"], (n,)),
        category=torch.randint(0, _VOCAB_SIZES["category"], (n,)),
        subcategory=torch.randint(0, _VOCAB_SIZES["subcategory"], (n,)),
        brand=torch.randint(0, _VOCAB_SIZES["brand"], (n,)),
        price_tier=torch.randint(0, _VOCAB_SIZES["price_tier"], (n,)),
        tags=torch.zeros(n, num_tags),
        text_embedding=torch.randn(n, text_dim),
    )


def _tiny_encoder(num_tags: int = 5, text_dim: int = 8, embedding_dim: int = 16) -> ItemEncoder:
    return ItemEncoder(_VOCAB_SIZES, num_tags=num_tags, text_dim=text_dim, embedding_dim=embedding_dim)


def test_item_encoder_output_shape_and_l2_normalization():
    model = _tiny_encoder(embedding_dim=16)
    features = _tiny_features(n=4)
    output = model(features)
    assert output.shape == (4, 16)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-5)
