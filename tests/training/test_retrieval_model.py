"""Unit tests for training/retrieval_model.py's ItemEncoder."""

import torch

from training.retrieval_model import ItemEncoder
from training.retrieval_preprocessing import ItemFeatures

_VOCAB_SIZES = {"item_id": 5, "category": 3, "subcategory": 3, "brand": 3, "price_tier": 3}


def _tiny_features(n: int, num_tags: int = 5, text_dim: int = 8, image_dim: int = 8, image_missing: bool = True) -> ItemFeatures:
    return ItemFeatures(
        item_id=torch.randint(0, _VOCAB_SIZES["item_id"], (n,)),
        category=torch.randint(0, _VOCAB_SIZES["category"], (n,)),
        subcategory=torch.randint(0, _VOCAB_SIZES["subcategory"], (n,)),
        brand=torch.randint(0, _VOCAB_SIZES["brand"], (n,)),
        price_tier=torch.randint(0, _VOCAB_SIZES["price_tier"], (n,)),
        tags=torch.zeros(n, num_tags),
        text_embedding=torch.randn(n, text_dim),
        image_embedding=torch.zeros(n, image_dim),
        image_missing=torch.full((n,), image_missing, dtype=torch.bool),
    )


def _tiny_encoder(num_tags: int = 5, text_dim: int = 8, image_dim: int = 8, embedding_dim: int = 16) -> ItemEncoder:
    return ItemEncoder(
        _VOCAB_SIZES, num_tags=num_tags, text_dim=text_dim, image_dim=image_dim, embedding_dim=embedding_dim
    )


def test_item_encoder_output_shape_and_l2_normalization():
    model = _tiny_encoder(embedding_dim=16)
    features = _tiny_features(n=4)
    output = model(features)
    assert output.shape == (4, 16)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_item_encoder_missing_image_uses_one_shared_learned_vector():
    model = _tiny_encoder(embedding_dim=16)
    model.eval()

    features = _tiny_features(n=2, image_missing=True)
    # Force every other input identical across the two rows -- image
    # substitution is then the only thing that could make outputs differ.
    features.item_id = torch.zeros(2, dtype=torch.long)
    features.category = torch.zeros(2, dtype=torch.long)
    features.subcategory = torch.zeros(2, dtype=torch.long)
    features.brand = torch.zeros(2, dtype=torch.long)
    features.price_tier = torch.zeros(2, dtype=torch.long)
    features.tags = torch.zeros(2, 5)
    features.text_embedding = torch.zeros(2, 8)

    with torch.no_grad():
        output = model(features)
    assert torch.allclose(output[0], output[1], atol=1e-6)


def test_item_encoder_present_image_is_used_not_ignored():
    model = _tiny_encoder(embedding_dim=16)
    model.eval()

    features_missing = _tiny_features(n=1, image_missing=True)
    features_present = _tiny_features(n=1, image_missing=False)
    features_present.item_id = features_missing.item_id.clone()
    features_present.category = features_missing.category.clone()
    features_present.subcategory = features_missing.subcategory.clone()
    features_present.brand = features_missing.brand.clone()
    features_present.price_tier = features_missing.price_tier.clone()
    features_present.tags = features_missing.tags.clone()
    features_present.text_embedding = features_missing.text_embedding.clone()
    features_present.image_embedding = torch.ones(1, 8)  # a real, non-zero embedding

    with torch.no_grad():
        output_missing = model(features_missing)
        output_present = model(features_present)
    assert not torch.allclose(output_missing, output_present, atol=1e-6)
