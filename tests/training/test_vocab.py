"""Unit tests for training/vocab.py."""

import pandas as pd

from common.taxonomy import ALL_BRANDS, CATEGORIES, PRICE_TIERS, TAXONOMY
from training.vocab import Vocabulary, build_item_id_vocab, build_taxonomy_vocabs, multi_hot_encode


def test_vocabulary_encode_round_trip():
    vocab = Vocabulary(["a", "b", "c"])
    assert vocab.encode("a") == vocab.token_to_index["a"]
    assert len({vocab.encode("a"), vocab.encode("b"), vocab.encode("c")}) == 3


def test_vocabulary_unknown_token_falls_back_to_zero():
    vocab = Vocabulary(["a", "b"])
    assert vocab.encode("never-seen") == 0
    assert vocab.token_to_index["<UNK>"] == 0


def test_vocabulary_to_dict_from_dict_round_trip():
    vocab = Vocabulary(["a", "b", "c"])
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.token_to_index == vocab.token_to_index
    assert restored.encode("b") == vocab.encode("b")


def test_vocabulary_deduplicates_tokens():
    vocab = Vocabulary(["a", "a", "b"])
    assert len(vocab) == 3  # <UNK> + a + b


def test_build_taxonomy_vocabs_covers_every_entry():
    vocabs = build_taxonomy_vocabs()
    for category in CATEGORIES:
        assert vocabs["category"].encode(category) != 0
    for brand in ALL_BRANDS:
        assert vocabs["brand"].encode(brand) != 0
    for tier in PRICE_TIERS:
        assert vocabs["price_tier"].encode(tier) != 0
    for spec in TAXONOMY.values():
        for sub in spec["subcategories"]:
            assert vocabs["subcategory"].encode(sub) != 0
        for tag in spec["tags"]:
            assert vocabs["tags"].encode(tag) != 0


def test_build_item_id_vocab_is_deterministic_and_deduplicated():
    ids = pd.Series(["item_3", "item_1", "item_2", "item_1"])
    vocab_a = build_item_id_vocab(ids)
    vocab_b = build_item_id_vocab(ids)
    assert vocab_a.token_to_index == vocab_b.token_to_index
    assert len(vocab_a) == 4  # <UNK> + 3 unique items


def test_multi_hot_encode_shape_and_none_is_all_zero():
    vocab = Vocabulary(["x", "y", "z"])
    values = [["x", "y"], None, []]
    matrix = multi_hot_encode(values, vocab)
    assert matrix.shape == (3, len(vocab))
    assert matrix[1].sum() == 0  # None -> all-zero
    assert matrix[2].sum() == 0  # empty list -> also all-zero
    assert matrix[0, vocab.encode("x")] == 1
    assert matrix[0, vocab.encode("y")] == 1
