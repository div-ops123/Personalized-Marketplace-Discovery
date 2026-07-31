"""Unit tests for serving/ranking_service/model_store.py.

build_candidate_rows is the correctness-critical piece: its cross-feature
formulas (same_brand/category/subcategory, price_ratio, price_diff) must
match pipelines/spark_jobs/ranking_dataset.py:249-255 exactly (training/
serving parity, docs/LLD.md:5-6) -- getting the ratio direction or the
abs() wrong would silently skew ranking scores relative to training.
rank() itself is exercised end to end against a tiny real LightGBM
booster, trained here on synthetic rows via the same encode_ranking_examples
path training uses.
"""

import json

import lightgbm as lgb
import pandas as pd
import pytest

from serving.ranking_service.model_store import ModelStore, UnknownItemError
from training.ranking_preprocessing import build_query_groups, encode_ranking_examples
from training.vocab import build_taxonomy_vocabs

_ITEM_CATALOG = {
    "anchor_1": {"category": "Footwear", "subcategory": "Sneakers", "brand": "Nike", "price": 100.0},
    "cand_same_brand": {"category": "Footwear", "subcategory": "Sneakers", "brand": "Nike", "price": 150.0},
    "cand_diff_brand": {"category": "Apparel", "subcategory": "Jackets", "brand": "Adidas", "price": 50.0},
}

_USER_FEATURES = {
    "user_known": {
        "avg_purchase_price": 75.0,
        "preferred_brands": ["Nike"],
        "historical_category_affinity": ["Footwear"],
    }
}

_CANDIDATE_FEATURES = {
    "cand_same_brand": {"recommendation_ctr": 0.1, "recommendation_cvr": 0.01, "recommendation_impressions": 100},
}


def _write_snapshot(tmp_path):
    ranking_dir = tmp_path / "ranking"
    features_dir = tmp_path / "features"
    ranking_dir.mkdir()
    features_dir.mkdir()

    (features_dir / "item_catalog.json").write_text(json.dumps(_ITEM_CATALOG))
    (features_dir / "user_features.json").write_text(json.dumps(_USER_FEATURES))
    (features_dir / "candidate_features.json").write_text(json.dumps(_CANDIDATE_FEATURES))

    _train_tiny_booster(ranking_dir)
    return ranking_dir, features_dir


def _train_tiny_booster(ranking_dir):
    df = pd.DataFrame(
        {
            "recommendation_impression_id": ["impr_1", "impr_1", "impr_2", "impr_2"],
            "device": ["mobile"] * 4,
            "country": ["US"] * 4,
            "anchor_category": ["Footwear"] * 4,
            "anchor_subcategory": ["Sneakers"] * 4,
            "anchor_brand": ["Nike"] * 4,
            "candidate_category": ["Footwear", "Apparel"] * 2,
            "candidate_subcategory": ["Sneakers", "Jackets"] * 2,
            "candidate_brand": ["Nike", "Adidas"] * 2,
            "position": [1, 2, 1, 2],
            "retrieval_similarity_score": [0.9, 0.5, 0.8, 0.4],
            "user_avg_purchase_price": [75.0] * 4,
            "anchor_price": [100.0] * 4,
            "candidate_price": [150.0, 50.0, 150.0, 50.0],
            "candidate_recommendation_ctr": [0.1, 0.05, 0.1, 0.05],
            "candidate_recommendation_cvr": [0.01, 0.005, 0.01, 0.005],
            "candidate_recommendation_impressions": [100, 50, 100, 50],
            "same_brand": [1, 0, 1, 0],
            "same_category": [1, 0, 1, 0],
            "same_subcategory": [1, 0, 1, 0],
            "price_ratio": [1.5, 0.5, 1.5, 0.5],
            "price_diff": [50.0, 50.0, 50.0, 50.0],
            "user_preferred_brands": [["Nike"]] * 4,
            "user_historical_category_affinity": [["Footwear"]] * 4,
            "label": [1, 0, 1, 0],
        }
    )
    sorted_df, groups = build_query_groups(df)
    taxonomy_vocabs = build_taxonomy_vocabs()
    vocabs = {"brand": taxonomy_vocabs["brand"], "category": taxonomy_vocabs["category"]}
    x, y = encode_ranking_examples(sorted_df, vocabs)
    train_set = lgb.Dataset(
        x, label=y, group=groups, categorical_feature=[c for c in x.columns if str(x[c].dtype) == "category"]
    )
    booster = lgb.train({"objective": "lambdarank", "verbosity": -1}, train_set, num_boost_round=5)
    booster.save_model(str(ranking_dir / "model.txt"))


def test_build_candidate_rows_cross_feature_formulas(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    store = ModelStore(ranking_dir, features_dir)

    rows = store.build_candidate_rows(
        "anchor_1",
        "user_known",
        "US",
        "mobile",
        [
            {"item_id": "cand_same_brand", "retrieval_similarity_score": 0.9},
            {"item_id": "cand_diff_brand", "retrieval_similarity_score": 0.5},
        ],
    )
    by_id = {row["item_id"]: row for row in rows}

    # anchor price=100, cand_same_brand price=150 -- must match
    # pipelines/spark_jobs/ranking_dataset.py:254-255 exactly:
    # price_ratio = candidate_price / anchor_price, price_diff = abs(diff).
    assert by_id["cand_same_brand"]["price_ratio"] == pytest.approx(150.0 / 100.0)
    assert by_id["cand_same_brand"]["price_diff"] == pytest.approx(abs(150.0 - 100.0))
    assert by_id["cand_same_brand"]["same_brand"] == 1
    assert by_id["cand_same_brand"]["same_category"] == 1
    assert by_id["cand_same_brand"]["same_subcategory"] == 1

    assert by_id["cand_diff_brand"]["price_ratio"] == pytest.approx(50.0 / 100.0)
    assert by_id["cand_diff_brand"]["price_diff"] == pytest.approx(abs(50.0 - 100.0))
    assert by_id["cand_diff_brand"]["same_brand"] == 0
    assert by_id["cand_diff_brand"]["same_category"] == 0
    assert by_id["cand_diff_brand"]["same_subcategory"] == 0


def test_build_candidate_rows_position_is_retrieval_rank_order(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    store = ModelStore(ranking_dir, features_dir)

    rows = store.build_candidate_rows(
        "anchor_1",
        "user_known",
        "US",
        "mobile",
        [
            {"item_id": "cand_diff_brand", "retrieval_similarity_score": 0.5},
            {"item_id": "cand_same_brand", "retrieval_similarity_score": 0.9},
        ],
    )
    by_id = {row["item_id"]: row for row in rows}
    assert by_id["cand_diff_brand"]["position"] == 1
    assert by_id["cand_same_brand"]["position"] == 2


def test_build_candidate_rows_unknown_user_passes_through_as_none(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    store = ModelStore(ranking_dir, features_dir)

    rows = store.build_candidate_rows(
        "anchor_1",
        "user_never_seen",
        "US",
        "mobile",
        [{"item_id": "cand_same_brand", "retrieval_similarity_score": 0.9}],
    )
    assert rows[0]["user_avg_purchase_price"] is None
    assert rows[0]["user_preferred_brands"] is None
    assert rows[0]["user_historical_category_affinity"] is None


def test_build_candidate_rows_unknown_candidate_missing_stats_passes_through_as_none(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    store = ModelStore(ranking_dir, features_dir)

    # cand_diff_brand has no entry in _CANDIDATE_FEATURES.
    rows = store.build_candidate_rows(
        "anchor_1",
        "user_known",
        "US",
        "mobile",
        [{"item_id": "cand_diff_brand", "retrieval_similarity_score": 0.5}],
    )
    assert rows[0]["candidate_recommendation_ctr"] is None
    assert rows[0]["candidate_recommendation_cvr"] is None
    assert rows[0]["candidate_recommendation_impressions"] is None


def test_build_candidate_rows_drops_candidate_missing_from_catalog(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    store = ModelStore(ranking_dir, features_dir)

    rows = store.build_candidate_rows(
        "anchor_1",
        "user_known",
        "US",
        "mobile",
        [
            {"item_id": "cand_same_brand", "retrieval_similarity_score": 0.9},
            {"item_id": "not_in_catalog", "retrieval_similarity_score": 0.8},
        ],
    )
    assert [row["item_id"] for row in rows] == ["cand_same_brand"]


def test_build_candidate_rows_unknown_anchor_raises(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    store = ModelStore(ranking_dir, features_dir)

    with pytest.raises(UnknownItemError):
        store.build_candidate_rows("not_a_real_item", "user_known", "US", "mobile", [])


def test_rank_returns_all_candidates_sorted_descending(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    store = ModelStore(ranking_dir, features_dir)

    ranked = store.rank(
        "anchor_1",
        "user_known",
        "US",
        "mobile",
        [
            {"item_id": "cand_diff_brand", "retrieval_similarity_score": 0.5},
            {"item_id": "cand_same_brand", "retrieval_similarity_score": 0.9},
        ],
    )
    assert {row["item_id"] for row in ranked} == {"cand_diff_brand", "cand_same_brand"}
    scores = [row["rank_score"] for row in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_empty_candidates_returns_empty_list(tmp_path):
    ranking_dir, features_dir = _write_snapshot(tmp_path)
    store = ModelStore(ranking_dir, features_dir)
    assert store.rank("anchor_1", "user_known", "US", "mobile", []) == []
