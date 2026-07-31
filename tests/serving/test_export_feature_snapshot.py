"""Unit tests for serving/export_feature_snapshot.py's pure logic: the
dedup-to-latest-snapshot rule and the three JSON-shape builders. The
Postgres IO (read_user_daily_features, read_candidate_daily_features,
read_full_item_catalog) is not unit-tested here, matching
serving/build_retrieval_index.py's existing precedent of no e2e test for
thin IO wrappers over a real server.
"""

import datetime

import pandas as pd

from serving.export_feature_snapshot import (
    build_candidate_features,
    build_item_catalog,
    build_user_features,
    latest_snapshot,
)


def test_latest_snapshot_keeps_max_date_per_key():
    df = pd.DataFrame(
        {
            "snapshot_date": [
                datetime.date(2026, 1, 1),
                datetime.date(2026, 1, 3),
                datetime.date(2026, 1, 2),
            ],
            "user_id": ["u1", "u1", "u2"],
            "avg_purchase_price": [10.0, 30.0, 20.0],
        }
    )
    result = latest_snapshot(df, "user_id")

    assert len(result) == 2
    u1_row = result[result["user_id"] == "u1"].iloc[0]
    assert u1_row["avg_purchase_price"] == 30.0
    assert u1_row["snapshot_date"] == datetime.date(2026, 1, 3)


def test_build_user_features_shape():
    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "avg_purchase_price": [42.5],
            "preferred_brands": [["Nike", "Adidas"]],
            "historical_category_affinity": [["Footwear"]],
        }
    )
    result = build_user_features(df)
    assert result == {
        "u1": {
            "avg_purchase_price": 42.5,
            "preferred_brands": ["Nike", "Adidas"],
            "historical_category_affinity": ["Footwear"],
        }
    }


def test_build_candidate_features_shape():
    df = pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "recommendation_ctr": [0.1],
            "recommendation_cvr": [0.01],
            "recommendation_impressions": [100],
        }
    )
    result = build_candidate_features(df)
    assert result == {
        "c1": {"recommendation_ctr": 0.1, "recommendation_cvr": 0.01, "recommendation_impressions": 100}
    }


def test_build_item_catalog_shape_drops_embeddings():
    df = pd.DataFrame(
        {
            "item_id": ["i1"],
            "category": ["Footwear"],
            "subcategory": ["Sneakers"],
            "brand": ["Nike"],
            "price": [99.99],
            "tags": [["casual"]],
            "image_embedding": [None],
            "text_embedding": [[0.1, 0.2]],
        }
    )
    result = build_item_catalog(df)
    assert result == {"i1": {"category": "Footwear", "subcategory": "Sneakers", "brand": "Nike", "price": 99.99}}
