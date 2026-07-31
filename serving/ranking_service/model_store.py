"""Loads the exported LambdaMART booster and the static feature snapshot
once, and scores ranking requests. No live Postgres/MLflow dependency at
request time -- see serving/export_ranking_model.py and
serving/export_feature_snapshot.py.

Reuses training/ranking_preprocessing.py:encode_ranking_examples as-is:
it's pure pandas over an already-shaped DataFrame and doesn't care
whether that DataFrame came from an offline batch join or was assembled
here for one online request.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from training.ranking_preprocessing import encode_ranking_examples
from training.vocab import build_taxonomy_vocabs


class UnknownItemError(KeyError):
    """Raised when an item_id isn't present in the exported item catalog."""


class ModelStore:
    """A loaded LambdaMART booster plus the static feature snapshot."""

    def __init__(self, ranking_dir: Path, features_dir: Path):
        """Loads model.txt and the feature-snapshot JSON files.

        Args:
            ranking_dir: Directory containing model.txt, as written by
                serving/export_ranking_model.py.
            features_dir: Directory containing user_features.json,
                candidate_features.json, item_catalog.json, as written by
                serving/export_feature_snapshot.py.
        """
        self._booster = lgb.Booster(model_file=str(ranking_dir / "model.txt"))

        # brand/category are deterministic (common/taxonomy.py's fixed
        # constants), not data-derived per training run -- no MLflow round
        # trip needed, see serving/export_ranking_model.py's docstring.
        taxonomy_vocabs = build_taxonomy_vocabs()
        self._vocabs = {"brand": taxonomy_vocabs["brand"], "category": taxonomy_vocabs["category"]}

        self._user_features: dict = json.loads((features_dir / "user_features.json").read_text())
        self._candidate_features: dict = json.loads((features_dir / "candidate_features.json").read_text())
        self._item_catalog: dict = json.loads((features_dir / "item_catalog.json").read_text())

    def build_candidate_rows(
        self,
        anchor_item_id: str,
        user_id: str,
        country: str,
        device: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Builds one ranking-feature row per candidate, pre-encoding.

        Split out from rank() so the row-construction logic -- especially
        the cross-feature formulas, which must match
        pipelines/spark_jobs/ranking_dataset.py:249-255 exactly
        (training/serving parity, LLD.md:5-6) -- is directly unit
        testable without needing a real trained booster.

        NULL handling matches training/ranking_preprocessing.py's
        encode_ranking_examples exactly: an unknown user_id or a candidate
        with no feature snapshot yet passes through as None/NaN (never
        synthesized as 0), so the model sees the same "unknown" signal it
        was trained on -- not a fabricated default.

        Args:
            anchor_item_id: The PDP item being viewed.
            user_id: The requesting user (may have no feature snapshot).
            country: Read live from the request, per docs/data-schema.md.
            device: Read live from the request, per docs/data-schema.md.
            candidates: Retrieval Service output -- list of
                {"item_id": str, "retrieval_similarity_score": float}, in
                retrieval-ranked order (used as the "position" feature).

        Returns:
            list[dict]: One row per candidate, ready to pass to
                pd.DataFrame(...) and then encode_ranking_examples.
                Candidates missing from the item catalog are dropped
                (defensive -- the catalog and the retrieval index should
                always agree, but a stale index shouldn't 500 the whole
                request).

        Raises:
            UnknownItemError: If anchor_item_id isn't in the item catalog.
        """
        anchor = self._item_catalog.get(anchor_item_id)
        if anchor is None:
            raise UnknownItemError(anchor_item_id)

        user = self._user_features.get(user_id)

        rows = []
        for position, candidate in enumerate(candidates, start=1):
            candidate_item_id = candidate["item_id"]
            candidate_meta = self._item_catalog.get(candidate_item_id)
            if candidate_meta is None:
                continue

            candidate_stats = self._candidate_features.get(candidate_item_id, {})
            anchor_price = anchor["price"]
            candidate_price = candidate_meta["price"]

            rows.append(
                {
                    "item_id": candidate_item_id,
                    "device": device,
                    "country": country,
                    "anchor_category": anchor["category"],
                    "anchor_subcategory": anchor["subcategory"],
                    "anchor_brand": anchor["brand"],
                    "candidate_category": candidate_meta["category"],
                    "candidate_subcategory": candidate_meta["subcategory"],
                    "candidate_brand": candidate_meta["brand"],
                    "position": position,
                    "retrieval_similarity_score": candidate["retrieval_similarity_score"],
                    "user_avg_purchase_price": user["avg_purchase_price"] if user else None,
                    "anchor_price": anchor_price,
                    "candidate_price": candidate_price,
                    "candidate_recommendation_ctr": candidate_stats.get("recommendation_ctr"),
                    "candidate_recommendation_cvr": candidate_stats.get("recommendation_cvr"),
                    "candidate_recommendation_impressions": candidate_stats.get("recommendation_impressions"),
                    # Cross features -- must match
                    # pipelines/spark_jobs/ranking_dataset.py:249-255
                    # exactly (training/serving parity, LLD.md:5-6).
                    "same_brand": int(anchor["brand"] == candidate_meta["brand"]),
                    "same_category": int(anchor["category"] == candidate_meta["category"]),
                    "same_subcategory": int(anchor["subcategory"] == candidate_meta["subcategory"]),
                    "price_ratio": candidate_price / anchor_price,
                    "price_diff": abs(candidate_price - anchor_price),
                    "user_preferred_brands": user["preferred_brands"] if user else None,
                    "user_historical_category_affinity": user["historical_category_affinity"] if user else None,
                    "label": 0,  # unused at inference -- encode_ranking_examples requires the column to exist
                }
            )
        return rows

    def rank(
        self,
        anchor_item_id: str,
        user_id: str,
        country: str,
        device: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Scores and ranks a set of retrieval candidates for one request.

        Args:
            anchor_item_id: The PDP item being viewed.
            user_id: The requesting user (may have no feature snapshot).
            country: Read live from the request, per docs/data-schema.md.
            device: Read live from the request, per docs/data-schema.md.
            candidates: Retrieval Service output -- list of
                {"item_id": str, "retrieval_similarity_score": float}, in
                retrieval-ranked order (used as the "position" feature).

        Returns:
            list[dict]: {"item_id", "rank_score"} pairs, descending by
                rank_score.

        Raises:
            UnknownItemError: If anchor_item_id isn't in the item catalog.
        """
        rows = self.build_candidate_rows(anchor_item_id, user_id, country, device, candidates)
        if not rows:
            return []

        df = pd.DataFrame(rows)
        x, _ = encode_ranking_examples(df, self._vocabs)
        scores = self._booster.predict(x)
        ranked = sorted(zip(df["item_id"], scores), key=lambda pair: pair[1], reverse=True)
        return [{"item_id": item_id, "rank_score": float(score)} for item_id, score in ranked]
