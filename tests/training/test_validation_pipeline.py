"""Unit tests for training/validation_pipeline.py's pure logic and the
frozen-test-set freeze/reuse behavior. No live Postgres/MLflow required --
MLflow/DB calls are mocked, matching this repo's existing convention of
not unit-testing thin IO wrappers over a real server.
"""

import math
from unittest.mock import MagicMock

import pandas as pd

from training import validation_pipeline as vp


def test_load_or_freeze_test_set_freezes_once_then_reuses(tmp_path, monkeypatch):
    frozen_path = tmp_path / "ranking_test_set.json"
    monkeypatch.setattr(vp, "_FROZEN_TEST_SET_PATH", frozen_path)

    base = pd.Timestamp("2025-04-01")
    df = pd.DataFrame(
        {
            "recommendation_impression_id": ["impr_a", "impr_a", "impr_b"],
            "timestamp": [base, base, base - pd.Timedelta(days=1)],
            "label": [1, 0, 1],
        }
    )
    read_calls = MagicMock(return_value=df)
    monkeypatch.setattr(vp, "read_ranking_training_examples", read_calls)
    monkeypatch.setattr(vp, "SPLIT_VAL_DAYS", 0)
    monkeypatch.setattr(vp, "SPLIT_TEST_DAYS", 0)

    first = vp.load_or_freeze_test_set(engine=None)
    assert frozen_path.exists()
    assert read_calls.call_count == 1
    assert set(first["recommendation_impression_id"]) == {"impr_a"}

    second = vp.load_or_freeze_test_set(engine=None)
    assert read_calls.call_count == 1  # not recomputed -- read from disk
    assert list(second["recommendation_impression_id"]) == list(first["recommendation_impression_id"])


def test_gate_passes_cold_start_clears_floor():
    assert vp.gate_passes(candidate_score=0.30, champion_score=None, floor=0.25) is True


def test_gate_passes_cold_start_below_floor_rejected():
    assert vp.gate_passes(candidate_score=0.10, champion_score=None, floor=0.25) is False


def test_gate_passes_must_beat_existing_champion():
    assert vp.gate_passes(candidate_score=0.30, champion_score=0.35, floor=0.25) is False
    assert vp.gate_passes(candidate_score=0.36, champion_score=0.35, floor=0.25) is True


def test_gate_passes_tie_with_champion_is_a_pass():
    assert vp.gate_passes(candidate_score=0.35, champion_score=0.35, floor=0.25) is True


def test_gate_passes_nan_candidate_always_rejected():
    assert vp.gate_passes(candidate_score=math.nan, champion_score=None, floor=0.25) is False
    assert vp.gate_passes(candidate_score=math.nan, champion_score=0.10, floor=0.0) is False


def test_get_champion_version_returns_none_when_no_alias_set():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = vp.MlflowException("not found")
    assert vp.get_champion_version(client, "ranking_lambdamart") is None


def test_get_champion_version_returns_version_when_alias_exists():
    client = MagicMock()
    version = MagicMock(version="3", run_id="abc123")
    client.get_model_version_by_alias.return_value = version
    assert vp.get_champion_version(client, "ranking_lambdamart") is version


def test_promote_challenger_moves_alias_and_clears_challenger():
    client = MagicMock()
    challenger = MagicMock(version="5")
    client.get_model_version_by_alias.return_value = challenger

    vp.promote_challenger(client, "ranking_lambdamart")

    client.set_registered_model_alias.assert_called_once_with("ranking_lambdamart", "champion", "5")
    client.delete_registered_model_alias.assert_called_once_with("ranking_lambdamart", "challenger")
