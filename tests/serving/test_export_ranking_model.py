"""Unit tests for serving/export_ranking_model.py's pure logic. The
MLflow IO (search_runs fallback, lightgbm.load_model, booster.save_model)
is not unit-tested here, matching serving/build_retrieval_index.py's
existing precedent of no e2e test for thin IO wrappers over a real server.
"""

from serving.export_ranking_model import resolve_run_id


def test_resolve_run_id_passes_through_explicit_id():
    assert resolve_run_id("abc123") == "abc123"
