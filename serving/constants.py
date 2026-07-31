"""Serving-specific constants -- scoped to serving/, kept out of
training/constants.py since these govern the ANN index and the Phase 6
serving path (docs/build-phases.md), not anything training/ itself
produces or consumes.
"""

from pathlib import Path

# HNSW index-build defaults (docs/build-phases.md Phase 5 step 16). Index
# type/metric choice (HNSW over IVF, cosine via L2-normalize + inner
# product) is justified in docs/LLD.md's "ANN Index" section and is not
# reconsidered here -- these are just the graph-construction knobs.
# Standard FAISS starting points for a small-to-mid catalog; all
# CLI-overridable in serving/build_retrieval_index.py.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

DEFAULT_INDEX_DIR = Path("serving") / "index"

# Phase 6 serving-path artifact dirs -- static exports the serving
# containers mount read-only, produced by export_ranking_model.py and
# export_feature_snapshot.py. No live Postgres/MLflow dependency at
# request time (see docs/build-phases.md Phase 6's "clone and click
# around without touching Postgres/Airflow/Spark at all").
DEFAULT_RANKING_DIR = Path("serving") / "ranking"
DEFAULT_FEATURES_DIR = Path("serving") / "features"

# Internal docker-compose network ports (docker-compose.serving.yml).
RETRIEVAL_SERVICE_PORT = 8001
RANKING_SERVICE_PORT = 8002
API_GATEWAY_PORT = 8000

DEFAULT_RETRIEVAL_K = 200
DEFAULT_RESULT_K = 20
