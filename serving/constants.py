"""Serving-specific constants -- scoped to serving/, kept out of
training/constants.py since these govern the ANN index this repo's future
Retrieval Service (docs/build-phases.md Phase 6) will read, not anything
training/ itself produces or consumes.
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
