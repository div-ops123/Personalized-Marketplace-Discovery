"""Training-specific constants -- scoped to training/, not shared with
data_gen/ or pipelines/ (see common/constants.py's own docstring for that
module's narrower scope).
"""

from pathlib import Path

# Bumped whenever a change to the two training tables' columns or to the
# preprocessing/vocab logic here would make an old model artifact's
# preprocessing incompatible with newly-read training data. Logged as an
# MLflow tag on every run for reproducibility.
FEATURE_SCHEMA_VERSION = "v1"

# Retrieval encoder defaults (docs/build-phases.md Phase 5). Not documented
# anywhere in docs/ -- standard choices for a small MLP-plus-embeddings
# model trained locally on CPU. All CLI-overridable in retrieval_train.py.
RETRIEVAL_EMBEDDING_DIM = 128
RETRIEVAL_LEARNING_RATE = 1e-3
RETRIEVAL_WEIGHT_DECAY = 1e-5
RETRIEVAL_BATCH_SIZE = 256
RETRIEVAL_EPOCHS = 20
RETRIEVAL_EARLY_STOP_PATIENCE = 3
RETRIEVAL_HARD_NEGATIVES_PER_ANCHOR = 1
RETRIEVAL_RECALL_K = 200

# Ranking (LambdaMART) defaults. LightGBM's own well-tested defaults for
# num_leaves/feature_fraction/bagging_*; learning_rate/num_boost_round/
# early_stopping_rounds tuned for a modest synthetic dataset trained fast
# locally. All CLI-overridable in ranking_train.py.
RANKING_NDCG_K = 20
DEFAULT_LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [RANKING_NDCG_K],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}
RANKING_NUM_BOOST_ROUND = 500
RANKING_EARLY_STOPPING_ROUNDS = 50

# Temporal train/val/test split defaults (see splits.py) -- last TEST_DAYS
# held out as test, the TEST_DAYS-to-(TEST_DAYS+VAL_DAYS) window before that
# as validation, everything earlier as train.
SPLIT_VAL_DAYS = 7
SPLIT_TEST_DAYS = 7

# Validation Pipeline / champion-challenger gate (docs/LLD.md's "Validation
# Pipeline -- Champion/Challenger Gate", mirrored locally by
# validation_pipeline.py). Conservative floors set below this project's
# current best measured scores (val_ndcg_at_20=0.3855, best_val_recall_at_200
# =0.9906) -- a candidate must clear these even with no champion to beat yet.
# Both CLI-overridable (--ndcg-floor / --recall-floor).
RANKING_NDCG_FLOOR = 0.25
RETRIEVAL_RECALL_FLOOR = 0.80

# Where validation_pipeline.py freezes ranking_training_examples's test
# split on first run, so every later gate run scores champion and
# challenger on identical rows regardless of when either was trained.
FROZEN_TEST_SET_DIR = Path("training") / "frozen_test_sets"

# MLflow Model Registry names -- match each training script's own
# mlflow.set_experiment(...) string, reused as the registered-model name
# too so there's only one name to keep in sync.
RANKING_MODEL_NAME = "ranking_lambdamart"
RETRIEVAL_MODEL_NAME = "retrieval_encoder"
