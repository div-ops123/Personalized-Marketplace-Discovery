"""Entrypoint: train the LambdaMART ranker (docs/build-phases.md Phase 5,
step 15's second half) against ranking_training_examples, log to MLflow.

Requires the `training` optional-dependency group (`uv sync --extra
training`), the docker-compose Postgres warehouse populated by
pipelines/spark_jobs/run_dataset_builders.py --builder both, and MLflow
running (infra/docker-compose.mlflow.yml):

    WAREHOUSE_BACKEND=postgres MLFLOW_TRACKING_URI=http://localhost:5001 \\
      python training/ranking_train.py --num-boost-round 500

TODO (v2): docs/data-schema.md's "Position Bias Correction" section calls
for IPS-weighting each row by the inverse probability it was actually
seen. No formula is documented anywhere in this repo. Per the same "no
cold-start bootstrap first" precedent already applied to retrieval, v1
ships this ranker UNWEIGHTED (every row weight = 1.0). Do not invent a
formula here; revisit once the propensity model / logging policy is
decided.
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.lightgbm
from dotenv import load_dotenv

from common.warehouse import get_engine
from training.constants import (
    DEFAULT_LGBM_PARAMS,
    RANKING_EARLY_STOPPING_ROUNDS,
    RANKING_NDCG_K,
    RANKING_NUM_BOOST_ROUND,
    SPLIT_TEST_DAYS,
    SPLIT_VAL_DAYS,
)
from training.db_io import read_ranking_training_examples
from training.mlflow_utils import log_preprocessing_artifacts, log_reproducibility_metadata
from training.ranking_eval import ndcg_at_k
from training.ranking_model import train_lambdamart
from training.ranking_preprocessing import build_query_groups, encode_ranking_examples
from training.splits import temporal_split
from training.vocab import build_taxonomy_vocabs

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for a ranking training run.

    Returns:
        argparse.Namespace: Parsed hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Train the LambdaMART ranker.")
    parser.add_argument("--num-boost-round", type=int, default=RANKING_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=RANKING_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--ndcg-k", type=int, default=RANKING_NDCG_K)
    parser.add_argument("--val-days", type=int, default=SPLIT_VAL_DAYS)
    parser.add_argument("--test-days", type=int, default=SPLIT_TEST_DAYS)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Trains the LambdaMART ranker end to end and logs the run to MLflow.

    Args:
        args: Parsed hyperparameters from parse_args().
    """
    start = time.perf_counter()

    engine = get_engine()
    df = read_ranking_training_examples(engine)
    train_df, val_df, test_df = temporal_split(df, "timestamp", args.val_days, args.test_days)
    logger.info("Ranking rows: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # Applied independently per split -- a global group array computed
    # before splitting would silently corrupt lgb.Dataset's group
    # boundaries across the train/val boundary.
    train_sorted, group_train = build_query_groups(train_df)
    val_sorted, group_val = build_query_groups(val_df)

    vocabs = build_taxonomy_vocabs()
    x_train, y_train = encode_ranking_examples(train_sorted, vocabs)
    x_val, y_val = encode_ranking_examples(val_sorted, vocabs)

    params = dict(DEFAULT_LGBM_PARAMS)
    params["eval_at"] = [args.ndcg_k]

    mlflow.set_experiment("ranking_lambdamart")
    with mlflow.start_run(run_name="ranking_lambdamart"):
        # LightGBM is a natively-supported MLflow autolog flavor -- this
        # genuinely satisfies LLD.md's "MLflow autologging covers optimizer
        # parameters and loss curves" line. Contrast with retrieval_train.py,
        # whose bespoke PyTorch loop has no such autolog flavor and logs
        # train_loss/val_recall_at_200 manually -- an intentional asymmetry
        # between the two scripts, not an inconsistency.
        # log_models=False: autolog's own model artifact is unnamed ("model"),
        # indistinguishable from retrieval_train.py's in the MLflow UI --
        # logged explicitly below instead, under a descriptive name.
        mlflow.lightgbm.autolog(log_models=False)

        booster = train_lambdamart(
            x_train,
            y_train,
            group_train,
            x_val,
            y_val,
            group_val,
            params=params,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )

        val_predictions = booster.predict(x_val)
        val_ndcg = ndcg_at_k(y_val, val_predictions, group_val, k=args.ndcg_k)
        mlflow.log_metric("val_ndcg_at_20", val_ndcg)
        mlflow.lightgbm.log_model(booster, name="ranking_lambdamart")

        log_reproducibility_metadata(engine, "ranking_training_examples")
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_preprocessing_artifacts(vocabs, Path(tmp_dir))

        elapsed = time.perf_counter() - start
        mlflow.log_metric("training_duration_seconds", elapsed)

    logger.info("Ranking training complete in %.1fs. val_ndcg_at_%d=%.4f", elapsed, args.ndcg_k, val_ndcg)


if __name__ == "__main__":
    # force=True: mlflow's own imports (above) already attach a handler to
    # the root logger, which makes a plain basicConfig() a silent no-op --
    # this script's own INFO logs would never print.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    # override=False (the default): see run_dataset_builders.py's docstring --
    # container-injected env vars must win over the bind-mounted .env.
    load_dotenv()
    main(parse_args())
