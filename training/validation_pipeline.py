"""Entrypoint: local Validation Pipeline -- Champion/Challenger Gate
(docs/LLD.md's "Validation Pipeline -- Champion/Challenger Gate", mirrored
locally). Evaluates a newly-trained candidate model against the current
champion-aliased model on a fixed, held-out test set -- frozen to a local
file on first run (training/frozen_test_sets/) so every later gate run
scores both models on identical rows regardless of when either was
trained. A candidate that clears the absolute floor (and, if a champion
already exists, matches or beats its freshly re-measured score) is
registered to the MLflow Model Registry and aliased "challenger".

Promoting challenger -> champion is a separate, explicit step
(--promote-challenger) -- the manual-review gate LLD.md requires before
Staging -> Production, done here as a deliberate second command rather
than an automatic one.

Requires the `training` optional-dependency group (`uv sync --extra
training`), the docker-compose Postgres warehouse, and MLflow running
(infra/docker-compose.mlflow.yml):

    WAREHOUSE_BACKEND=postgres MLFLOW_TRACKING_URI=http://localhost:5001 \\
      python training/validation_pipeline.py --model-type ranking

    python training/validation_pipeline.py --model-type ranking --promote-challenger

--run-id may be omitted to auto-select the most recently started run in
the model's experiment -- this script evaluates the run someone just
produced, not the best-scoring run to date (contrast with
export_ranking_model.py/build_retrieval_index.py's best-metric fallback).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.lightgbm
import mlflow.pytorch
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from common.warehouse import get_engine
from serving.build_retrieval_index import load_vocabs as load_retrieval_vocabs
from training.constants import (
    FROZEN_TEST_SET_DIR,
    RANKING_MODEL_NAME,
    RANKING_NDCG_FLOOR,
    RANKING_NDCG_K,
    RETRIEVAL_MODEL_NAME,
    RETRIEVAL_RECALL_FLOOR,
    RETRIEVAL_RECALL_K,
    SPLIT_TEST_DAYS,
    SPLIT_VAL_DAYS,
)
from training.db_io import read_full_item_catalog, read_ranking_training_examples
from training.ranking_eval import ndcg_at_k
from training.ranking_preprocessing import build_query_groups, encode_ranking_examples
from training.retrieval_eval import recall_at_k
from training.retrieval_model import ItemEncoder  # noqa: F401 -- required for mlflow.pytorch.load_model to unpickle
from training.splits import temporal_split
from training.vocab import build_taxonomy_vocabs

logger = logging.getLogger(__name__)

_EXPERIMENT_NAME = {"ranking": "ranking_lambdamart", "retrieval": "retrieval_encoder"}
_ARTIFACT_NAME = {"ranking": "ranking_lambdamart", "retrieval": "retrieval_encoder"}
_REGISTRY_NAME = {"ranking": RANKING_MODEL_NAME, "retrieval": RETRIEVAL_MODEL_NAME}
_FROZEN_TEST_SET_PATH = FROZEN_TEST_SET_DIR / "ranking_test_set.json"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for a validation-gate run.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Champion/challenger validation gate.")
    parser.add_argument("--model-type", choices=["ranking", "retrieval"], required=True)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run_id of the candidate to evaluate. Omit to auto-select the most recent run.",
    )
    parser.add_argument("--ndcg-floor", type=float, default=RANKING_NDCG_FLOOR, help="Ranking only.")
    parser.add_argument("--recall-floor", type=float, default=RETRIEVAL_RECALL_FLOOR, help="Retrieval only.")
    parser.add_argument(
        "--promote-challenger",
        action="store_true",
        help="Skip evaluation; move the current 'challenger' alias to 'champion' for --model-type.",
    )
    return parser.parse_args()


def resolve_run_id(run_id: str | None, experiment_name: str) -> str:
    """Returns run_id as given, or the most recently started run.

    Unlike export_ranking_model.py/build_retrieval_index.py's best-metric
    fallback, this picks the *newest* run -- the validation gate's job is
    to judge the candidate someone just trained, not to re-select whatever
    already scored highest historically.

    Args:
        run_id: An explicit MLflow run_id, or None to search by recency.
        experiment_name: The training experiment to search.

    Returns:
        str: The run_id to evaluate.

    Raises:
        ValueError: If run_id is None and no runs exist in the experiment.
    """
    if run_id is not None:
        return run_id
    runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        raise ValueError(
            f"No runs found in the '{experiment_name}' MLflow experiment -- train a model first."
        )
    return runs.iloc[0]["run_id"]


def load_or_freeze_test_set(engine) -> pd.DataFrame:
    """Loads the frozen ranking_training_examples test split, freezing it first if needed.

    Covers both models' evaluation: ndcg_at_k scores the ranker directly
    off these rows, and recall_at_k derives its purchase-labeled ground
    truth (label==1) from these same rows -- retrieval_training_examples's
    own click pairs are never read for evaluation (see retrieval_eval.py).
    Frozen once, on whichever model type's gate runs first; every run
    after that (either model type) reads the same file, so champion and
    challenger are always compared on identical data.

    Args:
        engine: A SQLAlchemy engine for the warehouse -- only used if the
            frozen file doesn't exist yet.

    Returns:
        pd.DataFrame: The frozen test-split rows.
    """
    if not _FROZEN_TEST_SET_PATH.exists():
        df = read_ranking_training_examples(engine)
        _train_df, _val_df, test_df = temporal_split(df, "timestamp", SPLIT_VAL_DAYS, SPLIT_TEST_DAYS)
        _FROZEN_TEST_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FROZEN_TEST_SET_PATH.write_text(test_df.to_json(orient="records", date_format="iso"))
        logger.info("Froze %d test rows to %s.", len(test_df), _FROZEN_TEST_SET_PATH)
    return pd.read_json(_FROZEN_TEST_SET_PATH, orient="records")


def evaluate_ranking(booster, test_df: pd.DataFrame) -> float:
    """Scores a LambdaMART booster's NDCG@K on the frozen test set.

    Args:
        booster: A loaded lightgbm.Booster.
        test_df: The frozen ranking_training_examples test split.

    Returns:
        float: NDCG@K, or NaN if no query group has a positive label.
    """
    vocabs = build_taxonomy_vocabs()  # deterministic (brand/category only) -- safe to rebuild fresh
    sorted_df, groups = build_query_groups(test_df)
    x, y = encode_ranking_examples(sorted_df, vocabs)
    predictions = booster.predict(x)
    return ndcg_at_k(y, predictions, groups, k=RANKING_NDCG_K)


def evaluate_retrieval(model, run_id: str, test_df: pd.DataFrame, engine) -> float:
    """Scores a retrieval encoder's Recall@K on the frozen test set.

    Args:
        model: A loaded, eval-mode ItemEncoder.
        run_id: The source run whose own item_id vocab this model was
            trained with -- must NOT be rebuilt fresh (see module docs).
        test_df: The frozen ranking_training_examples test split, used
            only for its purchase-labeled ground truth.
        engine: A SQLAlchemy engine for the (current-state, unfrozen)
            item catalog gallery.

    Returns:
        float: Recall@K, or NaN if no anchor has a purchased positive.
    """
    vocabs = load_retrieval_vocabs(run_id)
    run = mlflow.get_run(run_id)
    image_dim = int(run.data.params["image_dim"])
    catalog_df = read_full_item_catalog(engine)
    return recall_at_k(model, test_df, catalog_df, vocabs, image_dim, k=RETRIEVAL_RECALL_K)


def evaluate(model_type: str, run_id: str, test_df: pd.DataFrame, engine) -> float:
    """Loads and scores a run's logged model, dispatching on model_type.

    Args:
        model_type: "ranking" or "retrieval".
        run_id: The MLflow run to load the model from.
        test_df: The frozen test set.
        engine: A SQLAlchemy engine.

    Returns:
        float: The model's score on its model-type's metric.
    """
    artifact_name = _ARTIFACT_NAME[model_type]
    if model_type == "ranking":
        booster = mlflow.lightgbm.load_model(f"runs:/{run_id}/{artifact_name}")
        return evaluate_ranking(booster, test_df)
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/{artifact_name}")
    model.eval()
    return evaluate_retrieval(model, run_id, test_df, engine)


def get_champion_version(client: MlflowClient, registry_name: str) -> ModelVersion | None:
    """Returns the model version aliased "champion", or None if none exists yet.

    Args:
        client: An MlflowClient.
        registry_name: The registered model name.

    Returns:
        ModelVersion | None: The champion version, or None (cold start).
    """
    try:
        return client.get_model_version_by_alias(registry_name, "champion")
    except MlflowException:
        return None


def gate_passes(candidate_score: float, champion_score: float | None, floor: float) -> bool:
    """Decides promotion per LLD.md champion/challenger rule.

    Pulled out as a pure function (no MLflow/DB calls) so the actual
    decision logic -- cold start vs. must-beat-champion, and the NaN case
    (no positives in the frozen set for this metric) -- is unit-testable
    on its own.

    Args:
        candidate_score: The candidate's freshly measured score.
        champion_score: The current champion's freshly re-measured score
            on the same frozen set, or None if no champion is registered
            yet (cold start).
        floor: The absolute floor threshold for this metric.

    Returns:
        bool: True if the candidate should be registered and aliased
            "challenger". NaN scores (float comparisons are always False)
            correctly fail both the floor and champion checks.
    """
    return candidate_score >= floor and (champion_score is None or candidate_score >= champion_score)


def promote_challenger(client: MlflowClient, registry_name: str) -> None:
    """Moves the current "challenger" alias to "champion" (the manual-review step).

    Args:
        client: An MlflowClient.
        registry_name: The registered model name.
    """
    challenger = client.get_model_version_by_alias(registry_name, "challenger")
    client.set_registered_model_alias(registry_name, "champion", challenger.version)
    # Cleared so a stale alias doesn't keep pointing at a version that's
    # now champion, which would misleadingly read as "still pending review."
    client.delete_registered_model_alias(registry_name, "challenger")
    logger.info("Promoted %s v%s: challenger -> champion.", registry_name, challenger.version)


def main(args: argparse.Namespace) -> None:
    """Runs the validation gate (or the challenger->champion promotion) end to end.

    Args:
        args: Parsed CLI arguments from parse_args().
    """
    start = time.perf_counter()
    client = MlflowClient()
    registry_name = _REGISTRY_NAME[args.model_type]

    if args.promote_challenger:
        promote_challenger(client, registry_name)
        return

    engine = get_engine()
    experiment_name = _EXPERIMENT_NAME[args.model_type]
    artifact_name = _ARTIFACT_NAME[args.model_type]
    floor = args.ndcg_floor if args.model_type == "ranking" else args.recall_floor

    run_id = resolve_run_id(args.run_id, experiment_name)
    test_df = load_or_freeze_test_set(engine)

    candidate_score = evaluate(args.model_type, run_id, test_df, engine)
    logger.info("Candidate run_id=%s score=%.4f (floor=%.4f)", run_id, candidate_score, floor)

    champion_version = get_champion_version(client, registry_name)
    champion_score = None
    if champion_version is not None:
        champion_score = evaluate(args.model_type, champion_version.run_id, test_df, engine)
        logger.info(
            "Champion %s v%s (run_id=%s) freshly re-measured score=%.4f",
            registry_name, champion_version.version, champion_version.run_id, champion_score,
        )
    else:
        logger.info("No champion registered yet for %s -- cold-start gate (floor only).", registry_name)

    passed = gate_passes(candidate_score, champion_score, floor)

    if passed:
        result = mlflow.register_model(f"runs:/{run_id}/{artifact_name}", name=registry_name)
        client.set_registered_model_alias(registry_name, "challenger", result.version)
        client.set_tag(run_id, "validation_gate_result", "promoted_to_challenger")
        logger.info("PASSED: registered %s v%s, aliased 'challenger'.", registry_name, result.version)
    else:
        reason = "rejected_below_floor" if not (candidate_score >= floor) else "rejected_below_champion"
        client.set_tag(run_id, "validation_gate_result", reason)
        logger.info(
            "REJECTED (%s): candidate=%.4f floor=%.4f champion=%s",
            reason, candidate_score, floor, champion_score,
        )

    elapsed = time.perf_counter() - start
    logger.info("Validation gate finished in %.1fs.", elapsed)


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
