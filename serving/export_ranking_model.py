"""Entrypoint: export a trained LambdaMART ranker to a plain local file
(docs/build-phases.md Phase 6) so the Ranking Service never needs MLflow
live at request time.

Requires the `training` optional-dependency group (`uv sync --extra
training`) and MLflow running (infra/docker-compose.mlflow.yml) with a
completed training/ranking_train.py run:

    MLFLOW_TRACKING_URI=http://localhost:5001 \\
      python serving/export_ranking_model.py --run-id <RUN_ID>

--run-id may be omitted to auto-select the best run in the
"ranking_lambdamart" experiment by val_ndcg_at_20.

Writes model.txt and metadata.json to --output-dir (default
serving/ranking/). Unlike the retrieval encoder, the ranker's brand/
category vocabs are not exported here -- training/vocab.py's
build_taxonomy_vocabs() is a pure, deterministic function over
common/taxonomy.py's fixed constants (no MLflow/DB round trip needed,
same reasoning documented in training/mlflow_utils.py for why only
item_id's data-derived vocab needs per-run persistence) -- the Ranking
Service calls it directly at startup instead.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.lightgbm
from dotenv import load_dotenv

from serving.constants import DEFAULT_RANKING_DIR

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for a ranking-model export run.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Export the trained LambdaMART ranker to a local file.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "MLflow run_id (from the 'ranking_lambdamart' experiment) to export. "
            "Omit to auto-select the run with the highest val_ndcg_at_20."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RANKING_DIR)
    return parser.parse_args()


def resolve_run_id(run_id: str | None) -> str:
    """Returns run_id as given, or finds the best ranking_lambdamart run.

    Same rationale as serving/build_retrieval_index.py's resolve_run_id --
    no existing convention in this repo tags a "latest"/"production" run,
    so an explicit --run-id is the primary, consistent way to pick one;
    this search is only a convenience fallback.

    Args:
        run_id: An explicit MLflow run_id, or None to search for the best
            run in the "ranking_lambdamart" experiment.

    Returns:
        str: The run_id to export.

    Raises:
        ValueError: If run_id is None and no runs exist in the experiment.
    """
    if run_id is not None:
        return run_id
    runs = mlflow.search_runs(
        experiment_names=["ranking_lambdamart"],
        order_by=["metrics.val_ndcg_at_20 DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            "No runs found in the 'ranking_lambdamart' MLflow experiment -- "
            "train a model first (training/ranking_train.py) or pass --run-id explicitly."
        )
    return runs.iloc[0]["run_id"]


def main(args: argparse.Namespace) -> None:
    """Exports the trained booster and run metadata end to end.

    Args:
        args: Parsed CLI arguments from parse_args().
    """
    start = time.perf_counter()

    run_id = resolve_run_id(args.run_id)
    logger.info("Exporting ranking model from run_id=%s", run_id)

    booster = mlflow.lightgbm.load_model(f"runs:/{run_id}/ranking_lambdamart")
    run = mlflow.get_run(run_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.output_dir / "model.txt"))
    metadata = {
        "source_run_id": run_id,
        "val_ndcg_at_20": run.data.metrics.get("val_ndcg_at_20"),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    elapsed = time.perf_counter() - start
    logger.info("Ranking model exported in %.1fs -> %s", elapsed, args.output_dir)


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
