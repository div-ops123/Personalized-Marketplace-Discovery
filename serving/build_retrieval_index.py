"""Entrypoint: build the FAISS/HNSW retrieval index from a trained retrieval encoder's item embeddings.

Requires the `training` and `serving` optional-dependency groups (`uv sync
--extra training --extra serving`), the docker-compose Postgres warehouse
populated by pipelines/spark_jobs/run_dataset_builders.py, and MLflow
running (infra/docker-compose.mlflow.yml) with a completed
training/retrieval_train.py run:

    WAREHOUSE_BACKEND=postgres MLFLOW_TRACKING_URI=http://localhost:5001 \\
      python serving/build_retrieval_index.py --run-id <RUN_ID>

--run-id may be omitted to auto-select the best run in the
"retrieval_encoder" experiment by best_val_recall_at_200.

Writes items.faiss, item_ids.json, and metadata.json to --output-dir
(default serving/index/) -- not the Retrieval Service itself, just 
the index a future service will mount and query.

Index type/metric (HNSW, cosine via L2-normalize + inner product) is
decided in docs/LLD.md's "ANN Index" section, not reconsidered here.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import mlflow
import mlflow.artifacts
import mlflow.pytorch
import numpy as np
from dotenv import load_dotenv

from common.warehouse import get_engine
from serving.constants import DEFAULT_INDEX_DIR, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M
from training.db_io import read_full_item_catalog
from training.retrieval_model import ItemEncoder  # noqa: F401 -- required for mlflow.pytorch.load_model to unpickle
from training.retrieval_preprocessing import embed_catalog
from training.vocab import Vocabulary

logger = logging.getLogger(__name__)

_VOCAB_NAMES = ("item_id", "category", "subcategory", "brand", "price_tier", "tags")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for an index-build run.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Build the FAISS/HNSW retrieval index.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "MLflow run_id (from the 'retrieval_encoder' experiment) to build the index from. "
            "Omit to auto-select the run with the highest best_val_recall_at_200."
        ),
    )
    parser.add_argument("--hnsw-m", type=int, default=HNSW_M)
    parser.add_argument("--hnsw-ef-construction", type=int, default=HNSW_EF_CONSTRUCTION)
    parser.add_argument("--hnsw-ef-search", type=int, default=HNSW_EF_SEARCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INDEX_DIR)
    return parser.parse_args()


def resolve_run_id(run_id: str | None) -> str:
    """Returns run_id as given, or finds the best retrieval_encoder run.

    No existing convention in this repo tags a "latest"/"production" run
    (retrieval_train.py reuses the same experiment/run_name every run) --
    an explicit --run-id is the primary, consistent way to pick one; this
    search is only a convenience fallback.

    Args:
        run_id: An explicit MLflow run_id, or None to search for the best
            run in the "retrieval_encoder" experiment.

    Returns:
        str: The run_id to build the index from.

    Raises:
        ValueError: If run_id is None and no runs exist in the experiment.
    """
    if run_id is not None:
        return run_id
    runs = mlflow.search_runs(
        experiment_names=["retrieval_encoder"],
        order_by=["metrics.best_val_recall_at_200 DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            "No runs found in the 'retrieval_encoder' MLflow experiment -- "
            "train a model first (training/retrieval_train.py) or pass --run-id explicitly."
        )
    return runs.iloc[0]["run_id"]


def load_vocabs(run_id: str) -> dict[str, Vocabulary]:
    """Reloads every vocabulary logged by the given retrieval_train.py run.

    item_id's vocabulary is data-derived per training run (only item IDs
    seen in that run's train split, see training/vocab.py), so it must come
    from that run's own artifacts -- rebuilding it from the current catalog
    would silently misalign the loaded model's item_id_embedding indices
    with what it was actually trained on. The other five vocabs are
    deterministic (training/vocab.py's build_taxonomy_vocabs would give an
    identical result) but are reloaded the same way here for one consistent
    code path rather than a second, divergent one.

    Args:
        run_id: The MLflow run to load preprocessing artifacts from.

    Returns:
        dict[str, Vocabulary]: Keyed by _VOCAB_NAMES.
    """
    vocabs = {}
    for name in _VOCAB_NAMES:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"preprocessing/{name}_vocab.json"
        )
        vocabs[name] = Vocabulary.from_dict(json.loads(Path(path).read_text()))
    return vocabs


def build_hnsw_index(embeddings: np.ndarray, m: int, ef_construction: int, ef_search: int) -> faiss.IndexHNSWFlat:
    """Builds a cosine-similarity HNSW index over a set of embeddings.

    Cosine similarity is implemented via L2-normalization + FAISS inner
    product, not a native cosine metric -- the standard FAISS pattern.

    Args:
        embeddings: (n, dim) array. Re-normalized here even though
            ItemEncoder.forward already L2-normalizes its output -- a
            cheap safety/consistency step, not because drift is expected.
        m: HNSW graph degree (neighbors per node).
        ef_construction: HNSW build-time search width.
        ef_search: HNSW query-time search width, baked in as the index's
            default; a future Retrieval Service may override this per-query.

    Returns:
        faiss.IndexHNSWFlat: Populated index, ready to write_index() or query.
    """
    # copy=True: np.ascontiguousarray would return the *same* array (no
    # copy) if it's already C-contiguous float32, and normalize_L2 mutates
    # in place -- forcing a copy here avoids silently rewriting the
    # caller's own embeddings array as a side effect.
    embeddings = np.array(embeddings, dtype=np.float32, copy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexHNSWFlat(embeddings.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    index.hnsw.efSearch = ef_search
    return index


def main(args: argparse.Namespace) -> None:
    """Builds and persists the retrieval index end to end.

    Args:
        args: Parsed CLI arguments from parse_args().
    """
    start = time.perf_counter()

    run_id = resolve_run_id(args.run_id)
    logger.info("Building index from run_id=%s", run_id)

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/retrieval_encoder")
    model.eval()
    vocabs = load_vocabs(run_id)

    run = mlflow.get_run(run_id)
    try:
        image_dim = int(run.data.params["image_dim"])
    except KeyError as exc:
        raise ValueError(
            f"Run {run_id} has no logged 'image_dim' param -- retrain with a current "
            "training/retrieval_train.py before building an index."
        ) from exc

    engine = get_engine()
    catalog_df = read_full_item_catalog(engine)
    logger.info("Loaded item catalog: %d items", len(catalog_df))

    # No batching over the catalog -- fine at this repo's actual catalog
    # size (a few thousand items); docs/LLD.md's 2M-item production scale
    # is out of scope for this local, one-shot build step.
    embeddings = embed_catalog(model, catalog_df, vocabs, image_dim).detach().cpu().numpy()
    item_ids = catalog_df["item_id"].tolist()  # row-aligned with embeddings; embed_catalog never reorders/drops rows

    index = build_hnsw_index(embeddings, args.hnsw_m, args.hnsw_ef_construction, args.hnsw_ef_search)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output_dir / "items.faiss"))
    (args.output_dir / "item_ids.json").write_text(json.dumps(item_ids))
    metadata = {
        "source_run_id": run_id,
        "item_count": len(item_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "hnsw_m": args.hnsw_m,
        "hnsw_ef_construction": args.hnsw_ef_construction,
        "hnsw_ef_search": args.hnsw_ef_search,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    elapsed = time.perf_counter() - start
    logger.info("Index built in %.1fs: %d items -> %s", elapsed, len(item_ids), args.output_dir)


if __name__ == "__main__":
    # force=True: mlflow/torch's own imports (above) already attach a
    # handler to the root logger, which makes a plain basicConfig() a
    # silent no-op -- this script's own INFO logs would never print.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    # override=False (the default): see run_dataset_builders.py's docstring --
    # container-injected env vars must win over the bind-mounted .env.
    load_dotenv()
    main(parse_args())
