"""MLflow logging helpers shared by both training scripts.

Implements LLD.md's 8-field run-logging spec with local substitutes for
the two fields that are inherently cloud-specific ("dataset S3 path and
timestamp", "training duration/cost" -- see _dataset_snapshot_tag and each
training script's own training_duration_seconds metric).
"""

import json
import logging
import os
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from sqlalchemy.engine import Engine

from training.constants import FEATURE_SCHEMA_VERSION
from training.vocab import Vocabulary

logger = logging.getLogger(__name__)


def _git_commit_hash() -> str | None:
    """Returns the current git commit hash, or None outside a git checkout.

    Returns:
        str | None: The commit hash, or None if `git rev-parse` fails.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _dataset_snapshot_tag(engine: Engine, table_name: str) -> str:
    """Builds a local substitute for LLD.md's "dataset S3 path and timestamp".

    No S3 path exists for a local run -- this captures the table's row
    count and its own max timestamp instead, a reproducibility anchor
    scoped to what's actually available locally.

    Args:
        engine: A SQLAlchemy engine for the warehouse.
        table_name: retrieval_training_examples or ranking_training_examples.

    Returns:
        str: "{table} rows={n} max_timestamp={ts} backend={backend}".
    """
    n = int(pd.read_sql(f"SELECT COUNT(*) AS n FROM {table_name}", engine).iloc[0]["n"])
    columns = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 0", engine).columns
    if "timestamp" in columns:
        max_ts = pd.read_sql(f"SELECT MAX(timestamp) AS max_ts FROM {table_name}", engine).iloc[0]["max_ts"]
    else:
        max_ts = "n/a"
    backend = os.environ.get("WAREHOUSE_BACKEND", "postgres")
    return f"{table_name} rows={n} max_timestamp={max_ts} backend={backend}"


def log_reproducibility_metadata(
    engine: Engine, table_name: str, extra_tables: list[str] | None = None
) -> None:
    """Logs the feature-schema-version, git-commit, and dataset-snapshot tags.

    Args:
        engine: A SQLAlchemy engine for the warehouse.
        table_name: The primary training table this run reads from.
        extra_tables: Any additional tables read (e.g. retrieval eval also
            reads ranking_training_examples for its purchase-label ground
            truth -- see retrieval_eval.py).
    """
    mlflow.set_tag("feature_schema_version", FEATURE_SCHEMA_VERSION)
    commit = _git_commit_hash()
    if commit is not None:
        mlflow.set_tag("git_commit", commit)
    else:
        logger.warning("Not running inside a git checkout; skipping git_commit tag.")

    for table in [table_name, *(extra_tables or [])]:
        mlflow.set_tag(f"dataset_snapshot__{table}", _dataset_snapshot_tag(engine, table))


def log_preprocessing_artifacts(vocabs: dict[str, Vocabulary], tmp_dir: Path) -> None:
    """Persists every vocabulary as a JSON artifact under preprocessing/.

    Serving-time inference must use the identical token->index mapping
    used at training time -- logging these as MLflow artifacts is what
    makes that reproducible without re-deriving vocabularies later.

    Args:
        vocabs: Name -> Vocabulary, e.g. {"category": ..., "item_id": ...}.
        tmp_dir: A local directory to stage the JSON files in before upload.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for name, vocab in vocabs.items():
        path = tmp_dir / f"{name}_vocab.json"
        path.write_text(json.dumps(vocab.to_dict()))
        mlflow.log_artifact(str(path), artifact_path="preprocessing")
