"""Entrypoint: train the Siamese retrieval encoder (docs/build-phases.md
Phase 5, step 15's first half) against retrieval_training_examples, log to
MLflow, and save the best (by validation Recall@200) checkpoint.

Requires the `training` optional-dependency group (`uv sync --extra
training`), the docker-compose Postgres warehouse populated by
pipelines/spark_jobs/run_dataset_builders.py --builder both, and MLflow
running (infra/docker-compose.mlflow.yml):

    WAREHOUSE_BACKEND=postgres MLFLOW_TRACKING_URI=http://localhost:5001 \\
      python training/retrieval_train.py --epochs 20 --batch-size 256

Does not build the FAISS/HNSW index (docs/build-phases.md step 16, a
separate later step) -- this script's job stops at a trained checkpoint
plus its offline Recall@200.
"""

import argparse
import copy
import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from common.warehouse import get_engine
from training.constants import (
    RETRIEVAL_BATCH_SIZE,
    RETRIEVAL_EARLY_STOP_PATIENCE,
    RETRIEVAL_EMBEDDING_DIM,
    RETRIEVAL_EPOCHS,
    RETRIEVAL_HARD_NEGATIVES_PER_ANCHOR,
    RETRIEVAL_LEARNING_RATE,
    RETRIEVAL_RECALL_K,
    RETRIEVAL_WEIGHT_DECAY,
    SPLIT_TEST_DAYS,
    SPLIT_VAL_DAYS,
)
from training.db_io import (
    read_full_item_catalog,
    read_impression_timestamps,
    read_ranking_training_examples,
    read_retrieval_training_examples,
)
from training.mlflow_utils import log_preprocessing_artifacts, log_reproducibility_metadata
from training.retrieval_eval import recall_at_k
from training.retrieval_model import ItemEncoder, contrastive_loss
from training.retrieval_preprocessing import (
    encode_hard_negatives,
    encode_item_side,
    infer_embedding_dims,
    sample_training_batch,
)
from training.splits import temporal_split
from training.vocab import build_item_id_vocab, build_taxonomy_vocabs

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for a retrieval training run.

    Returns:
        argparse.Namespace: Parsed hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Train the Siamese retrieval encoder.")
    parser.add_argument("--epochs", type=int, default=RETRIEVAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=RETRIEVAL_BATCH_SIZE)
    parser.add_argument("--hard-negatives-per-anchor", type=int, default=RETRIEVAL_HARD_NEGATIVES_PER_ANCHOR)
    parser.add_argument("--lr", type=float, default=RETRIEVAL_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=RETRIEVAL_WEIGHT_DECAY)
    parser.add_argument("--embedding-dim", type=int, default=RETRIEVAL_EMBEDDING_DIM)
    parser.add_argument("--patience", type=int, default=RETRIEVAL_EARLY_STOP_PATIENCE)
    parser.add_argument("--recall-k", type=int, default=RETRIEVAL_RECALL_K)
    parser.add_argument("--val-days", type=int, default=SPLIT_VAL_DAYS)
    parser.add_argument("--test-days", type=int, default=SPLIT_TEST_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Trains the retrieval encoder end to end and logs the run to MLflow.

    Args:
        args: Parsed hyperparameters from parse_args().
    """
    start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    engine = get_engine()
    retrieval_df = read_retrieval_training_examples(engine)
    timestamps_df = read_impression_timestamps(engine)
    retrieval_df = retrieval_df.merge(timestamps_df, on="recommendation_impression_id", how="left")
    train_df, val_df, _test_df = temporal_split(retrieval_df, "timestamp", args.val_days, args.test_days)
    logger.info("Retrieval rows: train=%d val=%d test=%d", len(train_df), len(val_df), len(_test_df))

    ranking_df = read_ranking_training_examples(engine)
    _ranking_train, ranking_val, _ranking_test = temporal_split(ranking_df, "timestamp", args.val_days, args.test_days)
    item_catalog_df = read_full_item_catalog(engine)

    vocabs = build_taxonomy_vocabs()
    vocabs["item_id"] = build_item_id_vocab(
        pd.concat([train_df["anchor_item_id"], train_df["candidate_item_id"]])
    )
    text_dim, image_dim = infer_embedding_dims(train_df)

    vocab_sizes = {
        "item_id": len(vocabs["item_id"]),
        "category": len(vocabs["category"]),
        "subcategory": len(vocabs["subcategory"]),
        "brand": len(vocabs["brand"]),
        "price_tier": len(vocabs["price_tier"]),
    }
    model = ItemEncoder(
        vocab_sizes=vocab_sizes,
        num_tags=len(vocabs["tags"]),
        text_dim=text_dim,
        image_dim=image_dim,
        embedding_dim=args.embedding_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_positive = int((train_df["label"] == 1).sum())
    steps_per_epoch = max(1, num_positive // args.batch_size)

    mlflow.set_experiment("retrieval_encoder")
    with mlflow.start_run(run_name="retrieval_encoder"):
        mlflow.log_params(vars(args))
        mlflow.log_param("text_dim", text_dim)
        mlflow.log_param("image_dim", image_dim)
        mlflow.log_param("item_id_vocab_size", vocab_sizes["item_id"])
        log_reproducibility_metadata(
            engine, "retrieval_training_examples", extra_tables=["ranking_training_examples"]
        )

        best_recall = float("-inf")
        best_state = None
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            epoch_losses = []
            for _ in range(steps_per_epoch):
                batch = sample_training_batch(train_df, args.batch_size, args.hard_negatives_per_anchor, rng)
                anchor_features = encode_item_side(batch.anchor_rows, "anchor", vocabs, image_dim)
                pos_features = encode_item_side(batch.anchor_rows, "candidate", vocabs, image_dim)
                hard_neg_features, hard_neg_valid = encode_hard_negatives(batch, vocabs, image_dim)

                anchor_emb = model(anchor_features)
                pos_emb = model(pos_features)
                hard_neg_emb = model(hard_neg_features).view(len(batch.anchor_rows), args.hard_negatives_per_anchor, -1)

                loss = contrastive_loss(anchor_emb, pos_emb, hard_neg_emb, hard_neg_valid)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            val_recall = recall_at_k(model, ranking_val, item_catalog_df, vocabs, image_dim, k=args.recall_k)
            mlflow.log_metric("train_loss", mean_loss, step=epoch)
            mlflow.log_metric("val_recall_at_200", val_recall, step=epoch)
            logger.info("epoch=%d train_loss=%.4f val_recall_at_%d=%.4f", epoch, mean_loss, args.recall_k, val_recall)

            if val_recall > best_recall:
                best_recall = val_recall
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping at epoch=%d (best val_recall_at_%d=%.4f)", epoch, args.recall_k, best_recall)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        elapsed = time.perf_counter() - start
        mlflow.log_metric("training_duration_seconds", elapsed)
        mlflow.log_metric("best_val_recall_at_200", best_recall)

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_preprocessing_artifacts(vocabs, Path(tmp_dir))
        mlflow.pytorch.log_model(model, "model")

    logger.info("Retrieval training complete in %.1fs. best_val_recall_at_%d=%.4f", elapsed, args.recall_k, best_recall)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # override=False (the default): see run_dataset_builders.py's docstring --
    # container-injected env vars must win over the bind-mounted .env.
    load_dotenv()
    main(parse_args())
