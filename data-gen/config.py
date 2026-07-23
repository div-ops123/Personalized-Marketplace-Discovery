"""Reproducibility and scale constants for synthetic reference data generation."""

import os

GLOBAL_SEED = int(os.environ.get("SYNTHETIC_DATA_SEED", "42"))

N_ITEMS = 5000
N_USERS = 3000

COLD_START_FRACTION = 0.07

TEXT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
