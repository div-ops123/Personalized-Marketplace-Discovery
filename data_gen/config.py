"""Reproducibility and scale constants for synthetic reference/event data generation."""

import datetime
import os

from common.constants import ATTRIBUTION_WINDOW_HOURS  # noqa: F401 -- re-exported

GLOBAL_SEED = int(os.environ.get("SYNTHETIC_DATA_SEED", "42"))

N_ITEMS = 5000
N_USERS = 3000

COLD_START_FRACTION = 0.07

TEXT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Event simulation (raw historical log) ---

# Fixed (not wall-clock "today") so the simulated log is exactly reproducible
# regardless of what date the generator is actually run on.
SIMULATION_START_DATE = datetime.date(2025, 1, 1)

SIMULATION_WINDOW_DAYS = 90

# Last N days of the window, reserved as a time-based holdout. Not applied by
# the simulator itself (events are generated identically across the whole
# window) -- this is a recorded cutoff for Phase 4 / training to use later.
HOLDOUT_DAYS = 14

# Poisson mean; most user-days have zero PDP views, some have one or more.
PDP_VIEWS_PER_USER_PER_DAY_MEAN = 0.3

CANDIDATE_POOL_SIZE = 200
TOP_K_IMPRESSIONS = 20

# Gaussian noise added to cosine similarity when scoring/ranking candidates,
# so candidate order isn't perfectly deterministic content-similarity rank.
SIMILARITY_NOISE_STD = 0.05

# Click probability = BASE_CLICK_RATE * position_decay(position) *
# affinity_multiplier, clipped to [0, 1]. Tuned so aggregate CTR lands in a
# realistic single-digit-percent range.
BASE_CLICK_RATE = 0.12
AFFINITY_MULTIPLIER_CLIP = (0.2, 3.0)

# Fraction of clicks that convert to an attributed purchase within
# ATTRIBUTION_WINDOW_HOURS, before the same affinity multiplier is applied.
CLICK_TO_PURCHASE_BASE_RATE = 0.06

# Per user, per day, probability of an organic purchase (no preceding click)
# -- kept low so organic purchases are a real but minority case.
ORGANIC_PURCHASE_RATE = 0.0005

DEVICE_DISTRIBUTION = {"mobile": 0.60, "desktop": 0.32, "tablet": 0.08}
COUNTRY_DISTRIBUTION = {
    "US": 0.45,
    "UK": 0.15,
    "CA": 0.10,
    "DE": 0.10,
    "FR": 0.08,
    "IN": 0.07,
    "AU": 0.05,
}
