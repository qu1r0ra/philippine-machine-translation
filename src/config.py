"""
Global configuration file for the Philippine Machine Translation project.
Defines shared paths, constants, and helper functions for all notebooks and modules.
"""

import random
from pathlib import Path

import numpy as np

# ============================================================
# Project directories
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUGMENTED_DIR = DATA_DIR / "augmented"
MODELS_DIR = DATA_DIR / "models"

# ============================================================
# Language configuration
# ============================================================

LANGUAGE_PAIRS = [
    ("cebuano", "spanish"),
    ("cebuano", "tausug"),
    ("chavacano", "spanish"),
    ("ivatan", "yami"),
    ("pangasinense", "ilokano"),
    ("tagalog", "bikolano"),
    ("tagalog", "kapampangan"),
    ("tagalog", "spanish"),
]

# Default pair to train
SOURCE_LANG = LANGUAGE_PAIRS[0][0]
TARGET_LANG = LANGUAGE_PAIRS[0][1]

SOURCE_COL = "language1"
TARGET_COL = "language2"

# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 26


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Ensure reproducibility across modules."""
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Preprocessing
# ============================================================

MIN_SENT_LEN = 2
MAX_SENT_LEN = 100
LOWERCASE = True
REMOVE_PUNCT = True
REMOVE_NUMBERS = True
STRIP_EXTRA_SPACES = True

TRAIN_SPLIT = 0.9

# ============================================================
# Data augmentation
# ============================================================

AUGMENT_N_COPIES = 1  # how many noisy copies per sentence
SWAP_PROB = 0.05  # probability of swapping adjacent tokens
DROP_PROB = 0.03  # probability of dropping a token
DUP_PROB = 0.01  # probability of duplicating a token

# ============================================================
# Utility
# ============================================================


def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        AUGMENTED_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


ensure_dirs()
set_seed()

print("[CONFIG] Directories ensured and random seed set.")
