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
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

# Subdirectories for intermediate artifacts
ALIGNMENTS_DIR = PROCESSED_DIR / "alignments"
PHRASE_TABLES_DIR = MODELS_DIR / "phrase_tables"
EVAL_OUTPUT_DIR = RESULTS_DIR / "translations"

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
]

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
# Feature engineering
# ============================================================

SMOOTHING_ALPHA = 0.1

# ============================================================
# Modeling
# ============================================================

MAX_ITERS = 5

# ============================================================
# Evaluation
# ============================================================

EVAL_SAMPLE_SIZE = 100
BLEU_SMOOTHING = True
SAVE_TRANSLATIONS = True

# ============================================================
# Utility
# ============================================================


def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for path in [
        LOGS_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        ALIGNMENTS_DIR,
        PHRASE_TABLES_DIR,
        EVAL_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


ensure_dirs()
set_seed()

print("[CONFIG] Directories ensured and random seed set.")
