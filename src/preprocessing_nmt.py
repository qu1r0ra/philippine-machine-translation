"""
Preprocessing utilities for Neural Machine Translation (OpenNMT-py).
Handles text cleaning, normalization, filtering, and export to .src / .tgt files.
"""

import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR,
    SOURCE_COL,
    TARGET_COL,
    MIN_SENT_LEN,
    MAX_SENT_LEN,
    TRAIN_SPLIT,
)

# ============================================================
# Text normalization
# ============================================================


def normalize_text(text: str) -> str:
    """Lowercase, remove unwanted characters, and normalize spacing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", "", text)  # keep Spanish/Philippine diacritics
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Main preprocessing pipeline
# ============================================================


def preprocess_corpus(
    df: pd.DataFrame, src_col: str = SOURCE_COL, tgt_col: str = TARGET_COL
) -> pd.DataFrame:
    """Clean, normalize, and filter parallel text pairs."""
    print(f"\n[Preprocessing] Cleaning and filtering {len(df):,} sentence pairs...")

    df = df.copy()

    # Drop invalid rows
    invalid_values = ["N/A", "n/a", "na", "", None]
    df = df[~df[src_col].isin(invalid_values)]
    df = df[~df[tgt_col].isin(invalid_values)]
    df = df.dropna(subset=[src_col, tgt_col])

    # Drop metadata if present
    for col in ["usfm", "book", "chapter", "verse"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Normalize text
    df["src_tokens"] = df[src_col].apply(normalize_text)
    df["tgt_tokens"] = df[tgt_col].apply(normalize_text)

    # Filter by sentence length
    df = df[df.apply(_valid_length, axis=1)]

    df = df.drop_duplicates(subset=["src_tokens", "tgt_tokens"])
    df = df.reset_index(drop=True)

    print(f"[Preprocessing] {len(df):,} valid sentence pairs remain after cleaning.")
    return df


def _valid_length(row) -> bool:
    """Check if source and target sentence lengths are within thresholds."""
    src_len = len(row["src_tokens"].split())
    tgt_len = len(row["tgt_tokens"].split())
    return (
        MIN_SENT_LEN <= src_len <= MAX_SENT_LEN
        and MIN_SENT_LEN <= tgt_len <= MAX_SENT_LEN
    )


# ============================================================
# Export utilities
# ============================================================


def export_opennmt_files(
    df: pd.DataFrame,
    split_name: str,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Exports to OpenNMT-compatible .src and .tgt files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    src_path = output_dir / f"{split_name}.src"
    tgt_path = output_dir / f"{split_name}.tgt"

    print(f"[Exporting] Writing {split_name} split to {output_dir}")
    df["src_tokens"].to_csv(src_path, index=False, header=False)
    df["tgt_tokens"].to_csv(tgt_path, index=False, header=False)
    print(f"[Export] {split_name}.src and {split_name}.tgt written.")


# ============================================================
# Split and save function
# ============================================================


def split_and_export(df: pd.DataFrame) -> None:
    """Split dataset into train/valid and export both splits."""
    print(f"[Splitting] Train ratio = {TRAIN_SPLIT}")
    train_df, valid_df = train_test_split(
        df, train_size=TRAIN_SPLIT, random_state=26, shuffle=True
    )

    export_opennmt_files(train_df, "train")
    export_opennmt_files(valid_df, "valid")

    print("[Done] Train/valid splits exported successfully.")
