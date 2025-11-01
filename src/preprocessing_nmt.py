"""
Preprocessing utilities for Neural Machine Translation (OpenNMT-py).
Handles text cleaning, normalization, and export to .src / .tgt files.
"""

import re
from pathlib import Path
import pandas as pd
from src.config import PROCESSED_DIR, SOURCE_COL, TARGET_COL


def normalize_text(text: str) -> str:
    """Lowercase and remove unwanted characters (keeps accents, ñ, etc)."""
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_corpus(
    df: pd.DataFrame, src_col: str = SOURCE_COL, tgt_col: str = TARGET_COL
) -> pd.DataFrame:
    """
    Preprocess both source and target texts in a DataFrame.
    Drops rows with missing or invalid translations (e.g., "N/A").
    """
    print(f"\n[Preprocessing] Cleaning and tokenizing columns: {src_col}, {tgt_col}")
    df = df.copy()

    invalid_values = ["N/A", "n/a", "na", "", None]
    df = df[~df[src_col].isin(invalid_values)]
    df = df[~df[tgt_col].isin(invalid_values)]
    df = df.dropna(subset=[src_col, tgt_col])
    df = df.drop(["usfm", "book", "verse", "chapter"], axis=1)

    df["src_tokens"] = df[src_col].apply(normalize_text)
    df["tgt_tokens"] = df[tgt_col].apply(normalize_text)

    print(f"[Preprocessing] {len(df):,} sentence pairs remaining.")
    return df


def export_opennmt_files(
    df: pd.DataFrame,
    split_name: str,
    output_dir: Path = PROCESSED_DIR,
    src_col: str = SOURCE_COL,
    tgt_col: str = TARGET_COL,
) -> None:
    """Exports source and target sentences to OpenNMT .src/.tgt files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    src_path = output_dir / f"{split_name}.src"
    tgt_path = output_dir / f"{split_name}.tgt"

    print(f"\n[Exporting] Writing to {src_path} and {tgt_path}")
    df[src_col].to_csv(src_path, index=False, header=False)
    df[tgt_col].to_csv(tgt_path, index=False, header=False)

    print(f"[Export] Saved {split_name}.src and {split_name}.tgt to {output_dir}")
