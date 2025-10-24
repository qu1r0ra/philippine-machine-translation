"""
Preprocessing utilities for the Philippine MT project.
Handles text cleaning, tokenization, and word class creation via FastText embeddings.
"""

import json
import multiprocessing
import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.cluster import KMeans

from src.config import PROCESSED_DIR, RANDOM_SEED

# ============================================================
# Normalization and tokenization
# ============================================================


def normalize_text(text: str) -> str:
    """
    Lowercase and remove non-letter characters.
    Adjust regex to accommodate accent marks and ñ common in PH languages.
    """
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_sentence(text: str) -> list[str]:
    """Simple work tokenizer."""
    return nltk.word_tokenize(text)


def preprocess_sentence(text: str) -> list[str]:
    """Full text normalization and tokenization pipeline."""
    return tokenize_sentence(normalize_text(text))


# ============================================================
# Word class creation using FastText + KMeans
# ============================================================


def train_fasttext(
    sentences: list[list[str]],
    min_count: int = 3,
    vector_size: int = 100,
    epochs: int = 10,
    min_n: int = 3,
    max_n: int = 6,
    model_path: Path | None = None,
) -> FastText:
    """
    Train a FastText model on the tokenized corpus.
    Saves model if a path is provided.
    """
    print(f"\n[FastText] Training on {len(sentences):,} sentences...")

    model = FastText(
        sentences=sentences,
        min_count=min_count,
        vector_size=vector_size,
        workers=max(1, multiprocessing.cpu_count() - 1),
        epochs=epochs,
        min_n=min_n,
        max_n=max_n,
    )

    if model_path:
        model.save(str(model_path))
        print(f"[FastText] Model saved to {model_path}")

    return model


def cluster_words(
    model: FastText, n_clusters: int = 100, random_state: int = RANDOM_SEED
) -> dict[str, str]:
    """
    Cluster word embeddings into word classes using KMeans.
    Returns a mapping from word -> class ID.
    """
    vocab = [*map(str, model.wv.key_to_index)]

    vectors = np.array([model.wv[w] for w in vocab])

    print(
        f"[Clustering] Running KMeans on {len(vocab):,} "
        f"word vectors ({vectors.shape[1]} dims) ..."
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(vectors)

    word2class = {word: f"c{label}" for word, label in zip(vocab, labels, strict=True)}
    print(f"[Clustering] Done — created {n_clusters} clusters.")

    return word2class


def save_word_classes(
    word2class: dict[str, str],
    output_path: Path = PROCESSED_DIR / "word_classes.json",
):
    """Save word-to-class mapping as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(word2class, f, ensure_ascii=False, indent=2)
    print(f"[Save] Word classes saved to {output_path}")


# ============================================================
# Corpus-level preprocessing pipeline
# ============================================================


def preprocess_corpus(df: pd.DataFrame, src_col: str, tgt_col: str) -> pd.DataFrame:
    """
    Preprocess both source and target texts in a DataFrame.
    Returns a new DataFrame with tokenized columns.
    """
    print(f"\n[Preprocessing] Cleaning and tokenizing columns: {src_col}, {tgt_col}")
    df = df.copy()

    df["src_tokens"] = df[src_col].apply(preprocess_sentence)
    df["tgt_tokens"] = df[tgt_col].apply(preprocess_sentence)

    print("[Preprocessing] Done.")
    return df


def build_word_classes(
    df: pd.DataFrame, output_path: Path, vector_size: int = 100, n_clusters: int = 100
) -> dict[str, str]:
    """
    Builds word embeddings using FastText and clusters them into word classes.
    Saves the mapping to disk.
    """
    if "src_tokens" not in df or "tgt_tokens" not in df:
        raise ValueError(
            "DataFrame must contain 'src_tokens' and 'tgt_tokens' columns. "
            "Run preprocess_corpus() first."
        )

    all_sentences = df["src_tokens"].tolist() + df["tgt_tokens"].tolist()

    model = train_fasttext(all_sentences, vector_size=vector_size)
    word2class = cluster_words(model, n_clusters=n_clusters)

    save_word_classes(word2class, output_path)
    return word2class
