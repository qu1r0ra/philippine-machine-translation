import random

import pandas as pd

from src.config import (
    AUGMENT_N_COPIES,
    DROP_PROB,
    DUP_PROB,
    SWAP_PROB,
)


def inject_noise(
    tokens: list[str],
    swap_prob: float = SWAP_PROB,
    drop_prob: float = DROP_PROB,
    dup_prob: float = DUP_PROB,
) -> list[str]:
    """
    Apply simple noise injection to a list of tokens.

    Operations:
      - Randomly swap neighboring tokens
      - Randomly drop tokens
      - Randomly duplicate tokens
    """
    new_tokens = tokens.copy()

    # 1. Swap neighboring tokens with probability `swap_prob`
    for i in range(len(new_tokens) - 1):
        if random.random() < swap_prob:
            new_tokens[i], new_tokens[i + 1] = new_tokens[i + 1], new_tokens[i]

    # 2. Drop tokens with probability `drop_prob`
    new_tokens = [tok for tok in new_tokens if random.random() > drop_prob]

    # 3. Duplicate tokens with probability `dup_prob`
    i = 0
    while i < len(new_tokens):
        if random.random() < dup_prob:
            new_tokens.insert(i, new_tokens[i])
            i += 1  # skip over the duplicate
        i += 1

    return new_tokens


def augment_dataset(
    df: pd.DataFrame,
    src_col: str = "src_tokens",
    tgt_col: str = "tgt_tokens",
    n_copies: int = AUGMENT_N_COPIES,
) -> pd.DataFrame:
    """
    Apply noise injection to the source sentences of a dataset.

    Args:
        df: Input DataFrame containing source and target columns.
        src_col: Name of the source column.
        tgt_col: Name of the target column.
        n_copies: Number of noisy variants to generate per sentence.

    Returns:
        A new DataFrame containing both original and augmented sentence pairs.
    """
    if n_copies <= 0:
        print(
            "[INFO] No augmentation requested (n_copies=0). Returning original dataset."
        )
        return df

    print(f"[INFO] Augmenting dataset: {len(df):,} rows with {n_copies} copies each...")
    rows: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        src = row[src_col]
        tgt = row[tgt_col]
        rows.append((src, tgt))
        for _ in range(n_copies):
            noisy = " ".join(inject_noise(src.split()))
            rows.append((noisy, tgt))

    out_df = pd.DataFrame(rows, columns=[src_col, tgt_col])
    print(
        f"[INFO] Augmentation complete — total rows: "
        f"{len(out_df):,} ({(len(out_df) / len(df)):.1f}x increase)"
    )
    return out_df


def mix_datasets(
    base_df: pd.DataFrame,
    mix_df: pd.DataFrame,
    mix_ratio: float,
    src_col: str = "src_tokens",
    tgt_col: str = "tgt_tokens",
) -> pd.DataFrame:
    """
    Combine the base dataset with a fraction of another dataset
    (e.g., Chavacano–Spanish) for cross-lingual augmentation.

    Args:
        base_df: The main dataset (e.g., Cebuano–Spanish).
        mix_df: The auxiliary dataset to draw from (e.g., Chavacano–Spanish).
        mix_ratio: Proportion of base size to sample from mix_df (e.g., 0.1 for 10%).
        src_col: Source column name.
        tgt_col: Target column name.

    Returns:
        A new DataFrame containing base data + sampled mix data.
    """
    n_mix = int(len(base_df) * mix_ratio)
    sampled_mix = mix_df.sample(n=min(n_mix, len(mix_df)), random_state=42)
    combined = pd.concat([base_df, sampled_mix], ignore_index=True)
    print(
        f"[INFO] Mixed datasets: base={len(base_df):,}, "
        f"mix added={len(sampled_mix):,}, "
        f"total={len(combined):,} ({mix_ratio*100:.0f}% mix ratio)"
    )
    return combined[[src_col, tgt_col]]
