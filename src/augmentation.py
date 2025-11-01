import random
import pandas as pd


def inject_noise(
    tokens: list[str],
    swap_prob: float = 0.05,
    drop_prob: float = 0.03,
    dup_prob: float = 0.01,
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
    src_col: str = "src",
    tgt_col: str = "tgt",
    n_copies: int = 1,
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
    rows: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        src: str = row[src_col]
        tgt: str = row[tgt_col]
        rows.append((src, tgt))
        for _ in range(n_copies):
            noisy: str = " ".join(inject_noise(src.split()))
            rows.append((noisy, tgt))

    return pd.DataFrame(rows, columns=[src_col, tgt_col])
