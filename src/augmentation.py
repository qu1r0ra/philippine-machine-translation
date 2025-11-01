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
            i += 1  # skip the duplicate
        i += 1

    return new_tokens


def augment_dataset(df, src_col="src", tgt_col="tgt", n_copies=1):
    """Apply noise injection to source sentences."""
    rows = []
    for _, row in df.iterrows():
        src = row[src_col]
        tgt = row[tgt_col]
        rows.append((src, tgt))
        for _ in range(n_copies):
            noisy = " ".join(inject_noise(src.split()))
            rows.append((noisy, tgt))
    return pd.DataFrame(rows, columns=[src_col, tgt_col])
