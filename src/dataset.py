import json
from pathlib import Path
import pandas as pd

from .config import LABELS


def load_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    data = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)

    df = pd.DataFrame(data)

    df["has_transfer"] = df["labels"].apply(lambda ls: len(ls) > 0)

    # per-cue columns (multi-label targets)
    for label in LABELS:
        df[f"label_{label}"] = df["labels"].apply(lambda ls, l=label: int(l in ls))

    return df


def get_multilabel_targets(df: pd.DataFrame):
    """Return y as an (N, 4) numpy array in LABELS order."""
    cols = [f"label_{l}" for l in LABELS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing multi-label columns: {missing}")
    return df[cols].astype(int).values


if __name__ == "__main__":
    df = load_jsonl("data/anotations/narrative_cues.jsonl")
    print(df.head())
    print(df[["id", "labels", "has_transfer"] + [f"label_{l}" for l in LABELS]].head())