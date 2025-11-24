from pathlib import Path
import spacy
import pandas as pd
from .dataset import load_jsonl

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.max_length = 2_000_000  # just to be safe for longer segments


def preprocess_texts(texts: list[str]) -> list[dict]:
    """
    Process a list of texts with nlp.pipe (much faster than nlp(text) in a loop).
    Returns a list of dicts with tokens / lemmas / pos.
    """
    records = []
    for doc in nlp.pipe(texts, batch_size=32):
        tokens = [t.text for t in doc]
        lemmas = [t.lemma_ for t in doc]
        pos = [t.pos_ for t in doc]
        records.append(
            {
                "text": doc.text,
                "tokens": tokens,
                "lemmas": lemmas,
                "pos": pos,
            }
        )
    return records


def preprocess_dataset(input_path: str, output_path: str):
    print(f"Loading JSONL from {input_path} ...")
    df = load_jsonl(input_path)
    print(f"Loaded {len(df)} rows")

    texts = df["text"].tolist()

    print("Running spaCy preprocessing...")
    proc_records = preprocess_texts(texts)
    print("spaCy preprocessing done.")

    out_records = []
    for base, (_, row) in zip(proc_records, df.iterrows()):
        base["id"] = row["id"]
        base["labels"] = row.get("labels", [])
        base["has_transfer"] = row.get("has_transfer", bool(base["labels"]))
        # keep one-hot label_* columns if exist
        for col in df.columns:
            if col.startswith("label_"):
                base[col] = row[col]
        out_records.append(base)

    out_df = pd.DataFrame(out_records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(output_path)
    print(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    preprocess_dataset(
        "data/annotations/narrative_cues.jsonl",
        "data/processed/narrative_cues.pkl",
    )
