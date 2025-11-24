from pathlib import Path
import spacy
import pandas as pd
from .dataset import load_jsonl


nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> dict:
    doc = nlp(text)
    tokens = [t.text for t in doc]
    lemmas = [t.lemma_ for t in doc]
    pos = [t.pos_ for t in doc]

    return {
        "text": text,
        "tokens": tokens,
        "lemmas": lemmas,
        "pos": pos,
    }

def preprocess_dataset(input_path: str, output_path: str):
    df = load_jsonl(input_path)
    records = []

    for _, row in df.iterrows():
        proc = preprocess_text(row["text"])
        proc["id"] = row["id"]
        proc["labels"] = row["labels"]
        proc["has_transfer"] = row["has_transfer"]

        # also keep the multi-label columns if you want
        for col in df.columns:
            if col.startswith("label_"):
                proc[col] = row[col]

        records.append(proc)

    out_df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(output_path)  # binary Pandas file
    print(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    preprocess_dataset(
        "data/anotations/narrative_cues.jsonl",
        "data/processed/narrative_cues.pkl",
    )
