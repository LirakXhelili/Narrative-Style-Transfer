import json
from pathlib import Path
import pandas
from .config import LABELS

def load_josnl(path: str | Path) -> pandas.DataFrame:
    path = Path(path)
    data = []
    with path.open("r",encoding="utf-8") as f:
        for rreshti in f:
            rreshti = rreshti.strip()
            if not rreshti:
                continue
            objekti = json.loads(rreshti)
            data.append(objekti)
    dataframe = pandas.DataFrame(data)

    dataframe["has_transfer"] = dataframe["labels"].apply(lambda ls: len(ls) > 0)


    for label in LABELS:
        dataframe[f"label_{label}"] = dataframe["labels"].apply(
            lambda ls, l=label: int(l in ls)
        )
    return dataframe

if __name__ == "__main__":
    dataframe = load_josnl("data/anotations/narrative_cues.jsonl")
    print(dataframe.head())
    print(dataframe[["id","labels","has_transfer"]])