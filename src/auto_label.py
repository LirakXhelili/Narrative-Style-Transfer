
from pathlib import Path
import json
import re

EMOTION_POS = {
    "delighted", "glad", "happy", "relieved", "enjoy", "enjoyed",
    "pleased", "excited", "curious", "eager",
}

EMOTION_NEG = {
    "afraid", "frightened", "scared", "terrified", "worried",
    "anxious", "sad", "cry", "cried", "lonely", "miserable",
    "angry", "furious", "desperate", "low-spirited", "upset",
}

FORMAL_MARKERS = {
    "therefore", "however", "moreover", "consequently",
    "in conclusion", "thus",
}

INFORMAL_MARKERS = {
    "oh", "dear", "yeah", "ugh", "wow", "sort of", "kind of",
    "you know",
}

FIRST_PERSON = {" i ", " me ", " my ", " we ", " our ", " us "}
THIRD_PERSON = {" he ", " she ", " they ", " him ", " her ", " them "}


def guess_labels(text: str) -> list[str]:
    """
    Heuristic guesses for:
    - NARRATOR_SHIFT
    - TENSE_SHIFT
    - REGISTER_SHIFT
    - EMOTION_SHIFT

    You will later *correct* them in the JSONL file.
    """
    labels: list[str] = []

    t = " " + text.lower() + " "

   
    if any(fp in t for fp in FIRST_PERSON) and any(tp in t for tp in THIRD_PERSON):
        if not (t.strip().startswith("“") and t.strip().endswith("”")):
            labels.append("NARRATOR_SHIFT")

    has_past = bool(re.search(r"\b(was|were|had|did|went|said|came|looked|felt)\b", t))
    has_present = bool(re.search(r"\b(am|is|are|do|does|go|say|come|feel|think)\b", t))
    has_future = " will " in t or " shall " in t

    if (has_past and has_present) or (has_past and has_future):
        labels.append("TENSE_SHIFT")

    if any(w in t for w in FORMAL_MARKERS) and any(w in t for w in INFORMAL_MARKERS):
        labels.append("REGISTER_SHIFT")

    has_pos = any(w in t for w in EMOTION_POS)
    has_neg = any(w in t for w in EMOTION_NEG)

    if has_pos and has_neg:
        labels.append("EMOTION_SHIFT")
    else:
        # Negative emotion plus contrast word
        if " but " in t and has_neg:
            labels.append("EMOTION_SHIFT")

    return labels


def main():
    in_path = Path("data/annotations/candidate_segments.jsonl")
    out_path = Path("data/annotations/narrative_cues.jsonl")

    print(f"Reading from {in_path}")
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"]
            auto = guess_labels(text)

            # if you already hand-labeled something, keep it
            labels = obj.get("labels", [])
            if not labels:
                labels = auto

            # derive has_transfer and one-hot label columns
            obj["labels"] = labels
            obj["has_transfer"] = bool(labels)

            for cue in ["NARRATOR_SHIFT", "TENSE_SHIFT", "REGISTER_SHIFT", "EMOTION_SHIFT"]:
                obj[f"label_{cue}"] = 1 if cue in labels else 0

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote labeled segments to {out_path}")


if __name__ == "__main__":
    main()
