from pathlib import Path


LABELS = ["narrator_shift", "tense_shift", "register_shift", "emotion_shift"]


PROCESSED_PKL_PRIMARY = Path("data/processed/narrative_cues.pkl")
PROCESSED_PKL_FALLBACK = Path("processed/narrative_cues.pkl")


def resolve_processed_pkl() -> Path:
    if PROCESSED_PKL_PRIMARY.exists():
        return PROCESSED_PKL_PRIMARY
    if PROCESSED_PKL_FALLBACK.exists():
        return PROCESSED_PKL_FALLBACK
   
    return PROCESSED_PKL_PRIMARY