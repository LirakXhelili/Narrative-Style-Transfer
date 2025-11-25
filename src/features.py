import numpy
import pandas
from collections import Counter
from typing import List


FIRST_PERSON = {"i", "we", "me", "us", "my", "our"}
THIRD_PERSON = {"he", "she", "they", "him", "her", "them", "his", "their"}
NEGATIVE_WORDS = {"fear", "afraid", "anger", "sad", "terrified", "panic"}  
POS_VERBS = {"VERB", "AUX"}

FORMAL_MARKERS = {"therefore", "however", "moreover"}
INFORMAL_MARKERS = {"gonna", "wanna", "kinda", "lol", "dude", "yeah"}

def compute_features_row(row: pandas.Series)->numpy.ndarray:
    tokens = [t.lower() for t in row["tokens"]]
    pos = row["pos"]
    length = len(tokens) or 1

    counts = Counter(tokens)
    fp_count = sum(counts[w] for w in FIRST_PERSON)
    tp_count = sum(counts[w] for w in THIRD_PERSON)
    fp_ratio = fp_count / length
    tp_ratio = tp_count / length
    verb_count = sum(1 for p in pos if p in POS_VERBS)
    verb_ratio = verb_count / length
    neg_count = sum(counts[w] for w in NEGATIVE_WORDS)
    neg_ratio = neg_count / length
    formal_count = sum(counts[w] for w in FORMAL_MARKERS)
    informal_count = sum(counts[w] for w in INFORMAL_MARKERS)
    formal_ratio = formal_count / length
    informal_ratio = informal_count / length
    avg_token_len = sum(len(t) for t in tokens) / length

    return numpy.array([
        fp_ratio,
        tp_ratio,
        verb_ratio,
        neg_ratio,
        formal_ratio,
        informal_ratio,
        avg_token_len,
        length,         
    ], dtype=float)


def build_feature_matrix(df: pandas.DataFrame) -> numpy.ndarray:
    feats = [compute_features_row(row) for _, row in df.iterrows()]
    return numpy.vstack(feats)