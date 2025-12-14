from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from .config import LABELS, resolve_processed_pkl
from .dataset import get_multilabel_targets
from .features import build_feature_matrix


RANDOM_SEED = 42


def main():
    pkl_path = resolve_processed_pkl()
    df = pd.read_pickle(pkl_path)

    X = build_feature_matrix(df)
    y = get_multilabel_targets(df)  # (N, 4)

   
    strat = df["has_transfer"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=strat,
    )

    base = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    clf = OneVsRestClassifier(base)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("=== Traditional baseline: MULTI-LABEL cue detection (One-vs-Rest LogReg) ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )

    micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"Micro-F1: {micro:.3f}")
    print(f"Macro-F1: {macro:.3f}")

    Path("models").mkdir(exist_ok=True)
    out_path = Path("models/traditional_logreg_multilabel.joblib")
    joblib.dump(clf, out_path)
    print(f"Saved model to {out_path.as_posix()}")


if __name__ == "__main__":
    main()