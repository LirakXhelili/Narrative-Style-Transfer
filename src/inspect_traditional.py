import joblib
import pandas as pd

from .config import LABELS, resolve_processed_pkl
from .features import build_feature_matrix


TOP_K = 20


def _get_feature_names(X):

    if hasattr(X, "get_feature_names_out"):
        try:
            return X.get_feature_names_out()
        except Exception:
            pass
    if hasattr(X, "feature_names_out"):
        return X.feature_names_out
    return [f"feat_{i}" for i in range(X.shape[1])]


def main():
    df = pd.read_pickle(resolve_processed_pkl())
    X = build_feature_matrix(df)
    feature_names = _get_feature_names(X)

    clf = joblib.load("models/traditional_logreg_multilabel.joblib")

    print("=== Top coefficients per cue (LogReg One-vs-Rest) ===")
    for label, est in zip(LABELS, clf.estimators_):
        coefs = est.coef_[0]
        idx_sorted = coefs.argsort()

        top_pos = idx_sorted[-TOP_K:][::-1]
        top_neg = idx_sorted[:TOP_K]

        print(f"\n--- Cue: {label} ---")
        print("Top positive (pushes prediction toward 1):")
        for i in top_pos:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")

        print("Top negative (pushes prediction toward 0):")
        for i in top_neg:
            print(f"  {feature_names[i]}: {coefs[i]:.4f}")


if __name__ == "__main__":
    main()