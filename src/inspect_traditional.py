
import joblib
import pandas as pd
from .features import build_feature_matrix


def main():
    df = pd.read_pickle("data/processed/narrative_cues.pkl")
    X = build_feature_matrix(df)

    clf = joblib.load("models/traditional_logreg.joblib")

    feature_names = getattr(X, "feature_names_out", None)
    if feature_names is None:
       
        try:
            feature_names = X.get_feature_names_out()
        except Exception:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    coefs = clf.coef_[0]

        
    idx_sorted = coefs.argsort()
    top_pos = idx_sorted[-20:][::-1]
    top_neg = idx_sorted[:20]

    print("=== Top features for STYLE TRANSFER (positive weights) ===")
    for i in top_pos:
        print(f"{feature_names[i]}: {coefs[i]:.3f}")

    print("\n=== Top features for NO TRANSFER (negative weights) ===")
    for i in top_neg:
        print(f"{feature_names[i]}: {coefs[i]:.3f}")


if __name__ == "__main__":
    main()
