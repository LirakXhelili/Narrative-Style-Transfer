
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from .features import build_feature_matrix
import joblib

RANDOM_SEED = 42


def main():
    df = pd.read_pickle("data/processed/narrative_cues.pkl")

    X = build_feature_matrix(df)
    y = df["has_transfer"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=== Traditional baseline: binary style-transfer detection ===")
    print(classification_report(y_test, y_pred, digits=3))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, Path("models/traditional_logreg.joblib"))
    print("Saved model to models/traditional_logreg.joblib")


if __name__ == "__main__":
    main()
