"""
train_eval.py - first keystroke classifier: train on one session, test on a DIFFERENT session.

Run from the project root (venv active):
    python src/train_eval.py
    python src/train_eval.py --train session_07 --test session_06
    python src/train_eval.py --features results/features_s06_s07_pre20_post80.csv

What it does
  * Loads a features csv made by extract_features.py.
  * Trains on every press of the --train session(s) and tests on every press of the --test
    session(s). Train and test sessions must be different (session-separated evaluation).
  * Models: SVM (RBF kernel) and Random Forest. Features are standardised using the
    TRAINING data only. SVM settings (C) are picked by 5-fold cross-validation inside the
    training session only; the test session is never used for any choice.
  * Reports top-1, top-3 and top-5 accuracy (chance for 26 keys: top-1 3.8%, top-3 11.5%,
    top-5 19.2%), per-key accuracy for the SVM and the most common confusions.

It only reads the features csv and prints to the terminal. It saves nothing.
Caveat to quote in the report: the cross-validation used to choose C is within one session,
so it can be optimistic; the real, honest number is the test-session accuracy.
"""

import argparse
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", message="The number of unique classes")

SEED = 42
DEFAULT_FEATURES = "results/features_s06_s07_pre20_post80.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Session-separated keystroke classification.")
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--train", nargs="+", default=["session_06"], help="training session(s)")
    p.add_argument("--test", nargs="+", default=["session_07"], help="test session(s)")
    p.add_argument("--exclude-flagged", action="store_true",
                   help="drop presses from recordings flagged POSSIBLE_CLIPPING")
    return p.parse_args()


def topk_accuracy(scores, classes, y_true, k):
    order = np.argsort(-scores, axis=1)[:, :k]
    top = classes[order]
    return float(np.mean([y in row for y, row in zip(y_true, top)]))


def main():
    args = parse_args()
    overlap = set(args.train) & set(args.test)
    if overlap:
        raise SystemExit(f"Train and test sessions overlap: {sorted(overlap)}. "
                         "Use different sessions (session-separated evaluation).")

    df = pd.read_csv(args.features)
    feat_cols = [c for c in df.columns if c.startswith("mfcc")]
    if args.exclude_flagged:
        before = len(df)
        df = df[df["quality_flag"] != "POSSIBLE_CLIPPING"]
        print(f"Excluded {before - len(df)} presses from clipped recordings.")

    train = df[df["session"].isin(args.train)]
    test = df[df["session"].isin(args.test)]
    for name, part, wanted in (("train", train, args.train), ("test", test, args.test)):
        missing = set(wanted) - set(part["session"].unique())
        if missing:
            raise SystemExit(f"{name} session(s) not found in {args.features}: {sorted(missing)}")

    print(f"Train: {args.train} ({sorted(train['condition'].unique())}), {len(train)} presses")
    print(f"Test : {args.test} ({sorted(test['condition'].unique())}), {len(test)} presses")
    print(f"Features: {len(feat_cols)} MFCC statistics per press\n")

    scaler = StandardScaler().fit(train[feat_cols].values)
    Xtr = scaler.transform(train[feat_cols].values)
    Xte = scaler.transform(test[feat_cols].values)
    ytr = train["key"].values
    yte = test["key"].values

    # ---- SVM ---------------------------------------------------------------
    n_splits = int(min(5, Counter(ytr).most_common()[-1][1]))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    grid = GridSearchCV(SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr"),
                        {"C": [1, 10, 100]}, cv=cv)
    grid.fit(Xtr, ytr)
    svm = grid.best_estimator_
    svm_scores = svm.decision_function(Xte)
    svm_pred = svm.predict(Xte)

    # ---- Random Forest -----------------------------------------------------
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    rf_scores = rf.predict_proba(Xte)
    rf_pred = rf.predict(Xte)

    classes_svm = svm.classes_
    classes_rf = rf.classes_
    n_keys = len(classes_svm)
    print(f"Chance level: top-1 {100 / n_keys:.1f}%   top-3 {300 / n_keys:.1f}%   top-5 {500 / n_keys:.1f}%\n")
    print(f"SVM chosen C = {grid.best_params_['C']} (cv accuracy inside train session: "
          f"{grid.best_score_ * 100:.1f}%)\n")

    print(f"{'model':<14}{'top-1':>8}{'top-3':>8}{'top-5':>8}")
    for name, scores, classes, pred in (("SVM (RBF)", svm_scores, classes_svm, svm_pred),
                                        ("Random Forest", rf_scores, classes_rf, rf_pred)):
        t1 = float(np.mean(pred == yte)) * 100
        t3 = topk_accuracy(scores, classes, yte, 3) * 100
        t5 = topk_accuracy(scores, classes, yte, 5) * 100
        print(f"{name:<14}{t1:>7.1f}%{t3:>7.1f}%{t5:>7.1f}%")

    # ---- per-key accuracy and confusions (SVM) ------------------------------
    print("\nSVM per-key accuracy on the test session:")
    per_key = {}
    for k in sorted(set(yte)):
        m = yte == k
        per_key[k] = float(np.mean(svm_pred[m] == k)) * 100
    line = ""
    for k, v in per_key.items():
        line += f"{k}:{v:3.0f}%  "
        if len(line) > 70:
            print("  " + line)
            line = ""
    if line:
        print("  " + line)

    conf = Counter((t, p) for t, p in zip(yte, svm_pred) if t != p)
    print("\nMost common SVM mistakes (true -> predicted, count):")
    for (t, p), c in conf.most_common(8):
        print(f"  {t} -> {p}: {c}")


if __name__ == "__main__":
    main()
