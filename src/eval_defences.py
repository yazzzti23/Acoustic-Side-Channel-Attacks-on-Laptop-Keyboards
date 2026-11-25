"""
eval_defences.py - how much does each defence hurt the keystroke attack?

Run from the project root (venv active):
    python src/eval_defences.py
    python src/eval_defences.py --features results/features_s06_s07_s08_s09_s10_s11_pre20_post80.csv

What it does
  * Trains the attacker ONLY on the clean training sessions (default: session_06 + session_07).
  * Tests that one trained attacker on each remaining session, one at a time:
      - session_11 (clean)  -> the baseline: a clean session the attacker has never seen
      - session_08 (dampened, cloth under the stand)
      - session_09 (dampened, cloth over the keys)
      - session_10 (noisy, white noise from a phone)
    All test sessions are different from the training sessions (session-separated).
  * Models: SVM (RBF) and Random Forest, features standardised on the training data only,
    SVM C chosen by cross-validation inside the training data only. Test sessions are never
    used for any choice.
  * Prints top-1 / top-3 / top-5 accuracy with an approximate 95% interval for top-1, and how many
    percentage points each defence costs the attacker compared with the clean baseline.

It only reads the features csv and prints to the terminal. It saves nothing.
Caveat for the report: every test session has 130 presses (5 per key), so the 95% interval
on top-1 is about +/- 8 points. Differences smaller than that are not reliable.
"""

import argparse
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
DEFAULT_FEATURES = "results/features_s06_s07_s08_s09_s10_s11_pre20_post80.csv"
DEFAULT_TEST = ["session_11", "session_08", "session_09", "session_10"]
# Human-readable setup of each test session (from the setup notes you typed when recording).
DESCRIPTION = {
    "session_08": "dampened: cloth under the stand",
    "session_09": "dampened: cloth over the keys",
    "session_10": "noisy: white noise from a phone",
    "session_11": "clean (baseline)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Clean-trained attacker vs defended sessions.")
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--train", nargs="+", default=["session_06", "session_07"])
    p.add_argument("--test", nargs="+", default=DEFAULT_TEST)
    return p.parse_args()


def topk(scores, classes, y_true, k):
    order = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([y in classes[row] for y, row in zip(y_true, order)]))


def ci95(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) * 100


def main():
    args = parse_args()
    overlap = set(args.train) & set(args.test)
    if overlap:
        raise SystemExit(f"Train and test sessions overlap: {sorted(overlap)}.")

    df = pd.read_csv(args.features)
    feat = [c for c in df.columns if c.startswith("mfcc")]
    train = df[df["session"].isin(args.train)]
    missing = set(args.train) - set(train["session"].unique())
    if missing:
        raise SystemExit(f"Training session(s) not in features file: {sorted(missing)}")
    conds = sorted(train["condition"].unique())
    if conds != ["clean"]:
        print(f"WARNING: training sessions have conditions {conds}, not only 'clean'.")

    scaler = StandardScaler().fit(train[feat].values)
    Xtr = scaler.transform(train[feat].values)
    ytr = train["key"].values

    n_splits = int(min(5, Counter(ytr).most_common()[-1][1]))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    grid = GridSearchCV(SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr"),
                        {"C": [1, 10, 100]}, cv=cv).fit(Xtr, ytr)
    svm = grid.best_estimator_
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)

    print(f"Attacker trained on {args.train}: {len(train)} clean presses, "
          f"SVM C = {grid.best_params_['C']}")
    print("Chance: top-1 3.8%, top-3 11.5%, top-5 19.2%\n")

    results = {}
    for s in args.test:
        part = df[df["session"] == s]
        if part.empty:
            print(f"WARNING: {s} not found in the features file, skipped.")
            continue
        X = scaler.transform(part[feat].values)
        y = part["key"].values
        n = len(y)
        for name, model, scores in (("SVM", svm, svm.decision_function(X)),
                                    ("RF", rf, rf.predict_proba(X))):
            acc1 = float(np.mean(model.predict(X) == y))
            results[(s, name)] = (acc1 * 100, ci95(acc1, n), topk(scores, model.classes_, y, 3) * 100,
                                  topk(scores, model.classes_, y, 5) * 100, n,
                                  part["condition"].iloc[0])

    baseline_session = args.test[0]
    for name in ("SVM", "RF"):
        print(f"=== {name} ===")
        print(f"{'test session':<14}{'setup':<34}{'n':>4}{'top-1':>9}{'+/-':>6}{'top-3':>8}{'top-5':>8}"
              f"{'top-1 vs baseline':>20}")
        base = results.get((baseline_session, name))
        for s in args.test:
            r = results.get((s, name))
            if r is None:
                continue
            drop = "" if (base is None or s == baseline_session) else f"{r[0] - base[0]:+.1f} pts"
            desc = DESCRIPTION.get(s, r[5])
            print(f"{s:<14}{desc:<34}{r[4]:>4}{r[0]:>8.1f}%{r[1]:>5.1f}{r[2]:>7.1f}%{r[3]:>7.1f}%{drop:>20}")
        print()
    print("(+/- is the approximate 95% interval of top-1; differences smaller than it are not reliable)")


if __name__ == "__main__":
    main()
