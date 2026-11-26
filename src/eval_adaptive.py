"""
eval_adaptive.py - adaptive-attacker experiment: does retraining on defended typing recover the attack?

Run from the project root (venv active):
    python src/eval_adaptive.py
    python src/eval_adaptive.py --features results/features_s06_s07_s08_s09_s10_s11_s12_s13_pre20_post80.csv

For each defence that has TWO recorded sessions (default: white noise = session_10 + session_12,
cloth over keys = session_09 + session_13) it compares, always testing on a session that was NOT
used for training:

  1. Non-adaptive  : attacker trained on clean sessions only (session_06 + session_07),
                     tested on each defended session.
  2. Adaptive      : attacker trained on ONE defended session, tested on the OTHER (both directions).
  3. Adaptive+clean: attacker trained on clean (06 + 07) + one defended session, tested on the other
                     (both directions).
  4. Reference     : no defence, same small training size: one clean session -> another clean session
                     (06 -> 11 and 07 -> 11), so the adaptive rows can be judged against an
                     undefended attacker with equally little training data.

Models: SVM (RBF) and Random Forest. Features are standardised on the training data only, the SVM's C is
chosen by cross-validation inside the training data only, and test sessions are never used for any choice.
Every number is a top-1 accuracy over 130-131 test presses, so the 95% interval is about +/- 8 points
(top-1 only is shown with its interval; top-3 and top-5 are given for context).

It only reads the features csv and prints to the terminal. It saves nothing.
Caveat for the report: the two sessions of each defence are not perfectly identical (for example the
cloth-over-keys sessions differ in loudness, and the white-noise sessions differ in high-frequency level),
so the adaptive attacker is tested under a small extra mismatch.
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
DEFAULT_FEATURES = "results/features_s06_s07_s08_s09_s10_s11_s12_s13_pre20_post80.csv"
CLEAN_TRAIN = ["session_06", "session_07"]
CLEAN_TEST = "session_11"
DEFENCES = {
    "White noise (phone)": ["session_10", "session_12"],
    "Cloth over the keys": ["session_09", "session_13"],
}


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive-attacker experiment.")
    p.add_argument("--features", default=DEFAULT_FEATURES)
    return p.parse_args()


def ci95(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) * 100


def topk(scores, classes, y_true, k):
    order = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([y in classes[row] for y, row in zip(y_true, order)])) * 100


def run(df, feat, train_sessions, test_session):
    """Train on train_sessions, test on test_session. Returns {model: (top1, ci, top3, top5)}, n_train, n_test."""
    assert test_session not in train_sessions, "test session must differ from the training sessions"
    train = df[df["session"].isin(train_sessions)]
    test = df[df["session"] == test_session]
    if train.empty or test.empty:
        raise SystemExit(f"Missing data for train {train_sessions} or test {test_session} in the features file.")
    scaler = StandardScaler().fit(train[feat].values)
    Xtr, Xte = scaler.transform(train[feat].values), scaler.transform(test[feat].values)
    ytr, yte = train["key"].values, test["key"].values

    n_splits = int(min(5, Counter(ytr).most_common()[-1][1]))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    svm = GridSearchCV(SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr"),
                       {"C": [1, 10, 100]}, cv=cv).fit(Xtr, ytr).best_estimator_
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)

    out = {}
    for name, model, scores in (("SVM", svm, svm.decision_function(Xte)), ("RF", rf, rf.predict_proba(Xte))):
        acc = float(np.mean(model.predict(Xte) == yte))
        out[name] = (acc * 100, ci95(acc, len(yte)), topk(scores, model.classes_, yte, 3),
                     topk(scores, model.classes_, yte, 5))
    return out, len(train), len(test)


def short(sessions):
    return "+".join(s.replace("session_", "") for s in sessions)


def print_rows(title, rows):
    print(title)
    print(f"  {'train -> test':<26}{'n_train':>8}{'SVM top-1':>12}{'+/-':>6}{'SVM top-5':>11}{'RF top-1':>11}{'RF top-5':>10}")
    for label, res, ntr in rows:
        s, r = res["SVM"], res["RF"]
        print(f"  {label:<26}{ntr:>8}{s[0]:>11.1f}%{s[1]:>6.1f}{s[3]:>10.1f}%{r[0]:>10.1f}%{r[3]:>9.1f}%")
    print()


def main():
    args = parse_args()
    df = pd.read_csv(args.features)
    feat = [c for c in df.columns if c.startswith("mfcc")]
    print("Chance: top-1 3.8%, top-5 19.2%.  Each test session has about 130 presses (+/- 8 points on top-1).\n")

    rows = []
    for tr in CLEAN_TRAIN:
        res, ntr, _ = run(df, feat, [tr], CLEAN_TEST)
        rows.append((f"{short([tr])} -> {short([CLEAN_TEST])}  (clean)", res, ntr))
    print_rows("REFERENCE: no defence, one clean training session -> a different clean session", rows)

    for dname, (d1, d2) in DEFENCES.items():
        print("=" * 78)
        print(f"DEFENCE: {dname}   (sessions {short([d1])} and {short([d2])})")
        print("=" * 78)

        rows = []
        for d in (d1, d2):
            res, ntr, _ = run(df, feat, CLEAN_TRAIN, d)
            rows.append((f"clean {short(CLEAN_TRAIN)} -> {short([d])}", res, ntr))
        print_rows("1) NON-ADAPTIVE: attacker trained on clean typing only", rows)

        rows = []
        for tr, te in ((d1, d2), (d2, d1)):
            res, ntr, _ = run(df, feat, [tr], te)
            rows.append((f"{short([tr])} -> {short([te])}", res, ntr))
        print_rows("2) ADAPTIVE: attacker trained on one defended session, tested on the other", rows)

        rows = []
        for tr, te in ((d1, d2), (d2, d1)):
            res, ntr, _ = run(df, feat, CLEAN_TRAIN + [tr], te)
            rows.append((f"clean+{short([tr])} -> {short([te])}", res, ntr))
        print_rows("3) ADAPTIVE + CLEAN: trained on clean sessions plus one defended session", rows)


if __name__ == "__main__":
    main()
