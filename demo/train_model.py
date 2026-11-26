"""
train_model.py - train the attacker used by the live guessing demo and save it.

Run from the project root (venv active):
    python demo/train_model.py
    python demo/train_model.py --exclude session_17 session_18      # keep sessions out of training
    python demo/train_model.py --features results/features_....csv

What it does
  * Reads a features file made by src/extract_features.py (default: the most recently
    changed results/features_*.csv).
  * Trains an SVM (RBF) with probability estimates on the CLEAN sessions in that file
    (all of them, minus any --exclude sessions), standardising features on training data only.
    C is chosen by cross-validation inside the training data.
  * Saves the model with its settings (clip window, which sessions it saw) to
    demo/model/keystroke_model.joblib. It will not overwrite an existing model: use --out.

Keep your held-out test sessions out of the model with --exclude, so the demo and the final
test are honest. The live demo refuses to replay a session the model was trained on.

The clip window (--pre / --post, seconds) and --highpass must match how the features file
was made (they are part of its file name, e.g. ..._pre20_post80.csv).
"""

import argparse
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", message="The number of unique classes")
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Train and save the demo attacker.")
    p.add_argument("--features", default=None)
    p.add_argument("--exclude", nargs="*", default=[], help="sessions to keep OUT of training")
    p.add_argument("--pre", type=float, default=0.02)
    p.add_argument("--post", type=float, default=0.08)
    p.add_argument("--highpass", type=float, default=0.0)
    p.add_argument("--out", default="demo/model/keystroke_model.joblib")
    return p.parse_args()


def main():
    args = parse_args()
    if args.features:
        feat_path = Path(args.features)
    else:
        found = sorted(Path("results").glob("features_*.csv"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit("No results/features_*.csv found. Run this from the project root.")
        feat_path = found[-1]
    if not feat_path.exists():
        raise SystemExit(f"{feat_path} not found.")

    tag = f"pre{int(args.pre * 1000)}_post{int(args.post * 1000)}"
    if args.highpass > 0:
        tag += f"_hp{int(args.highpass)}"
    if tag not in feat_path.name:
        raise SystemExit(f"The features file name ({feat_path.name}) does not contain '{tag}'. "
                         "Check --pre / --post / --highpass match how it was made.")

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists. Not overwriting. Use --out for a different name.")

    df = pd.read_csv(feat_path)
    feat = [c for c in df.columns if c.startswith("mfcc")]
    clean = df[(df["condition"] == "clean") & (~df["session"].isin(args.exclude))]
    sessions = sorted(clean["session"].unique())
    if len(sessions) < 2:
        raise SystemExit(f"Need at least 2 clean sessions to train; found {sessions}.")

    X, y = clean[feat].values, clean["key"].values
    n_splits = int(min(5, Counter(y).most_common()[-1][1]))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale")),
                        {"svc__C": [1, 10, 100]}, cv=cv).fit(X, y)
    best_c = grid.best_params_["svc__C"]
    model = make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", gamma="scale", C=best_c, probability=True, random_state=SEED))
    model.fit(X, y)

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "classes": list(model.classes_), "feature_names": feat,
        "pre": args.pre, "post": args.post, "highpass": args.highpass,
        "sessions": sessions, "excluded": sorted(args.exclude), "svm_C": best_c,
        "features_file": feat_path.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(clean)),
    }, out)

    print(f"Features file : {feat_path}")
    print(f"Trained on    : {sessions}  ({len(clean)} clean presses)")
    print(f"Left out      : {sorted(args.exclude) if args.exclude else 'nothing'}")
    print(f"SVM C         : {best_c}")
    print(f"Saved         : {out}")
    print("Note: no accuracy is printed here on purpose; test it on sessions it did not train on.")


if __name__ == "__main__":
    main()
