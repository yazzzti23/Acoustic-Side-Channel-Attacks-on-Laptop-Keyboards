"""
train_demo_model.py - train the neural-network model used by the live demo.

Run from the project root (venv active):
    python demo/train_demo_model.py

What it does
  * Uses ONLY clean sessions that have a key log (default: the five development sessions
    06 07 11 14 15). Session 16, the held-out final test, is NOT used, so the replay of session 16
    stays an honest test.
  * Cuts a 60 ms clip (10 ms before to 50 ms after each logged key-down), turns it into a log-mel vector.
  * Adds three augmented copies of every TRAINING clip (8 ms earlier, 8 ms later, light noise).
    No outside audio and no synthetic keys are used.
  * Trains 5 small neural networks (different random seeds) and averages their probabilities
    (soft voting), so the demo can show a confidence for each guess.
  * Saves everything to demo/model/keystroke_model.joblib. It refuses to overwrite an existing file.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))                 # audio_features.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # extract_features.py
import audio_features as af  # noqa: E402
import extract_features as ef  # noqa: E402

DEFAULT_SESSIONS = ["session_06", "session_07", "session_11", "session_14", "session_15"]


def parse_args():
    p = argparse.ArgumentParser(description="Train the demo's neural-network model.")
    p.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    p.add_argument("--pre", type=float, default=0.01)
    p.add_argument("--post", type=float, default=0.05)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--no-aug", action="store_true", help="do not add augmented copies of the training clips")
    p.add_argument("--out", default="demo/model/keystroke_model.joblib")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists. Not overwriting. Use --out for a different name.")
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = Path("dataset/raw")
    meta = pd.read_csv("dataset/metadata_master.csv")
    for s in args.sessions:
        conds = meta.loc[meta.session == s, "condition"].unique()
        if list(conds) != ["clean"]:
            raise SystemExit(f"{s}: condition is {list(conds)}, the demo model is trained on clean sessions only.")
        if not (raw / s / "events").is_dir():
            raise SystemExit(f"{s} has no key log (events/ folder).")

    rng = np.random.default_rng(0)
    X, y = [], []
    n_orig = 0
    for s in args.sessions:
        for key in ef.KEYS:
            audio, sr = sf.read(raw / s / f"{key}_burst.wav", dtype="float32")
            audio = audio[:, 0] if audio.ndim == 2 else audio
            n = af.n_samples(sr, args.pre, args.post)
            for t in ef.read_down_times(raw / s / "events" / f"{key}_events.csv", key):
                variants = [dict()]
                if not args.no_aug:
                    variants += [dict(shift=-0.008), dict(shift=0.008), dict(noise_db=25)]
                for v in variants:
                    s0 = int(round((t + v.get("shift", 0.0) - args.pre) * sr))
                    if s0 < 0 or s0 + n > len(audio):
                        continue
                    clip = audio[s0:s0 + n].copy()
                    if "noise_db" in v:
                        clip = clip + rng.normal(0, np.sqrt(np.mean(clip ** 2)) * 10 ** (-v["noise_db"] / 20),
                                                 len(clip)).astype(np.float32)
                    X.append(af.logmel_vector(clip, sr))
                    y.append(key)
                    n_orig += 0 if v else 1
        print(f"  {s}: done", flush=True)
    X, y = np.array(X), np.array(y)
    print(f"{n_orig} real presses, {len(y)} training vectors (incl. augmented copies), {X.shape[1]} features each")

    members = [(f"mlp{k}", make_pipeline(StandardScaler(), PCA(100, random_state=0),
                                         MLPClassifier((128,), alpha=0.1, max_iter=400, random_state=k)))
               for k in range(args.seeds)]
    model = VotingClassifier(members, voting="soft").fit(X, y)

    joblib.dump(dict(kind="mlp_logmel", model=model, classes=list(model.classes_), pre=args.pre, post=args.post,
                     highpass=0.0, sessions=list(args.sessions), excluded=["session_16"], n_train=n_orig,
                     n_vectors=len(y), augmentation=not args.no_aug, seeds=args.seeds), out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
