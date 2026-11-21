"""
extract_features.py - cut one short clip around every logged key press and compute MFCC features.

Run from the project root (venv active):
    python src/extract_features.py
    python src/extract_features.py --sessions session_06 session_07 --highpass 300

What it does
  * For each requested session, reads every accepted <KEY>_burst.wav and its
    events/<KEY>_events.csv (key-down times measured from the first audio sample).
  * Cuts one clip per key-down: [down - pre, down + post] seconds.
    (Segmentation uses the LOGGED press times, i.e. ground truth. A real attacker would
    not have these; say so in the report.)
  * Per clip: peak-normalise, compute MFCCs, drop coefficient 0 (energy), then
    summarise each remaining coefficient by its mean and standard deviation over time.
  * Writes ONE csv to results/ with a row per press: session, condition, key, press index,
    time, source file, quality flag, raw clip peak/rms, then the feature columns.

What it never does
  * It never modifies anything in dataset/. Rejected takes (rejected/ folder) are not used.
  * It never overwrites an existing results file.
  * It never drops a press silently: presses whose clip would run outside the recording
    are listed and counted in the summary.

Defaults: clip = 0.02 s before to 0.08 s after the key-down (0.10 s total), no high-pass,
20 MFCCs (19 kept), n_fft 1024, hop 256, 40 mel bands.
"""

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

PROJECT_RAW = Path("dataset") / "raw"
MASTER_PATH = Path("dataset") / "metadata_master.csv"
RESULTS_DIR = Path("results")
KEYS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

N_MFCC = 20
N_FFT = 1024
HOP = 256
N_MELS = 40


def parse_args():
    p = argparse.ArgumentParser(description="MFCC features from logged key presses.")
    p.add_argument("--sessions", nargs="+", default=["session_06", "session_07"],
                   help="session folders under dataset/raw (default: session_06 session_07)")
    p.add_argument("--pre", type=float, default=0.02, help="seconds before key-down (default 0.02)")
    p.add_argument("--post", type=float, default=0.08, help="seconds after key-down (default 0.08)")
    p.add_argument("--highpass", type=float, default=0.0,
                   help="high-pass cutoff in Hz applied to the whole recording first (default 0 = off)")
    p.add_argument("--out", default=None,
                   help="output csv path (default results/features_<sessions>_pre..._post....csv)")
    return p.parse_args()


def highpass(audio, sr, cutoff):
    sos = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio)


def read_down_times(path, expected_key):
    times = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["event"] != "down":
                continue
            if row["key"].upper() != expected_key:
                raise SystemExit(f"{path}: event for key {row['key']} found in the {expected_key} file.")
            times.append(float(row["t_seconds_from_audio_start"]))
    return sorted(times)


def clip_features(clip, sr):
    peak = float(np.max(np.abs(clip)))
    rms = float(np.sqrt(np.mean(clip ** 2)))
    norm = clip / peak if peak > 0 else clip
    mfcc = librosa.feature.mfcc(y=norm.astype(np.float32), sr=sr, n_mfcc=N_MFCC,
                                n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    mfcc = mfcc[1:, :]  # drop coefficient 0 (energy)
    feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feats, peak, rms


def main():
    args = parse_args()

    if not MASTER_PATH.exists() or not RAW_OK():
        raise SystemExit("Run this from the project root (the folder that contains dataset/ and src/).")

    master = pd.read_csv(MASTER_PATH)
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.out:
        out_path = Path(args.out)
    else:
        tag = "_".join(s.replace("session_", "s") for s in args.sessions)
        hp = f"_hp{int(args.highpass)}" if args.highpass > 0 else ""
        out_path = RESULTS_DIR / f"features_{tag}_pre{int(args.pre * 1000)}_post{int(args.post * 1000)}{hp}.csv"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists. Not overwriting. Use --out for a different name.")

    feat_names = ([f"mfcc{i}_mean" for i in range(1, N_MFCC)] +
                  [f"mfcc{i}_std" for i in range(1, N_MFCC)])

    rows = []
    skipped = []
    print(f"Clip window: -{args.pre:.3f} s to +{args.post:.3f} s around each key-down; "
          f"high-pass: {'off' if args.highpass <= 0 else str(args.highpass) + ' Hz'}")

    for session in args.sessions:
        folder = PROJECT_RAW / session
        if not folder.is_dir():
            raise SystemExit(f"{folder} not found.")
        if not (folder / "events").is_dir():
            raise SystemExit(f"{folder} has no events/ folder (no key log), so it cannot be used here.")

        meta = master[(master["session"] == session) & (master["quality_flag"] != "REJECTED")]
        conds = meta["condition"].unique()
        if len(conds) != 1:
            raise SystemExit(f"{session}: expected one condition in metadata_master.csv, found {list(conds)}.")
        condition = conds[0]
        flags = dict(zip(meta["key"], meta["quality_flag"]))

        n_session = 0
        for key in KEYS:
            wav = folder / f"{key}_burst.wav"
            ev = folder / "events" / f"{key}_events.csv"
            if not wav.exists() or not ev.exists():
                print(f"WARNING: {session} key {key}: missing {'wav' if not wav.exists() else 'events'} file.")
                continue

            audio, sr = sf.read(wav, dtype="float32")
            if audio.ndim == 2:
                audio = audio[:, 0]
            if args.highpass > 0:
                audio = highpass(audio, sr, args.highpass)

            downs = read_down_times(ev, key)
            for idx, t in enumerate(downs, start=1):
                s0 = int(round((t - args.pre) * sr))
                s1 = int(round((t + args.post) * sr))
                if s0 < 0 or s1 > len(audio):
                    skipped.append((session, key, idx, round(t, 3)))
                    continue
                feats, peak, rms = clip_features(audio[s0:s1], sr)
                row = {
                    "session": session, "condition": condition, "key": key,
                    "press_index": idx, "t_down": round(t, 4),
                    "source_file": wav.as_posix(), "quality_flag": flags.get(key, ""),
                    "clip_peak": round(peak, 6), "clip_rms": round(rms, 6),
                }
                row.update({n: float(v) for n, v in zip(feat_names, feats)})
                rows.append(row)
                n_session += 1
        print(f"{session} ({condition}): {n_session} clips")

    if not rows:
        raise SystemExit("No clips produced. Nothing written.")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print("\nPer-session / per-key check (expected 5 clips per key if no press was skipped):")
    counts = df.groupby(["session", "key"]).size().unstack(fill_value=0)
    for session in counts.index:
        odd = {k: int(v) for k, v in counts.loc[session].items() if v != 5}
        print(f"  {session}: {int(counts.loc[session].sum())} clips, keys with a count other than 5: "
              f"{odd if odd else 'none'}")
    if skipped:
        print(f"\nSkipped (clip would run outside the recording): {len(skipped)}")
        for s in skipped:
            print("  ", s)
    else:
        print("\nSkipped presses: 0")
    print(f"\nFeature columns: {len(feat_names)}   Rows: {len(df)}")
    print(f"Saved: {out_path}")


def RAW_OK():
    return PROJECT_RAW.is_dir()


if __name__ == "__main__":
    main()
