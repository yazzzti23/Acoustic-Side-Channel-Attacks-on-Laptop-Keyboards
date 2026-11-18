"""
audit_wavs.py - READ-ONLY audit of every WAV file under dataset/raw.

Run from the project root:
    python src/audit_wavs.py

It does not create, modify, move or delete any file. It only prints.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RAW_DIR = Path("dataset") / "raw"
METADATA_PATH = Path("dataset") / "metadata.csv"

EXPECTED_SR = 44100
EXPECTED_CHANNELS = 1
LOW_PEAK_THRESHOLD = 0.01   # heuristic: below this the clip is probably silent / missed press
CLIP_THRESHOLD = 0.99       # heuristic: at or above this the clip may be clipped


def audit_file(path):
    """Return a dict of properties and flags for one WAV file."""
    info = {"path": path, "session": path.parent.name, "name": path.name, "flags": []}
    try:
        data, sr = sf.read(str(path), always_2d=True)
    except Exception as exc:  # unreadable / corrupt file
        info["flags"].append(f"UNREADABLE ({exc})")
        return info

    channels = data.shape[1]
    frames = data.shape[0]
    duration = frames / sr if sr else 0.0
    peak = float(np.max(np.abs(data))) if frames else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if frames else 0.0

    info.update(sr=sr, channels=channels, duration=duration, peak=peak, rms=rms)

    if sr != EXPECTED_SR:
        info["flags"].append(f"SR={sr}")
    if channels != EXPECTED_CHANNELS:
        info["flags"].append(f"CHANNELS={channels}")
    if peak < LOW_PEAK_THRESHOLD:
        info["flags"].append("LOW_PEAK")
    if peak >= CLIP_THRESHOLD:
        info["flags"].append("POSSIBLE_CLIPPING")
    if frames == 0:
        info["flags"].append("EMPTY")
    return info


def main():
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Executable  : {sys.executable}")
    print(f"NumPy       : {np.__version__}")
    print(f"SoundFile   : {sf.__version__}")
    print(f"Working dir : {Path.cwd()}")

    if not RAW_DIR.exists():
        print(f"\nERROR: {RAW_DIR} not found. Run this from the project root.")
        return

    wav_paths = sorted(RAW_DIR.rglob("*.wav"))
    print(f"\nFound {len(wav_paths)} WAV files under {RAW_DIR}")

    results = [audit_file(p) for p in wav_paths]

    # ---- per-session summary -------------------------------------------
    print("\n" + "=" * 70)
    print("PER-SESSION SUMMARY")
    print("=" * 70)
    sessions = sorted({r["session"] for r in results})
    for s in sessions:
        rs = [r for r in results if r["session"] == s and "duration" in r]
        if not rs:
            print(f"{s}: no readable files")
            continue
        flagged = sum(1 for r in rs if r["flags"])
        durs = [r["duration"] for r in rs]
        peaks = [r["peak"] for r in rs]
        print(
            f"{s:<14} files={len(rs):<3} "
            f"duration={min(durs):.2f}-{max(durs):.2f}s  "
            f"peak={min(peaks):.4f}-{max(peaks):.4f}  "
            f"flagged={flagged}"
        )

    # ---- flagged files --------------------------------------------------
    flagged_results = [r for r in results if r["flags"]]
    print("\n" + "=" * 70)
    print(f"FLAGGED FILES ({len(flagged_results)})")
    print("=" * 70)
    if not flagged_results:
        print("None.")
    for r in flagged_results:
        if "peak" in r:
            print(
                f"{r['session']}/{r['name']:<18} "
                f"peak={r['peak']:.4f} rms={r['rms']:.5f} "
                f"dur={r['duration']:.2f}s  -> {', '.join(r['flags'])}"
            )
        else:
            print(f"{r['session']}/{r['name']:<18} -> {', '.join(r['flags'])}")

    # ---- anything in raw/ that is not a WAV inside a session folder -----
    print("\n" + "=" * 70)
    print("OTHER FILES (non-WAV, or WAV not inside a session folder)")
    print("=" * 70)
    others = []
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            if p.suffix.lower() != ".wav" or p.parent == RAW_DIR:
                others.append(p)
    if not others:
        print("None.")
    for p in others:
        print(p)

    # ---- metadata.csv header --------------------------------------------
    print("\n" + "=" * 70)
    print("METADATA FILE")
    print("=" * 70)
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"{METADATA_PATH}: {max(len(lines) - 1, 0)} data rows")
        print(f"Header: {lines[0].strip() if lines else '(empty file)'}")
    else:
        print(f"{METADATA_PATH} does not exist.")

    print("\nAudit complete. No files were modified.")


if __name__ == "__main__":
    main()