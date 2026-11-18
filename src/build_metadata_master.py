"""
build_metadata_master.py - build a clean master metadata file from the WAV
files that actually exist on disk.

Run from the project root:
    python src/build_metadata_master.py

READS : dataset/raw/session_*/*.wav and dataset/metadata.csv
WRITES: dataset/metadata_master.csv  (one row per real keystroke recording)
        dataset/metadata_notes.txt   (stale rows, mismatches, anything odd)

It never modifies or deletes metadata.csv or any WAV file, and it refuses to
run if either output file already exists.
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

RAW_DIR = Path("dataset") / "raw"
OLD_METADATA = Path("dataset") / "metadata.csv"
MASTER_PATH = Path("dataset") / "metadata_master.csv"
NOTES_PATH = Path("dataset") / "metadata_notes.txt"

LOW_PEAK_THRESHOLD = 0.01   # heuristic, same as the audit script
CLIP_THRESHOLD = 0.99
PEAK_MATCH_TOL = 1e-3       # old-metadata peak vs measured peak
INTENDED_BURST_PRESSES = 5

SINGLE_RE = re.compile(r"^([A-Za-z])_(\d+)\.wav$")
BURST_RE = re.compile(r"^([A-Za-z])_burst\.wav$")

COLUMNS = [
    "filename", "key", "session", "condition", "recording_type",
    "repetition", "sample_rate", "duration_s", "peak_amplitude",
    "rms_amplitude", "quality_flag", "intended_presses", "actual_presses",
    "notes",
]


def normalise(fname):
    """Metadata filename -> path relative to dataset/raw, using '/'."""
    s = str(fname).strip().replace("\\", "/")
    for prefix in ("dataset/raw/", "raw/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def measure(path):
    """Return (sample_rate, duration_s, peak, rms) for a WAV file."""
    data, sr = sf.read(str(path), always_2d=True)
    frames = data.shape[0]
    duration = frames / sr if sr else 0.0
    peak = float(np.max(np.abs(data))) if frames else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if frames else 0.0
    return sr, duration, peak, rms


def quality_flag(peak):
    if peak < LOW_PEAK_THRESHOLD:
        return "LOW_PEAK"
    if peak >= CLIP_THRESHOLD:
        return "POSSIBLE_CLIPPING"
    return "OK"


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found. Run this from the project root.")
        return
    if not OLD_METADATA.exists():
        print(f"ERROR: {OLD_METADATA} not found.")
        return
    for out in (MASTER_PATH, NOTES_PATH):
        if out.exists():
            print(f"STOP: {out} already exists. Not overwriting it.")
            print("Rename or move it yourself if you want to regenerate.")
            return

    old = pd.read_csv(OLD_METADATA)
    old["_norm"] = old["filename"].map(normalise)

    # Only WAVs inside a session_* folder count as keystroke recordings.
    wavs = sorted(
        p for p in RAW_DIR.glob("session_*/*.wav") if p.is_file()
    )

    notes = []
    rows = []
    matched_old_index = set()

    for path in wavs:
        rel = path.relative_to(RAW_DIR).as_posix()      # e.g. session_01/A_001.wav
        session = path.parent.name
        name = path.name

        m_single = SINGLE_RE.match(name)
        m_burst = BURST_RE.match(name)
        if m_single:
            key = m_single.group(1).upper()
            rec_type = "single"
            repetition = int(m_single.group(2))
            intended = 1
        elif m_burst:
            key = m_burst.group(1).upper()
            rec_type = "burst"
            repetition = ""
            intended = INTENDED_BURST_PRESSES
        else:
            key, rec_type, repetition, intended = "", "unknown", "", ""
            notes.append(f"UNRECOGNISED FILENAME PATTERN: {rel}")

        sr, duration, peak, rms = measure(path)

        # Find the old metadata row(s) for this file.
        candidates = old[old["_norm"] == rel]
        condition = "unknown"
        row_note = ""
        if len(candidates) == 0:
            notes.append(f"NO OLD METADATA ROW for {rel} (condition set to 'unknown')")
        else:
            # choose the old row whose peak is closest to the measured peak
            diffs = (candidates["peak_amplitude"] - peak).abs()
            best_idx = diffs.idxmin()
            best = candidates.loc[best_idx]
            matched_old_index.add(best_idx)
            condition = str(best["condition"])
            if diffs.min() > PEAK_MATCH_TOL:
                row_note = "old-metadata peak does not match file"
                notes.append(
                    f"PEAK MISMATCH {rel}: old={best['peak_amplitude']:.6f} "
                    f"measured={peak:.6f}"
                )
            # any other old rows for the same file are stale
            for idx in candidates.index:
                if idx != best_idx:
                    matched_old_index.add(idx)
                    stale = candidates.loc[idx]
                    notes.append(
                        f"STALE ROW (file was overwritten) {rel}: "
                        f"old row peak={stale['peak_amplitude']:.6f}, "
                        f"file on disk peak={peak:.6f}"
                    )
                    row_note = (row_note + "; " if row_note else "") + \
                        "file re-recorded under same name; earlier take lost"

        rows.append({
            "filename": f"dataset/raw/{rel}",
            "key": key,
            "session": session,
            "condition": condition,
            "recording_type": rec_type,
            "repetition": repetition,
            "sample_rate": sr,
            "duration_s": round(duration, 3),
            "peak_amplitude": round(peak, 6),
            "rms_amplitude": round(rms, 6),
            "quality_flag": quality_flag(peak),
            "intended_presses": intended,
            "actual_presses": "",   # fill in after listening / validation
            "notes": row_note,
        })

    # Old rows that never matched any file on disk
    for idx, r in old.iterrows():
        if idx not in matched_old_index:
            notes.append(f"OLD ROW WITH NO FILE ON DISK: {r['filename']}")

    # Files in raw/ but outside session folders (e.g. microphone test)
    for p in sorted(RAW_DIR.glob("*.wav")):
        notes.append(
            f"EXCLUDED (not a keystroke recording): {p.relative_to(RAW_DIR).as_posix()}"
        )

    # ---- write outputs ---------------------------------------------------
    with open(MASTER_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.write("metadata_notes.txt - generated by build_metadata_master.py\n")
        f.write("The original dataset/metadata.csv was NOT modified.\n\n")
        if notes:
            f.write("\n".join(notes) + "\n")
        else:
            f.write("No issues found.\n")

    # ---- summary ---------------------------------------------------------
    df = pd.DataFrame(rows)
    print(f"Wrote {MASTER_PATH} with {len(df)} rows")
    print(f"Wrote {NOTES_PATH} with {len(notes)} note(s)\n")
    print("Rows per session / condition / type:")
    summary = (
        df.groupby(["session", "condition", "recording_type"])
        .size().reset_index(name="rows")
    )
    print(summary.to_string(index=False))
    print("\nQuality flags:")
    print(df["quality_flag"].value_counts().to_string())
    print("\nNotes:")
    for n in notes:
        print(f"  - {n}")
    print("\nDone. metadata.csv and all WAV files were left untouched.")


if __name__ == "__main__":
    main()
    