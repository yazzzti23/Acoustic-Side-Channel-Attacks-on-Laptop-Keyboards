"""
audit_metadata.py - READ-ONLY check of dataset/metadata.csv against the WAV
files on disk under dataset/raw.

Run from the project root:
    python src/audit_metadata.py

It does not create, modify, move or delete any file. It only prints.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("dataset") / "raw"
METADATA_PATH = Path("dataset") / "metadata.csv"
MAX_LIST = 25  # cap on how many problem rows to print per section


def normalise(fname):
    """Turn a metadata filename into a path relative to dataset/raw, using '/'."""
    s = str(fname).strip().replace("\\", "/")
    for prefix in ("dataset/raw/", "raw/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def resolve(fname, session, disk):
    """Find which on-disk file a metadata row refers to, or None."""
    s = normalise(fname)
    base = s.split("/")[-1]
    candidates = [s]
    if session is not None and str(session).strip() not in ("", "nan"):
        candidates.append(f"{str(session).strip()}/{base}")
    candidates.append(base)
    for c in candidates:
        if c in disk:
            return c
    return None


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found. Run this from the project root.")
        return
    if not METADATA_PATH.exists():
        print(f"ERROR: {METADATA_PATH} not found.")
        return

    disk = {p.relative_to(RAW_DIR).as_posix() for p in RAW_DIR.rglob("*.wav")}
    df = pd.read_csv(METADATA_PATH)

    print("=" * 70)
    print("METADATA OVERVIEW")
    print("=" * 70)
    print(f"Rows            : {len(df)}")
    print(f"Columns ({len(df.columns)})   : {', '.join(df.columns)}")
    print(f"WAV files on disk (under dataset/raw): {len(disk)}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("ROWS PER SESSION / CONDITION")
    print("=" * 70)
    group_cols = [c for c in ("session", "condition") if c in df.columns]
    if group_cols:
        counts = df.groupby(group_cols, dropna=False).size().reset_index(name="rows")
        print(counts.to_string(index=False))
    else:
        print("No 'session' or 'condition' column found.")

    if "key" in df.columns:
        print(f"\nDistinct key values: {df['key'].nunique(dropna=False)}")

    # ---- match each metadata row to a file on disk -----------------------
    resolved = []
    for _, row in df.iterrows():
        session = row["session"] if "session" in df.columns else None
        resolved.append(resolve(row["filename"], session, disk))
    df["_resolved"] = resolved

    missing = df[df["_resolved"].isna()]
    print("\n" + "=" * 70)
    print(f"METADATA ROWS WITH NO MATCHING FILE ON DISK ({len(missing)})")
    print("=" * 70)
    if missing.empty:
        print("None.")
    for _, row in missing.head(MAX_LIST).iterrows():
        sess = row["session"] if "session" in df.columns else "?"
        print(f"{sess} | {row['filename']}")
    if len(missing) > MAX_LIST:
        print(f"... and {len(missing) - MAX_LIST} more")

    matched = set(df["_resolved"].dropna())
    unlisted = sorted(disk - matched)
    print("\n" + "=" * 70)
    print(f"WAV FILES ON DISK WITH NO METADATA ROW ({len(unlisted)})")
    print("=" * 70)
    if not unlisted:
        print("None.")
    for p in unlisted[:MAX_LIST]:
        print(p)
    if len(unlisted) > MAX_LIST:
        print(f"... and {len(unlisted) - MAX_LIST} more")

    dup = df[df["_resolved"].notna() & df["_resolved"].duplicated(keep=False)]
    print("\n" + "=" * 70)
    print(f"DUPLICATE ROWS POINTING AT THE SAME FILE ({len(dup)})")
    print("=" * 70)
    if dup.empty:
        print("None.")
    for _, row in dup.head(MAX_LIST).iterrows():
        print(row["_resolved"])

    print("\nAudit complete. No files were modified.")


if __name__ == "__main__":
    main()
    