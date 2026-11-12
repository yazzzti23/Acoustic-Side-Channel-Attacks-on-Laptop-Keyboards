
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks

# Configuration
DATASET_DIR = Path("dataset/raw/session_03")
EXPECTED_PRESSES = 5

WINDOW_MS = 10
HOP_MS = 5
MIN_DISTANCE_SECONDS = 0.4
MIN_HEIGHT = 0.002
MIN_PROMINENCE = 0.001

results = []

for audio_path in sorted(DATASET_DIR.glob("*_burst.wav")):

    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    window_size = int(sr * WINDOW_MS / 1000)
    hop_size = int(sr * HOP_MS / 1000)

    rms = []

    for start in range(0, len(audio) - window_size + 1, hop_size):
        frame = audio[start:start + window_size]
        rms.append(np.sqrt(np.mean(frame ** 2)))

    rms = np.array(rms)

    min_distance = int(
        MIN_DISTANCE_SECONDS * sr / hop_size
    )

    peaks, _ = find_peaks(
        rms,
        height=MIN_HEIGHT,
        prominence=MIN_PROMINENCE,
        distance=min_distance
    )

    times = peaks * hop_size / sr
    detected = len(peaks)

    results.append({
        "key": audio_path.stem.split("_")[0],
        "detected": detected,
        "expected": EXPECTED_PRESSES,
        "status": "REVIEW" if detected != EXPECTED_PRESSES else "COUNT_OK",
        "timestamps": ", ".join(
            f"{t:.3f}" for t in times
        )
    })

df = pd.DataFrame(results)

print("\n=== BATCH SEGMENTATION VALIDATION ===\n")
print(df.to_string(index=False))

print("\nSummary:")
print(f"Recordings checked: {len(df)}")
print(f"Count matched: {(df['status'] == 'COUNT_OK').sum()}")
print(f"Need review: {(df['status'] == 'REVIEW').sum()}")

Path("results").mkdir(exist_ok=True)
df.to_csv("results/batch_validation.csv", index=False)

print("\nSaved: results/batch_validation.csv")
