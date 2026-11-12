
import sounddevice as sd
import soundfile as sf
import numpy as np
import csv
import time
from pathlib import Path
import string

# Recording configuration
SAMPLE_RATE = 44100
DURATION = 5
DEVICE_ID = 1
PRESSES_PER_RECORDING = 5

#KEYS = list(string.ascii_uppercase)
KEYS = ["B"]

# Session details
session_id = input("Enter session ID: ").strip()
condition = input("Enter condition (clean/noisy_pilot/dampened): ").strip()

session_folder = Path("dataset/raw") / session_id
session_folder.mkdir(parents=True, exist_ok=True)

metadata_path = Path("dataset/metadata.csv")
file_exists = metadata_path.exists()

with open(metadata_path, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)

    if not file_exists:
        writer.writerow([
            "filename", "key", "session",
            "condition", "sample_rate",
            "duration", "peak_amplitude"
        ])

    for key in KEYS:
        print(f"\nPrepare to press: {key}")

        input(f"Press ENTER when ready to record {key}...")

        print("Recording starts in 3 seconds...")
        time.sleep(3)

        print(f"Press {key} {PRESSES_PER_RECORDING} times, "
              "with roughly 1 second between presses!")

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=DEVICE_ID
        )

        sd.wait()

        peak = float(np.max(np.abs(audio)))

        filename = f"{key}_burst.wav"
        filepath = session_folder / filename

        sf.write(filepath, audio, SAMPLE_RATE)

        writer.writerow([
            str(filepath).replace("\\", "/"),
            key,
            session_id,
            condition,
            SAMPLE_RATE,
            DURATION,
            peak
        ])

        csvfile.flush()

        print(f"Saved: {filepath}")

print("\nDATA COLLECTION COMPLETED!")
