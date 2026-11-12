
import sounddevice as sd
import soundfile as sf
import numpy as np
import os

# Configuration
SAMPLE_RATE = 44100
DURATION = 10
DEVICE_ID = 1

# Output location
OUTPUT_PATH = "dataset/raw/microphone_test.wav"

print("Keyboard Audio Recording Test")
print("--------------------------------")

print("Selected microphone:")
print(sd.query_devices(DEVICE_ID))

input("\nPress ENTER when you are ready...")

print("\nRecording starts in 3 seconds...")
sd.sleep(3000)

print("RECORDING STARTED!")
print("Press the A key approximately 8-10 times.")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=DEVICE_ID
)

sd.wait()

# Save audio
os.makedirs("dataset/raw", exist_ok=True)

sf.write(
    OUTPUT_PATH,
    audio,
    SAMPLE_RATE
)

print("\nRecording completed!")
print("Saved at:", OUTPUT_PATH)
print("Audio duration:", len(audio) / SAMPLE_RATE, "seconds")
print("Peak amplitude:", np.max(np.abs(audio)))
