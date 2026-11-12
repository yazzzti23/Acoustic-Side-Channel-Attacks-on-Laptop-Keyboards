
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ==========================================
# 1. LOAD AUDIO
# ==========================================

audio_path = "dataset/raw/session_04/B_burst.wav"

audio, sr = sf.read(audio_path)

print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio) / sr:.2f} seconds")


# ==========================================
# 2. CALCULATE SHORT-TIME RMS ENERGY
# ==========================================

window_size = 441   # 10 ms
hop_size = 220      # 5 ms

rms_values = []

for start in range(0, len(audio) - window_size, hop_size):
    window = audio[start:start + window_size]

    rms = np.sqrt(np.mean(window ** 2))
    rms_values.append(rms)

rms_values = np.array(rms_values)

times = np.arange(len(rms_values)) * hop_size / sr


# ==========================================
# 3. DETECT CANDIDATE KEYPRESSES
# ==========================================

minimum_distance = int(0.4 * sr / hop_size)

peaks, properties = find_peaks(
    rms_values,
    height=0.002,
    distance=minimum_distance,
    prominence=0.001
)

print("\nDetected keypress candidates:")
print("--------------------------------")

for i, peak in enumerate(peaks, start=1):
    print(
        f"Candidate {i}: "
        f"Time = {times[peak]:.3f}s, "
        f"RMS = {rms_values[peak]:.5f}"
    )

print("--------------------------------")
print(f"Total candidates detected: {len(peaks)}")


# ==========================================
# 4. PLOT RMS ENERGY AND DETECTED PEAKS
# ==========================================

plt.figure(figsize=(14, 5))

plt.plot(
    times,
    rms_values,
    label="RMS Energy"
)

plt.plot(
    times[peaks],
    rms_values[peaks],
    "rx",
    markersize=10,
    label="Detected candidates"
)

plt.title("Keypress Detection - A Key")
plt.xlabel("Time (seconds)")
plt.ylabel("RMS Energy")

plt.legend()
plt.grid()
plt.tight_layout()

plt.show()




