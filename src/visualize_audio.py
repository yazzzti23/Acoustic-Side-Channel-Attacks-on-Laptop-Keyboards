
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load our test recording
audio_path = "dataset/raw/microphone_test.wav"

audio, sr = librosa.load(
    audio_path,
    sr=None,
    mono=True
)

# Time axis
time = np.arange(len(audio)) / sr

# Create plots
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# 1. Waveform
axes[0].plot(time, audio)
axes[0].set_title("Keyboard Recording - Waveform")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

# 2. Mel Spectrogram
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128
)

mel_db = librosa.power_to_db(mel, ref=np.max)

img = librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis="time",
    y_axis="mel",
    ax=axes[1]
)

axes[1].set_title("Mel Spectrogram")
fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

plt.tight_layout()
plt.show()
