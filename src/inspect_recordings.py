import soundfile as sf
import sounddevice as sd
import numpy as np

audio, sr = sf.read("dataset/raw/session_01/M_002.wav")
# Normalize for listening only
audio = audio / np.max(np.abs(audio)) * 0.8

print("Playing amplified M_001...")
sd.play(audio, sr)
sd.wait()