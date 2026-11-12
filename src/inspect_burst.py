
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

audio, sr = sf.read("dataset/raw/session_04/B_burst.wav")

time_axis = np.arange(len(audio)) / sr

plt.figure(figsize=(14, 5))
plt.plot(time_axis, audio)
plt.title("A key - 5 press burst")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
