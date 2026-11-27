"""
audio_features.py - log-mel features for the neural-network demo model.

Shared by demo/train_demo_model.py (training) and demo/live_guess.py (live guessing) so that both use
exactly the same recipe as the notebook's final test: peak-normalise the clip, 40-band log-mel
spectrogram (FFT 1024, hop 256), flattened into one vector.
"""

import librosa
import numpy as np

N_FFT, HOP, N_MELS = 1024, 256, 40


def n_samples(sr, pre, post):
    """Exact number of samples in a clip, so every clip gives a vector of the same length."""
    return int(round((pre + post) * sr))


def logmel_vector(clip, sr):
    clip = np.asarray(clip, dtype=np.float32)
    peak = float(np.max(np.abs(clip)))
    if peak > 0:
        clip = clip / peak
    m = librosa.feature.melspectrogram(y=clip, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return np.log(m + 1e-6).ravel()
