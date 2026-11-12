
import numpy as np
import pandas as pd
import scipy
import matplotlib
import librosa
import sounddevice as sd
import soundfile as sf
import sklearn

print("ALL LIBRARIES IMPORTED SUCCESSFULLY!")

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Librosa:", librosa.__version__)
print("Scikit-learn:", sklearn.__version__)

print("\nAvailable audio devices:")
print(sd.query_devices())
