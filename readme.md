# 🎧 KeyLeak — Acoustic Side-Channel Attacks on Laptop Keyboards

![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-in%20development-orange)

**Can a microphone tell which key you pressed, and can you stop it?**

Every key on a keyboard sounds slightly different. This project builds a machine-learning pipeline that identifies keystrokes from audio alone, then tests simple defences to see how much of that leakage they remove.

---

## What it does

**Attack**
- Records keystroke audio from a laptop's built-in keyboard and microphone
- Detects individual key presses in continuous audio
- Extracts MFCC features and classifies the pressed key (A–Z) with SVM and Random Forest

**Defence**
- Tests physical dampening and controlled background noise
- Measures how much each reduces classification accuracy
- Runs an adaptive-attacker experiment: can an attacker retrain on defended audio and recover?

## Pipeline

```text
Audio recording → Event detection → Segmentation → MFCC features
      → SVM / Random Forest → Evaluation → Defence experiments
```

## Tech stack

Python 3.12 · NumPy · SciPy · Librosa · scikit-learn · Matplotlib · Seaborn · SoundDevice

## Quick start

```bash
git clone <your-repo-url>
cd AcousticKeyword

python -m venv acoustic
acoustic\Scripts\activate        # Windows

pip install numpy pandas scipy matplotlib seaborn librosa sounddevice soundfile scikit-learn jupyter
```

```bash
python src/test_microphone.py    # check your microphone
python src/recording.py          # record keystroke sessions
python src/segment_test.py       # run the event-detection prototype
```

## Project structure

```text
├── src/          # recording, segmentation and analysis scripts
├── dataset/      # recordings and extracted clips (not tracked)
├── notebooks/    # exploration and experiments
├── models/       # trained models
├── results/      # figures and metrics
└── report/       # write-up
```

## Roadmap

- [x] Recording setup and audio analysis tools
- [x] Keystroke event-detection prototype
- [ ] Reliable segmentation pipeline
- [ ] MFCC feature extraction
- [ ] SVM and Random Forest classifiers
- [ ] Defence experiments
- [ ] Adaptive-attacker evaluation
- [ ] Final report and results

## Ethics

Built for academic study of a known vulnerability class. All audio is recorded from the author's own keyboard. No third-party typing, passwords or private data is captured or used.

## Author

**Yashti Mittal** — B.Tech Computer Engineering, Thapar Institute of Engineering and Technology