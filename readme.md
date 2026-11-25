# 🎧 KeyLeak — Acoustic Side-Channel Attacks on Laptop Keyboards

![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-in%20development-orange)

**Can a microphone tell which key you pressed, and can you stop it?**

Every key on a keyboard sounds slightly different. This project builds a machine-learning pipeline that identifies keystrokes from audio alone, then tests simple defences to see how much of that leakage they remove.

---

## What it does

**Attack**
- Records keystroke audio from a laptop's built-in keyboard and microphone
- Cuts a short clip around each key press (timing taken from a key-press log during data collection)
- Extracts MFCC features and classifies the pressed key (A–Z) with SVM and Random Forest

**Defence**
- Tests physical dampening and controlled background noise
- Measures how much each reduces classification accuracy
- Runs an adaptive-attacker experiment: can an attacker retrain on defended audio and recover?

## Pipeline

```text
Recording + key log → Clip around each press → MFCC features
      → SVM / Random Forest → Session-separated evaluation → Defence experiments
```

## Evaluation approach

- The attacker is always trained on some recording sessions and tested on **different** sessions, never on the same recording.
- Every session is labelled with its condition (`clean`, `dampened`, `noisy`).
- Rejected or unreliable takes are kept and labelled, not silently dropped.

## Preliminary results (work in progress)

Attacker trained on two clean sessions, tested on sessions it never heard (26 keys, 130 presses per test session, so roughly ±7 points of uncertainty):

| Test setup | SVM top-1 | SVM top-5 |
|---|---|---|
| Clean (baseline) | 21.5% | 57.7% |
| Cloth under the laptop stand | 17.8% | 50.4% |
| Cloth over the keys | 7.7% | 30.0% |
| White background noise | 3.8% | 18.5% |

Random guessing is 3.8% top-1 and 19.2% top-5. These numbers are from a first pass and will change as the adaptive-attacker and ablation experiments are added.

## Tech stack

Python 3.12 · NumPy · SciPy · Librosa · scikit-learn · Pandas · SoundDevice · pynput

## Quick start

```bash
git clone <your-repo-url>
cd AcousticKeyword

python -m venv acoustic
acoustic\Scripts\activate        # Windows

pip install numpy pandas scipy matplotlib seaborn librosa sounddevice soundfile scikit-learn pynput jupyter
```

```bash
python src/test_microphone.py     # check your microphone
python src/recording.py           # record a keystroke session (with key-press log)
python src/extract_features.py    # clips + MFCC features -> results/
python src/train_eval.py          # train on one session, test on another
python src/eval_defences.py       # clean-trained attacker vs defended sessions
```

The raw recordings are not included in the repository.

## Project structure

```text
├── src/          # recording, feature extraction and evaluation scripts
├── dataset/      # recording metadata (raw audio not tracked)
├── notebooks/    # exploration and experiments
├── models/       # trained models
├── results/      # feature files, figures and metrics
└── report/       # write-up
```

## Roadmap

- [x] Recording setup with key-press logging and audio alignment
- [x] Clean, dampened and noisy recording sessions
- [x] MFCC feature extraction
- [x] SVM and Random Forest classifiers (session-separated)
- [x] First defence comparison
- [ ] Adaptive-attacker evaluation
- [ ] High-pass filter and clip-window ablations
- [ ] Live demo (top-3 guesses with confidence)
- [ ] Final report and results

## Limitations

One laptop, one typist and a small number of presses per key. Segmentation currently uses logged key-press times, which a real attacker would not have. Results show the shape of the effect, not exact figures for other keyboards or people.

## Ethics

Built for academic study of a known vulnerability class. All audio is recorded from the author's own keyboard. No third-party typing, passwords or private data is captured or used.

## Author

**Yashti Mittal** — B.Tech Computer Engineering, Thapar Institute of Engineering and Technology
