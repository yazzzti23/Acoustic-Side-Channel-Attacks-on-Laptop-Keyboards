# Acoustic Side-Channel Attacks on Laptop Keyboards

**An ML-based attack and defence study** — can machine learning identify which key was pressed from the sound of typing, and can simple countermeasures stop it?

Undergraduate Machine Learning course project, B.Tech Computer Engineering, Thapar Institute of Engineering and Technology.

> **Status: work in progress.** The audio-processing pipeline is under development. No classification results have been produced yet. See [Current status](#current-status).

---

## Ethics and scope

This is an academic study of an acoustic information-leakage vulnerability and its defences.

- All recordings are made on **my own laptop keyboard** with **my own microphone**.
- No one else's typing, passwords or private information is recorded or used.
- The project does **not** include sentence reconstruction, real-time keystroke prediction, or any attack on third parties.

---

## Overview

Different keys on a keyboard sound slightly different because of their position, keycap geometry and the mechanics beneath them. This project investigates whether those differences are enough for a classifier to recover the pressed key.

The study has two parts:

1. **Attack:** record keystroke audio, detect individual key presses, extract MFCC features, and train classical ML models to identify one of 26 keys (A–Z).
2. **Defence:** apply countermeasures (physical dampening, controlled background noise) and measure how much they reduce the attack's accuracy, including whether an attacker who retrains on defended recordings can recover performance (adaptive attacker).

## Scope

| Component | Decision |
|---|---|
| Task | Single-key identification |
| Classes | 26 English alphabet keys (A–Z) |
| Keyboard | Laptop built-in keyboard |
| Microphone | Laptop built-in microphone array |
| Audio | Mono WAV, 44,100 Hz |
| Features | MFCC |
| Models | SVM, Random Forest (CNN only if time permits) |
| Defences | Physical dampening, controlled background noise |
| Extension | Adaptive attacker |
| Language | Python 3.12 |

## Pipeline

```text
Recording -> Raw WAV -> Quality inspection -> Keypress event detection
   -> Segmentation -> MFCC features -> Session-based train/test split
   -> SVM / Random Forest -> Evaluation -> Defence experiments -> Analysis
```

## Current status

| Stage | Status |
|---|---|
| Environment and microphone test | Done |
| Single-press pilot recordings | Done (noisy pilot) |
| Burst recording (5 presses per file) | Done (noisy pilot, A–Z) |
| Waveform and short-time RMS analysis | Done |
| Peak-based event detection | Prototype, **not yet validated** |
| Clean-room baseline recordings | Pending |
| Reusable segmentation pipeline | Pending |
| MFCC extraction | Pending |
| SVM / Random Forest training | Pending |
| Defence recordings and evaluation | Pending |

### Findings so far

Confirmed or directly observed:

- The laptop microphone captures keystroke transients.
- Burst recording (several presses in one file) works and is faster than one file per press.
- A single physical press can produce more than one acoustic event (press and release sounds were heard on playback).
- On one A-key burst, an RMS + peak detector found 8 candidate events, and adding a 0.4 s minimum peak separation reduced this to 5.

Not yet established:

- That those 5 candidates correspond to the real key-down events. Matching the expected count does not prove correct segmentation.
- That the method generalises across keys, pressing force, typing speed or noise levels.
- That the extracted clips contain enough information to classify keys.

The next planned experiment is to log key-down and key-up timestamps during recording to get ground truth for validating the detector.

### Known limitations

- Early sessions were recorded with environmental noise (festival fireworks). They are kept as **noisy pilot data** and are not a clean baseline.
- The fixed 0.4 s separation and fixed RMS threshold are heuristics that may drop real presses or fail in different noise conditions.
- Built-in microphone processing (noise suppression, automatic gain) may affect transients and must stay consistent across sessions.
- Clips from the same raw recording must never be split across train and test sets; evaluation will be separated by recording session.
- Public keystroke datasets use different keyboards, microphones and conditions. They will be evaluated separately (cross-dataset), not merged into training.

## Repository structure

```text
AcousticKeyword/
├── dataset/
│   ├── raw/              # raw WAV recordings, organised by session
│   ├── processed/        # extracted keystroke clips
│   └── metadata.csv
├── src/
│   ├── test_microphone.py   # checks that the microphone records
│   ├── visualize_audio.py   # waveform / spectrogram plots
│   ├── recording.py         # burst recording script
│   ├── inspect_burst.py     # waveform inspection of a burst file
│   └── segment_test.py      # RMS energy + peak detection prototype
├── notebooks/
├── models/
├── results/
└── report/
```

## Setup

Requires Python 3.12.

```bash
# create and activate a virtual environment (Windows)
python -m venv acoustic
acoustic\Scripts\activate

# install dependencies
pip install numpy pandas scipy matplotlib seaborn librosa sounddevice soundfile scikit-learn jupyter
```

## Usage

Run scripts from the project root with the environment activated.

```bash
python src/test_microphone.py     # confirm the microphone works
python src/recording.py           # record burst sessions
python src/inspect_burst.py       # plot a recorded burst
python src/segment_test.py        # RMS energy + peak detection test
```

## Roadmap

- [x] Environment and microphone testing
- [x] Noisy pilot recordings (single-press and burst)
- [x] RMS energy and initial peak detection
- [ ] Validate segmentation against logged key timestamps
- [ ] Record clean baseline sessions in a quiet environment
- [ ] Reusable segmentation script with metadata and rejected-event logging
- [ ] MFCC feature extraction
- [ ] SVM and Random Forest baselines with session-separated evaluation
- [ ] Defence experiments (dampening, background noise)
- [ ] Adaptive attacker experiment
- [ ] Final report

## Author

Yashti Mittal — B.Tech Computer Engineering, Thapar Institute of Engineering and Technology.

## License

Add a license before making the repository public (for example MIT).