"""
live_guess.py - real-time guessing session for the acoustic keystroke attack.

Run from the project root (venv active):
    python demo/live_guess.py                          # live, microphone, 10 random letters
    python demo/live_guess.py --rounds 15
    python demo/live_guess.py --replay session_17      # no microphone: replay a recorded session

How a round works
  1. The script picks a random letter and shows it to YOU. The model never sees it.
  2. You press that key 3 times, about 1 second apart, while it records 5 seconds.
  3. The script finds the 3 loudest clicks (no key log is used), cuts the same short clip around each
     click as in the experiments, and asks the model about each one.
  4. The 3 sets of probabilities are multiplied together (they are the same key), and the top 3
     guesses are shown with a confidence for each.
  5. It checks the guesses against the real letter and keeps a running score.

Everything is saved to demo/logs/<time>/ (the 5-second recordings in live mode, and results.csv).
Replay mode refuses to use a session the model was trained on.

Honest limits: one key at a time, a quiet room like the clean recordings is needed, and the numbers
from a handful of rounds wobble by 10+ points. The confidence is the model's own estimate; with 26
similar-sounding keys it is usually low, which is accurate, not a bug.
"""

import argparse
import csv
import random
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import extract_features as ef  # noqa: E402  (reuses the exact feature code used in the experiments)

SAMPLE_RATE = 44100
DURATION = 5
N_PRESSES = 3
CLICK_TO_KEYDOWN = 0.021      # measured: the click peak is ~21 ms after the logged key-down
MIN_SEPARATION = 0.4          # seconds between presses
MAD_K = 8.0
ENV_HOP = 44                  # 1 ms envelope step
COUNTDOWN = 3
DEFAULT_DEVICE = 1


def parse_args():
    p = argparse.ArgumentParser(description="Live keystroke guessing session.")
    p.add_argument("--model", default="demo/model/keystroke_model.joblib")
    p.add_argument("--rounds", type=int, default=None, help="letters to try (default 10 live, 26 replay)")
    p.add_argument("--replay", default=None, help="session folder under dataset/raw to replay instead of the mic")
    p.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-dir", default="demo/logs")
    return p.parse_args()


# ---------------------------------------------------------------- audio capture
def record_round(device, announce):
    import sounddevice as sd
    total = int(DURATION * SAMPLE_RATE)
    blocks, count = [], [0]
    started, done = threading.Event(), threading.Event()

    def cb(indata, frames, t, status):
        started.set()
        take = min(frames, total - count[0])
        if take > 0:
            blocks.append(indata[:take, 0].copy())
            count[0] += take
        if count[0] >= total:
            done.set()
            raise sd.CallbackStop

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device, callback=cb)
    try:
        stream.start()
        if not started.wait(5.0):
            raise SystemExit("The microphone delivered no audio.")
        announce()
        done.wait(DURATION + 5.0)
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


def flush_input():
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    except ImportError:
        try:
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass


# ---------------------------------------------------------------- click finding
def find_clicks(audio, sr, n=N_PRESSES):
    """Return the times (s) of the n loudest clicks, sorted, or fewer if not found."""
    sos = butter(4, 300, btype="highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, audio)
    m = (len(y) // ENV_HOP) * ENV_HOP
    env = np.sqrt((y[:m].reshape(-1, ENV_HOP) ** 2).mean(axis=1))
    med = np.median(env)
    mad = np.median(np.abs(env - med)) * 1.4826
    thr = med + MAD_K * mad
    idx, props = find_peaks(env, height=thr, distance=max(1, int(MIN_SEPARATION * sr / ENV_HOP)))
    if len(idx) == 0:
        return []
    top = idx[np.argsort(props["peak_heights"])[::-1][:n]]
    return sorted(float((i * ENV_HOP + ENV_HOP / 2) / sr) for i in top)


def clip_at(audio, sr, click_t, pre, post):
    down = click_t - CLICK_TO_KEYDOWN
    s0, s1 = int(round((down - pre) * sr)), int(round((down + post) * sr))
    if s0 < 0 or s1 > len(audio):
        return None
    return audio[s0:s1]


def guess(payload, audio, sr):
    """Return dict with click times, per-press top-1, and combined probabilities."""
    clicks = find_clicks(audio, sr)
    if len(clicks) < N_PRESSES:
        return {"clicks": clicks, "ok": False}
    if payload["highpass"] > 0:
        sos = butter(4, payload["highpass"], btype="highpass", fs=sr, output="sos")
        audio = sosfiltfilt(sos, audio)
    feats = []
    for t in clicks:
        c = clip_at(audio, sr, t, payload["pre"], payload["post"])
        if c is None:
            return {"clicks": clicks, "ok": False}
        feats.append(ef.clip_features(c, sr)[0])
    model, classes = payload["model"], np.array(payload["classes"])
    P = model.predict_proba(np.array(feats))
    logp = np.log(np.clip(P, 1e-6, 1.0)).sum(axis=0)
    comb = np.exp(logp - logp.max())
    comb /= comb.sum()
    order = np.argsort(-comb)
    return {"clicks": clicks, "ok": True, "per_press_top1": [str(classes[int(np.argmax(p))]) for p in P],
            "top3": [(str(classes[i]), float(comb[i])) for i in order[:3]]}


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    mp = Path(args.model)
    if not mp.exists():
        raise SystemExit(f"{mp} not found. Run demo/train_model.py first.")
    payload = joblib.load(mp)
    trained_on = set(payload["sessions"])

    if args.replay:
        if args.replay in trained_on:
            raise SystemExit(f"{args.replay} was used to train this model. Replay a session it has not seen.")
        if not (Path("dataset/raw") / args.replay).is_dir():
            raise SystemExit(f"dataset/raw/{args.replay} not found.")

    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    keys = list(payload["classes"])
    rng.shuffle(keys)
    rounds = args.rounds or (26 if args.replay else 10)
    keys = keys[:rounds]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / (("replay_" if args.replay else "live_") + stamp)
    log_dir.mkdir(parents=True, exist_ok=False)
    rows = []

    print("=" * 56)
    print("KEYSTROKE GUESSING SESSION" + (f"  (replay of {args.replay})" if args.replay else "  (live microphone)"))
    print(f"Model trained on {len(trained_on)} clean sessions, {payload['n_train']} presses. Chance: 1 in 26.")
    print(f"Each round: {N_PRESSES} presses of the same key. {len(keys)} rounds. Logs: {log_dir}")
    print("=" * 56)

    for i, target in enumerate(keys, start=1):
        result, attempt = None, 0
        while attempt < 3:
            attempt += 1
            print(f"\n--- Round {i}/{len(keys)} ---")
            if args.replay:
                wav = Path("dataset/raw") / args.replay / f"{target}_burst.wav"
                if not wav.exists():
                    print(f"{wav} missing, skipping.")
                    break
                audio, sr = sf.read(wav, dtype="float32")
                audio = audio[:, 0] if audio.ndim == 2 else audio
                print(f"(replaying a recorded press of key {target})")
            else:
                flush_input()
                input(f"Next key: >>> {target} <<<   Press ENTER, then press {target} {N_PRESSES} times...")
                for c in range(COUNTDOWN, 0, -1):
                    print(f"  {c}...", flush=True)
                    time.sleep(1)
                print("Starting the microphone...", flush=True)
                audio = record_round(args.device, lambda t=target: print(
                    f">>> PRESS {t} {N_PRESSES} TIMES, about 1 second apart <<<", flush=True))
                sr = SAMPLE_RATE
                flush_input()
                sf.write(log_dir / f"round{i:02d}_{target}_try{attempt}.wav", audio, sr, subtype="PCM_16")
            result = guess(payload, audio, sr)
            if result["ok"]:
                break
            print(f"Found only {len(result['clicks'])} clear click(s) instead of {N_PRESSES}."
                  + (" Try again." if (attempt < 3 and not args.replay) else ""))
            if args.replay:
                break
        if result is None or not result["ok"]:
            rows.append({"round": i, "target": target, "detected": "no", "top1": "", "top1_conf": "",
                         "top2": "", "top3": "", "hit_top1": "", "hit_top3": "", "per_press_top1": ""})
            print("No usable detection this round (counted separately, not hidden).")
            continue

        t3 = result["top3"]
        hit1 = t3[0][0] == target
        hit3 = target in [k for k, _ in t3]
        print("\nThe model's guesses:")
        for rank, (k, p) in enumerate(t3, start=1):
            print(f"  {rank}. {k}   {p * 100:5.1f}%")
        print(f"Per-press first guesses: {' '.join(result['per_press_top1'])}")
        print(f"Real key: {target}   ->  " + ("TOP-1 CORRECT" if hit1 else ("in top 3" if hit3 else "missed")))
        rows.append({"round": i, "target": target, "detected": "yes", "top1": t3[0][0],
                     "top1_conf": round(t3[0][1], 4), "top2": t3[1][0], "top3": t3[2][0],
                     "hit_top1": int(hit1), "hit_top3": int(hit3),
                     "per_press_top1": "".join(result["per_press_top1"])})
        done = [r for r in rows if r["detected"] == "yes"]
        print(f"Running score: top-1 {sum(r['hit_top1'] for r in done)}/{len(done)}   "
              f"top-3 {sum(r['hit_top3'] for r in done)}/{len(done)}")

    with open(log_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["round", "target", "detected", "top1", "top1_conf", "top2", "top3",
                                          "hit_top1", "hit_top3", "per_press_top1"])
        w.writeheader()
        w.writerows(rows)

    done = [r for r in rows if r["detected"] == "yes"]
    print("\n" + "=" * 56)
    print("SESSION SUMMARY")
    print(f"Rounds: {len(rows)}   with a usable detection: {len(done)}   without: {len(rows) - len(done)}")
    if done:
        print(f"Top-1 correct: {sum(r['hit_top1'] for r in done)}/{len(done)} "
              f"({100 * sum(r['hit_top1'] for r in done) / len(done):.0f}%)    chance 4%")
        print(f"Top-3 correct: {sum(r['hit_top3'] for r in done)}/{len(done)} "
              f"({100 * sum(r['hit_top3'] for r in done) / len(done):.0f}%)    chance 12%")
        print("Small sample: expect this to wobble by 10+ points between sessions.")
    print(f"Saved: {log_dir / 'results.csv'}")
    print("=" * 56)


if __name__ == "__main__":
    main()
