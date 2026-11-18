"""
recording.py (v4) - record one burst-recording session (A-Z, ~5 presses per key)
with automatic press counting and audio-aligned key timestamps.

Run from the project root (venv active):
    python src/recording.py            # real session -> dataset/raw/<session_id>/
    python src/recording.py --test     # 3-key dry run -> dataset/test_runs/<session_id>/

Requires:  pip install pynput

What it writes (real session):
    dataset/raw/<session_id>/<KEY>_burst.wav          accepted recordings
    dataset/raw/<session_id>/events/<KEY>_events.csv  key-down / key-up times of the target key
    dataset/raw/<session_id>/rejected/...             takes you chose to redo (kept, not deleted)
    dataset/raw/<session_id>/ambient/noise_floor.wav  a few seconds of room noise
    dataset/raw/<session_id>/session_info.txt         all session parameters
    dataset/metadata_master.csv                       one row per saved WAV (same 14-column schema)

What changed in v4 (compared with v3):
  * The microphone stream is opened FIRST. "PRESS NOW" is only shown once audio is
    actually flowing, so early presses are no longer lost to stream start-up delay.
  * Key times are measured relative to the FIRST AUDIO SAMPLE (using the audio
    stream's own clock where Windows provides it), so a logged key-down time is
    directly comparable with its position in the WAV file.
  * Room-noise check reports a robust median level and offers a redo if a loud
    transient (bump, click) was recorded during the "silent" seconds.

Key logger privacy: during a take only the TARGET key's down/up times are stored.
Any other key is only counted (no key identity is saved). The logger runs only
while a take is being recorded.

It never overwrites an existing session folder and never touches the legacy
dataset/metadata.csv. It refuses to append to a metadata file whose columns differ.
"""

import csv
import random
import re
import string
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    raise SystemExit(
        "pynput is not installed in this environment.\n"
        "Run:  pip install pynput\n"
        "then start this script again."
    )

# ==========================================
# 1. CONFIGURATION
# ==========================================

SAMPLE_RATE = 44100
DURATION = 5                 # seconds per burst recording
CHANNELS = 1
DEVICE_ID = 1                # Microphone Array (Intel Smart Sound)
INTENDED_PRESSES = 5
NOISE_FLOOR_SECONDS = 3
COUNTDOWN_SECONDS = 3
LOGGER_WARMUP_SECONDS = 0.2  # let the key listener start before audio starts
STREAM_START_TIMEOUT = 5.0   # seconds to wait for the first audio block
LOW_PEAK_THRESHOLD = 0.01    # same heuristic as the audit / master metadata
CLIP_THRESHOLD = 0.99
TRANSIENT_RATIO = 15.0       # noise-check peak / median-envelope above this => warn
ALLOWED_CONDITIONS = ["clean", "noisy", "noisy_pilot", "dampened"]

TEST_MODE = "--test" in sys.argv

if TEST_MODE:
    BASE_DIR = Path("dataset") / "test_runs"
    MASTER_PATH = Path("dataset") / "test_runs" / "metadata_test.csv"
else:
    BASE_DIR = Path("dataset") / "raw"
    MASTER_PATH = Path("dataset") / "metadata_master.csv"

COLUMNS = [
    "filename", "key", "session", "condition", "recording_type",
    "repetition", "sample_rate", "duration_s", "peak_amplitude",
    "rms_amplitude", "quality_flag", "intended_presses", "actual_presses",
    "notes",
]

# ==========================================
# 2. KEY LOGGER
# ==========================================


class KeyLogger:
    """Logs down/up times of ONE target key; only counts every other key."""

    def __init__(self, target):
        self.target = target.lower()
        self.raw_events = []    # (event, time.perf_counter() value)
        self.other_count = 0    # key events from non-target keys (identity not stored)
        self._is_down = False   # ignores OS auto-repeat while the key is held
        self._lock = threading.Lock()
        self._listener = None

    def _is_target(self, key):
        char = getattr(key, "char", None)
        return isinstance(char, str) and char.lower() == self.target

    def _on_press(self, key):
        t = time.perf_counter()
        with self._lock:
            if self._is_target(key):
                if not self._is_down:
                    self._is_down = True
                    self.raw_events.append(("down", t))
            else:
                self.other_count += 1

    def _on_release(self, key):
        t = time.perf_counter()
        with self._lock:
            if self._is_target(key):
                if self._is_down:
                    self._is_down = False
                    self.raw_events.append(("up", t))

    def start(self):
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def relative_events(self, audio_start_perf):
        """Events as (name, seconds since the first audio sample)."""
        with self._lock:
            return [(e, t - audio_start_perf) for e, t in self.raw_events]

    def summary(self, audio_start_perf):
        """Return (n_downs_in_window, n_ups_in_window, n_events_outside_window)."""
        events = self.relative_events(audio_start_perf)
        downs = [t for e, t in events if e == "down" and 0 <= t <= DURATION]
        ups = [t for e, t in events if e == "up" and 0 <= t <= DURATION]
        outside = sum(1 for _, t in events if t < 0 or t > DURATION)
        return len(downs), len(ups), outside


# ==========================================
# 3. AUDIO CAPTURE
# ==========================================


class Capture:
    """Records a fixed number of seconds from an already-running input stream."""

    def __init__(self, seconds):
        self.total = int(seconds * SAMPLE_RATE)
        self.blocks = []
        self.count = 0
        self.first = None
        self.overflows = 0
        self.started = threading.Event()
        self.done = threading.Event()

    def callback(self, indata, frames, time_info, status):
        if self.first is None:
            self.first = {
                "perf": time.perf_counter(),
                "frames": frames,
                "adc": float(time_info.inputBufferAdcTime),
                "current": float(time_info.currentTime),
            }
            self.started.set()
        if status.input_overflow:
            self.overflows += 1
        remaining = self.total - self.count
        if remaining > 0:
            take = min(frames, remaining)
            self.blocks.append(indata[:take, 0].copy())
            self.count += take
        if self.count >= self.total:
            self.done.set()
            raise sd.CallbackStop

    def audio_start_perf(self):
        """time.perf_counter() value at which the first audio sample was captured.

        Preferred: the stream's own clock (currentTime - inputBufferAdcTime = how
        long ago the first sample of the first block was captured). If Windows does
        not provide sensible values, fall back to 'one block ago'.
        Returns (perf_value, method, lag_seconds).
        """
        f = self.first
        lag = f["current"] - f["adc"]
        if f["current"] > 0 and 0.0 <= lag <= 1.0:
            return f["perf"] - lag, "adc", lag
        lag = f["frames"] / SAMPLE_RATE
        return f["perf"] - lag, "block", lag

    def audio(self):
        data = np.concatenate(self.blocks) if self.blocks else np.zeros(0, dtype=np.float32)
        return data.reshape(-1, 1).astype(np.float32)


def capture(seconds, logger=None, announce=None):
    """Open the stream, wait until audio flows, call announce(), record, close.

    Returns (audio, Capture). If a logger is given it is started before the stream
    and stopped after it.
    """
    cap = Capture(seconds)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
        device=DEVICE_ID, callback=cap.callback,
    )
    if logger is not None:
        logger.start()
    try:
        time.sleep(LOGGER_WARMUP_SECONDS)
        stream.start()
        if not cap.started.wait(timeout=STREAM_START_TIMEOUT):
            raise RuntimeError("The microphone stream delivered no audio within "
                               f"{STREAM_START_TIMEOUT:.0f} s.")
        if announce is not None:
            announce()
        if not cap.done.wait(timeout=seconds + 5.0):
            raise RuntimeError("Recording did not finish in time.")
    finally:
        if logger is not None:
            logger.stop()
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
    return cap.audio(), cap


# ==========================================
# 4. HELPERS
# ==========================================


def flush_input():
    """Throw away anything typed into the terminal during a take."""
    try:
        import msvcrt  # Windows
        while msvcrt.kbhit():
            msvcrt.getwch()
    except ImportError:
        try:
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass


def check_device():
    """Make sure the configured microphone exists and can record."""
    try:
        info = sd.query_devices(DEVICE_ID)
    except Exception as exc:
        raise SystemExit(f"Cannot open audio device {DEVICE_ID}: {exc}")
    if info["max_input_channels"] < 1:
        raise SystemExit(f"Device {DEVICE_ID} ({info['name']}) has no input channels.")
    return info


def check_master_header():
    """Refuse to append to a metadata file with a different column layout."""
    if MASTER_PATH.exists() and MASTER_PATH.stat().st_size > 0:
        with open(MASTER_PATH, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            raise SystemExit(
                f"{MASTER_PATH} has a different column layout than this script "
                f"expects. Not appending.\nFound   : {header}\nExpected: {COLUMNS}"
            )


def append_row(row):
    """Append one row to the master metadata file (writes the header if new)."""
    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = (not MASTER_PATH.exists()) or MASTER_PATH.stat().st_size == 0
    with open(MASTER_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def measure(audio):
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return peak, rms


def noise_stats(audio):
    """Return (peak, rms, median 10 ms envelope) of a silent-room recording."""
    x = audio[:, 0] if audio.ndim == 2 else audio
    n = (len(x) // 441) * 441
    env = np.sqrt(np.mean(x[:n].reshape(-1, 441) ** 2, axis=1))
    return (float(np.max(np.abs(x))), float(np.sqrt(np.mean(x ** 2))),
            float(np.median(env)))


def quality_flag(peak):
    if peak < LOW_PEAK_THRESHOLD:
        return "LOW_PEAK"
    if peak >= CLIP_THRESHOLD:
        return "POSSIBLE_CLIPPING"
    return "OK"


def countdown():
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)


def write_events(path, key, events):
    """Save the target key's down/up times (seconds since the first audio sample)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event", "key", "t_seconds_from_audio_start"])
        for event, t in events:
            writer.writerow([event, key, f"{t:.4f}"])


def ask_accept():
    """ENTER accepts the take, 'r' redoes it. Returns 'accept' or 'r'."""
    while True:
        answer = input("Press ENTER to accept this take, or type 'r' to redo: ").strip().lower()
        if answer == "":
            return "accept"
        if answer == "r":
            return "r"
        print("Just press ENTER to accept, or type r to redo.")


def ask_manual_count():
    """Fallback if the key logger saw nothing at all. Returns an int, or 'r'."""
    while True:
        answer = input(
            "How many presses did you actually make? (number, or 'r' to redo): "
        ).strip().lower()
        if answer == "r":
            return "r"
        if answer.isdigit() and 0 <= int(answer) <= 20:
            return int(answer)
        print("Please type a number from 0 to 20, or 'r'.")


def row_dict(filepath, key, session_id, condition, peak, rms, flag,
             actual_presses, notes):
    return {
        "filename": filepath.as_posix(),
        "key": key,
        "session": session_id,
        "condition": condition,
        "recording_type": "burst",
        "repetition": "",
        "sample_rate": SAMPLE_RATE,
        "duration_s": DURATION,
        "peak_amplitude": round(peak, 6),
        "rms_amplitude": round(rms, 6),
        "quality_flag": flag,
        "intended_presses": INTENDED_PRESSES,
        "actual_presses": actual_presses,
        "notes": notes,
    }


def noise_check():
    """Record the silent-room check; offer a redo if a loud transient was captured."""
    while True:
        print(f"\nStay completely silent for {NOISE_FLOOR_SECONDS} seconds. Do NOT type.")
        countdown()
        audio, _ = capture(NOISE_FLOOR_SECONDS)
        peak, rms, med = noise_stats(audio)
        print(f"Room noise  -> peak: {peak:.5f}   rms: {rms:.6f}   "
              f"median level: {med:.6f}")
        if med > 0 and peak > TRANSIENT_RATIO * med:
            print("NOTE: a loud transient (bump, click or similar) was recorded during "
                  "the silent check, so the peak and RMS above are inflated.")
            flush_input()
            again = input("Redo the room-noise check? (y/n): ").strip().lower()
            if again == "y":
                continue
        return audio, peak, rms, med


# ==========================================
# 5. MAIN
# ==========================================


def main():
    print("=" * 50)
    if TEST_MODE:
        print("TEST MODE: 3 keys only, nothing goes into the real dataset")
    else:
        print("REAL SESSION")
    print("=" * 50)

    dev_info = check_device()
    device_name = dev_info["name"]
    print(f"Microphone: [{DEVICE_ID}] {device_name}")
    check_master_header()

    # ---- session details -------------------------------------------------
    session_id = input("\nEnter session ID (e.g. session_06): ").strip()
    if not re.fullmatch(r"session_[A-Za-z0-9_]+", session_id):
        raise SystemExit("Session ID must look like session_06 (letters/digits/underscore).")

    condition = input(f"Enter condition ({'/'.join(ALLOWED_CONDITIONS)}): ").strip()
    if condition not in ALLOWED_CONDITIONS:
        raise SystemExit(f"Invalid condition. Choose one of: {ALLOWED_CONDITIONS}")

    session_folder = BASE_DIR / session_id
    if session_folder.exists():
        raise SystemExit(f"{session_folder} already exists. Choose a new session ID.")

    setup_notes = input(
        "Setup notes (laptop position, surface, noise source, mic settings; optional): "
    ).strip()

    # ---- pre-flight checklist --------------------------------------------
    print("\nBefore you start:")
    print("  - Same laptop position, same sitting posture, same typing style every session")
    print("  - Windows microphone enhancements / noise suppression OFF, input volume 100")
    print("  - Close other apps, silence your phone, no music or video")
    print("  - Do not touch the laptop except to type")
    print("  - Press ONLY the requested key during a take (other keys are counted as errors)")
    flush_input()
    input("Press ENTER when ready for the room-noise check...")

    # ---- noise floor (kept in memory until we know the folder is created) -
    noise_audio, noise_peak, noise_rms, noise_med = noise_check()

    if condition == "clean":
        flush_input()
        confirm = input(
            "Is the room genuinely quiet and is this your standard setup? (y/n): "
        ).strip().lower()
        if confirm != "y":
            raise SystemExit(
                "Not labelling this 'clean'. Nothing was saved. "
                "Fix the room or use another condition label."
            )

    # ---- create the session folder ---------------------------------------
    keys = list(string.ascii_uppercase)
    seed = int(time.time())
    random.Random(seed).shuffle(keys)
    if TEST_MODE:
        keys = keys[:3]

    session_folder.mkdir(parents=True)
    (session_folder / "ambient").mkdir()
    (session_folder / "events").mkdir()
    sf.write(session_folder / "ambient" / "noise_floor.wav", noise_audio,
             SAMPLE_RATE, subtype="PCM_16")

    info_path = session_folder / "session_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"session: {session_id}\n")
        f.write(f"condition: {condition}\n")
        f.write(f"mode: {'TEST' if TEST_MODE else 'REAL'}\n")
        f.write("script_version: v4 (audio-aligned key logger)\n")
        f.write(f"started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"device: [{DEVICE_ID}] {device_name}\n")
        f.write(f"device_default_low_input_latency_s: {dev_info.get('default_low_input_latency')}\n")
        f.write(f"device_default_high_input_latency_s: {dev_info.get('default_high_input_latency')}\n")
        f.write(f"sample_rate: {SAMPLE_RATE}\n")
        f.write(f"duration_s: {DURATION}\n")
        f.write(f"intended_presses: {INTENDED_PRESSES}\n")
        f.write(f"key_order_seed: {seed}\n")
        f.write(f"key_order: {' '.join(keys)}\n")
        f.write(f"noise_floor_peak: {noise_peak:.6f}\n")
        f.write(f"noise_floor_rms: {noise_rms:.7f}\n")
        f.write(f"noise_floor_median_level: {noise_med:.7f}\n")
        f.write(f"setup_notes: {setup_notes}\n")

    print("\nRandomised recording order:")
    print(" -> ".join(keys))

    saved = 0
    off_count = []   # keys where logged presses != intended

    try:
        for index, key in enumerate(keys, start=1):
            attempt = 0
            while True:
                attempt += 1
                print("\n" + "=" * 40)
                print(f"KEY {index}/{len(keys)}: {key}   (attempt {attempt})")
                print("=" * 40)
                flush_input()
                input(f"Press ENTER when ready to record {key}...")

                print("\nGet ready...")
                countdown()
                print("Starting the microphone...", flush=True)

                def announce(k=key):
                    print(f">>> PRESS {k} {INTENDED_PRESSES} TIMES, about 1 second apart <<<",
                          flush=True)

                logger = KeyLogger(key)
                audio, cap = capture(DURATION, logger=logger, announce=announce)
                flush_input()   # discard the letters typed during the take

                start_perf, method, lag = cap.audio_start_perf()
                events = logger.relative_events(start_perf)
                n_down, n_up, outside = logger.summary(start_perf)
                other = logger.other_count

                peak, rms = measure(audio)
                flag = quality_flag(peak)

                print(f"\nPeak: {peak:.5f}   RMS: {rms:.5f}   Quality: {flag}")
                if flag == "LOW_PEAK":
                    print("WARNING: very quiet recording, the presses may be missing.")
                if flag == "POSSIBLE_CLIPPING":
                    print("WARNING: possible clipping.")
                if cap.overflows:
                    print(f"WARNING: the audio stream reported {cap.overflows} input overflow(s).")
                if len(audio) < cap.total:
                    print(f"WARNING: only {len(audio)} of {cap.total} samples were captured.")

                print(f"Key logger: {n_down} press(es) of {key}, "
                      f"{n_up} release(s), {other} other key event(s).")
                print(f"(audio clock: {method}, first-block lag {lag * 1000:.0f} ms)")
                if outside:
                    print(f"NOTE: {outside} {key} event(s) fell outside the recording window.")
                if n_down != INTENDED_PRESSES:
                    print(f"NOTE: expected {INTENDED_PRESSES} presses, logger counted {n_down}.")
                if other:
                    print("NOTE: other keys were pressed during this take.")

                manual = False
                if n_down == 0 and other == 0:
                    print("WARNING: the key logger saw NO key events at all. "
                          "It may not be working (is the terminal running as admin?).")
                    decision = ask_manual_count()
                    manual = True
                else:
                    decision = ask_accept()

                notes_parts = [f"logger_count={n_down}", f"releases={n_up}",
                               f"other_key_events={other}", f"align={method}"]
                if outside:
                    notes_parts.append(f"events_outside_window={outside}")
                if cap.overflows:
                    notes_parts.append(f"input_overflows={cap.overflows}")
                if attempt > 1:
                    notes_parts.append(f"attempt {attempt}")

                if decision == "r":
                    rejected_dir = session_folder / "rejected"
                    rejected_dir.mkdir(exist_ok=True)
                    rpath = rejected_dir / f"{key}_burst_rejected_{attempt}.wav"
                    sf.write(rpath, audio, SAMPLE_RATE, subtype="PCM_16")
                    write_events(session_folder / "events" /
                                 f"{key}_rejected_{attempt}_events.csv", key, events)
                    append_row(row_dict(
                        rpath, key, session_id, condition, peak, rms, "REJECTED",
                        "", "REJECTED_BY_USER; " + "; ".join(notes_parts)))
                    print(f"Take kept in rejected/ ({rpath.name}). Redoing {key}.")
                    continue

                if manual:
                    actual = decision
                    notes_parts.append("MANUAL_COUNT (logger saw no events)")
                else:
                    actual = n_down

                path = session_folder / f"{key}_burst.wav"
                sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
                write_events(session_folder / "events" / f"{key}_events.csv", key, events)
                append_row(row_dict(
                    path, key, session_id, condition, peak, rms, flag,
                    actual, "; ".join(notes_parts)))
                saved += 1
                if actual != INTENDED_PRESSES:
                    off_count.append((key, actual))
                print(f"Saved: {path}")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted. Everything saved so far is kept.")
        with open(info_path, "a", encoding="utf-8") as f:
            f.write(f"INTERRUPTED after {saved} of {len(keys)} keys\n")
        print(f"Session {session_id} is INCOMPLETE ({saved}/{len(keys)} keys).")
        return

    with open(info_path, "a", encoding="utf-8") as f:
        f.write(f"completed: {saved} of {len(keys)} keys at "
                f"{datetime.now().isoformat(timespec='seconds')}\n")

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("SESSION COMPLETE")
    print(f"Session: {session_id}   Condition: {condition}")
    print(f"Keys saved: {saved}/{len(keys)}")
    if off_count:
        print(f"Keys where the logger counted a number other than {INTENDED_PRESSES}: "
              + ", ".join(f"{k}={n}" for k, n in off_count))
    print(f"Metadata: {MASTER_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    main()
