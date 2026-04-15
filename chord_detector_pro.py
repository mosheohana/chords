"""
chord_detector_pro.py

High-accuracy chord recognition pipeline.

Primary engine:  madmom CNN+CRF deep learning model (~80% accuracy on benchmarks).
Fallback engine: Basic Pitch notes + Krumhansl-Schmuckler key estimation +
                 amplitude-weighted template matching over 180 chord types.

Full vocabulary: maj, min, 7, maj7, min7, dim, dim7, aug, m7b5 (half-dim),
                 sus2, sus4, maj6, min6

Usage:
    python chord_detector_pro.py song.mp3
    python chord_detector_pro.py song.mp3 --json out.json
    python chord_detector_pro.py song.mp3 --beats 4
    python chord_detector_pro.py song.mp3 --engine basic-pitch
"""

""".venv\Scripts\python.exe chord_detector_madmom.py
 "song2.mp3" --json "song2_chords_madmom_only.json" --min-duration 0.5
 thid if for creating new file"""

import argparse
import json
import sys

import librosa
import numpy as np

# madmom 0.16.1 uses deprecated np.float/int/complex/bool removed in NumPy 1.24+
for _attr in ("float", "int", "complex", "bool"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, eval(_attr)))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
HOP_LENGTH = 512

# Chord types: internal_key -> (intervals, display_suffix)
CHORD_TYPES = {
    "maj":    ([0, 4, 7],          ""),
    "min":    ([0, 3, 7],          "m"),
    "7":      ([0, 4, 7, 10],      "7"),
    "maj7":   ([0, 4, 7, 11],      "maj7"),
    "min7":   ([0, 3, 7, 10],      "m7"),
    "dim":    ([0, 3, 6],          "dim"),
    "dim7":   ([0, 3, 6, 9],       "dim7"),
    "aug":    ([0, 4, 8],          "aug"),
    "m7b5":   ([0, 3, 6, 10],      "m7b5"),
    "sus4":   ([0, 5, 7],          "sus4"),
    "sus2":   ([0, 2, 7],          "sus2"),
    "maj6":   ([0, 4, 7, 9],       "maj6"),
    "min6":   ([0, 3, 7, 9],       "m6"),
}

# Krumhansl-Schmuckler key profiles
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Diatonic chord types per scale degree for key-weighting
# Maps (scale_degree, mode) -> list of chord types that are diatonic
_MAJOR_DIATONIC = {
    0: ["maj", "maj7"],
    2: ["min", "min7"],
    4: ["min", "min7"],
    5: ["maj", "maj7"],
    7: ["maj", "7"],
    9: ["min", "min7"],
    11: ["dim", "m7b5"],
}
_MINOR_DIATONIC = {
    0: ["min", "min7"],
    2: ["dim", "m7b5"],
    3: ["maj", "maj7"],
    5: ["min", "min7"],
    7: ["min", "7"],
    8: ["maj", "maj7"],
    10: ["maj", "7"],
}


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def _build_template(root_idx: int, intervals: list[int]) -> np.ndarray:
    """Chord template with interval-specific weights."""
    weights = {0: 1.3, 3: 0.9, 4: 0.9, 5: 0.85, 7: 1.0, 9: 0.75, 10: 0.7, 11: 0.7}
    t = np.zeros(12)
    for iv in intervals:
        t[(root_idx + iv) % 12] = weights.get(iv, 0.75)
    return t


def build_all_templates() -> dict[str, tuple[np.ndarray, str]]:
    """Return {internal_name: (template_vector, display_name)}."""
    templates = {}
    for root_idx, root in enumerate(NOTES):
        for ctype, (intervals, suffix) in CHORD_TYPES.items():
            key = f"{root}:{ctype}"
            display = root + suffix
            templates[key] = (_build_template(root_idx, intervals), display)
    return templates


TEMPLATES = build_all_templates()


# ---------------------------------------------------------------------------
# Key estimation (Krumhansl-Schmuckler)
# ---------------------------------------------------------------------------

def estimate_key(chroma_mean: np.ndarray) -> tuple[int, str]:
    """
    Return (root_idx, mode) — the most likely key given the mean chroma vector.
    mode is 'major' or 'minor'.
    """
    best_r = -2.0
    best_root = 0
    best_mode = "major"

    for i in range(12):
        profile_major = np.roll(KS_MAJOR, i)
        profile_minor = np.roll(KS_MINOR, i)
        r_major = float(np.corrcoef(chroma_mean, profile_major)[0, 1])
        r_minor = float(np.corrcoef(chroma_mean, profile_minor)[0, 1])
        if r_major > best_r:
            best_r = r_major
            best_root = i
            best_mode = "major"
        if r_minor > best_r:
            best_r = r_minor
            best_root = i
            best_mode = "minor"

    return best_root, best_mode


def diatonic_bonus(root_idx: int, ctype: str, key_root: int, key_mode: str) -> float:
    """Return a small bonus for chords that are diatonic to the estimated key."""
    scale = _MAJOR_DIATONIC if key_mode == "major" else _MINOR_DIATONIC
    degree = (root_idx - key_root) % 12
    if degree in scale and ctype in scale[degree]:
        return 0.12
    return 0.0


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------

def pick_best_chord(
    chroma_vec: np.ndarray,
    key_root: int,
    key_mode: str,
) -> tuple[str, float]:
    """Match a chroma vector against all templates; return (display_name, score)."""
    norm = np.linalg.norm(chroma_vec)
    if norm < 1e-6:
        return "N/C", 0.0

    frame = chroma_vec / norm

    best_score = -1.0
    best_display = "N/C"

    for key, (template, display) in TEMPLATES.items():
        root_name, ctype = key.split(":", 1)
        root_idx = NOTES.index(root_name)

        t_norm = np.linalg.norm(template)
        t = template / t_norm if t_norm > 0 else template

        # Cosine similarity
        sim = float(np.dot(frame, t))

        # Penalty for energy outside the chord tones
        extra = float(np.sum(frame[template == 0]))
        penalty = 0.15 * extra

        # Root presence bonus
        root_bonus = 0.15 * frame[root_idx]

        # Diatonic bonus
        diatonic = diatonic_bonus(root_idx, ctype, key_root, key_mode)

        score = sim + root_bonus + diatonic - penalty

        if score > best_score:
            best_score = score
            best_display = display

    return best_display, round(best_score, 3)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def collapse_runs(ranges: list) -> list:
    """Merge consecutive identical chords."""
    out = []
    for start, end, chord in ranges:
        if out and out[-1][2] == chord:
            out[-1] = (out[-1][0], end, chord)
        else:
            out.append([start, end, chord])
    return [tuple(r) for r in out]


def merge_short(ranges: list, min_dur: float) -> list:
    """Absorb very short chord segments into their predecessor."""
    if not ranges or min_dur <= 0:
        return ranges
    out = list(ranges)
    changed = True
    while changed:
        changed = False
        new = []
        i = 0
        while i < len(out):
            start, end, chord = out[i]
            if new and (end - start) < min_dur:
                new[-1] = (new[-1][0], end, new[-1][2])
                changed = True
            else:
                new.append((start, end, chord))
            i += 1
        out = new
    return collapse_runs(out)


# ---------------------------------------------------------------------------
# madmom engine
# ---------------------------------------------------------------------------

def _convert_madmom_label(label: str) -> str:
    """Convert madmom label 'C:min7' → 'Cm7', 'N' → 'N/C'."""
    if label in ("N", "X"):
        return "N/C"
    if ":" not in label:
        return label
    root, quality = label.split(":", 1)
    suffix_map = {
        "maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7",
        "dim": "dim", "dim7": "dim7", "aug": "aug", "hdim7": "m7b5",
        "sus2": "sus2", "sus4": "sus4", "maj6": "maj6", "min6": "m6",
    }
    return root + suffix_map.get(quality, quality)


def _audio_to_wav(file_path: str) -> str:
    """Write a temporary 44.1 kHz mono WAV and return its path."""
    import tempfile
    import soundfile as sf

    y, sr = librosa.load(file_path, sr=44100, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, y.T, sr)
    return tmp_path


def _madmom_beat_grid(wav_path: str, duration: float) -> list[float]:
    """Use madmom's DBN beat tracker — more accurate than librosa."""
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

    activations = RNNBeatProcessor()(wav_path)
    beat_times = DBNBeatTrackingProcessor(fps=100)(activations)
    grid = [0.0]
    grid.extend(float(t) for t in beat_times if 0 < t < duration)
    grid.append(duration)
    return sorted(set(round(t, 4) for t in grid))


def detect_with_madmom(file_path: str, min_dur: float, beats_per_chord: int = 4) -> tuple[list[tuple], float]:
    """
    Hybrid pipeline:
      - madmom DBN beat tracker  (best-in-class rhythm detection)
      - librosa chroma_cqt + HPSS on each beat window
      - Full 156-chord template matching with key estimation
    """
    import os

    print("   Converting audio...")
    y, sr = librosa.load(file_path, sr=44100)
    duration = librosa.get_duration(y=y, sr=sr)

    wav_path = _audio_to_wav(file_path)
    try:
        print("   Running madmom beat tracker...")
        grid = _madmom_beat_grid(wav_path, duration)
    finally:
        os.unlink(wav_path)

    tempo = 60.0 / np.mean(np.diff(grid[1:-1])) if len(grid) > 3 else 120.0

    if len(grid) < 3:
        raise RuntimeError("Beat tracking found too few beats")

    # Harmonic source for cleaner chroma
    y_harmonic = librosa.effects.harmonic(y, margin=4)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH, bins_per_octave=36)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    # Key estimation from full-song chroma
    key_root, key_mode = estimate_key(chroma.mean(axis=1))
    key_name = NOTES[key_root] + (" major" if key_mode == "major" else " minor")
    print(f"   Estimated key: {key_name}  |  Tempo: {tempo:.1f} BPM")

    ranges = []
    last = len(grid) - 1
    for i in range(0, last, beats_per_chord):
        j = min(i + beats_per_chord, last)
        start, end = grid[i], grid[j]

        mask = (times >= start) & (times < end)
        if not mask.any():
            mask[np.argmin(np.abs(times - start))] = True

        window_chroma = chroma[:, mask].mean(axis=1)
        chord, _score = pick_best_chord(window_chroma, key_root, key_mode)
        if chord and chord != "N/C":
            ranges.append((start, end, chord))

    return merge_short(collapse_runs(ranges), min_dur), tempo


# ---------------------------------------------------------------------------
# Basic Pitch engine
# ---------------------------------------------------------------------------

def _get_note_events(file_path: str):
    from basic_pitch.inference import predict
    _, _, note_events = predict(file_path)
    return note_events


def _weighted_chroma_window(
    note_events,
    start: float,
    end: float,
    min_amp: float = 0.10,
    min_dur: float = 0.06,
    bass_cutoff_midi: int = 55,
    bass_weight: float = 1.6,
) -> np.ndarray:
    chroma = np.zeros(12)
    for n_start, n_end, pitch, amplitude, *_ in note_events:
        if (n_end - n_start) < min_dur or amplitude < min_amp:
            continue
        overlap = min(n_end, end) - max(n_start, start)
        if overlap <= 0:
            continue
        pc = int(pitch) % 12
        w = bass_weight if int(pitch) <= bass_cutoff_midi else 1.0
        chroma[pc] += overlap * float(amplitude) * w
    return chroma


def detect_with_basic_pitch(
    file_path: str,
    beats_per_chord: int,
    min_dur: float,
) -> tuple[list[tuple], float]:
    print("   Loading audio and extracting notes with Basic Pitch...")
    note_events = _get_note_events(file_path)
    print(f"   Detected {len(note_events)} notes")

    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo = float(np.asarray(tempo_arr).mean())
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    # Build time grid
    grid = sorted({0.0} | {float(t) for t in beat_times if 0 < t < duration} | {duration})

    # Key estimation from full-song chroma
    full_chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr)
    key_root, key_mode = estimate_key(full_chroma.mean(axis=1))
    key_name = NOTES[key_root] + (" major" if key_mode == "major" else " minor")
    print(f"   Estimated key: {key_name}")

    ranges = []
    last = len(grid) - 1
    for i in range(0, last, beats_per_chord):
        j = min(i + beats_per_chord, last)
        start, end = grid[i], grid[j]
        chroma = _weighted_chroma_window(note_events, start, end)
        chord, _score = pick_best_chord(chroma, key_root, key_mode)
        if chord and chord != "N/C":
            ranges.append((start, end, chord))

    return merge_short(collapse_runs(ranges), min_dur), tempo


# ---------------------------------------------------------------------------
# Shared output helpers (also used by server.py if needed)
# ---------------------------------------------------------------------------

def chords_to_dicts(chords: list[tuple]) -> list[dict]:
    return [
        {"start": round(s, 2), "end": round(e, 2), "chord": c}
        for s, e, c in chords
    ]


def write_json(chords: list[tuple], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chords_to_dicts(chords), f, ensure_ascii=False, indent=2)


def print_chords(chords: list[tuple]) -> None:
    print("\nChords detected:\n")
    for start, end, chord in chords:
        bar = "#" * max(1, int((end - start) * 4))
        print(f"  {start:6.2f}s - {end:6.2f}s  {chord:<10} {bar}")
    print(f"\n  Total: {len(chords)} chord segments")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="High-accuracy chord detector")
    parser.add_argument("file", help="Audio file (mp3, wav, flac, etc.)")
    parser.add_argument("--json", dest="json_path", help="Write output to JSON file")
    parser.add_argument(
        "--beats", dest="beats_per_chord", type=int, default=4,
        help="Beats per chord window (Basic Pitch engine only, default 4)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=0.5,
        help="Merge chord changes shorter than this many seconds (default 0.5)",
    )
    parser.add_argument(
        "--engine", choices=["auto", "madmom", "basic-pitch"], default="auto",
        help="Force a specific engine (default: auto - tries madmom first)",
    )
    args = parser.parse_args()

    chords = []
    tempo = None

    if args.engine in ("auto", "madmom"):
        try:
            print("Trying madmom hybrid engine (beat tracking + full vocabulary)...")
            chords, tempo = detect_with_madmom(args.file, args.min_duration, args.beats_per_chord)
            print("   madmom engine OK")
        except ImportError:
            if args.engine == "madmom":
                print("ERROR: madmom is not installed. Run: pip install madmom", file=sys.stderr)
                sys.exit(1)
            print("   madmom not available — falling back to Basic Pitch engine")
        except Exception as exc:
            if args.engine == "madmom":
                raise
            print(f"   madmom failed ({exc}) — falling back to Basic Pitch engine")

    if not chords and args.engine in ("auto", "basic-pitch"):
        try:
            chords, tempo = detect_with_basic_pitch(
                args.file, args.beats_per_chord, args.min_duration
            )
        except ImportError:
            print("ERROR: basic-pitch is not installed. Run: pip install basic-pitch", file=sys.stderr)
            sys.exit(1)

    if not chords:
        print("No chords detected.", file=sys.stderr)
        sys.exit(1)

    if tempo:
        print(f"   Estimated tempo: {tempo:.1f} BPM")

    if args.json_path:
        write_json(chords, args.json_path)
        print(f"   Wrote {args.json_path}")

    print_chords(chords)


if __name__ == "__main__":
    main()
