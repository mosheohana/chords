"""
chord_detector_madmom.py

Madmom-only chord recognition.

This script uses madmom's CNN chord feature extractor and CRF chord recognizer.
It does not use Basic Pitch and does not fall back to another detector.

Usage:
    python chord_detector_madmom.py song.mp3
    python chord_detector_madmom.py song.mp3 --json chords_madmom.json
"""

import argparse
import json

import numpy as np


# madmom 0.16.1 still references deprecated NumPy aliases.
for _attr in ("float", "int", "complex", "bool"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, eval(_attr)))


def convert_madmom_label(label):
    """Convert madmom labels like 'A:min' or 'G:maj' to site labels."""
    if label in ("N", "X"):
        return "N/C"

    if ":" not in label:
        return label

    root, quality = label.split(":", 1)
    suffix_map = {
        "maj": "",
        "min": "m",
        "7": "7",
        "maj7": "maj7",
        "min7": "m7",
        "dim": "dim",
        "dim7": "dim7",
        "aug": "aug",
        "hdim7": "m7b5",
        "sus2": "sus2",
        "sus4": "sus4",
        "maj6": "maj6",
        "min6": "m6",
    }
    return root + suffix_map.get(quality, quality)


def collapse_runs(chords):
    collapsed = []
    for start, end, chord in chords:
        if collapsed and collapsed[-1][2] == chord:
            prev_start, _prev_end, prev_chord = collapsed[-1]
            collapsed[-1] = (prev_start, end, prev_chord)
        else:
            collapsed.append((start, end, chord))
    return collapsed


def merge_short(chords, min_duration):
    if min_duration <= 0 or len(chords) <= 1:
        return chords

    merged = []
    for start, end, chord in chords:
        if merged and (end - start) < min_duration:
            prev_start, _prev_end, prev_chord = merged[-1]
            merged[-1] = (prev_start, end, prev_chord)
        else:
            merged.append((start, end, chord))

    return collapse_runs(merged)


def detect_chords(file_path, min_duration=0.5, include_no_chord=False):
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

    print("Running madmom CNN chord feature processor...")
    features = CNNChordFeatureProcessor()(file_path)

    print("Running madmom CRF chord recognizer...")
    raw_chords = CRFChordRecognitionProcessor()(features)

    chords = []
    for start, end, label in raw_chords:
        chord = convert_madmom_label(str(label))
        if chord == "N/C" and not include_no_chord:
            continue
        chords.append((float(start), float(end), chord))

    return merge_short(collapse_runs(chords), min_duration)


def chords_to_dicts(chords):
    return [
        {"start": round(start, 2), "end": round(end, 2), "chord": chord}
        for start, end, chord in chords
    ]


def write_json(chords, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chords_to_dicts(chords), file, ensure_ascii=False, indent=2)


def print_chords(chords):
    print("\nChords detected:\n")
    for start, end, chord in chords:
        bar = "#" * max(1, int((end - start) * 4))
        print(f"  {start:6.2f}s - {end:6.2f}s  {chord:<8} {bar}")
    print(f"\n  Total: {len(chords)} chord segments")


def main():
    parser = argparse.ArgumentParser(description="Madmom-only chord detector")
    parser.add_argument("file", help="Audio file, for example song.mp3")
    parser.add_argument("--json", dest="json_path", help="Write output to JSON")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Merge chord changes shorter than this many seconds",
    )
    parser.add_argument(
        "--include-no-chord",
        action="store_true",
        help="Keep madmom's N/C segments in the output",
    )
    args = parser.parse_args()

    chords = detect_chords(
        args.file,
        min_duration=args.min_duration,
        include_no_chord=args.include_no_chord,
    )

    if args.json_path:
        write_json(chords, args.json_path)
        print(f"   Wrote {args.json_path}")

    print_chords(chords)


if __name__ == "__main__":
    main()
