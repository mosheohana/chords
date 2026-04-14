import argparse
import json

import numpy as np
from basic_pitch.inference import predict


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def build_chord_templates():
    templates = {}
    for i, root in enumerate(NOTES):
        major = np.zeros(12)
        major[i] = 1
        major[(i + 4) % 12] = 1
        major[(i + 7) % 12] = 1
        templates[root] = major

        minor = np.zeros(12)
        minor[i] = 1
        minor[(i + 3) % 12] = 1
        minor[(i + 7) % 12] = 1
        templates[root + "m"] = minor

    return templates


def midi_to_chroma(midi_note):
    return midi_note % 12


def get_notes_from_audio(file_path):
    """Return Basic Pitch note events: start, end, pitch, amplitude, pitch bend."""
    _model_output, _midi_data, note_events = predict(file_path)
    return note_events


def group_notes_to_chords(note_events, onset_tolerance=0.1):
    """
    Group nearby note onsets and estimate the best matching major/minor chord.
    Each returned item is (start_time, chord_name, score).
    """
    if not note_events:
        return []

    templates = build_chord_templates()
    sorted_notes = sorted(note_events, key=lambda n: n[0])

    onset_groups = []
    current_group = [sorted_notes[0]]

    for note in sorted_notes[1:]:
        if note[0] - current_group[0][0] <= onset_tolerance:
            current_group.append(note)
        else:
            onset_groups.append(current_group)
            current_group = [note]
    onset_groups.append(current_group)

    chords = []
    for group in onset_groups:
        onset_time = group[0][0]
        active_pitches = set()

        for start, end, pitch, *_ in note_events:
            if start <= onset_time < end:
                active_pitches.add(midi_to_chroma(int(pitch)))

        if not active_pitches:
            continue

        chroma_vec = np.zeros(12)
        for pitch in active_pitches:
            chroma_vec[pitch] = 1

        norm = np.linalg.norm(chroma_vec)
        frame = chroma_vec / norm if norm > 0 else chroma_vec

        best_score = -1
        best_chord = None
        for chord, template in templates.items():
            score = np.dot(frame, template)
            if score > best_score:
                best_score = score
                best_chord = chord

        chords.append((round(onset_time, 2), best_chord, round(best_score, 2)))

    return chords


def smooth_chords(chords, audio_duration=None, min_duration=0.8):
    """Remove consecutive duplicate chords and turn onsets into time ranges."""
    if not chords:
        return []

    smoothed = []
    start_time, current_chord, _score = chords[0]

    for i in range(1, len(chords)):
        time, chord, _score = chords[i]
        if chord != current_chord:
            smoothed.append((start_time, time, current_chord))
            start_time = time
            current_chord = chord

    end_time = audio_duration if audio_duration is not None else chords[-1][0]
    if end_time > start_time:
        smoothed.append((start_time, round(end_time, 2), current_chord))

    if min_duration <= 0 or len(smoothed) <= 1:
        return smoothed

    merged = []
    for start, end, chord in smoothed:
        duration = end - start
        if merged and duration < min_duration:
            prev_start, _prev_end, prev_chord = merged[-1]
            merged[-1] = (prev_start, end, prev_chord)
        else:
            merged.append((start, end, chord))

    return merged


def chords_to_dicts(chords):
    return [
        {"start": round(start, 2), "end": round(end, 2), "chord": chord}
        for start, end, chord in chords
    ]


def write_chords_json(chords, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chords_to_dicts(chords), file, ensure_ascii=False, indent=2)


def print_chords(chords):
    print("\nChords:\n")
    for start, end, chord in chords:
        print(f"  {start:.2f}s - {end:.2f}s : {chord}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Detect chords with Basic Pitch.")
    parser.add_argument("file", help="Audio file to analyze, for example song.mp3")
    parser.add_argument("--json", dest="json_path", help="Write detected chords to JSON")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.8,
        help="Merge chord changes shorter than this many seconds",
    )
    parser.add_argument(
        "--onset-tolerance",
        type=float,
        default=0.35,
        help="Group note onsets within this many seconds",
    )
    args = parser.parse_args()

    print("Analyzing with Basic Pitch...")
    note_events = get_notes_from_audio(args.file)
    print(f"   Found {len(note_events)} notes")

    chords = group_notes_to_chords(note_events, onset_tolerance=args.onset_tolerance)
    audio_duration = max((event[1] for event in note_events), default=None)
    smoothed = smooth_chords(
        chords,
        audio_duration=audio_duration,
        min_duration=args.min_duration,
    )

    if args.json_path:
        write_chords_json(smoothed, args.json_path)
        print(f"   Wrote {args.json_path}")

    print_chords(smoothed)


if __name__ == "__main__":
    main()
