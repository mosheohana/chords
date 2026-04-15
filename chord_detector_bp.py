import argparse
import json

import librosa
import numpy as np
from basic_pitch.inference import predict


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
HOP_LENGTH = 512


def build_chord_templates(vocabulary="triads"):
    templates = {}
    for i, root in enumerate(NOTES):
        templates[root] = chord_template(i, [0, 4, 7])
        templates[root + "m"] = chord_template(i, [0, 3, 7])
        if vocabulary == "extended":
            templates[root + "7"] = chord_template(i, [0, 4, 7, 10])
            templates[root + "maj7"] = chord_template(i, [0, 4, 7, 11])
            templates[root + "m7"] = chord_template(i, [0, 3, 7, 10])
            templates[root + "sus4"] = chord_template(i, [0, 5, 7])

    return templates


def chord_template(root_index, intervals):
    template = np.zeros(12)
    weights = {
        0: 1.25,
        3: 0.95,
        4: 0.95,
        5: 0.85,
        7: 0.95,
        10: 0.65,
        11: 0.65,
    }

    for interval in intervals:
        template[(root_index + interval) % 12] = weights.get(interval, 0.75)

    return template


def midi_to_chroma(midi_note):
    return midi_note % 12


def get_notes_from_audio(file_path):
    """Return Basic Pitch note events: start, end, pitch, amplitude, pitch bend."""
    _model_output, _midi_data, note_events = predict(file_path)
    return note_events


def get_audio_duration(file_path, note_events):
    note_duration = max((event[1] for event in note_events), default=0)
    try:
        return max(note_duration, librosa.get_duration(path=file_path))
    except TypeError:
        return note_duration


def collapse_duplicate_ranges(ranges):
    collapsed = []
    for start, end, chord in ranges:
        if collapsed and collapsed[-1][2] == chord:
            prev_start, _prev_end, prev_chord = collapsed[-1]
            collapsed[-1] = (prev_start, end, prev_chord)
        else:
            collapsed.append((start, end, chord))
    return collapsed


def score_chord(chroma_vec, chord, template):
    norm = np.linalg.norm(chroma_vec)
    if norm <= 0:
        return -1

    frame = chroma_vec / norm
    template_norm = np.linalg.norm(template)
    normalized_template = template / template_norm if template_norm > 0 else template
    matched = float(np.dot(frame, normalized_template))
    extra_energy = float(np.sum(frame[template == 0]))

    root = chord.replace("maj7", "").replace("sus4", "").replace("m7", "")
    root = root.replace("m", "").replace("7", "")
    root_index = NOTES.index(root)
    root_bonus = 0.2 * frame[root_index]

    return matched + root_bonus - (0.18 * extra_energy)


def pick_chord_from_chroma(chroma_vec, templates):
    best_score = -1
    best_chord = None
    for chord, template in templates.items():
        score = score_chord(chroma_vec, chord, template)
        if score > best_score:
            best_score = score
            best_chord = chord

    return best_chord, best_score


def build_weighted_chroma_for_window(
    note_events,
    start_time,
    end_time,
    min_note_duration=0.08,
    min_amplitude=0.12,
    bass_cutoff=55,
    bass_weight=1.7,
):
    chroma_vec = np.zeros(12)

    for note_start, note_end, pitch, amplitude, *_rest in note_events:
        duration = note_end - note_start
        if duration < min_note_duration or amplitude < min_amplitude:
            continue

        overlap = min(note_end, end_time) - max(note_start, start_time)
        if overlap <= 0:
            continue

        pitch = int(pitch)
        pitch_class = midi_to_chroma(pitch)
        pitch_weight = bass_weight if pitch <= bass_cutoff else 1.0
        chroma_vec[pitch_class] += overlap * float(amplitude) * pitch_weight

    return chroma_vec


def detect_chords_by_beats(
    file_path,
    note_events,
    beats_per_chord=4,
    min_note_duration=0.08,
    min_amplitude=0.12,
    vocabulary="triads",
):
    y, sr = librosa.load(file_path)
    duration = max(librosa.get_duration(y=y, sr=sr), get_audio_duration(file_path, note_events))
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    grid = [0.0]
    grid.extend(float(time) for time in beat_times if 0 < time < duration)
    grid.append(duration)
    grid = sorted(set(round(time, 4) for time in grid))

    if len(grid) < 3:
        print("   Beat tracking found too few beats, falling back to onset mode.")
        chords = group_notes_to_chords(note_events, onset_tolerance=0.35)
        return smooth_chords(chords, audio_duration=duration, min_duration=0.8), tempo

    templates = build_chord_templates(vocabulary=vocabulary)
    ranges = []
    last_grid_index = len(grid) - 1
    for start_index in range(0, last_grid_index, beats_per_chord):
        end_index = min(start_index + beats_per_chord, last_grid_index)
        start = grid[start_index]
        end = grid[end_index]
        chroma_vec = build_weighted_chroma_for_window(
            note_events,
            start,
            end,
            min_note_duration=min_note_duration,
            min_amplitude=min_amplitude,
        )
        chord, _score = pick_chord_from_chroma(chroma_vec, templates)
        if chord:
            ranges.append((start, end, chord))

    return collapse_duplicate_ranges(ranges), tempo


def group_notes_to_chords(note_events, onset_tolerance=0.1, vocabulary="triads"):
    """
    Group nearby note onsets and estimate the best matching major/minor chord.
    Each returned item is (start_time, chord_name, score).
    """
    if not note_events:
        return []

    templates = build_chord_templates(vocabulary=vocabulary)
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

        best_chord, best_score = pick_chord_from_chroma(chroma_vec, templates)

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

    return collapse_duplicate_ranges(merged)


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
    parser.add_argument(
        "--grid",
        choices=["beats", "onsets"],
        default="beats",
        help="Align Basic Pitch notes to beat windows, or use the old onset mode",
    )
    parser.add_argument(
        "--beats-per-chord",
        type=int,
        default=4,
        help="When using --grid beats, choose a chord every N beats",
    )
    parser.add_argument(
        "--min-note-duration",
        type=float,
        default=0.08,
        help="Ignore Basic Pitch notes shorter than this many seconds",
    )
    parser.add_argument(
        "--min-amplitude",
        type=float,
        default=0.12,
        help="Ignore Basic Pitch notes quieter than this value",
    )
    parser.add_argument(
        "--vocabulary",
        choices=["triads", "extended"],
        default="triads",
        help="Use simple major/minor chords, or include 7/maj7/m7/sus4 templates",
    )
    args = parser.parse_args()

    print("Analyzing with Basic Pitch...")
    note_events = get_notes_from_audio(args.file)
    print(f"   Found {len(note_events)} notes")

    if args.grid == "beats":
        smoothed, tempo = detect_chords_by_beats(
            args.file,
            note_events,
            beats_per_chord=max(1, args.beats_per_chord),
            min_note_duration=args.min_note_duration,
            min_amplitude=args.min_amplitude,
            vocabulary=args.vocabulary,
        )
        print(f"   Estimated tempo: {float(np.asarray(tempo).mean()):.1f} BPM")
        print(f"   Grid: {args.beats_per_chord} beats per chord")
    else:
        chords = group_notes_to_chords(
            note_events,
            onset_tolerance=args.onset_tolerance,
            vocabulary=args.vocabulary,
        )
        audio_duration = get_audio_duration(args.file, note_events)
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
