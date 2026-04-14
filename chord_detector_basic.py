import argparse
import json

import librosa
import numpy as np


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
HOP_LENGTH = 512


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


def load_chroma(file_path):
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    frames = chroma.shape[1]
    times = librosa.frames_to_time(np.arange(frames), sr=sr, hop_length=HOP_LENGTH)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, sr, chroma, times, duration


def pick_chord(chroma_vector, templates):
    norm = np.linalg.norm(chroma_vector)
    frame = chroma_vector / norm if norm > 0 else chroma_vector

    best_score = -1
    best_chord = None
    for chord, template in templates.items():
        score = np.dot(frame, template)
        if score > best_score:
            best_score = score
            best_chord = chord

    return best_chord


def detect_chords(file_path):
    _y, _sr, chroma, times, _duration = load_chroma(file_path)
    templates = build_chord_templates()

    result = []
    for i, time in enumerate(times):
        result.append((time, pick_chord(chroma[:, i], templates)))

    return result


def collapse_duplicate_ranges(ranges):
    collapsed = []
    for start, end, chord in ranges:
        if collapsed and collapsed[-1][2] == chord:
            prev_start, _prev_end, prev_chord = collapsed[-1]
            collapsed[-1] = (prev_start, end, prev_chord)
        else:
            collapsed.append((start, end, chord))

    return collapsed


def detect_chords_by_beats(file_path, beats_per_chord=2):
    y, sr, chroma, times, duration = load_chroma(file_path)
    templates = build_chord_templates()
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    grid = [0.0]
    grid.extend(float(time) for time in beat_times if 0 < time < duration)
    grid.append(duration)
    grid = sorted(set(round(time, 4) for time in grid))

    if len(grid) < 3:
        print("   Beat tracking found too few beats, falling back to frame smoothing.")
        return smooth_chords(detect_chords(file_path), min_duration=0.25), tempo

    ranges = []
    last_grid_index = len(grid) - 1
    for start_index in range(0, last_grid_index, beats_per_chord):
        end_index = min(start_index + beats_per_chord, last_grid_index)
        start = grid[start_index]
        end = grid[end_index]

        frame_indexes = np.where((times >= start) & (times < end))[0]
        if len(frame_indexes) == 0:
            frame_indexes = [int(np.argmin(np.abs(times - start)))]

        average_chroma = np.mean(chroma[:, frame_indexes], axis=1)
        ranges.append((start, end, pick_chord(average_chroma, templates)))

    return collapse_duplicate_ranges(ranges), tempo


def smooth_chords(chords, min_duration=1.0):
    if not chords:
        return []

    ranges = []
    start_time, current = chords[0]

    for i in range(1, len(chords)):
        time, chord = chords[i]
        if chord != current:
            ranges.append((start_time, time, current))
            start_time = time
            current = chord

    last_time = chords[-1][0]
    if last_time > start_time:
        ranges.append((start_time, last_time, current))

    if min_duration <= 0 or len(ranges) <= 1:
        return ranges

    merged = []
    for start, end, chord in ranges:
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
        print(f"{start:.2f}s - {end:.2f}s : {chord}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Detect chords with librosa chroma.")
    parser.add_argument("file", help="Audio file to analyze, for example song.mp3")
    parser.add_argument("--json", dest="json_path", help="Write detected chords to JSON")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Merge chord changes shorter than this many seconds",
    )
    parser.add_argument(
        "--grid",
        choices=["frames", "beats"],
        default="frames",
        help="Detect freely per frame, or align chord ranges to song beats",
    )
    parser.add_argument(
        "--beats-per-chord",
        type=int,
        default=2,
        help="When using --grid beats, choose a chord every N beats",
    )
    args = parser.parse_args()

    print("Processing with librosa chroma...")
    if args.grid == "beats":
        smoothed, tempo = detect_chords_by_beats(
            args.file,
            beats_per_chord=max(1, args.beats_per_chord),
        )
        print(f"   Estimated tempo: {float(np.asarray(tempo).mean()):.1f} BPM")
        print(f"   Grid: {args.beats_per_chord} beats per chord")
    else:
        chords = detect_chords(args.file)
        smoothed = smooth_chords(chords, min_duration=args.min_duration)

    if args.json_path:
        write_chords_json(smoothed, args.json_path)
        print(f"   Wrote {args.json_path}")

    print_chords(smoothed)


if __name__ == "__main__":
    main()
