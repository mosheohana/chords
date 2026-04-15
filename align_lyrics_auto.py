"""
align_lyrics_auto.py

Heuristic line-level lyric alignment.

This is not speech recognition. It assumes you already have the lyric lines and
tries to place line boundaries on musically plausible low-energy points in the
audio. It is meant as an automatic first draft for later manual correction.

Usage:
    python align_lyrics_auto.py song.mp3 lyrics.json --json lyrics_aligned.json
"""

import argparse
import json
import re

import librosa
import numpy as np


HOP_LENGTH = 512


def load_lyrics(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        lines = [item["text"] if isinstance(item, dict) else str(item) for item in data]
    else:
        raise ValueError("lyrics file must contain a JSON list")

    return [line.strip() for line in lines if line and line.strip()]


def line_weight(line):
    words = re.findall(r"\S+", line)
    letters = re.sub(r"[\s,.;:!?\"'()\[\]{}-]+", "", line)
    return max(1.0, (0.75 * len(words)) + (0.08 * len(letters)))


def smooth(values, size):
    if size <= 1:
        return values
    kernel = np.ones(size) / size
    return np.convolve(values, kernel, mode="same")


def pick_song_bounds(times, energy):
    threshold = np.percentile(energy, 38)
    active = energy > threshold
    active_indexes = np.where(active)[0]
    if len(active_indexes) == 0:
        return float(times[0]), float(times[-1])

    start_index = max(0, active_indexes[0] - 8)
    end_index = min(len(times) - 1, active_indexes[-1] + 8)
    return float(times[start_index]), float(times[end_index])


def best_boundary_near(target, times, energy, search_radius, min_time, max_time):
    mask = (times >= max(min_time, target - search_radius)) & (times <= min(max_time, target + search_radius))
    indexes = np.where(mask)[0]
    if len(indexes) == 0:
        return target

    local_energy = energy[indexes]
    distance = np.abs(times[indexes] - target)
    distance = distance / max(search_radius, 1e-6)
    energy_norm = local_energy / max(np.percentile(energy, 95), 1e-6)

    # Prefer a quiet point near the expected boundary.
    score = (0.72 * energy_norm) + (0.28 * distance)
    return float(times[indexes[int(np.argmin(score))]])


def align_lines(
    audio_path,
    lines,
    search_radius=2.5,
    min_line_duration=0.8,
    first_line_start=None,
    last_line_end=None,
):
    y, sr = librosa.load(audio_path, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    harmonic = librosa.effects.harmonic(y, margin=4)
    rms = librosa.feature.rms(y=harmonic, frame_length=2048, hop_length=HOP_LENGTH)[0]
    rms = smooth(rms, 19)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)

    song_start, song_end = pick_song_bounds(times, rms)
    song_start = max(0.0, song_start)
    song_end = min(duration, song_end)
    if first_line_start is not None:
        song_start = min(max(0.0, first_line_start), duration)
    if last_line_end is not None:
        song_end = min(max(song_start + min_line_duration, last_line_end), duration)

    usable_duration = max(song_end - song_start, len(lines) * min_line_duration)

    weights = np.array([line_weight(line) for line in lines], dtype=float)
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]

    expected_boundaries = [song_start]
    expected_boundaries.extend(song_start + usable_duration * value for value in cumulative[:-1])
    expected_boundaries.append(song_end)

    boundaries = [expected_boundaries[0]]
    for index, expected in enumerate(expected_boundaries[1:-1], start=1):
        min_time = boundaries[-1] + min_line_duration
        remaining = len(lines) - index
        max_time = song_end - (remaining * min_line_duration)
        boundary = best_boundary_near(
            expected,
            times,
            rms,
            search_radius=search_radius,
            min_time=min_time,
            max_time=max_time,
        )
        boundary = min(max(boundary, min_time), max_time)
        boundaries.append(boundary)
    boundaries.append(song_end)

    aligned = []
    for index, line in enumerate(lines):
        start = boundaries[index]
        end = boundaries[index + 1]
        aligned.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "text": line,
                "estimated": True,
            }
        )

    return aligned


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Create estimated line timestamps for lyrics.")
    parser.add_argument("audio", help="Audio file, for example song.mp3")
    parser.add_argument("lyrics", help="Lyrics JSON file with a list of text lines")
    parser.add_argument("--json", dest="json_path", default="lyrics_aligned_auto.json")
    parser.add_argument("--search-radius", type=float, default=2.5)
    parser.add_argument("--min-line-duration", type=float, default=0.8)
    parser.add_argument(
        "--first-line-start",
        type=float,
        default=None,
        help="Force the first lyric line to start at this time in seconds",
    )
    parser.add_argument(
        "--last-line-end",
        type=float,
        default=None,
        help="Force the final lyric line to end at this time in seconds",
    )
    args = parser.parse_args()

    lines = load_lyrics(args.lyrics)
    aligned = align_lines(
        args.audio,
        lines,
        search_radius=args.search_radius,
        min_line_duration=args.min_line_duration,
        first_line_start=args.first_line_start,
        last_line_end=args.last_line_end,
    )
    write_json(aligned, args.json_path)

    print(f"Wrote {args.json_path}")
    print(f"Aligned {len(aligned)} lines")
    for index, item in enumerate(aligned[:8], start=1):
        print(f"{index:02d}: {item['start']:6.2f}s - {item['end']:6.2f}s")


if __name__ == "__main__":
    main()
