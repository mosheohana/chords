"""
chord_detector_hmm.py

Beat-synchronous chord recognition with an explicit Hidden Markov Model.

This module is intentionally separate from the older detectors.  The pipeline is:

1. Load audio and split harmonic/percussive content with HPSS.
2. Track beats and aggregate chroma per beat interval.
3. Estimate the global key with Krumhansl-Schmuckler profiles.
4. Build chord-template emission probabilities.
5. Decode a musically plausible chord sequence with Viterbi.
6. Merge unstable or very short chord regions.

The output format is a list of dictionaries:
    [{"start": float, "end": float, "chord": str, "confidence": float}]

Usage:
    python chord_detector_hmm.py media/audio/song2.mp3
    python chord_detector_hmm.py media/audio/song2.mp3 --json data/chords/song2_chords_hmm.json
    python chord_detector_hmm.py media/audio/song2.mp3 --compare-naive
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable

import librosa
import numpy as np


# madmom 0.16.1 expects aliases removed from Python 3.10+.
for _name in (
    "Callable", "Iterable", "Iterator", "Generator",
    "Mapping", "MutableMapping", "MutableSequence", "MutableSet",
    "Sequence", "Set",
):
    if sys.version_info >= (3, 10) and not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

# madmom 0.16.1 also references NumPy aliases removed in NumPy 1.24+.
for _attr in ("float", "int", "complex", "bool"):
    if _attr not in np.__dict__:
        setattr(np, _attr, getattr(__builtins__, _attr, eval(_attr)))


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
HOP_LENGTH = 512
EPS = 1e-10

# This vocabulary is deliberately compact enough for local Viterbi decoding while
# covering the common triads, sevenths, suspended chords, and altered triads.
CHORD_QUALITIES = {
    "maj": ([0, 4, 7], ""),
    "min": ([0, 3, 7], "m"),
    "7": ([0, 4, 7, 10], "7"),
    "maj7": ([0, 4, 7, 11], "maj7"),
    "min7": ([0, 3, 7, 10], "m7"),
    "dim": ([0, 3, 6], "dim"),
    "aug": ([0, 4, 8], "aug"),
    "sus2": ([0, 2, 7], "sus2"),
    "sus4": ([0, 5, 7], "sus4"),
}

# Krumhansl-Schmuckler profiles.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
MAJOR_DIATONIC = {
    0: {"maj", "maj7"},
    2: {"min", "min7", "sus2", "sus4"},
    4: {"min", "min7"},
    5: {"maj", "maj7", "sus2", "sus4"},
    7: {"maj", "7", "sus2", "sus4"},
    9: {"min", "min7"},
    11: {"dim"},
}
MINOR_DIATONIC = {
    0: {"min", "min7", "sus2", "sus4"},
    2: {"dim"},
    3: {"maj", "maj7"},
    5: {"min", "min7", "sus2", "sus4"},
    7: {"min", "7", "sus2", "sus4"},
    8: {"maj", "maj7"},
    10: {"maj", "7"},
}


@dataclass(frozen=True)
class ChordState:
    """Single HMM state representing one root and chord quality."""

    root: int
    quality: str
    intervals: tuple[int, ...]
    template: np.ndarray
    label: str


@dataclass
class BeatFeatures:
    """Beat-synchronous feature matrix and metadata."""

    chroma: np.ndarray
    bass_chroma: np.ndarray
    beat_times: np.ndarray
    tempo: float
    duration: float


@dataclass
class DecodedBeat:
    """Decoded chord state for one beat interval."""

    start: float
    end: float
    state_index: int
    chord: str
    confidence: float


def _normalize_vector(vector: np.ndarray, norm: str = "l2") -> np.ndarray:
    """Return a finite normalized vector while preserving all-zero silence."""
    vector = np.asarray(vector, dtype=float)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    vector = np.maximum(vector, 0.0)
    denom = np.linalg.norm(vector) if norm == "l2" else np.sum(vector)
    return vector / denom if denom > EPS else np.zeros_like(vector)


def _softmax(log_values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for posterior-like confidence scores."""
    shifted = log_values - np.max(log_values)
    exp_values = np.exp(shifted)
    return exp_values / max(float(np.sum(exp_values)), EPS)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation helper that tolerates flat vectors."""
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _display_chord(root: int, quality: str) -> str:
    """Return the display label used by the web app."""
    return NOTES[root] + CHORD_QUALITIES[quality][1]


def _build_weighted_template(root: int, intervals: Iterable[int]) -> np.ndarray:
    """Create a chord template with root/fifth slightly emphasized."""
    weights = {
        0: 1.30, 2: 0.82, 3: 0.95, 4: 0.98, 5: 0.86, 6: 0.82,
        7: 1.08, 8: 0.82, 10: 0.76, 11: 0.76,
    }
    template = np.zeros(12, dtype=float)
    for interval in intervals:
        template[(root + interval) % 12] = weights.get(interval, 0.75)
    return _normalize_vector(template)


def build_chord_states() -> list[ChordState]:
    """Build all HMM chord states from the expanded vocabulary."""
    states: list[ChordState] = []
    for root in range(12):
        for quality, (intervals, _suffix) in CHORD_QUALITIES.items():
            states.append(
                ChordState(
                    root=root,
                    quality=quality,
                    intervals=tuple(intervals),
                    template=_build_weighted_template(root, intervals),
                    label=_display_chord(root, quality),
                )
            )
    return states


def estimate_key(chroma_mean: np.ndarray) -> tuple[int, str, float]:
    """
    Estimate global key with Krumhansl-Schmuckler profiles.

    Returns (root_index, mode, correlation_score).  The HMM uses this only as a
    soft prior in transitions and emissions; it never hard-rejects out-of-key
    chords because real songs borrow chords constantly.
    """
    chroma_mean = _normalize_vector(chroma_mean)
    best_root = 0
    best_mode = "major"
    best_score = -2.0

    for root in range(12):
        major_score = _safe_corrcoef(chroma_mean, np.roll(KS_MAJOR, root))
        minor_score = _safe_corrcoef(chroma_mean, np.roll(KS_MINOR, root))
        if major_score > best_score:
            best_root, best_mode, best_score = root, "major", major_score
        if minor_score > best_score:
            best_root, best_mode, best_score = root, "minor", minor_score

    return best_root, best_mode, best_score


def is_diatonic(state: ChordState, key_root: int, key_mode: str) -> bool:
    """Return whether a chord state fits the estimated major/minor key."""
    degree = (state.root - key_root) % 12
    table = MAJOR_DIATONIC if key_mode == "major" else MINOR_DIATONIC
    return state.quality in table.get(degree, set())


def key_scale_contains(root: int, key_root: int, key_mode: str) -> bool:
    """Return whether a pitch class belongs to the estimated key scale."""
    scale = MAJOR_SCALE if key_mode == "major" else MINOR_SCALE
    return (root - key_root) % 12 in scale


class BeatSynchronousFeatureExtractor:
    """Extract beat-level chroma and bass chroma from an audio file."""

    def __init__(
        self,
        hop_length: int = HOP_LENGTH,
        prefer_madmom: bool = True,
        chroma_kind: str = "cqt",
        sample_rate: int = 44100,
    ) -> None:
        self.hop_length = hop_length
        self.prefer_madmom = prefer_madmom
        self.chroma_kind = chroma_kind
        self.sample_rate = sample_rate

    def extract(self, file_path: str) -> BeatFeatures:
        """Load audio and return normalized beat-synchronous features."""
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0:
            raise RuntimeError("Audio file has zero duration")

        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # CQT is the default because it preserves chord-tone detail for sevenths,
        # suspended chords, augmented/diminished triads, and root evidence.  CENS
        # is available for very noisy material, but its smoothing can blur those
        # chord qualities before the HMM gets a chance to model time.
        if self.chroma_kind == "cens":
            chroma = librosa.feature.chroma_cens(
                y=y_harmonic,
                sr=sr,
                hop_length=self.hop_length,
                bins_per_octave=36,
            )
        else:
            chroma = librosa.feature.chroma_cqt(
                y=y_harmonic,
                sr=sr,
                hop_length=self.hop_length,
                bins_per_octave=36,
            )

        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]), sr=sr, hop_length=self.hop_length
        )
        beat_times, tempo = self._beat_grid(file_path, y_percussive, sr, duration)
        bass_chroma = self._extract_bass_chroma(y_harmonic, sr, chroma.shape[1])

        beat_chroma = self._aggregate_to_beats(chroma, times, beat_times, norm="l2")
        beat_bass = self._aggregate_to_beats(bass_chroma, times, beat_times, norm="l1")

        return BeatFeatures(
            chroma=beat_chroma,
            bass_chroma=beat_bass,
            beat_times=beat_times,
            tempo=tempo,
            duration=duration,
        )

    def _beat_grid(
        self,
        file_path: str,
        y_percussive: np.ndarray,
        sr: int,
        duration: float,
    ) -> tuple[np.ndarray, float]:
        """Return a beat boundary grid using madmom when available, librosa otherwise."""
        beat_times: list[float] = []
        tempo = 120.0

        if self.prefer_madmom:
            try:
                beat_times = self._madmom_beats(file_path, duration)
            except Exception as exc:
                print(f"   madmom beat tracking unavailable ({exc}); using librosa beats.")

        if len(beat_times) < 2:
            tempo_raw, beat_frames = librosa.beat.beat_track(
                y=y_percussive, sr=sr, hop_length=self.hop_length
            )
            tempo = float(np.asarray(tempo_raw).mean())
            beat_times = [
                float(t)
                for t in librosa.frames_to_time(
                    beat_frames, sr=sr, hop_length=self.hop_length
                )
                if 0.0 < float(t) < duration
            ]
        elif len(beat_times) > 2:
            tempo = 60.0 / max(float(np.median(np.diff(beat_times))), EPS)

        grid = sorted({0.0, duration} | {round(t, 4) for t in beat_times if 0 < t < duration})
        if len(grid) < 3:
            raise RuntimeError("Beat tracking found too few beats for beat-synchronous HMM")
        return np.asarray(grid, dtype=float), tempo

    def _madmom_beats(self, file_path: str, duration: float) -> list[float]:
        """Run madmom's recurrent-network beat tracker on a temporary WAV."""
        import soundfile as sf
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            sf.write(wav_path, y, sr)
            activations = RNNBeatProcessor()(wav_path)
            beats = DBNBeatTrackingProcessor(fps=100)(activations)
            return [float(t) for t in beats if 0 < float(t) < duration]
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _extract_bass_chroma(self, y_harmonic: np.ndarray, sr: int, target_frames: int) -> np.ndarray:
        """
        Extract low-frequency pitch-class energy.

        This is not full transcription.  It gives the HMM a root cue from the
        bass register, which helps distinguish C from Am/C-like ambiguity and
        can optionally label inversions after decoding.
        """
        cqt = np.abs(
            librosa.cqt(
                y_harmonic,
                sr=sr,
                hop_length=self.hop_length,
                fmin=librosa.note_to_hz("C1"),
                n_bins=48,
                bins_per_octave=12,
            )
        )
        bass = np.zeros((12, cqt.shape[1]), dtype=float)
        for bin_index in range(cqt.shape[0]):
            freq = librosa.note_to_hz("C1") * (2.0 ** (bin_index / 12.0))
            midi = int(round(float(librosa.hz_to_midi(freq))))
            if midi <= 55:  # roughly G3 and below: bass/root region.
                bass[midi % 12] += cqt[bin_index]

        if bass.shape[1] < target_frames:
            pad = np.zeros((12, target_frames - bass.shape[1]))
            bass = np.hstack([bass, pad])
        return bass[:, :target_frames]

    def _aggregate_to_beats(
        self,
        feature: np.ndarray,
        frame_times: np.ndarray,
        beat_times: np.ndarray,
        norm: str,
    ) -> np.ndarray:
        """Aggregate frame features inside each beat interval."""
        rows = []
        for start, end in zip(beat_times[:-1], beat_times[1:]):
            mask = (frame_times >= start) & (frame_times < end)
            if not np.any(mask):
                mask[int(np.argmin(np.abs(frame_times - start)))] = True
            # Median resists transient harmonic spikes; mean keeps sustained tones.
            median_part = np.median(feature[:, mask], axis=1)
            mean_part = np.mean(feature[:, mask], axis=1)
            rows.append(_normalize_vector(0.65 * mean_part + 0.35 * median_part, norm=norm))
        return np.vstack(rows)


class ChordHMM:
    """Key-aware HMM for beat-synchronous chord decoding."""

    def __init__(
        self,
        states: list[ChordState],
        key_root: int,
        key_mode: str,
        emission_temperature: float = 7.0,
        bass_weight: float = 1.1,
    ) -> None:
        self.states = states
        self.key_root = key_root
        self.key_mode = key_mode
        self.emission_temperature = emission_temperature
        self.bass_weight = bass_weight
        self.log_initial = self._build_initial_log_probabilities()
        self.log_transition = self._build_transition_log_probabilities()

    def emission_log_probabilities(
        self,
        chroma: np.ndarray,
        bass_chroma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert template similarity into log emission probabilities.

        Similarity is turned into a probability distribution with softmax.  The
        bass term boosts states whose root is supported in the low register, but
        does not dominate the harmonic chroma.
        """
        log_emissions = np.zeros((chroma.shape[0], len(self.states)), dtype=float)
        raw_scores = np.zeros_like(log_emissions)

        for t, (chroma_vec, bass_vec) in enumerate(zip(chroma, bass_chroma)):
            if np.linalg.norm(chroma_vec) < EPS:
                log_emissions[t] = -math.log(len(self.states))
                continue

            chord_energy = float(np.sum(chroma_vec))
            for s, state in enumerate(self.states):
                similarity = float(np.dot(chroma_vec, state.template))
                outside_penalty = 0.23 * float(np.sum(chroma_vec[state.template <= EPS]))
                root_bonus = 0.20 * float(chroma_vec[state.root])
                bass_root_bonus = self.bass_weight * float(bass_vec[state.root])
                diatonic_bonus = 0.18 if is_diatonic(state, self.key_root, self.key_mode) else -0.06

                # A silence/low-energy guard keeps tiny feature residues from
                # producing overconfident chord changes.
                energy_scale = min(1.0, chord_energy * 2.0)
                score = (
                    similarity
                    - outside_penalty
                    + root_bonus
                    + bass_root_bonus
                    + diatonic_bonus
                ) * energy_scale
                raw_scores[t, s] = score

            probabilities = _softmax(raw_scores[t] * self.emission_temperature)
            log_emissions[t] = np.log(probabilities + EPS)

        return log_emissions, raw_scores

    def viterbi(self, chroma: np.ndarray, bass_chroma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode the most likely state sequence and per-beat confidence."""
        log_emission, _raw_scores = self.emission_log_probabilities(chroma, bass_chroma)
        n_steps, n_states = log_emission.shape
        trellis = np.full((n_steps, n_states), -np.inf)
        backptr = np.zeros((n_steps, n_states), dtype=int)

        trellis[0] = self.log_initial + log_emission[0]
        for t in range(1, n_steps):
            scores = trellis[t - 1][:, None] + self.log_transition
            backptr[t] = np.argmax(scores, axis=0)
            trellis[t] = scores[backptr[t], np.arange(n_states)] + log_emission[t]

        path = np.zeros(n_steps, dtype=int)
        path[-1] = int(np.argmax(trellis[-1]))
        for t in range(n_steps - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]

        # Local posterior approximation.  It is not a full forward-backward
        # posterior, but it is useful for filtering uncertain chord regions.
        confidence = np.zeros(n_steps, dtype=float)
        for t in range(n_steps):
            posterior = _softmax(trellis[t])
            confidence[t] = float(posterior[path[t]])

        return path, confidence

    def _build_initial_log_probabilities(self) -> np.ndarray:
        """Prefer diatonic opening chords without making them mandatory."""
        scores = np.zeros(len(self.states), dtype=float)
        for i, state in enumerate(self.states):
            if is_diatonic(state, self.key_root, self.key_mode):
                scores[i] += 0.45
            if state.root == self.key_root:
                scores[i] += 0.30
            if state.quality in {"maj", "min"}:
                scores[i] += 0.10
        return np.log(_softmax(scores) + EPS)

    def _build_transition_log_probabilities(self) -> np.ndarray:
        """Build non-uniform transition matrix from musical heuristics."""
        n_states = len(self.states)
        scores = np.zeros((n_states, n_states), dtype=float)

        for i, prev in enumerate(self.states):
            prev_pcs = {(prev.root + interval) % 12 for interval in prev.intervals}
            for j, nxt in enumerate(self.states):
                next_pcs = {(nxt.root + interval) % 12 for interval in nxt.intervals}
                root_distance = min((nxt.root - prev.root) % 12, (prev.root - nxt.root) % 12)
                common_tones = len(prev_pcs & next_pcs)

                score = -0.25
                if i == j:
                    score += 3.20  # Chords usually persist across adjacent beats.
                if prev.root == nxt.root and prev.quality != nxt.quality:
                    score += 1.00  # C -> C7 or Csus4 -> C is common.
                if root_distance in {5, 7}:
                    score += 0.75  # Circle-of-fifths/fourths motion.
                if root_distance in {2, 3, 4}:
                    score += 0.22  # Stepwise/third motion is plausible.
                if root_distance == 6:
                    score -= 0.55  # Tritone root jumps are less common.

                score += 0.22 * common_tones

                if is_diatonic(nxt, self.key_root, self.key_mode):
                    score += 0.60
                elif key_scale_contains(nxt.root, self.key_root, self.key_mode):
                    score += 0.12
                else:
                    score -= 0.45

                if prev.quality in {"sus2", "sus4"} and nxt.root == prev.root:
                    score += 0.45
                if nxt.quality in {"dim", "aug"} and i != j:
                    score -= 0.20

                scores[i, j] = score

        log_transition = np.zeros_like(scores)
        for i in range(n_states):
            log_transition[i] = np.log(_softmax(scores[i]) + EPS)
        return log_transition


def decode_beats(
    features: BeatFeatures,
    states: list[ChordState],
    include_inversions: bool = False,
) -> tuple[list[DecodedBeat], tuple[int, str, float], ChordHMM]:
    """Estimate key and decode one HMM state per beat interval."""
    key = estimate_key(np.mean(features.chroma, axis=0))
    key_root, key_mode, _key_score = key
    hmm = ChordHMM(states, key_root=key_root, key_mode=key_mode)
    path, confidence = hmm.viterbi(features.chroma, features.bass_chroma)

    decoded: list[DecodedBeat] = []
    for i, state_index in enumerate(path):
        state = states[int(state_index)]
        chord = state.label
        if include_inversions:
            bass_pc = int(np.argmax(features.bass_chroma[i]))
            bass_strength = float(features.bass_chroma[i, bass_pc])
            chord_pcs = {(state.root + interval) % 12 for interval in state.intervals}
            if bass_strength > 0.35 and bass_pc in chord_pcs and bass_pc != state.root:
                chord = f"{chord}/{NOTES[bass_pc]}"
        decoded.append(
            DecodedBeat(
                start=float(features.beat_times[i]),
                end=float(features.beat_times[i + 1]),
                state_index=int(state_index),
                chord=chord,
                confidence=float(confidence[i]),
            )
        )

    return decoded, key, hmm


def naive_decode(
    features: BeatFeatures,
    states: list[ChordState],
    key_root: int,
    key_mode: str,
) -> list[DecodedBeat]:
    """Decode each beat independently for debugging against the HMM."""
    hmm = ChordHMM(states, key_root=key_root, key_mode=key_mode)
    log_emission, _raw_scores = hmm.emission_log_probabilities(features.chroma, features.bass_chroma)
    decoded: list[DecodedBeat] = []
    for i in range(log_emission.shape[0]):
        posterior = _softmax(log_emission[i])
        state_index = int(np.argmax(posterior))
        decoded.append(
            DecodedBeat(
                start=float(features.beat_times[i]),
                end=float(features.beat_times[i + 1]),
                state_index=state_index,
                chord=states[state_index].label,
                confidence=float(posterior[state_index]),
            )
        )
    return decoded


def beats_to_segments(beats: list[DecodedBeat]) -> list[dict]:
    """Collapse equal adjacent beat labels into chord segments."""
    if not beats:
        return []

    segments = []
    current = {
        "start": beats[0].start,
        "end": beats[0].end,
        "chord": beats[0].chord,
        "confidence_values": [beats[0].confidence],
    }
    for beat in beats[1:]:
        if beat.chord == current["chord"]:
            current["end"] = beat.end
            current["confidence_values"].append(beat.confidence)
        else:
            segments.append(current)
            current = {
                "start": beat.start,
                "end": beat.end,
                "chord": beat.chord,
                "confidence_values": [beat.confidence],
            }
    segments.append(current)
    return [_finalize_segment(segment) for segment in segments]


def _finalize_segment(segment: dict) -> dict:
    """Convert an internal segment into the public output shape."""
    confidence_values = segment.pop("confidence_values")
    segment["confidence"] = float(np.mean(confidence_values))
    return segment


def post_process_segments(
    segments: list[dict],
    min_duration: float = 0.5,
    min_confidence: float = 0.0,
    timeline_end: float | None = None,
) -> list[dict]:
    """Merge short or weak chord regions into the most plausible neighbor."""
    if not segments:
        return []

    cleaned = [dict(segment) for segment in segments if segment["confidence"] >= min_confidence]
    if not cleaned:
        return []

    changed = True
    while changed and len(cleaned) > 1:
        changed = False
        merged: list[dict] = []
        i = 0
        while i < len(cleaned):
            current = cleaned[i]
            duration = current["end"] - current["start"]
            if duration < min_duration:
                changed = True
                if merged and i + 1 < len(cleaned):
                    prev_duration = merged[-1]["end"] - merged[-1]["start"]
                    next_duration = cleaned[i + 1]["end"] - cleaned[i + 1]["start"]
                    if next_duration > prev_duration and cleaned[i + 1]["chord"] != merged[-1]["chord"]:
                        cleaned[i + 1]["start"] = current["start"]
                    else:
                        merged[-1]["end"] = current["end"]
                        merged[-1]["confidence"] = min(merged[-1]["confidence"], current["confidence"])
                elif merged:
                    merged[-1]["end"] = current["end"]
                    merged[-1]["confidence"] = min(merged[-1]["confidence"], current["confidence"])
                elif i + 1 < len(cleaned):
                    cleaned[i + 1]["start"] = current["start"]
                i += 1
                continue
            merged.append(current)
            i += 1
        cleaned = _merge_equal_neighbors(merged)

    cleaned = _ensure_continuous_timeline(cleaned, timeline_end=timeline_end)

    return [
        {
            "start": round(float(segment["start"]), 2),
            "end": round(float(segment["end"]), 2),
            "chord": str(segment["chord"]),
            "confidence": round(float(np.clip(segment["confidence"], 0.0, 1.0)), 3),
        }
        for segment in cleaned
        if segment["end"] > segment["start"]
    ]


def _ensure_continuous_timeline(
    segments: list[dict],
    timeline_end: float | None = None,
    tolerance: float = 0.02,
) -> list[dict]:
    """
    Guarantee that the public output has no silent holes between segments.

    Filtering by confidence or aggressive short-segment merging can otherwise
    leave timeline gaps.  A chord viewer is easier to reason about when every
    second is covered, so gaps are assigned to the stronger neighboring chord.
    """
    if not segments:
        return []

    ordered = sorted((dict(segment) for segment in segments), key=lambda item: item["start"])
    repaired: list[dict] = [ordered[0]]

    for segment in ordered[1:]:
        previous = repaired[-1]
        gap = float(segment["start"]) - float(previous["end"])
        if gap > tolerance:
            prev_conf = float(previous.get("confidence", 0.0))
            next_conf = float(segment.get("confidence", 0.0))
            if next_conf > prev_conf:
                segment["start"] = previous["end"]
            else:
                previous["end"] = segment["start"]
        elif gap < -tolerance:
            midpoint = (float(previous["end"]) + float(segment["start"])) / 2.0
            previous["end"] = midpoint
            segment["start"] = midpoint
        else:
            segment["start"] = previous["end"]
        repaired.append(segment)

    if timeline_end is not None and repaired:
        end_gap = float(timeline_end) - float(repaired[-1]["end"])
        if end_gap > tolerance:
            repaired[-1]["end"] = float(timeline_end)

    return _merge_equal_neighbors(repaired)


def _merge_equal_neighbors(segments: list[dict]) -> list[dict]:
    """Merge adjacent segments with the same chord label."""
    merged: list[dict] = []
    for segment in segments:
        if merged and merged[-1]["chord"] == segment["chord"]:
            old_duration = merged[-1]["end"] - merged[-1]["start"]
            new_duration = segment["end"] - segment["start"]
            total_duration = max(old_duration + new_duration, EPS)
            merged[-1]["confidence"] = (
                merged[-1]["confidence"] * old_duration + segment["confidence"] * new_duration
            ) / total_duration
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(dict(segment))
    return merged


def detect_chords_hmm(
    file_path: str,
    min_duration: float = 0.5,
    min_confidence: float = 0.0,
    chroma_kind: str = "cqt",
    prefer_madmom: bool = True,
    include_inversions: bool = False,
    compare_naive: bool = False,
) -> tuple[list[dict], float]:
    """Run the full beat-synchronous HMM chord recognition pipeline."""
    print("Running beat-synchronous HMM chord detector...")
    extractor = BeatSynchronousFeatureExtractor(
        prefer_madmom=prefer_madmom,
        chroma_kind=chroma_kind,
    )
    features = extractor.extract(file_path)
    states = build_chord_states()
    decoded, key, hmm = decode_beats(features, states, include_inversions=include_inversions)

    key_root, key_mode, key_score = key
    print(
        f"   Estimated key: {NOTES[key_root]} {key_mode} "
        f"(score {key_score:.2f}) | Tempo: {features.tempo:.1f} BPM"
    )

    if compare_naive:
        naive = naive_decode(features, states, key_root, key_mode)
        naive_segments = post_process_segments(
            beats_to_segments(naive),
            min_duration=0.0,
            timeline_end=features.duration,
        )
        hmm_segments = post_process_segments(
            beats_to_segments(decoded),
            min_duration=0.0,
            timeline_end=features.duration,
        )
        print("\nNaive beat decisions:")
        _print_segment_preview(naive_segments)
        print("\nHMM beat decisions:")
        _print_segment_preview(hmm_segments)

    segments = beats_to_segments(decoded)
    return post_process_segments(
        segments,
        min_duration,
        min_confidence,
        timeline_end=features.duration,
    ), features.tempo


def chords_to_dicts(chords: list[dict] | list[tuple]) -> list[dict]:
    """Return the public JSON shape used by server.py and the web UI."""
    if not chords:
        return []
    first = chords[0]
    if isinstance(first, dict):
        return [
            {
                "start": round(float(chord["start"]), 2),
                "end": round(float(chord["end"]), 2),
                "chord": str(chord["chord"]),
                "confidence": round(float(chord.get("confidence", 0.0)), 3),
            }
            for chord in chords
        ]
    return [
        {
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "chord": str(chord),
            "confidence": round(float(confidence), 3) if len(item) > 3 else 0.0,
        }
        for item in chords
        for start, end, chord, *confidence in [item]
    ]


def write_json(chords: list[dict], output_path: str) -> None:
    """Write chord segments as UTF-8 JSON."""
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chords_to_dicts(chords), file, ensure_ascii=False, indent=2)


def _print_segment_preview(segments: list[dict], limit: int = 18) -> None:
    """Print a compact segment list for debugging."""
    for segment in segments[:limit]:
        print(
            f"  {segment['start']:6.2f}s - {segment['end']:6.2f}s  "
            f"{segment['chord']:<9} conf={segment['confidence']:.2f}"
        )
    if len(segments) > limit:
        print(f"  ... {len(segments) - limit} more")


def print_chords(chords: list[dict]) -> None:
    """Pretty-print chord segments."""
    print("\nChords detected:\n")
    _print_segment_preview(chords, limit=9999)
    print(f"\n  Total: {len(chords)} chord segments")


def main() -> None:
    parser = argparse.ArgumentParser(description="Beat-synchronous HMM chord detector")
    parser.add_argument("file", help="Audio file to analyze")
    parser.add_argument("--json", dest="json_path", help="Write output to JSON")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Merge chord segments shorter than this many seconds",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop segments below this confidence before short-segment merging",
    )
    parser.add_argument(
        "--chroma",
        choices=["cqt", "cens"],
        default="cqt",
        help="Chroma representation. cqt is detailed; cens is more smoothed/noise-robust.",
    )
    parser.add_argument(
        "--no-madmom",
        action="store_true",
        help="Use librosa beat tracking instead of trying madmom first",
    )
    parser.add_argument(
        "--include-inversions",
        action="store_true",
        help="Append slash-bass labels when the bass strongly supports an inversion",
    )
    parser.add_argument(
        "--compare-naive",
        action="store_true",
        help="Print independent beat decisions next to HMM-smoothed output",
    )
    args = parser.parse_args()

    chords, tempo = detect_chords_hmm(
        args.file,
        min_duration=args.min_duration,
        min_confidence=args.min_confidence,
        chroma_kind=args.chroma,
        prefer_madmom=not args.no_madmom,
        include_inversions=args.include_inversions,
        compare_naive=args.compare_naive,
    )

    print(f"   Estimated tempo: {tempo:.1f} BPM")
    if args.json_path:
        write_json(chords, args.json_path)
        print(f"   Wrote {args.json_path}")
    print_chords(chords)


if __name__ == "__main__":
    main()
