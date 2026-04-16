# ChordLab

ChordLab is a small web app for playing a song and viewing its chords on a timeline.

The app can load:
- an audio file
- a chord JSON file
- a lyrics JSON file


Live site:

[https://chords-chi.vercel.app/]

## Project Structure

```text
index.html              Page structure
style.css               Site styling
app.js                  Player, chord timeline, lyrics display

media/
  hero-video.mp4        Landing video
  audio/                Local audio files

data/
  chords/               Generated chord JSON files
  lyrics/               Lyrics and aligned lyrics JSON files

chord_detector_madmom.py
chord_detector_pro.py
chord_detector_basic.py
chord_detector_bp.py
align_lyrics_auto.py
server.py
```


## Main Tools Used

- HTML, CSS, and vanilla JavaScript for the frontend.
- Python for audio analysis scripts.
- librosa for chroma features, beat tracking, and audio utilities.
- madmom for chord recognition and beat tracking experiments.
- Basic Pitch for note-based chord detection experiments.
- NumPy for signal and vector calculations.
- Vercel for hosting the frontend.

## Detector Scripts

### `chord_detector_madmom.py`

Madmom-only chord detection. This is the cleanest current detector.

```powershell
.venv\Scripts\python.exe chord_detector_madmom.py "media/audio/song2.mp3" --json "data/chords/song2_chords_madmom_only.json" --min-duration 0.5
```

### `chord_detector_pro.py`

Hybrid detector. Tries madmom first, then falls back to Basic Pitch.

```powershell
.venv\Scripts\python.exe chord_detector_pro.py "media/audio/song2.mp3" --json "data/chords/song2_chords_pro.json" --engine auto --beats 4 --min-duration 0.5
```

### `chord_detector_basic.py`

Fast librosa/chroma-based detector.

### `chord_detector_bp.py`

Basic Pitch-based detector.

## Lyrics Alignment

`align_lyrics_auto.py` creates estimated line timestamps from an audio file and an existing lyrics JSON file.

```powershell
.venv\Scripts\python.exe align_lyrics_auto.py "media/audio/song.mp3" "data/lyrics/lyrics.json" --json "data/lyrics/lyrics_aligned_auto.json" --first-line-start 16
```

This is only an estimated alignment. A future version should use proper forced alignment.

## Future Work

- Add a real backend for automatic chord detection from uploaded audio.
- Add detector selection in the UI.
- Add job status for long audio processing.
- Add automatic lyrics alignment with Whisper or WhisperX.
- Add optional YouTube URL support through a backend.
- Add manual editing for chords and lyric timing.
