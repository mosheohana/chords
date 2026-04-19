"""
server.py  –  ChordLab backend

Endpoints
---------
GET  /api/health           Liveness probe
POST /api/analyze-audio    Upload audio, get back chord JSON

Form fields for POST /api/analyze-audio
----------------------------------------
audio          (file)    Required. Any format librosa can read (mp3, wav, flac …)
detector       (str)     "basic" | "madmom" | "pro" | "bp"   default: "madmom"
beatsPerChord  (int)     Beats per chord window               default: 4
minDuration    (float)   Merge segments shorter than N s      default: 0.5

Run locally
-----------
    python server.py            # http://localhost:8001
    python server.py 9000       # custom port
"""

import collections
import collections.abc

# madmom 0.16.1 uses removed aliases from collections (Python 3.10+)
for _name in (
    "Callable", "Iterable", "Iterator", "Generator",
    "Mapping", "MutableMapping", "MutableSequence", "MutableSet",
    "Sequence", "Set",
):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

import cgi
import json
import mimetypes
import shutil
import sys
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
UPLOADS_DIR = GENERATED_DIR / "uploads"
DEFAULT_PORT = 8001

ALLOWED_ORIGINS = ["*"]          # tighten to your Vercel domain if you like


# ---------------------------------------------------------------------------
# CORS helpers
# ---------------------------------------------------------------------------

def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ChordRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_OPTIONS(self):
        """Pre-flight CORS request."""
        self.send_response(204)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/api/health":
            self.send_json({"status": "ok"})
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/analyze-audio":
            self.handle_analyze_audio()
            return
        self.send_json({"error": "Not found"}, status=404)

    # ------------------------------------------------------------------
    # Audio analysis
    # ------------------------------------------------------------------

    def handle_analyze_audio(self):
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_json({"error": "Expected multipart/form-data"}, status=400)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )

        file_item = form["audio"] if "audio" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            self.send_json({"error": "Missing audio file"}, status=400)
            return

        detector      = (form.getfirst("detector", "madmom") or "madmom").strip().lower()
        beats_per_chord = self.parse_int(form, "beatsPerChord", default=4, minimum=1)
        min_duration    = self.parse_float(form, "minDuration",  default=0.5, minimum=0.1)

        upload_path = self.save_upload(file_item)

        try:
            chords, tempo = self.run_detector(
                str(upload_path), detector, beats_per_chord, min_duration
            )
        except Exception as exc:
            self.send_json({"error": f"Could not analyze audio: {exc}"}, status=500)
            return

        import numpy as np
        self.send_json({
            "audioUrl":     f"/generated/uploads/{upload_path.name}",
            "fileName":     upload_path.name,
            "detector":     detector,
            "tempo":        float(np.asarray(tempo).mean()) if tempo is not None else None,
            "beatsPerChord": beats_per_chord,
            "chords":       chords,
        })

    # ------------------------------------------------------------------
    # Detector dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def run_detector(file_path, detector, beats_per_chord, min_duration):
        """Return (chords_as_dicts, tempo)."""

        if detector == "madmom":
            from chord_detector_madmom import detect_chords, chords_to_dicts
            chords = detect_chords(file_path, min_duration=min_duration)
            return chords_to_dicts(chords), None

        if detector == "pro":
            from chord_detector_pro import detect_with_madmom, chords_to_dicts
            chords, tempo = detect_with_madmom(
                file_path, min_dur=min_duration, beats_per_chord=beats_per_chord
            )
            return chords_to_dicts(chords), tempo

        if detector == "bp":
            from chord_detector_bp import (
                get_notes_from_audio, detect_chords_by_beats, chords_to_dicts
            )
            import numpy as np
            note_events = get_notes_from_audio(file_path)
            chords, tempo = detect_chords_by_beats(
                file_path, note_events, beats_per_chord=beats_per_chord
            )
            return chords_to_dicts(chords), tempo

        # default: "basic"
        from chord_detector_basic import detect_chords_by_beats, chords_to_dicts
        chords, tempo = detect_chords_by_beats(
            file_path, beats_per_chord=beats_per_chord
        )
        return chords_to_dicts(chords), tempo

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def save_upload(self, file_item):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(file_item.filename).suffix.lower() or ".audio"
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        upload_path = UPLOADS_DIR / safe_name
        with upload_path.open("wb") as out:
            shutil.copyfileobj(file_item.file, out)
        return upload_path

    def parse_int(self, form, field, default, minimum):
        try:
            return max(minimum, int(form.getfirst(field, default)))
        except (TypeError, ValueError):
            return default

    def parse_float(self, form, field, default, minimum):
        try:
            return max(minimum, float(form.getfirst(field, default)))
        except (TypeError, ValueError):
            return default

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path):
        mime, _ = mimetypes.guess_type(path)
        return mime or super().guess_type(path)

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(("0.0.0.0", port), ChordRequestHandler)
    print(f"ChordLab API listening on port {port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
