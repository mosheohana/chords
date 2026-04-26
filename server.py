"""
server.py  –  ChordLab backend (async job processing)

Endpoints
---------
GET  /api/health              Liveness probe
POST /api/analyze-audio       Upload audio → returns {jobId} immediately
GET  /api/job/<id>            Poll job status → pending / processing / done / error

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
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
UPLOADS_DIR = GENERATED_DIR / "uploads"
DEFAULT_PORT = 8001

# In-memory job store: {job_id: {status, chords, error, created_at}}
_jobs: dict = {}
_jobs_lock = threading.Lock()

JOB_TTL = 60 * 30  # clean up jobs older than 30 minutes


def _log(message: str):
    print(f"[server] {message}", flush=True)


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, file_path: str, detector: str, beats_per_chord: int, min_duration: float):
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    started_at = time.time()
    _log(
        f"job {job_id} started detector={detector} "
        f"beats_per_chord={beats_per_chord} min_duration={min_duration} "
        f"file={Path(file_path).name} size_mb={file_size_mb:.2f}"
    )
    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"

    try:
        chords, tempo = _run_detector(file_path, detector, beats_per_chord, min_duration)
        import numpy as np
        tempo_val = float(np.asarray(tempo).mean()) if tempo is not None else None
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "done",
                "chords": chords,
                "tempo": tempo_val,
                "detector": detector,
            })
        elapsed = time.time() - started_at
        _log(
            f"job {job_id} finished status=done detector={detector} "
            f"segments={len(chords)} tempo={tempo_val} elapsed_s={elapsed:.2f}"
        )
    except Exception as exc:
        elapsed = time.time() - started_at
        _log(
            f"job {job_id} failed detector={detector} "
            f"error={exc!r} elapsed_s={elapsed:.2f}\n{traceback.format_exc()}"
        )
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "error",
                "error": str(exc),
            })


def _run_detector(file_path, detector, beats_per_chord, min_duration):
    if detector == "hmm":
        from chord_detector_hmm import detect_chords_hmm, chords_to_dicts
        chords, tempo = detect_chords_hmm(
            file_path,
            min_duration=min_duration,
            min_confidence=0.0,
            chroma_kind="cqt",
            prefer_madmom=True,
            include_inversions=False,
            compare_naive=False,
        )
        return chords_to_dicts(chords), tempo

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
        from chord_detector_bp import get_notes_from_audio, detect_chords_by_beats, chords_to_dicts
        note_events = get_notes_from_audio(file_path)
        chords, tempo = detect_chords_by_beats(
            file_path, note_events, beats_per_chord=beats_per_chord
        )
        return chords_to_dicts(chords), tempo

    # default: "basic"
    from chord_detector_basic import detect_chords_by_beats, chords_to_dicts
    chords, tempo = detect_chords_by_beats(file_path, beats_per_chord=beats_per_chord)
    return chords_to_dicts(chords), tempo


def _cleanup_old_jobs():
    while True:
        time.sleep(300)  # every 5 minutes
        cutoff = time.time() - JOB_TTL
        with _jobs_lock:
            stale = [jid for jid, j in _jobs.items() if j.get("created_at", 0) < cutoff]
            for jid in stale:
                del _jobs[jid]


threading.Thread(target=_cleanup_old_jobs, daemon=True).start()


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ChordRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/api/health":
            self.send_json({"status": "ok", "version": "async-v1"})
            return

        if path.startswith("/api/job/"):
            job_id = path[len("/api/job/"):]
            self.handle_job_status(job_id)
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/analyze-audio":
            self.handle_analyze_audio()
            return
        self.send_json({"error": "Not found"}, status=404)

    # ------------------------------------------------------------------
    # Upload → start job
    # ------------------------------------------------------------------

    def handle_analyze_audio(self):
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_json({"error": "Expected multipart/form-data"}, status=400)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
        )

        file_item = form["audio"] if "audio" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            self.send_json({"error": "Missing audio file"}, status=400)
            return

        detector       = (form.getfirst("detector", "madmom") or "madmom").strip().lower()
        beats_per_chord = self.parse_int(form, "beatsPerChord", default=4, minimum=1)
        min_duration    = self.parse_float(form, "minDuration", default=0.5, minimum=0.1)

        upload_path = self.save_upload(file_item)
        file_size_mb = upload_path.stat().st_size / (1024 * 1024)
        job_id = uuid.uuid4().hex
        _log(
            f"received upload job={job_id} original_name={file_item.filename!r} "
            f"saved_name={upload_path.name} size_mb={file_size_mb:.2f} detector={detector}"
        )

        with _jobs_lock:
            _jobs[job_id] = {
                "status": "pending",
                "created_at": time.time(),
                "audioUrl": f"/generated/uploads/{upload_path.name}",
                "fileName": upload_path.name,
            }

        threading.Thread(
            target=_run_job,
            args=(job_id, str(upload_path), detector, beats_per_chord, min_duration),
            daemon=True,
        ).start()

        self.send_json({"jobId": job_id})

    # ------------------------------------------------------------------
    # Poll job status
    # ------------------------------------------------------------------

    def handle_job_status(self, job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)

        if job is None:
            _log(f"job status requested but missing job={job_id}")
            self.send_json({"error": "Job not found"}, status=404)
            return

        self.send_json(job)

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
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = int(os.environ.get("PORT", DEFAULT_PORT))
    server = ThreadingHTTPServer(("0.0.0.0", port), ChordRequestHandler)
    _log(f"ChordLab API listening on port {port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
