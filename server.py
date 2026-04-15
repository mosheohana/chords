import cgi
import json
import mimetypes
import shutil
import sys
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chord_detector_basic import chords_to_dicts, detect_chords_by_beats


ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
UPLOADS_DIR = GENERATED_DIR / "uploads"
DEFAULT_PORT = 8001


class ChordRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path == "/api/analyze-audio":
            self.handle_analyze_audio()
            return

        self.send_json({"error": "Not found"}, status=404)

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
        if not file_item or not getattr(file_item, "filename", ""):
            self.send_json({"error": "Missing audio file"}, status=400)
            return

        beats_per_chord = self.parse_int(form, "beatsPerChord", default=4, minimum=1)
        upload_path = self.save_upload(file_item)

        try:
            chords, tempo = detect_chords_by_beats(
                str(upload_path),
                beats_per_chord=beats_per_chord,
            )
        except Exception as exc:
            self.send_json({"error": f"Could not analyze audio: {exc}"}, status=500)
            return

        self.send_json(
            {
                "audioUrl": f"/generated/uploads/{upload_path.name}",
                "fileName": upload_path.name,
                "tempo": float(tempo.mean()) if hasattr(tempo, "mean") else float(tempo),
                "beatsPerChord": beats_per_chord,
                "chords": chords_to_dicts(chords),
            }
        )

    def save_upload(self, file_item):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        original_name = Path(file_item.filename).name
        suffix = Path(original_name).suffix.lower() or ".audio"
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        upload_path = UPLOADS_DIR / safe_name

        with upload_path.open("wb") as output:
            shutil.copyfileobj(file_item.file, output)

        return upload_path

    def parse_int(self, form, field_name, default, minimum):
        try:
            value = int(form.getfirst(field_name, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, value)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path):
        mime_type, encoding = mimetypes.guess_type(path)
        if mime_type:
            return mime_type
        return super().guess_type(path)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(("localhost", port), ChordRequestHandler)
    print(f"Serving chord app at http://localhost:{port}/index.html")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
