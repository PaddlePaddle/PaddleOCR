#!/usr/bin/env python3
"""Local browser UI for uploading a scanned PDF and running PaddleOCR."""

from __future__ import annotations

import argparse
import html
import json
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_ocr = None
_ocr_lock = Lock()

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PaddleOCR PDF demo</title>
  <style>
    body { font: 16px/1.5 system-ui, sans-serif; max-width: 760px; margin: 8vh auto; padding: 0 20px; color: #172033; background: #f5f7fb; }
    main { background: white; border: 1px solid #e3e8f0; border-radius: 16px; padding: 32px; box-shadow: 0 12px 36px #17203312; }
    h1 { margin-top: 0; } label { display: block; font-weight: 600; margin: 18px 0 6px; }
    input, select, button { font: inherit; } input[type=file], select { width: 100%; padding: 10px; border: 1px solid #cbd3df; border-radius: 8px; background: white; }
    button { margin-top: 22px; background: #155eef; color: white; border: 0; border-radius: 8px; padding: 11px 18px; cursor: pointer; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f7fb; border-radius: 8px; padding: 16px; }
    .hint { color: #596579; font-size: .92rem; } .error { color: #b42318; }
  </style>
</head>
<body><main>
  <h1>Scanned PDF to text</h1>
  <p>Upload a scanned PDF to extract its text with PaddleOCR. This demo uses compact PP-OCRv6 models for CPU-friendly inference. The file is processed locally and is not retained.</p>
  <form method="post" enctype="multipart/form-data">
    <label for="pdf">PDF scan (max 20 MB)</label>
    <input id="pdf" name="pdf" type="file" accept="application/pdf,.pdf" required>
    <button type="submit">Run OCR</button>
  </form>
</main></body></html>"""


def _get_ocr():
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "PaddleOCR dependencies are missing. In a supported Python "
                    "environment, install PaddlePaddle with "
                    "`python -m pip install paddlepaddle`, then install this "
                    "checkout with `python -m pip install -e .`."
                ) from exc
            _ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv6_tiny_det",
                text_recognition_model_name="PP-OCRv6_tiny_rec",
                engine="paddle",
            )
        return _ocr


def _result_text(result) -> str:
    lines = []

    def visit(value):
        if isinstance(value, dict):
            texts = value.get("rec_texts")
            if isinstance(texts, list):
                lines.extend(str(text) for text in texts if text)
            for key, child in value.items():
                if key != "rec_texts":
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for page in result:
        payload = getattr(page, "json", None)
        if callable(payload):
            payload = payload()
        if payload is None and hasattr(page, "to_dict"):
            payload = page.to_dict()
        if isinstance(payload, str):
            payload = json.loads(payload)
        visit(payload)

    return "\n".join(lines)


class UploadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self._send_html(PAGE)

    def do_POST(self):
        if self.path != "/":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_error(413, "Upload must be between 1 byte and 20 MB")
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data;"):
            self.send_error(400, "Expected a multipart PDF upload")
            return

        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: "
            + content_type.encode("ascii", "strict")
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + self.rfile.read(length)
        )
        pdf_data = None
        filename = "uploaded.pdf"
        for part in message.iter_parts():
            field_name = part.get_param("name", header="content-disposition")
            data = part.get_payload(decode=True) or b""
            if field_name == "pdf":
                pdf_data = data
                filename = part.get_filename() or filename

        if not pdf_data or not filename.lower().endswith(".pdf") or not pdf_data.startswith(b"%PDF-"):
            self.send_error(400, "Please upload a valid PDF file")
            return

        import tempfile
        from pathlib import Path

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as upload:
                upload.write(pdf_data)
                upload.flush()
                result = _get_ocr().predict(upload.name)
            text = _result_text(result)
            if not text:
                text = "OCR completed, but no recognized text was returned."
            body = (
                "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                "<title>OCR result</title><style>body{font:16px/1.5 system-ui,sans-serif;"
                "max-width:760px;margin:8vh auto;padding:0 20px;color:#172033}"
                "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fb;"
                "padding:16px;border-radius:8px}a{color:#155eef}</style><h1>OCR result</h1>"
                f"<p>File: {html.escape(Path(filename).name)}</p>"
                f"<pre>{html.escape(text)}</pre><p><a href=\"/\">Process another PDF</a></p>"
                "</html>"
            )
            self._send_html(body)
        except Exception as exc:
            message = (
                "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
                "<title>OCR error</title><h1>OCR could not run</h1>"
                f"<p>{html.escape(str(exc))}</p><p><a href=\"/\">Try again</a></p></html>"
            )
            self._send_html(message, status=500)

    def _send_html(self, body: str, status: int = 200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), UploadHandler)
    print(f"PDF upload UI running at http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PDF upload UI.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
