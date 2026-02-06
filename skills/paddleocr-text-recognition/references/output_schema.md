# PaddleOCR Text Recognition Output Schema

This document defines the output format returned by `ocr_caller.py`, based on the actual PaddleOCR Text Recognition API response.

## Output Structure

`ocr_caller.py` wraps the raw API response in a unified envelope:

```json
{
  "ok": true,
  "text": "Extracted text from all pages",
  "result": [raw API result array],
  "error": null
}
```

On error:

```json
{
  "ok": false,
  "text": "",
  "result": null,
  "error": {"code": "ERROR_CODE", "message": "Human-readable message"}
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `INPUT_ERROR` | Invalid input (missing file, unsupported format) |
| `CONFIG_ERROR` | API not configured |
| `API_ERROR` | API call failed (auth, timeout, server error) |

---

## Raw API Result Structure

The `result` field contains an array with one object per page:

```json
[
  {
    "prunedResult": {
      "model_settings": {...},
      "dt_polys": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
      "text_det_params": {
        "limit_side_len": 64,
        "limit_type": "min",
        "thresh": 0.3,
        "max_side_limit": 4000,
        "box_thresh": 0.6,
        "unclip_ratio": 1.5
      },
      "text_type": "general",
      "rec_texts": ["First line of text", "Second line of text", ...],
      "rec_scores": [0.98, 0.95, ...],
      "rec_polys": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
      "rec_boxes": [[x1, y1, x2, y2], ...]
    },
    "ocrImage": "https://...",
    "inputImage": "https://..."
  }
]
```

### `prunedResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `rec_texts` | string[] | Recognized text for each detected text line |
| `rec_scores` | number[] | Confidence score (0-1) for each text line |
| `rec_boxes` | number[][] | Bounding boxes `[x1, y1, x2, y2]` for each text line |
| `rec_polys` | number[][][] | Polygon coordinates `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` for each text line |
| `dt_polys` | number[][][] | Detection polygons (same format as `rec_polys`) |
| `text_det_params` | object | Text detection parameters used |
| `text_type` | string | Text type (e.g., `"general"`) |
| `model_settings` | object | Model configuration used for this request |

### Top-level Fields

| Field | Type | Description |
|-------|------|-------------|
| `ocrImage` | string | URL to visualization image (when `visualize: true`) |
| `inputImage` | string | URL to the input image used by the API |

---

## Text Extraction

`ocr_caller.py` extracts text using this logic:

1. Iterate each page in `result` array
2. Get `prunedResult.rec_texts` (array of recognized text strings)
3. Join text lines with `\n` within each page
4. Join pages with `\n\n`

## Usage

```python
import json, subprocess

result = subprocess.run(
    ["python", "scripts/paddleocr-text-recognition/ocr_caller.py", "--file-url", "URL", "--pretty"],
    capture_output=True, text=True
)
data = json.loads(result.stdout)

if data["ok"]:
    # Quick: use extracted text
    print(data["text"])

    # Detailed: iterate pages
    for page in data["result"]:
        texts = page["prunedResult"]["rec_texts"]
        scores = page["prunedResult"]["rec_scores"]
        for text, score in zip(texts, scores):
            print(f"[{score:.2f}] {text}")
else:
    print(f"Error: {data['error']['message']}")
```

## Command Line

```bash
# Basic OCR
python scripts/paddleocr-text-recognition/ocr_caller.py --file-url "URL" --pretty

# OCR local file
python scripts/paddleocr-text-recognition/ocr_caller.py --file-path "doc.pdf" --pretty

# Save to file
python scripts/paddleocr-text-recognition/ocr_caller.py --file-url "URL" --output result.json
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (check `error` field) |
