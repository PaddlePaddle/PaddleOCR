# PaddleOCR Document Parsing Output Schema

This document defines the output format returned by `vl_caller.py`, based on the actual PaddleOCR Document Parsing API response.

## Output Structure

`vl_caller.py` wraps the raw API response in a unified envelope:

```json
{
  "ok": true,
  "text": "Extracted markdown text from all pages",
  "result": {complete raw API response},
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

The `result` field contains the complete raw API response:

```json
{
  "logId": "request-uuid",
  "errorCode": 0,
  "errorMsg": "Success",
  "result": {
    "layoutParsingResults": [
      {
        "prunedResult": {
          "page_count": 1,
          "width": 1200,
          "height": 800,
          "model_settings": {...},
          "parsing_res_list": [...],
          "layout_det_res": {"boxes": [...]}
        },
        "markdown": {
          "text": "Full page content in markdown/HTML format",
          "images": {"imgs/filename.jpg": "https://..."}
        },
        "outputImages": {"layout_det_res": "https://..."},
        "inputImage": "https://..."
      }
    ],
    "dataInfo": {
      "numPages": 1,
      "pages": [{"width": 1200, "height": 800}],
      "type": "pdf"
    },
    "preprocessedImages": ["https://..."]
  }
}
```

### `prunedResult.parsing_res_list`

Array of detected content blocks:

```json
{
  "block_label": "text",
  "block_content": "Paragraph text content here...",
  "block_bbox": [100, 200, 500, 230],
  "block_id": 3,
  "group_id": 3,
  "block_order": 2,
  "block_polygon_points": [[100.0, 200.0], [500.0, 200.0], [500.0, 230.0], [100.0, 230.0]]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `block_label` | string | Block type (see Block Labels below) |
| `block_content` | string | Text content or HTML (tables use `<table>` HTML) |
| `block_bbox` | number[4] | Bounding box `[x1, y1, x2, y2]` |
| `block_id` | number | Unique block identifier within page |
| `group_id` | number | Group identifier for related blocks |
| `block_order` | number | Reading order index (optional, not all blocks have it) |
| `block_polygon_points` | number[][] | Polygon coordinates for non-rectangular regions |

### Block Labels

| Label | Description |
|-------|-------------|
| `text` | Regular text content |
| `table` | Table (content is HTML `<table>`) |
| `image` | Embedded image |
| `seal` | Seal or stamp |
| `figure_title` | Figure/chart title or caption |
| `vision_footnote` | Footnote detected by vision model |
| `aside_text` | Side/margin text |

### `prunedResult.layout_det_res.boxes`

Layout detection results with confidence scores:

```json
{
  "cls_id": 22,
  "label": "text",
  "score": 0.877,
  "coordinate": [879, 66, 1142, 91],
  "order": 2,
  "polygon_points": [[879.0, 66.0], [1142.0, 66.0], [1142.0, 91.0], [879.0, 91.0]]
}
```

### `markdown`

| Field | Type | Description |
|-------|------|-------------|
| `markdown.text` | string | Full page content rendered as markdown/HTML |
| `markdown.images` | object | Map of image filenames to URLs |

---

## Text Extraction

`vl_caller.py` extracts text using this priority:

1. `markdown.text` — preferred (full page content in reading order)
2. `prunedResult.parsing_res_list` — fallback (concatenate `block_content` of all blocks)

For multi-page documents, text from each page is joined with `\n\n`.

## Usage

```python
import json, subprocess

result = subprocess.run(
    ["python", "scripts/paddleocr-doc-parsing/vl_caller.py", "--file-url", "URL", "--pretty"],
    capture_output=True, text=True
)
data = json.loads(result.stdout)

if data["ok"]:
    # Quick: use extracted text
    print(data["text"])

    # Detailed: iterate blocks
    for page in data["result"]["result"]["layoutParsingResults"]:
        for block in page["prunedResult"]["parsing_res_list"]:
            print(f"[{block['block_label']}] {block['block_content'][:50]}")
else:
    print(f"Error: {data['error']['message']}")
```

## Command Line

```bash
# Parse document
python scripts/paddleocr-doc-parsing/vl_caller.py --file-url "URL" --pretty

# Parse local file
python scripts/paddleocr-doc-parsing/vl_caller.py --file-path "doc.pdf" --pretty

# Save to file
python scripts/paddleocr-doc-parsing/vl_caller.py --file-url "URL" --output result.json
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (check `error` field) |
