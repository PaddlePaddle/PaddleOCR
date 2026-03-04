# PaddleOCR Document Parsing Output Schema

This document defines the output envelope returned by `vl_caller.py`.

## Output Envelope

`vl_caller.py` wraps provider response in a stable structure:

```json
{
  "ok": true,
  "text": "Extracted text from all pages",
  "result": { ... },  // raw provider response
  "error": null
}
```

On error:

```json
{
  "ok": false,
  "text": "",
  "result": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message"
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `INPUT_ERROR` | Invalid input (missing file, unsupported format) |
| `CONFIG_ERROR` | API not configured |
| `API_ERROR` | API call failed (auth, timeout, service error, or invalid response schema) |

## Raw Result Notes

The `result` field contains raw provider output.  
Raw fields may vary by model version and endpoint.

## Raw Result Example

```json
{
  "logId": "request-uuid",
  "errorCode": 0,
  "errorMsg": "Success",
  "result": {
    "layoutParsingResults": [
      {
        "prunedResult": { ... },  // layout elements with position/content/confidence information
        "markdown": {
          "text": "Full page content in markdown/HTML format",
          "images": {
            "imgs/filename.jpg": "https://..."
          },
          "...": "other model-specific fields"
        }
      }
    ]
  }
}
```

## Important Fields

- `result[n].prunedResult`  
  Structured parsing data for page `n` (layout elements, locations, content, confidence, and related metadata).

- `result[n].markdown`  
  Rendered output for page `n`.

- `result[n].markdown.text`  
  Full page markdown text.

## Text Extraction

`vl_caller.py` extracts top-level `text` from `result.layoutParsingResults[n].markdown.text` and joins pages with `\n\n`.

## Command Examples

```bash
# Parse document from URL
python scripts/paddleocr-doc-parsing/vl_caller.py --file-url "URL" --pretty

# Parse local file
python scripts/paddleocr-doc-parsing/vl_caller.py --file-path "doc.pdf" --pretty

# Save result to file
python scripts/paddleocr-doc-parsing/vl_caller.py --file-url "URL" --output result.json
```
