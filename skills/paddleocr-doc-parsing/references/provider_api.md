# Provider API Reference: PaddleOCR Document Parsing

This document describes the external provider API contract that this skill depends on.

## Endpoint

**POST** `<PADDLEOCR_DOC_PARSING_API_URL>`

Where the URL is obtained from [PaddleOCR official website](https://paddleocr.com).

Example: `https://xxxxx.aistudio-app.com/layout-parsing`

## Authentication

**Header:**
```
Authorization: token <ACCESS_TOKEN>
```

Where `<ACCESS_TOKEN>` is the API token obtained from [PaddleOCR official website](https://paddleocr.com).

## Request Body

```json
{
  "file": "<base64_encoded_content>",
  "fileType": 0,
  "useDocOrientationClassify": false,
  "useDocUnwarping": false,
  "useChartRecognition": false
}
```

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | string | Base64-encoded file content |
| `fileType` | number | `0` = PDF, `1` = Image |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `useDocOrientationClassify` | boolean | `false` | Enable document orientation correction |
| `useDocUnwarping` | boolean | `false` | Enable document unwarping/skew correction |
| `useChartRecognition` | boolean | `false` | Enable chart recognition |

## Response Structure

### Success Response

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
          "parsing_res_list": [
            {
              "block_label": "text",
              "block_content": "Recognized text content",
              "block_bbox": [x1, y1, x2, y2],
              "block_id": 0,
              "block_order": 1,
              "group_id": 0,
              "block_polygon_points": [[x1, y1], ...]
            }
          ],
          "layout_det_res": {
            "boxes": [
              {
                "cls_id": 22,
                "label": "text",
                "score": 0.87,
                "coordinate": [x1, y1, x2, y2],
                "order": 1,
                "polygon_points": [[x1, y1], ...]
              }
            ]
          }
        },
        "markdown": {
          "text": "Full page content in markdown/HTML",
          "images": {"imgs/filename.jpg": "https://..."}
        },
        "outputImages": {
          "layout_det_res": "https://..."
        },
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

Key points:
- `result.layoutParsingResults` is an array with one object per page
- Each page contains `prunedResult` (structured blocks), `markdown` (rendered text), and image URLs
- `dataInfo` contains document metadata (page count, dimensions, file type)

### Error Response

HTTP status codes indicate errors:

| HTTP Status | Meaning | Mapped Error Code |
|-------------|---------|-------------------|
| 200 | Success | - |
| 401 | Authentication failed | `PROVIDER_AUTH_ERROR` |
| 403 | Access forbidden | `PROVIDER_AUTH_ERROR` |
| 429 | Quota/rate limit exceeded | `PROVIDER_QUOTA_EXCEEDED` |
| 503 | Service overloaded | `PROVIDER_OVERLOADED` |
| 504 | Gateway timeout | `PROVIDER_TIMEOUT` |
| Other | Unknown error | `PROVIDER_ERROR` |

## Supported File Formats

| Format | Extension | fileType |
|--------|-----------|----------|
| PDF | .pdf | 0 |
| PNG | .png | 1 |
| JPEG | .jpg, .jpeg | 1 |
| BMP | .bmp | 1 |
| TIFF | .tiff, .tif | 1 |
| WebP | .webp | 1 |

## Limitations

- **Maximum file size**: No limit
- **Maximum pages**: 100 pages per request
- **Timeout**: 10 minutes

## Best Practices

1. **Handle timeouts**: For large documents, processing may take several minutes
2. **Retry on 503/504**: Use exponential backoff
3. **Never log tokens**: Keep credentials secure

## Request Example

```bash
curl -X POST "https://xxxxx.aistudio-app.com/layout-parsing" \
  -H "Authorization: token YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'$(base64 -w 0 document.pdf)'",
    "fileType": 0,
    "useDocOrientationClassify": true,
    "useDocUnwarping": true
  }'
```
