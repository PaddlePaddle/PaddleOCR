---
name: paddleocr-text-recognition
description: >-
  Use this skill whenever the user wants text extracted from images, photos, scans, screenshots,
  or scanned PDFs. Returns exact machine-readable strings with line-level text and optional bbox
  coordinates. Strong accuracy for CJK, small print, and handwritten text.
  Trigger terms: OCR, 文字识别, 图片转文字, 截图识字, 提取图中文字, 扫描识字, 识字, 纯文字,
  plain text extraction, 坐标, 检测框, bbox, bounding box, image to text, screenshot, photo scan,
  recognize text.
license: Apache-2.0
metadata:
  openclaw:
    requires:
      env:
        - PADDLEOCR_ACCESS_TOKEN
      bins:
        - paddleocr
    primaryEnv: PADDLEOCR_ACCESS_TOKEN
    emoji: "🔤"
    install:
      - kind: uv
        package: paddleocr
        bins: [paddleocr]
---

# PaddleOCR Text Recognition

## When to Use This Skill

**Use this skill for**:

- Extract text from images (screenshots, photos, scans)
- Extract text from PDFs or document images when the goal is **line/box-level text**
- Extract text from URLs or local files that point to images/PDFs

**Do not use for**:

- Documents with tables, formulas, charts, or complex layouts — use Document Parsing instead

## Usage

### Basic OCR

From URL:

```bash
paddleocr api \
  --model_type ocr \
  --file_url "https://example.com/image.png"
```

From local file:

```bash
paddleocr api \
  --model_type ocr \
  --file_path "./document.pdf"
```

### Common Options

```bash
# With specific model
paddleocr api \
  --model_type ocr \
  --model PP-OCRv5 \
  --file_path "./report.pdf"

# Disable preprocessing (faster, for flat/well-oriented images)
paddleocr api \
  --model_type ocr \
  --file_path "./document.pdf" \
  --use_doc_unwarping False \
  --use_doc_orientation_classify False

# Save result to file
paddleocr api \
  --model_type ocr \
  --file_url "https://..." \
  --output result.json

# Page ranges
paddleocr api \
  --model_type ocr \
  --file_path "./large.pdf" \
  --page_ranges "1-5,10,15-20"
```

### Output Format

```json
{
  "jobId": "job-xxx",
  "pages": [
    {
      "prunedResult": {
        "rec_texts": ["Line 1", "Line 2"],
        "rec_scores": [0.98, 0.95]
      },
      "ocrImageUrl": "https://..."
    }
  ]
}
```

## Important Notes

**Preprocessing options**: By default, the API enables document preprocessing (unwarping and orientation classification). For flat, well-oriented images (screenshots, properly scanned documents), you can disable preprocessing for faster results:

```bash
paddleocr api --model_type ocr --file_path "./document.pdf" --use_doc_unwarping False --use_doc_orientation_classify False
```

Keep preprocessing enabled when:
- The input is a photo of a curved or folded document
- The document has significant perspective distortion
- Orientation is uncertain (rotated 90/180/270 degrees)

**Display complete results**: Always show the full extracted content to users. Do not truncate with "..." unless content exceeds 10,000 characters. When multiple pages are processed, summarize if needed but provide complete results when explicitly requested.

**Handle errors gracefully**: When the CLI returns an error, inform the user of the specific issue rather than silently failing or falling back to your own vision capabilities. Common errors:
- Authentication: `PADDLEOCR_ACCESS_TOKEN` invalid or missing
- Quota: API rate limit exceeded
- No content detected: Image may be blank or contain no text

## CLI Reference

Run `paddleocr api --help` for all options.

For full documentation, see: [PaddleOCR Official Documentation](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/serving/paddleocr_official_api/cli.html)
