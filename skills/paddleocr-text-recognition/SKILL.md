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
compatibility: Requires paddleocr>=3.6.0
metadata:
  openclaw:
    requires:
      env:
        - PADDLEOCR_ACCESS_TOKEN
      bins:
        - paddleocr
    primaryEnv: PADDLEOCR_ACCESS_TOKEN
    emoji: "🔤"
---

# PaddleOCR Text Recognition

## When to Use This Skill

**Use this skill for**:

- Extract text from images (screenshots, photos, scans)
- Extract text from PDFs or document images when the goal is **line/box-level text**
- Extract text from URLs or local files that point to images/PDFs

**Do not use for**:

- Documents with tables, formulas, charts, or complex layouts — use Document Parsing instead

## Installation

Install PaddleOCR 3.6.0+:

```bash
pip install "paddleocr>=3.6.0"
```

## Configuration

Get your access token from [AI Studio](https://aistudio.baidu.com/account/accessToken), then set environment variable:

```bash
export PADDLEOCR_ACCESS_TOKEN=your_token_here
```

**Optional**: set custom base URL (defaults to official service):

```bash
export PADDLEOCR_BASE_URL=https://paddleocr.aistudio-app.com
```

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
# With preprocessing options
paddleocr api \
  --model_type ocr \
  --file_path "./document.pdf" \
  --use_doc_unwarping \
  --use_doc_orientation_classify

# Save result to file
paddleocr api \
  --model_type ocr \
  --file_url "https://..." \
  --output result.json

# Save visualized images
paddleocr api \
  --model_type ocr \
  --file_path "./image.png" \
  --visualize

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

## CLI Reference

For full documentation, see: [PaddleOCR 官方 API CLI](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/cli.md)

Run `paddleocr api --help` for all options.
