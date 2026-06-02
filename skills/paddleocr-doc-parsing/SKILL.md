---
name: paddleocr-doc-parsing
description: >-
  Use this skill to extract structured Markdown/JSON from PDFs and document images—tables with
  cell-level precision, formulas as LaTeX, figures, seals, charts, headers/footers, multi-column
  layout and correct reading order.
  Trigger terms: 文档解析, 版面分析, 版面还原, 表格提取, 公式识别, 多栏排版, 扫描件结构化,
  发票, 财报, 复杂 PDF, PDF转Markdown, 图表, 阅读顺序; reading order, formula, LaTeX,
  layout parsing, structure extraction, PP-StructureV3, PaddleOCR-VL.
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
    emoji: "📄"
---

# PaddleOCR Document Parsing

## When to Use This Skill

**Use this skill for**:

- Documents with tables (invoices, financial reports, spreadsheets)
- Documents with mathematical formulas (academic papers, scientific documents)
- Documents with charts and diagrams
- Multi-column layouts (newspapers, magazines, brochures)
- Complex document structures requiring layout analysis

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

### Basic Document Parsing

From URL:

```bash
paddleocr api \
  --model_type doc_parsing \
  --file_url "https://example.com/report.pdf"
```

From local file:

```bash
paddleocr api \
  --model_type doc_parsing \
  --file_path "./document.pdf"
```

### With Specific Model

PP-StructureV3 (better for tables and formulas):

```bash
paddleocr api \
  --model_type doc_parsing \
  --model PP-StructureV3 \
  --file_path "./report.pdf" \
  --use_table_recognition \
  --use_formula_recognition
```

PaddleOCR-VL-1.6 (better for general documents, default):

```bash
paddleocr api \
  --model_type doc_parsing \
  --model PaddleOCR-VL-1.6 \
  --file_url "https://..." \
  --use_chart_recognition
```

### Common Options

```bash
# With page ranges
paddleocr api \
  --model_type doc_parsing \
  --file_path "./large.pdf" \
  --page_ranges "1-5,10,15-20"

# Save result and resources
paddleocr api \
  --model_type doc_parsing \
  --file_url "https://..." \
  --output result.json \
  --save_resources ./resources

# With layout detection
paddleocr api \
  --model_type doc_parsing \
  --file_path "./document.pdf" \
  --use_layout_detection \
  --use_seal_recognition

# Prettify markdown output
paddleocr api \
  --model_type doc_parsing \
  --file_path "./document.pdf" \
  --prettify_markdown
```

### Output Format

```json
{
  "jobId": "job-xxx",
  "pages": [
    {
      "markdownText": "# Title\n\nContent...",
      "markdownImages": {
        "img1": "https://...",
        "img2": "https://..."
      },
      "outputImages": {
        "layout1": "https://..."
      }
    }
  ]
}
```

## CLI Reference

For full documentation, see: [PaddleOCR 官方 API CLI](../../docs/version3.x/inference_deployment/serving/paddleocr_official_api/cli.md)

Run `paddleocr api --help` for all options.