---
name: paddleocr-doc-parsing
description: >-
  Use this skill to extract structured Markdown/JSON from PDFs and document images—tables with
  cell-level precision, formulas as LaTeX, figures, seals, charts, headers/footers, multi-column
  layout and correct reading order. Supports local PaddleOCR-VL 1.8 and cloud API parsing.
  Trigger terms: 文档解析, 版面分析, 版面还原, 表格提取, 公式识别, 多栏排版, 扫描件结构化,
  发票, 财报, 复杂 PDF, PDF转Markdown, 图表, 阅读顺序; reading order, formula, LaTeX,
  layout parsing, structure extraction, PP-StructureV3, PaddleOCR-VL.
license: Apache-2.0
metadata:
  openclaw:
    requires:
      bins:
        - paddleocr
    primaryEnv: PADDLEOCR_ACCESS_TOKEN
    emoji: "📄"
    install:
      - kind: uv
        package: paddleocr
        bins: [paddleocr]
---

# PaddleOCR Document Parsing

## When to Use This Skill

**Use this skill for**:

- Documents with tables (invoices, financial reports, spreadsheets)
- Documents with mathematical formulas (academic papers, scientific documents)
- Documents with charts and diagrams
- Multi-column layouts (newspapers, magazines, brochures)
- Complex document structures requiring layout analysis

## Usage

Use local inference when the user requests local or offline processing, or explicitly asks for PaddleOCR-VL 1.8. For other requests, use the cloud API.

### Local PaddleOCR-VL 1.8

Use a local PDF or image and install `paddleocr[doc-parser]>=3.8.0` with a [supported inference engine](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html). Local inference requires no API token. For offline use, make the model files available locally in advance. If the local runtime or input file is unavailable, report what is missing and keep the task local.

```bash
paddleocr doc_parser -i "./document.pdf" --pipeline_version v1.8 --save_path "./output"
```

Read the Markdown or JSON files in the output directory. For a PDF, present the pages in document order. This pipeline uses PP-DocLayoutV4. To enable optional layout deformation correction, add `--use_layout_deform True`.

### Cloud API

Set `PADDLEOCR_ACCESS_TOKEN`. Passing a local file path to `paddleocr api` uploads that file to the configured API service.

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

#### Common Options

```bash
# With specific model
paddleocr api \
  --model_type doc_parsing \
  --model PP-StructureV3 \
  --file_path "./report.pdf"

# Disable preprocessing (faster, for flat/well-oriented images)
paddleocr api \
  --model_type doc_parsing \
  --file_path "./document.pdf" \
  --use_doc_unwarping False \
  --use_doc_orientation_classify False

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

# Prettify markdown output
paddleocr api \
  --model_type doc_parsing \
  --file_path "./document.pdf" \
  --prettify_markdown True
```

#### Output Format

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

## Important Notes

**Cloud preprocessing options**: For flat, well-oriented images (screenshots, properly scanned documents), you can disable preprocessing for faster results:

```bash
paddleocr api --model_type doc_parsing --file_path "./document.pdf" --use_doc_unwarping False --use_doc_orientation_classify False
```

Keep preprocessing enabled when:
- The input is a photo of a curved or folded document
- The document has significant perspective distortion
- Orientation is uncertain (rotated 90/180/270 degrees)

**Display complete results**: Always show the full extracted content to users. Do not truncate with "..." unless content exceeds 10,000 characters. When multiple pages are processed, summarize if needed but provide complete results when explicitly requested.

**Handle errors gracefully**: When the CLI returns an error, inform the user of the specific issue rather than silently failing. Common errors:
- Authentication: `PADDLEOCR_ACCESS_TOKEN` invalid or missing
- Quota: API rate limit exceeded
- No content detected: Document may be blank or contain no extractable text

## CLI Reference

Run `paddleocr doc_parser --help` for local inference options or `paddleocr api --help` for cloud API options.

For full documentation, see: [PaddleOCR Official Documentation](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/serving/paddleocr_official_api/cli.html)
