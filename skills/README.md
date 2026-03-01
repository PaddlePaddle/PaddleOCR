# PaddleOCR Skills

This directory contains AI agent skills for PaddleOCR cloud APIs.

## Included Skills

- `paddleocr-text-recognition`: OCR text extraction for images/PDFs.
- `paddleocr-doc-parsing`: document parsing for layout-aware extraction.

## Quick Start

1. Install dependencies for the skill you use.
2. Configure API credentials with the skill scripts.
3. Run smoke tests:

```bash
python skills/paddleocr-text-recognition/scripts/smoke_test.py
python skills/paddleocr-doc-parsing/scripts/smoke_test.py
```

## Documentation

- Text recognition: `skills/paddleocr-text-recognition/SKILL.md`
- Doc parsing: `skills/paddleocr-doc-parsing/SKILL.md`

## API Access

Get API credentials from Paddle AI Studio: <https://paddleocr.com>

## License

[Apache License 2.0](../LICENSE)

## Repository

- Source: <https://github.com/PaddlePaddle/PaddleOCR/tree/main/skills>
