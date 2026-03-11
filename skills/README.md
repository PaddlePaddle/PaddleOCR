# PaddleOCR Skills

This directory contains AI agent skills for PaddleOCR official APIs.

## Included Skills

- `paddleocr-text-recognition`: OCR text extraction for images/PDFs.
- `paddleocr-doc-parsing`: document parsing for layout-aware extraction.

## Required Environment Variables

- `paddleocr-text-recognition`: `PADDLEOCR_OCR_API_URL`, `PADDLEOCR_ACCESS_TOKEN`
  Optional: `PADDLEOCR_TIMEOUT`
- `paddleocr-doc-parsing`: `PADDLEOCR_DOC_PARSING_API_URL`, `PADDLEOCR_ACCESS_TOKEN`
  Optional: `PADDLEOCR_DOC_PARSING_TIMEOUT`

## Quick Start

1. Install dependencies for the skill you use.
2. Preferred: set the required environment variables in your shell, host application, or secret manager. If your runtime already injects them, the scripts use them directly.
3. For local debugging and smoke tests, you can use the helper scripts or the shared local fallback file:
   ```bash
   python skills/paddleocr-text-recognition/scripts/configure.py
   python skills/paddleocr-doc-parsing/scripts/configure.py
   cp skills/.env.example skills/.env
   ```
   Then fill in `skills/.env` as needed. `skills/.env` is a shared local fallback, not the recommended production configuration method.
4. Run smoke tests:

```bash
python skills/paddleocr-text-recognition/scripts/smoke_test.py
python skills/paddleocr-doc-parsing/scripts/smoke_test.py
```

## Documentation

- Text recognition: `skills/paddleocr-text-recognition/SKILL.md`
- Doc parsing: `skills/paddleocr-doc-parsing/SKILL.md`

## API Access

Get API credentials from the PaddleOCR official website: <https://www.paddleocr.com>

## License

[Apache License 2.0](../LICENSE)
