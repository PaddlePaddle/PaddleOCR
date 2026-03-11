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

## Prerequisites

- `python` and `pip` must be available in `PATH`.
- The local helper examples assume a shell environment that can run commands such as `cp`.
- If a skill is installed under a host application directory, follow that host application's environment-variable configuration best practices instead of creating local config files there.

## Quick Start

Run the following commands from the `skills/` directory.

1. Install dependencies for the skill you use.
2. Configure API credentials using one of the following options.

   Option A: run the helper script for the skill you want to test.
   ```bash
   python paddleocr-text-recognition/scripts/configure.py
   python paddleocr-doc-parsing/scripts/configure.py
   ```

   Option B: create a local `.env` file from the `.env.example` template and fill in the required variables.
   ```bash
   cp .env.example .env
   ```

   If the skill is installed under a host application directory (for example, `~/.claude/skills`), do not run `configure.py` or create `.env` there. Follow the host application's environment-variable configuration best practices instead.
3. Run the smoke test for the skill you want to verify:

```bash
python paddleocr-text-recognition/scripts/smoke_test.py
python paddleocr-doc-parsing/scripts/smoke_test.py
```

## Documentation

- Text recognition: `paddleocr-text-recognition/SKILL.md`
- Doc parsing: `paddleocr-doc-parsing/SKILL.md`

## API Access

Get API credentials from the PaddleOCR official website: <https://www.paddleocr.com>

## License

[Apache License 2.0](../LICENSE)
