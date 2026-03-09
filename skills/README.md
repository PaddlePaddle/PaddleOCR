# PaddleOCR Skills

This directory contains agent skills for PaddleOCR official APIs.

## Included Skills

- `paddleocr-text-recognition`: OCR text extraction for images/PDFs.
- `paddleocr-doc-parsing`: document parsing for layout-aware extraction.

## Supported Models

- `paddleocr-doc-parsing`: `PP-StructureV3`, `PaddleOCR-VL`, `PaddleOCR-VL-1.5`
- `paddleocr-text-recognition`: `PP-OCRv5`

## Quick Start

This workflow requires Node.js, `npm`, and `npx`. If `npx` is unavailable, install Node.js first.

1. List installable skills from this repository:
   ```bash
   npx skills add PaddlePaddle/PaddleOCR --list
   ```
2. Install skills globally. The examples below install both skills; install only the one you need if applicable:
   ```bash
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-text-recognition -y
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-doc-parsing -y
   ```
3. Verify installation:
   ```bash
   npx skills list -g
   ```
4. Install Python dependencies:
   ```bash
   python -m pip install -r ~/.agents/skills/paddleocr-text-recognition/scripts/requirements.txt
   python -m pip install -r ~/.agents/skills/paddleocr-doc-parsing/scripts/requirements.txt
   # Optional: required only when using document file optimization
   python -m pip install -r ~/.agents/skills/paddleocr-doc-parsing/scripts/requirements-optimize.txt
   ```
   If using Windows PowerShell, use equivalent paths under `$HOME\\.agents\\skills\\...`.
5. Configure API credentials interactively:
   ```bash
   python ~/.agents/skills/paddleocr-text-recognition/scripts/configure.py
   python ~/.agents/skills/paddleocr-doc-parsing/scripts/configure.py
   ```
   Shared env file location: `~/.agents/skills/.env`
6. Run smoke tests:
   ```bash
   python ~/.agents/skills/paddleocr-text-recognition/scripts/smoke_test.py
   python ~/.agents/skills/paddleocr-doc-parsing/scripts/smoke_test.py
   ```

## Using in AI Apps (for example, Claude Code)

Describe the OCR or document parsing task in natural language and provide a file URL or local path so the AI app can invoke the skill.

### paddleocr-text-recognition

URL example:
```text
Extract all text from this file: https://example.com/invoice.jpg
```

Local file example:
```text
Extract all text from local file C:\docs\invoice.pdf
```

### paddleocr-doc-parsing

URL example:
```text
Parse this PDF and return the main body plus all tables in structured format: https://example.com/report.pdf
```

Local file example:
```text
Parse local file C:\docs\report.pdf and return complete structured output.
```

## Verification & Troubleshooting

- Skill not installed: run `npx skills list -g` to confirm the required skill is present.
- Missing dependencies: rerun `python -m pip install -r ...` for the corresponding skill.
- Configuration errors: rerun the corresponding `configure.py` script.
- API URL and access token source: <https://www.paddleocr.com>

## Documentation

- Text recognition: `skills/paddleocr-text-recognition/SKILL.md`
- Doc parsing: `skills/paddleocr-doc-parsing/SKILL.md`

## API Access

Get API credentials from the PaddleOCR official website: <https://www.paddleocr.com>

## License

[Apache License 2.0](../LICENSE)
