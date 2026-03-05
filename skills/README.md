# PaddleOCR Skills

Beginner-friendly agent skills for PaddleOCR official APIs. Follow this document to install, configure, and run them end-to-end.

## Included Skills

- `paddleocr-text-recognition`: OCR text extraction for images/PDFs.
- `paddleocr-doc-parsing`: document parsing for layout-aware extraction.

## Supported Models

- `paddleocr-doc-parsing`: `PP-StructureV3`, `PaddleOCR-VL`, `PaddleOCR-VL-1.5`
- `paddleocr-text-recognition`: `PP-OCRv5`
- Note: actual model capabilities and supported file formats depend on the configured API endpoint.

## Quick Start (npx)

1. List installable skills from this repository:
   ```bash
   npx skills add PaddlePaddle/PaddleOCR --list
   ```
2. Install skills globally:
   ```bash
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-text-recognition -y
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-doc-parsing -y
   ```
3. Verify installation:
   ```bash
   npx skills list -g
   ```
4. Install Python dependencies right after installation:
   ```bash
   python -m pip install -r ~/.agents/skills/paddleocr-text-recognition/scripts/requirements.txt
   python -m pip install -r ~/.agents/skills/paddleocr-doc-parsing/scripts/requirements.txt
   # Optional: required only when using document file optimization
   python -m pip install -r ~/.agents/skills/paddleocr-doc-parsing/scripts/requirements-optimize.txt
   ```
   On Windows, use equivalent paths under `$HOME\\.agents\\skills\\...`.
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

## How to Use in Chat

You can use URL inputs or local file paths in your chat requests.

### Text Recognition (`paddleocr-text-recognition`)

Copy and send:
```bash
Extract all text from this file: https://example.com/invoice.jpg
```

Or:
```bash
Extract all text from local file C:\docs\invoice.pdf
```

### Document Parsing (`paddleocr-doc-parsing`)

Copy and send:
```bash
Parse this PDF and return the main body plus all tables in structured format: https://example.com/report.pdf
```

Or:
```bash
Parse local file C:\docs\report.pdf and return complete structured output.
```

## Verification & Troubleshooting

- Skill not installed: run `npx skills list -g` to confirm both skills are present.
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
