# PaddleOCR Skills

These skills let AI apps call PaddleOCR official APIs for OCR text extraction from images/PDFs and layout-aware document parsing.

## Included Skills

- `paddleocr-text-recognition`: OCR text extraction for images/PDFs.
- `paddleocr-doc-parsing`: document parsing for layout-aware extraction.

## Supported Models

- `paddleocr-doc-parsing`: `PP-StructureV3`, `PaddleOCR-VL`, `PaddleOCR-VL-1.5`
- `paddleocr-text-recognition`: `PP-OCRv5`

## Required Environment Variables

- `paddleocr-text-recognition`: `PADDLEOCR_OCR_API_URL`, `PADDLEOCR_ACCESS_TOKEN`
  Optional: `PADDLEOCR_TIMEOUT`
- `paddleocr-doc-parsing`: `PADDLEOCR_DOC_PARSING_API_URL`, `PADDLEOCR_ACCESS_TOKEN`
  Optional: `PADDLEOCR_DOC_PARSING_TIMEOUT`

## Install in AI Apps

1. Follow the installation mechanism supported by your AI app.
   - Claude Code skills: <https://code.claude.com/docs/en/skills>
   - Claude custom skills: <https://claude.com/docs/skills/overview>
   - Cursor rules / AGENTS.md: <https://docs.cursor.com/context/rules> and <https://docs.cursor.com/en/cli/using>
2. You can also install these skills through the `skills` CLI:
   ```bash
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-text-recognition -y
   npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-doc-parsing -y
   ```

   Note: this repository is relatively large. On slower networks or devices, `npx skills add` may hit the current 60-second clone timeout. If that happens, prefer the app-native installation flow above.

## Repository-Local Smoke Test

- Python 3.8+ must be installed, and `python` and `pip` must be available in `PATH`.
- Run the following commands from the `skills/` directory.
- The examples below cover both skills. Run only the commands for the skill you need.

1. Install dependencies for the skill you use.
   ```bash
   python -m pip install -r paddleocr-text-recognition/scripts/requirements.txt
   python -m pip install -r paddleocr-doc-parsing/scripts/requirements.txt
   # Optional: required only when using document file optimization
   python -m pip install -r paddleocr-doc-parsing/scripts/requirements-optimize.txt
   ```
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

- Missing dependencies: rerun the dependency installation command for the relevant file, such as `paddleocr-text-recognition/scripts/requirements.txt`, `paddleocr-doc-parsing/scripts/requirements.txt`, or `paddleocr-doc-parsing/scripts/requirements-optimize.txt`.
- Configuration issues: first check whether the required environment variables are available in the host application or runtime environment.
- For repository-local smoke tests, you can rerun the corresponding `configure.py` script or update the local `.env` file.

## Documentation

- Text recognition: `paddleocr-text-recognition/SKILL.md`
- Doc parsing: `paddleocr-doc-parsing/SKILL.md`

## API Access

Get API credentials from the PaddleOCR official website: <https://www.paddleocr.com>

## License

[Apache License 2.0](../LICENSE)
