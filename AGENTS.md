# AGENTS.md

> Instructions for AI coding agents working on `PaddlePaddle/PaddleOCR`.
> Human reviewers must understand and approve every AI-assisted change end-to-end.

## What This Is

PaddleOCR is a production-ready OCR and document AI engine built on PaddlePaddle. It provides text detection, recognition, document structure analysis, and information extraction.

---

## 1. Contribution Policy

### Duplicate-work checks

Before proposing a PR, check for existing work:

```bash
gh issue view <issue_number> --repo PaddlePaddle/PaddleOCR --comments
gh pr list --repo PaddlePaddle/PaddleOCR --state open --search "<issue_number> in:body"
gh pr list --repo PaddlePaddle/PaddleOCR --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- A human submitter must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work should include what AI tool was used.

---

## 2. Development Workflow

### Environment setup

```bash
# Install PaddlePaddle first (see https://www.paddlepaddle.org.cn/install/quick)
# Then install PaddleOCR in dev mode:
pip install -e ".[all]"

# Install pre-commit hooks:
pip install pre-commit
pre-commit install
```

### Running tests

```bash
# Run all tests (resource-intensive tests skipped by default):
pytest tests/

# Run a specific test file:
pytest tests/models/test_text_detection.py -v -s

# Run a specific test:
pytest tests/models/test_text_detection.py -v -s -k test_predict
```

Tests live in `tests/` with subdirectories `models/` and `pipelines/`. Use `@pytest.mark.resource_intensive` for tests that download models or need GPU.

### Running linters

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff --all-files
```

### Commit messages

Use `Co-authored-by:` trailers to attribute AI assistance. For example:

```text

Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
```

---

## 3. Project Structure

```
PaddleOCR/
├── paddleocr/              # Public API (3.x) — what users import
│   ├── __init__.py         # Top-level exports (__all__ is the source of truth)
│   ├── _pipelines/         # High-level pipelines (OCR, PPStructureV3, etc.)
│   ├── _models/            # Individual model wrappers (TextDetection, etc.)
│   └── _cli.py             # CLI entry point
├── ppocr/                  # Internal training framework (not user-facing)
│   ├── modeling/           # Model architectures (Backbone, Neck, Head)
│   ├── data/               # Data loading and augmentation
│   ├── losses/             # Loss functions
│   ├── metrics/            # Evaluation metrics
│   └── postprocess/        # Post-processing
├── tools/                  # Train/infer/eval scripts (tools/train.py)
├── configs/                # YAML configs organized by task (det/, rec/, table/, etc.)
├── deploy/                 # Deployment (C++, Docker, ONNX, mobile)
├── tests/                  # Tests (models/ + pipelines/)
└── agent_docs/             # Detailed AI-readable documentation
```

**Two layers** — understand which you're working in:

- **`paddleocr/`** — Public API (3.x). `_pipelines/` has high-level pipelines, `_models/` has individual model wrappers. Users import from here.
- **`ppocr/`** — Internal training framework. Used by `tools/train.py`, not by end users.

---

## 4. Critical: 3.x API Only

PaddleOCR 3.x is **not backwards compatible** with 2.x. Never generate 2.x-style code:

- Use `.predict()` not `.ocr()` (deprecated)
- Results are objects with `.print()`, `.save_to_img()`, `.save_to_json()` — not nested lists
- `PPStructure` is removed — use `PPStructureV3`
- For single-task inference, use model classes (`TextDetection`, `TextRecognition`) not `det`/`rec` params

---

## 5. Discovering Available Pipelines & Models

**Do NOT rely on hardcoded lists.** Always discover dynamically from source:

- **Pipelines**: Read `__all__` in `paddleocr/_pipelines/__init__.py`
- **Models**: Read `__all__` in `paddleocr/_models/__init__.py`
- **All public exports**: Read `__all__` in `paddleocr/__init__.py`

Each pipeline inherits from `PaddleXPipelineWrapper` (in `_pipelines/base.py`).
Each model inherits from `PaddleXPredictorWrapper` (in `_models/base.py`).

To understand a specific pipeline or model, read its source file in the corresponding directory.

---

## 6. Code Style & Conventions

- Follow existing patterns in the file you're modifying
- Use type hints for function signatures
- Error messages should be clear and actionable
- No `eval()`, `exec()`, or `pickle` on user-controlled input
- Always run `pre-commit run --all-files` before committing

---

## 7. Testing Guidelines

- When adding a new pipeline or model, add corresponding tests
- Test the public API (`.predict()`, result object methods), not internal implementation details

---

## 8. PR Guidelines

- PR titles: concise, lowercase, descriptive of what changed
- PR descriptions: explain the "why", not just the "what"
- Keep PRs focused — one logical change per PR
- Ensure `pre-commit run --all-files` passes before pushing

---

## 9. Detailed Docs

Read these as needed — don't load them all upfront:

- `agent_docs/inference_api.md` — Pipelines, models, constructor params, CLI, usage patterns
- `agent_docs/training.md` — Training commands, config YAML structure, internal framework
- `agent_docs/config_system.md` — Training YAML configs AND inference PaddleX configs (two separate systems)
