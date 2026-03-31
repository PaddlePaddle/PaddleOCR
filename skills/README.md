# PaddleOCR Skills

This directory contains official PaddleOCR Agent Skills. They integrate with AI apps such as Claude Code for OCR text extraction from images/PDFs and layout-aware document parsing.

## Included Skills

- `paddleocr-text-recognition`: extract text from images/PDFs.
- `paddleocr-doc-parsing`: document parsing that converts images/PDFs to Markdown.

## Prerequisites

1. Python 3.8 or later must be installed on the device that runs the skill.
2. These skills depend on PaddleOCR official APIs and require API credentials. Visit the [PaddleOCR website](https://www.paddleocr.com), click **API**, select the model you need, then copy the `API_URL` and `Token`. They correspond to the API URL and access token used for authentication. Supported models per skill:
   - `paddleocr-text-recognition`: `PP-OCRv5`
   - `paddleocr-doc-parsing`: `PP-StructureV3`, `PaddleOCR-VL`, `PaddleOCR-VL-1.5`

## Using in AI Apps

> The instructions below cover both skills. Install and configure only the skill(s) you need.

### Install to AI Apps

#### Option 1: Install via `skills` CLI

The `skills` CLI installs skills globally on the device so all AI apps can use them. [Node.js](https://nodejs.org/en/download) is required.

```shell
npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-text-recognition -y
npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-doc-parsing -y
```

> This repository is relatively large. On slower networks, `npx skills add` may time out. If that happens, clone the repository locally first, then install from the local path:
>
> ```shell
> git clone https://github.com/PaddlePaddle/PaddleOCR.git
> npx skills add ./PaddleOCR/skills/paddleocr-text-recognition
> ```

#### Option 2: Install via `clawhub` (OpenClaw)

```shell
clawhub install paddleocr-text-recognition
clawhub install paddleocr-doc-parsing
```

See the [OpenClaw Skills documentation](https://docs.openclaw.ai/tools/skills) for details.

#### Option 3: Manual installation

If the above options are not available, you can clone the repository and manually copy the skill directories to the location required by your AI app ([Git](https://git-scm.com/downloads) required):

```shell
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

After cloning, skill source code is located under `PaddleOCR/skills`. Refer to the documentation for your AI app to complete installation:

- Claude Code: <https://code.claude.com/docs/en/skills>
- claude.ai: <https://support.claude.com/en/articles/12512180-use-skills-in-claude>
- OpenClaw: <https://docs.openclaw.ai/tools/skills>

### Configure Environment Variables

After installation, configure the required environment variables so the skills can work properly. Each skill requires the following:

| Skill | Required | Optional |
| --- | --- | --- |
| `paddleocr-text-recognition` | `PADDLEOCR_OCR_API_URL` (API URL), `PADDLEOCR_ACCESS_TOKEN` (access token) | `PADDLEOCR_OCR_TIMEOUT` (API request timeout) |
| `paddleocr-doc-parsing` | `PADDLEOCR_DOC_PARSING_API_URL` (API URL), `PADDLEOCR_ACCESS_TOKEN` (access token) | `PADDLEOCR_DOC_PARSING_TIMEOUT` (API request timeout) |

Below are configuration methods for some AI apps:

- **Claude Code**: add an `env` field to `.claude/settings.local.json` in your project:

  ```json
  {
    "env": {
      "PADDLEOCR_ACCESS_TOKEN": "<ACCESS_TOKEN>",
      "PADDLEOCR_OCR_API_URL": "<OCR_API_URL>",
      "PADDLEOCR_DOC_PARSING_API_URL": "<DOC_PARSING_API_URL>"
    }
  }
  ```

- **OpenClaw**: add skill configuration to `~/.openclaw/openclaw.json`:

  ```json
  {
    "skills": {
      "entries": {
        "paddleocr-text-recognition": {
          "enabled": true,
          "env": {
            "PADDLEOCR_OCR_API_URL": "<OCR_API_URL>",
            "PADDLEOCR_ACCESS_TOKEN": "<ACCESS_TOKEN>"
          }
        },
        "paddleocr-doc-parsing": {
          "enabled": true,
          "env": {
            "PADDLEOCR_DOC_PARSING_API_URL": "<DOC_PARSING_API_URL>",
            "PADDLEOCR_ACCESS_TOKEN": "<ACCESS_TOKEN>"
          }
        }
      }
    }
  }
  ```

### Usage Examples

After configuration, describe the OCR or document parsing task in natural language and provide a file URL or local path so the AI app can invoke the corresponding skill.

**paddleocr-text-recognition**

URL example:

```text
Extract all text from this file: https://example.com/invoice.jpg
```

Local file example:

```text
Extract all text from local file C:\docs\invoice.pdf
```

**paddleocr-doc-parsing**

URL example:

```text
Parse this PDF and return the main body plus all tables: https://example.com/report.pdf
```

Local file example:

```text
Parse local file C:\docs\report.pdf and return complete structured output.
```

## Local Testing

This section describes how to run smoke tests locally to verify that the skills work correctly.

> The examples below cover both skills. Run only the commands for the skill(s) you need.

Make sure your working directory is the directory containing this file.

1. Install dependencies.

   ```shell
   python -m pip install -r paddleocr-text-recognition/scripts/requirements.txt
   python -m pip install -r paddleocr-doc-parsing/scripts/requirements.txt
   # Optional: required only when using document file optimization
   python -m pip install -r paddleocr-doc-parsing/scripts/requirements-optimize.txt
   ```

2. Configure environment variables (see [Configure Environment Variables](#configure-environment-variables) for the list of variables).

   ```shell
   export PADDLEOCR_OCR_API_URL="<OCR_API_URL>"
   export PADDLEOCR_ACCESS_TOKEN="<ACCESS_TOKEN>"
   export PADDLEOCR_DOC_PARSING_API_URL="<DOC_PARSING_API_URL>"
   ```

3. Run the smoke test scripts.

   ```shell
   python paddleocr-text-recognition/scripts/smoke_test.py
   python paddleocr-doc-parsing/scripts/smoke_test.py
   ```
