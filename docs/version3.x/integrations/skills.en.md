---
comments: true
---

# PaddleOCR Agent Skills

PaddleOCR provides official Agent Skills that package the routing rules, calling steps, configuration requirements, and best practices for handling results in OCR and document parsing tasks into on-demand modular capabilities, helping Skills-enabled AI apps complete text recognition and layout parsing more reliably.

## Choose the Right Skill First

| Need | Recommended Skill | Output |
| --- | --- | --- |
| Extract plain text from images or PDFs | `paddleocr-text-recognition` | line-level text with bounding boxes and confidence scores |
| Preserve headings, paragraphs, tables, formulas, and layout structure | `paddleocr-doc-parsing` | Markdown / structured output |

## Included Skills

- `paddleocr-text-recognition`: extracts text from images, scans, and PDF files.
- `paddleocr-doc-parsing`: parses complex document layouts and converts them to Markdown or structured output.

## Prerequisites

1. Install Python 3.9 or later on the machine that runs the skill.
2. Install PaddleOCR 3.6.0+: `pip install "paddleocr>=3.6.0"`
3. Get access token from [AI Studio](https://aistudio.baidu.com/account/accessToken)

## Install into AI Apps

> The instructions below cover both skills. Install and configure only the skill or skills you need.

### Option 1: Install via `skills` CLI

The `skills` CLI installs skills globally on the device so they can be used by supported AI apps. [Node.js](https://nodejs.org/en/download) is required.

```shell
npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-text-recognition -y
npx skills add PaddlePaddle/PaddleOCR -g --skill paddleocr-doc-parsing -y
```

> This repository is relatively large. On slower networks, `npx skills add` may time out. If that happens, clone the repository locally first and then install from the local path:
>
> ```shell
> git clone https://github.com/PaddlePaddle/PaddleOCR.git
> npx skills add ./PaddleOCR/skills/paddleocr-text-recognition
> npx skills add ./PaddleOCR/skills/paddleocr-doc-parsing
> ```

### Option 2: Install via `clawhub` (OpenClaw)

```shell
clawhub install paddleocr-text-recognition
clawhub install paddleocr-doc-parsing
```

See the [OpenClaw Skills documentation](https://docs.openclaw.ai/tools/skills) for details.

### Option 3: Manual Installation

If the options above do not fit your environment, clone the repository and copy the skill directories to the location required by your AI app:

```shell
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

The skill source code is located under `PaddleOCR/skills`. Refer to your AI app documentation to complete the installation:

- Claude Code: <https://code.claude.com/docs/en/skills>
- claude.ai: <https://support.claude.com/en/articles/12512180-use-skills-in-claude>
- OpenClaw: <https://docs.openclaw.ai/tools/skills>

## Configure Environment Variables

After installation, configure the following environment variables:

- Required: `PADDLEOCR_ACCESS_TOKEN` (access token)
- Optional: `PADDLEOCR_BASE_URL` (API base URL, defaults to official service)

Get access token: visit [AI Studio Access Token](https://aistudio.baidu.com/account/accessToken)

Examples for some AI apps:

- **Claude Code**: add an `env` field to `.claude/settings.local.json` in your project:

  ```json
  {
    "env": {
      "PADDLEOCR_ACCESS_TOKEN": "<ACCESS_TOKEN>"
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
            "PADDLEOCR_ACCESS_TOKEN": "<ACCESS_TOKEN>"
          }
        },
        "paddleocr-doc-parsing": {
          "enabled": true,
          "env": {
            "PADDLEOCR_ACCESS_TOKEN": "<ACCESS_TOKEN>"
          }
        }
      }
    }
  }
  ```

## Usage Examples

After configuration, describe the task in natural language and provide a file URL or local path so the AI app can invoke the corresponding skill.

### `paddleocr-text-recognition`

URL example:

```text
Extract all text from this file: https://example.com/invoice.jpg
```

Local file example:

```text
Extract all text from local file C:\docs\invoice.pdf
```

### `paddleocr-doc-parsing`

URL example:

```text
Parse this PDF and return the main body plus all tables: https://example.com/report.pdf
```

Local file example:

```text
Parse local file C:\docs\report.pdf and return complete structured output.
```
