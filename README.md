# PaddleOCR-SKILLs

<p align="center">
  <strong>Multi-Model OCR Skills Suite for Claude Code</strong>
</p>

<p align="center">
  Intelligent text extraction and document parsing powered by Baidu PaddlePaddle
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Claude%20Code-Skills-purple.svg" alt="Claude Code Skills">
</p>

<p align="center">
  <a href="./README-cn.md">简体中文</a> | <strong>English</strong>
</p>

---

## 🎯 Two Skills, One Solution

This repository provides two complementary OCR skills for different document processing needs:

### 1. paddleocr-text-recognition - Fast Text Extraction

**Best for**: Simple text recognition from images and PDFs

- ⚡ **Fast Recognition** - 3 quality modes (fast/quality/auto)
- 📝 **80+ Languages** - Comprehensive multilingual support
- 🎛️ **Adaptive Quality** - Automatic retry with progressive quality enhancement
- 📊 **Quality Scoring** - Built-in confidence metrics

**Use when**: You need quick text extraction from screenshots, scans, or simple documents

### 2. paddleocr-doc-parsing - Advanced Document Parsing

**Best for**: Complex documents with tables, formulas, and structured layouts

- 📊 **Table Recognition** - Extract structured data from tables
- 🔢 **Formula Detection** - Recognize mathematical equations (LaTeX output)
- 📐 **Layout Analysis** - Automatic document structure detection
- 🌍 **109 Languages** - Enhanced multilingual capabilities
- 📄 **Structured Output** - JSON or Markdown format

**Use when**: You need to parse invoices, academic papers, financial reports, or any document with complex structure

---

## 📦 Installation

> **Prerequisites**: Node.js >= 14, Python 3.8+, [Claude Code CLI](https://claude.ai/code)

### Install Skills

Install all skills:
```bash
npx skills add Aidenwu0209/PaddleOCR-Skills
```

Install a specific skill:
```bash
# Text recognition only
npx skills add Aidenwu0209/PaddleOCR-Skills --skill paddleocr-text-recognition

# Document parsing only
npx skills add Aidenwu0209/PaddleOCR-Skills --skill paddleocr-doc-parsing
```

After installation, the installer will prompt you to select which AI agents to install to (Claude Code, Cursor, Cline, etc.).

### Configure API Credentials

Get your API credentials at [Paddle AI Studio](https://paddleocr.com), then configure:

**paddleocr-text-recognition:**
```bash
python ~/.claude/skills/paddleocr-text-recognition/scripts/configure.py
```

**paddleocr-doc-parsing:**
```bash
python ~/.claude/skills/paddleocr-doc-parsing/scripts/configure.py
```

<details>
<summary>Alternative: Manual Installation</summary>

```bash
git clone https://github.com/Aidenwu0209/PaddleOCR-Skills.git
cd PaddleOCR-Skills

# paddleocr-text-recognition
pip install -r skills/paddleocr-text-recognition/scripts/requirements.txt
python skills/paddleocr-text-recognition/scripts/configure.py

# paddleocr-doc-parsing
pip install -r skills/paddleocr-doc-parsing/scripts/requirements.txt
python skills/paddleocr-doc-parsing/scripts/configure.py
```

</details>

---

## 🚀 Quick Start

After installation, just describe your need in natural language:

**Simple text extraction**:
> "Extract text from this image: screenshot.png"

Claude will use **paddleocr-text-recognition** for fast text recognition.

**Complex document parsing**:
> "Parse this invoice table: invoice.pdf"

Claude will use **paddleocr-doc-parsing** for structured data extraction.

---

## 📊 Feature Comparison

| Feature | paddleocr-text-recognition | paddleocr-doc-parsing |
|---------|:--------:|:------------:|
| **Primary Use Case** | Text extraction | Document parsing |
| **Speed** | Fast ⚡ | Medium 🐢 |
| **Languages** | 80+ | 109 |
| **Quality Modes** | 3 modes | Auto |
| **Table Recognition** | ❌ | ✅ |
| **Formula Detection** | ❌ | ✅ |
| **Layout Analysis** | ❌ | ✅ |
| **Output Format** | Plain text + JSON | JSON / Markdown |
| **Best For** | Screenshots, scans | Invoices, papers |

---

## 📚 Documentation

### paddleocr-text-recognition Documentation
- [Skill Guide](./skills/paddleocr-text-recognition/SKILL.md) - How to use paddleocr-text-recognition
- [Output Schema](./skills/paddleocr-text-recognition/references/output_schema.md) - Output format specification
- [Provider API](./skills/paddleocr-text-recognition/references/provider_api.md) - API contract details

### paddleocr-doc-parsing Documentation
- [Skill Guide](./skills/paddleocr-doc-parsing/SKILL.md) - How to use paddleocr-doc-parsing
- [Output Schema](./skills/paddleocr-doc-parsing/references/output_schema.md) - Output format specification
- [Provider API](./skills/paddleocr-doc-parsing/references/provider_api.md) - API contract details

> **Note**: Model versions and capabilities are determined by the API endpoint. Get the latest API at [Paddle AI Studio](https://paddleocr.com).

---

## 🔍 Which Skill Should I Use?

```
┌─────────────────────────────────────┐
│  What do you need to extract?      │
└───────────┬─────────────────────────┘
            │
    ┌───────┴────────┐
    │  Just text?    │
    └───┬────────┬───┘
        │        │
       Yes      No
        │        │
        ▼        ▼
   text-       ┌──────────────────────┐
   recognition │ Tables / Formulas /  │
               │ Complex Layout?      │
               └──────┬───────────────┘
                      │
                     Yes
                      │
                      ▼
                doc-parsing
```

### Quick Selection Guide

| Your Task | Recommended Skill |
|-----------|------------------|
| "Extract text from this screenshot" | **paddleocr-text-recognition** |
| "Read text from this scanned document" | **paddleocr-text-recognition** |
| "Parse this invoice table" | **paddleocr-doc-parsing** |
| "Extract data from this financial report" | **paddleocr-doc-parsing** |
| "Get text from this academic paper with formulas" | **paddleocr-doc-parsing** |
| "Quick OCR of a photo" | **paddleocr-text-recognition** |

---

## 🧪 Testing

**Test paddleocr-text-recognition**:
```bash
python skills/paddleocr-text-recognition/scripts/smoke_test.py
```

**Test paddleocr-doc-parsing**:
```bash
python skills/paddleocr-doc-parsing/scripts/smoke_test.py
```

---

## 💡 Usage Examples

### paddleocr-text-recognition Examples

**Basic text extraction**:
```bash
python skills/paddleocr-text-recognition/scripts/ocr_caller.py --file-url "https://example.com/image.jpg" --pretty
```

**Fast mode for clear images**:
```bash
python skills/paddleocr-text-recognition/scripts/ocr_caller.py --file-path "screenshot.png" --preset fast
```

**High quality mode**:
```bash
python skills/paddleocr-text-recognition/scripts/ocr_caller.py --file-path "scan.pdf" --preset quality
```

### paddleocr-doc-parsing Examples

**Parse document with tables**:
```bash
python skills/paddleocr-doc-parsing/scripts/vl_caller.py --file-path "invoice.pdf" --pretty
```

**Extract as Markdown**:
```bash
python skills/paddleocr-doc-parsing/scripts/vl_caller.py --file-url "URL" --format markdown --pretty
```

**Save result to file**:
```bash
python skills/paddleocr-doc-parsing/scripts/vl_caller.py --file-path "document.pdf" --output result.json
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

[MIT License](./LICENSE)

---

## 🙏 Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Baidu's PaddlePaddle OCR toolkit
- [Paddle AI Studio](https://paddleocr.com) - API service provider

---

## 📮 Support

- **Issues**: [GitHub Issues](https://github.com/Aidenwu0209/PaddleOCR-SKILLs/issues)
- **Documentation**: See the [skills](./skills/) directory
- **API Status**: [Paddle AI Studio](https://paddleocr.com)

---

<p align="center">
  Made with ❤️ for Claude Code
</p>
