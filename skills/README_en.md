# PaddleOCR Claude Code Skills

[中文](./README.md) | English

## Overview

This directory provides two Claude Code skills for OCR text recognition and document parsing via PaddleOCR official API.

| Skill | Purpose | Use Cases |
|-------|---------|-----------|
| **paddleocr-text-recognition** | Fast text extraction | Screenshots, scans, simple documents |
| **paddleocr-doc-parsing** | Advanced document parsing | Tables, formulas, seals, complex layouts |

## Relationship with MCP Server

The Skills in this directory complement the `mcp_server/` directory. Both invoke the same underlying models but serve different scenarios:

| Feature | MCP Server | Skills |
|---------|-----------|--------|
| Protocol | Model Context Protocol (MCP) | Claude Code Skill Protocol |
| Clients | Claude Desktop, VSCode, etc. | Claude Code CLI |
| Architecture | Long-running server process | Direct CLI invocation |

## Installation

### 1. Install Dependencies

```bash
pip install -r skills/paddleocr-text-recognition/scripts/requirements.txt
pip install -r skills/paddleocr-doc-parsing/scripts/requirements.txt
```

### 2. Get API Credentials

Visit [https://paddleocr.com](https://paddleocr.com) to obtain your API URL and Access Token.

### 3. Configure

```bash
python skills/paddleocr-text-recognition/scripts/configure.py
python skills/paddleocr-doc-parsing/scripts/configure.py
```

### 4. Verify

```bash
python skills/paddleocr-text-recognition/scripts/smoke_test.py
python skills/paddleocr-doc-parsing/scripts/smoke_test.py
```

## Usage

### paddleocr-text-recognition

```bash
# URL input
python skills/paddleocr-text-recognition/scripts/ocr_caller.py --file-url "https://example.com/image.png"

# Local file
python skills/paddleocr-text-recognition/scripts/ocr_caller.py --file-path "./document.pdf" --pretty
```

### paddleocr-doc-parsing

```bash
# URL input
python skills/paddleocr-doc-parsing/scripts/vl_caller.py --file-url "https://example.com/doc.pdf"

# Local file
python skills/paddleocr-doc-parsing/scripts/vl_caller.py --file-path "./invoice.pdf" --pretty
```

## Directory Structure

```
skills/
├── README.md                          # Chinese documentation
├── README_en.md                       # English documentation
├── paddleocr-text-recognition/        # Text recognition
│   ├── SKILL.md                       # Skill definition
│   ├── scripts/                       # Python scripts
│   └── references/                    # API reference docs
└── paddleocr-doc-parsing/             # Document parsing
    ├── SKILL.md                       # Skill definition
    ├── scripts/                       # Python scripts
    └── references/                    # API reference docs
```
