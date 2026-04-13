# PaddleOCR Developer Onboarding

Welcome! This documentation is designed for software engineers who are new to
OCR (Optical Character Recognition) and PaddleOCR. It covers everything you
need to understand the project architecture and start contributing.

## Reading Order

Start from the top and work your way down. Each document builds on the previous.

1. **[OCR Fundamentals](ocr-fundamentals.md)** — What OCR is, how detection
   and recognition work, key algorithms and metrics. Start here if you have
   zero OCR background.
2. **[Architecture Deep Dive](architecture.md)** — PaddleOCR's two-layer
   design: the high-level pipeline API (`paddleocr/`) and the core ML
   framework (`ppocr/`). Covers the config system, data pipeline, and model
   composition.
3. **[Codebase Map](codebase-map.md)** — Quick-reference directory guide.
   Skim once, return often.
4. **[Adding Models](adding-models.md)** — Step-by-step guide for adding a
   new backbone, head, neck, loss, or complete model configuration.
5. **[Training & Evaluation](training-evaluation.md)** — How to train models,
   prepare data, run evaluation, and interpret metrics.
6. **[Deployment & Export](deployment-export.md)** — Exporting trained models,
   high-performance inference, ONNX, serving, mobile, and Docker deployment.
7. **[Testing & Debugging](testing-debugging.md)** — Running the test suite,
   test patterns, debugging common training and inference issues.

## Quick Links

| I want to...                        | Go to                                          |
|-------------------------------------|-------------------------------------------------|
| Understand what OCR is              | [OCR Fundamentals](ocr-fundamentals.md)         |
| Learn how the code is structured    | [Architecture](architecture.md)                 |
| Find where a specific file lives    | [Codebase Map](codebase-map.md)                 |
| Add a new backbone or head          | [Adding Models](adding-models.md)               |
| Train or fine-tune a model          | [Training & Evaluation](training-evaluation.md) |
| Deploy a model to production        | [Deployment & Export](deployment-export.md)      |
| Run tests or debug an issue         | [Testing & Debugging](testing-debugging.md)      |

## Prerequisites

- **Python** 3.8 - 3.13
- **PaddlePaddle** 3.x (GPU version recommended for training)
- **Git** for version control
- A basic understanding of deep learning concepts (CNNs, loss functions,
  gradient descent) is helpful but not required — the OCR Fundamentals
  doc covers OCR-specific concepts from scratch.

Install PaddlePaddle:

```bash
# GPU (CUDA 12.6)
python -m pip install paddlepaddle-gpu==3.2.1

# CPU only
python -m pip install paddlepaddle==3.2.1
```

Install PaddleOCR:

```bash
python -m pip install paddleocr

# With document parsing (VLM) support
python -m pip install "paddleocr[doc-parser]"
```
