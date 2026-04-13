# Deployment and Export

Guide for exporting trained models and deploying PaddleOCR to production.
Covers Layer 1 (pipeline API) deployment, model export, high-performance
inference, ONNX, serving, and mobile/edge deployment.

## Export Overview

Before deployment, trained models (dynamic graph) must be exported to static
graph format for inference optimization:

```
Trained Model                      Static Model                Deploy Targets
(dynamic graph)                    (inference-ready)
                                                               ┌─────────────┐
┌─────────────┐  export_model.py   ┌─────────────────┐       │ Python API  │
│ .pdparams   │─────────────────>  │ inference.pdmodel│──────>│ (Layer 1)   │
│ .pdopt      │                    │ inference.pdiparams      ├─────────────┤
│ .states     │                    │ inference.yml    │──────>│ CLI         │
└─────────────┘                    └─────────────────┘       ├─────────────┤
                                          │                   │ C++ Infer   │
                                          │    Paddle2ONNX    ├─────────────┤
                                          ├──────────────────>│ ONNX Runtime│
                                          │                   ├─────────────┤
                                          │                   │ TensorRT    │
                                          │                   ├─────────────┤
                                          │                   │ REST API    │
                                          │                   ├─────────────┤
                                          └──────────────────>│ Mobile/Edge │
                                                              └─────────────┘
```

## Exporting a Trained Model

### Using tools/export_model.py

```bash
python tools/export_model.py \
    -c configs/det/det_mv3_db.yml \
    -o Global.checkpoints=./output/db_mv3/best_accuracy \
    -o Global.save_inference_dir=./inference/det_db/
```

This produces three files:
- `inference.pdmodel` — network structure
- `inference.pdiparams` — model weights
- `inference.yml` — configuration metadata (input shape, preprocessing)

The export process (`ppocr/utils/export_model.py`) converts the dynamic
PaddlePaddle graph to a static graph using `paddle.jit.to_static` with
algorithm-specific `InputSpec` configurations.

---

## Layer 1 Deployment (Pipeline API)

For most production use cases, Layer 1 is the recommended path. It handles
model downloading, pipeline orchestration, and result formatting.

### Python API

```python
from paddleocr import PaddleOCR

# General OCR (PP-OCRv5 by default)
ocr = PaddleOCR(lang="en")
result = ocr.predict("image.jpg")

for res in result:
    res.print()                        # Print results
    res.save_to_json("output/")        # Save as JSON
    res.save_to_img("output/")         # Save visualization

# Access structured data
for res in result:
    texts = res.json["rec_texts"]      # Recognized text strings
    scores = res.json["rec_scores"]    # Confidence scores
    polys = res.json["dt_polys"]       # Detection polygons
```

```python
from paddleocr import PPStructureV3

# Document structure analysis
structure = PPStructureV3()
result = structure.predict("document.pdf")
for res in result:
    res.save_to_markdown("output/")    # Markdown output
```

```python
from paddleocr import PaddleOCRVL

# Vision-Language document parsing
vl = PaddleOCRVL()
result = vl.predict("complex_doc.png")
for res in result:
    res.save_to_markdown("output/")
```

### Using Custom Model Directories

Point to your own exported models:

```python
ocr = PaddleOCR(
    text_detection_model_dir="./inference/my_det_model/",
    text_recognition_model_dir="./inference/my_rec_model/",
)
```

### CLI

```bash
# OCR pipeline
paddleocr ocr --input image.jpg --device gpu:0

# Document structure
paddleocr pp_structurev3 --input document.pdf

# Vision-Language parsing
paddleocr doc_parser --input page.png

# Single model inference
paddleocr text_detection --input image.jpg
paddleocr text_recognition --input cropped_text.jpg
```

### When to Use Layer 1 vs Layer 2

```
Use Layer 1 (paddleocr/)              Use Layer 2 (ppocr/ + tools/)
─────────────────────────              ──────────────────────────────

Production inference                   Training new models
REST API serving                       Evaluating checkpoints
Quick prototyping                      Adding new architectures
Custom model directories               Debugging model internals
CLI batch processing                   Exporting models
                                       Research experiments
```

---

## High-Performance Inference (HPI)

HPI automatically selects the best inference backend and applies
optimizations (TensorRT, ONNX Runtime, OpenVINO) based on your hardware.

### Installation

```bash
# CPU systems
paddleocr install_hpi_deps cpu

# GPU systems (requires CUDA 11.8 + cuDNN 8.9, or CUDA 12.6 + cuDNN 9.5)
paddleocr install_hpi_deps gpu
```

### Usage

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(enable_hpi=True)
result = ocr.predict("image.jpg")
```

```bash
paddleocr ocr --input image.jpg --enable_hpi true
```

**Supported backends**: Paddle Inference (with TensorRT), ONNX Runtime,
OpenVINO. HPI selects the optimal backend automatically.

**Note**: The first run builds the inference engine (slow). Subsequent runs
reuse the cached engine.

---

## ONNX Export

Convert PaddlePaddle models to ONNX format for cross-platform inference:

```bash
# Install the conversion tool
paddlex --install paddle2onnx

# Convert
paddlex \
    --paddle2onnx \
    --paddle_model_dir ./inference/det_db/ \
    --onnx_model_dir ./onnx_models/det_db/ \
    --opset_version 14
```

The converter handles operator mapping automatically. If conversion fails at
the specified opset version, it retries with higher versions.

---

## Serving Deployment

### Basic Serving (PaddleX)

```bash
# Install serving plugin
paddlex --install serving

# Start server
paddlex --serve --pipeline OCR --host 0.0.0.0 --port 8080
```

The server exposes REST API endpoints at `http://0.0.0.0:8080`. Send
images via HTTP POST and receive JSON results. Client code can be in any
language.

Parameters:

| Flag          | Purpose                  | Default     |
|---------------|--------------------------|-------------|
| `--pipeline`  | Pipeline name or config  | Required    |
| `--device`    | GPU/CPU selection        | Auto-detect |
| `--host`      | Bind address             | 0.0.0.0     |
| `--port`      | Listen port              | 8080        |
| `--use_hpip`  | Enable HPI               | false       |

### Production Serving (Triton)

For high-stability production environments, PaddleOCR supports deployment on
NVIDIA Triton Inference Server. See the PaddleX Serving Guide for details.

---

## PaddleOCR-VL Deployment

PaddleOCR-VL (the Vision-Language model) has specialized deployment options
due to its LLM component:

| Backend                 | Hardware              | Use case                |
|-------------------------|-----------------------|-------------------------|
| PaddlePaddle only       | All supported GPUs    | Default, broadest compat|
| PaddlePaddle + vLLM     | NVIDIA (CC >= 8.0)    | High-throughput GPU     |
| PaddlePaddle + SGLang   | NVIDIA                | Optimized serving       |
| PaddlePaddle + FastDeploy| Multi-backend        | Flexible acceleration   |
| MLX-VLM                 | Apple Silicon          | macOS native            |
| llama.cpp               | CPU (x64)             | CPU-optimized           |

**Docker is strongly recommended** for VL deployment to avoid environment
issues:

```bash
# Start vLLM-backed server
docker run --rm --gpus all --network host \
    <paddleocr-vl-image>:latest \
    paddleocr genai_server \
        --model_name PaddleOCR-VL-1.5-0.9B \
        --host 0.0.0.0 --port 8080 \
        --backend vllm

# Use with CLI
paddleocr doc_parser \
    --input image.png \
    --vl_rec_backend vllm-server \
    --vl_rec_server_url http://127.0.0.1:8080/v1
```

---

## C++ Inference

For native performance without Python overhead:

- Source: `deploy/cpp_infer/`
- Supports Linux and Windows
- Uses Paddle Inference C++ API
- Provides command-line tool for detection, recognition, classification

Build with CMake. See `deploy/cpp_infer/readme.md` for platform-specific
instructions.

---

## Mobile and Edge Deployment

### PaddleLite (Mobile)

`deploy/lite/` contains PaddleLite optimization configs for mobile devices.
PaddleLite provides model quantization and hardware-specific optimization.

### Android and iOS Demos

- `deploy/android_demo/` — Android OCR demo app
- `deploy/ios_demo/` — iOS OCR demo app

### ARM Virtual Hardware (Edge MCU)

`deploy/avh/` provides deployment to ARM Cortex-M/A targets for embedded
systems.

---

## Docker Deployment

Containerized deployment for serving:

```
deploy/docker/              # General-purpose Docker images
deploy/paddleocr_vl_docker/ # VL model Docker images
```

Docker images include all dependencies and provide ready-to-use serving
endpoints.

---

## Model Compression

`deploy/slim/` provides tools for reducing model size:

- **Quantization** — INT8 inference for smaller models and faster execution
- **Knowledge Distillation** — Train smaller models from larger teachers
- **Pruning** — Remove unnecessary model parameters

## What's Next?

- **[Testing & Debugging](testing-debugging.md)** — Running tests and
  debugging common issues
