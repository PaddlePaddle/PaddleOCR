# Codebase Map

Quick-reference guide to the PaddleOCR repository. Skim this once to build
spatial awareness, then return when you need to find something specific.

## Top-Level Directory Structure

```
PADDLEOCR/
├── paddleocr/          # Layer 1: Pipeline API (user-facing)
├── ppocr/              # Layer 2: Core ML framework (model training)
├── ppstructure/        # Legacy PP-Structure (superseded by PPStructureV3)
├── configs/            # YAML training configurations
├── tools/              # Training, eval, export scripts
├── deploy/             # Deployment: C++, ONNX, mobile, Docker, serving
├── tests/              # Pytest test suite
├── docs/               # User-facing documentation (MkDocs, multilingual)
├── doc/                # Font files for text rendering
├── benchmark/          # Performance benchmarking scripts
├── mcp_server/         # Claude MCP server integration
├── langchain-paddleocr/# LangChain integration wrapper
├── skills/             # Claude Code skills
├── test_tipc/          # Training & Inference Precision Comparison tests
├── applications/       # Application examples
├── readme/             # Multilingual README files
├── .github/            # CI/CD workflows, issue templates
├── pyproject.toml      # Python package metadata
├── setup.py            # Setuptools entry (delegates to pyproject.toml)
├── requirements.txt    # Core dependencies
└── mkdocs.yml          # Documentation site configuration
```

## paddleocr/ — Pipeline API (Layer 1)

```
paddleocr/
├── __init__.py          # Exports: 13 models + 10 pipelines + logger
├── __main__.py          # python -m paddleocr entry point
├── _cli.py              # CLI: registers subcommands for all models/pipelines
├── _common_args.py      # Shared CLI argument definitions
├── _abstract.py         # CLISubcommandExecutor ABC
├── _version.py          # Package version
├── _models/             # Model wrappers (one file per model)
│   ├── base.py          # PaddleXPredictorWrapper ABC
│   ├── text_detection.py
│   ├── text_recognition.py
│   ├── layout_detection.py
│   ├── table_structure_recognition.py
│   ├── formula_recognition.py
│   ├── doc_vlm.py
│   └── ...              # 13 model classes total
├── _pipelines/          # Pipeline orchestrators (one file per pipeline)
│   ├── base.py          # PaddleXPipelineWrapper ABC
│   ├── ocr.py           # PaddleOCR (general OCR pipeline)
│   ├── pp_structurev3.py# PPStructureV3 (document structure)
│   ├── paddleocr_vl.py  # PaddleOCRVL (Vision-Language)
│   ├── pp_chatocrv4.py  # PPChatOCRv4Doc
│   └── ...              # 10 pipeline classes total
└── _utils/              # Utilities: logging, CLI helpers, deprecation
```

## ppocr/ — Core ML Framework (Layer 2)

```
ppocr/
├── modeling/
│   ├── architectures/
│   │   ├── __init__.py          # build_model()
│   │   ├── base_model.py        # BaseModel: Transform→Backbone→Neck→Head
│   │   └── distillation_model.py# DistillationModel: teacher-student
│   ├── backbones/
│   │   ├── __init__.py          # build_backbone(config, model_type)
│   │   ├── det_mobilenet_v3.py  # Detection MobileNetV3
│   │   ├── rec_mobilenet_v3.py  # Recognition MobileNetV3
│   │   ├── rec_svtrnet.py       # SVTRNet backbone
│   │   └── ...                  # 30+ backbone implementations
│   ├── necks/
│   │   ├── __init__.py          # build_neck(config)
│   │   ├── db_fpn.py            # DBFPN for detection
│   │   ├── rnn.py               # SequenceEncoder for recognition
│   │   └── ...                  # 15+ neck implementations
│   ├── heads/
│   │   ├── __init__.py          # build_head(config)
│   │   ├── det_db_head.py       # DBHead for DB detection
│   │   ├── rec_ctc_head.py      # CTCHead for CTC recognition
│   │   ├── rec_multi_head.py    # MultiHead (CTC + attention combined)
│   │   ├── cls_head.py          # ClsHead for classification
│   │   └── ...                  # 30+ head implementations
│   └── transforms/
│       ├── __init__.py          # build_transform(config)
│       └── tps.py               # Thin Plate Spline (rec text rectification)
├── data/
│   ├── __init__.py              # build_dataloader()
│   ├── simple_dataset.py        # SimpleDataSet (file-based)
│   ├── lmdb_dataset.py          # LMDBDataSet (LMDB format)
│   └── imaug/
│       ├── __init__.py          # create_operators(), transform()
│       ├── operators.py         # DecodeImage, NormalizeImage, ToCHWImage
│       ├── label_ops.py         # DetLabelEncode, CTCLabelEncode
│       ├── iaa_augment.py       # IaaAugment, RandAugment
│       ├── make_border_map.py   # DB border map target generation
│       ├── make_shrink_map.py   # DB shrink map target generation
│       └── ...                  # 20+ augmentation operators
├── losses/
│   ├── __init__.py              # build_loss(): 37 loss classes
│   ├── det_db_loss.py           # DBLoss
│   ├── rec_ctc_loss.py          # CTCLoss
│   ├── combined_loss.py         # CombinedLoss (weighted sum)
│   └── ...                      # 35+ loss implementations
├── postprocess/
│   ├── __init__.py              # build_post_process(): 30+ decoders
│   ├── db_postprocess.py        # DBPostProcess (binary→contour→polygon)
│   ├── rec_postprocess.py       # CTCLabelDecode, AttnLabelDecode, etc.
│   ├── cls_postprocess.py       # ClsPostProcess
│   └── ...
├── metrics/
│   ├── __init__.py              # build_metric(): 15 metric classes
│   ├── det_metric.py            # DetMetric (precision, recall, hmean)
│   ├── rec_metric.py            # RecMetric (accuracy)
│   ├── cls_metric.py            # ClsMetric
│   └── ...
├── optimizer/
│   ├── __init__.py              # build_optimizer()
│   ├── optimizer.py             # Optimizer wrappers
│   ├── learning_rate.py         # 15+ LR schedulers (Cosine, Poly, Step)
│   └── regularizer.py           # L1/L2 regularization
└── utils/
    ├── save_load.py             # Checkpoint save/load
    ├── logging.py               # Logger configuration
    ├── utility.py               # Common utilities
    ├── export_model.py          # Dynamic→static graph export
    └── ...
```

## configs/ — Training Configurations

```
configs/
├── det/                         # Text detection configs
│   ├── PP-OCRv5/                # PP-OCRv5 detection (latest)
│   ├── PP-OCRv4/
│   ├── PP-OCRv3/
│   └── det_mv3_db.yml           # Classic MobileNetV3 + DB example
├── rec/                         # Text recognition configs
│   ├── PP-OCRv5/
│   │   ├── PP-OCRv5_mobile_rec.yml
│   │   ├── PP-OCRv5_server_rec.yml
│   │   └── multi_language/      # Per-language configs
│   ├── SVTRv2/
│   └── PP-FormulaNet/           # Formula recognition configs
├── cls/                         # Text angle classification configs
├── table/                       # Table recognition configs
├── kie/                         # Key Information Extraction configs
├── e2e/                         # End-to-end detection+recognition
└── sr/                          # Super-resolution configs
```

**Config naming convention**: `{algorithm}_{backbone}_{extras}.yml`
(e.g., `det_mv3_db.yml` = detection, MobileNetV3, DB algorithm)

## tools/ — Training & Inference Scripts

| File                | Purpose                                    |
|---------------------|--------------------------------------------|
| `train.py`          | Main training entry point                  |
| `eval.py`           | Model evaluation                           |
| `export_model.py`   | Export to static graph for deployment      |
| `program.py`        | Config loading, training loop, eval loop   |
| `infer_det.py`      | Detection inference (Layer 2 style)        |
| `infer_rec.py`      | Recognition inference                      |
| `infer_cls.py`      | Classification inference                   |
| `infer_kie.py`      | KIE inference                              |
| `infer/predict_system.py` | Full det+rec pipeline inference      |

## deploy/ — Deployment Targets

| Directory          | Target                                       |
|--------------------|----------------------------------------------|
| `cpp_infer/`       | C++ native inference (Linux/Windows)         |
| `paddle2onnx/`     | ONNX model conversion                       |
| `lite/`            | PaddleLite (mobile optimization)             |
| `android_demo/`    | Android app demo                             |
| `ios_demo/`        | iOS app demo                                 |
| `hubserving/`      | PaddleHub Serving (REST API)                 |
| `docker/`          | Docker images for serving                    |
| `paddleocr_vl_docker/` | Docker for VL model deployment           |
| `avh/`             | ARM Virtual Hardware (edge MCU)              |
| `slim/`            | Model compression (quantization, pruning)    |

## tests/ — Test Suite

```
tests/
├── testing_utils.py             # Test helpers: TEST_DATA_DIR, assertions
├── test_files/                  # Sample images for testing
├── models/                      # Per-model unit tests (13 files)
│   ├── test_text_detection.py
│   ├── test_text_recognition.py
│   ├── test_layout_detection.py
│   └── ...
└── pipelines/                   # Per-pipeline integration tests (10 files)
    ├── test_ocr.py
    ├── test_pp_structurev3.py
    └── ...
```

## ppstructure/ — Legacy (Being Superseded)

The `ppstructure/` directory contains the original PP-Structure
implementation for table recognition, layout analysis, and KIE. This is
being replaced by the `PPStructureV3` pipeline in `paddleocr/_pipelines/`.
**Do not build new features on ppstructure/.**

## Quick Reference Table

| I want to...                        | Look at                                                   |
|-------------------------------------|-----------------------------------------------------------|
| Add a new backbone                  | `ppocr/modeling/backbones/__init__.py`                     |
| Add a new head                      | `ppocr/modeling/heads/__init__.py`                         |
| Add a new neck                      | `ppocr/modeling/necks/__init__.py`                         |
| Add a new loss function             | `ppocr/losses/__init__.py`                                 |
| Add a new post-processor            | `ppocr/postprocess/__init__.py`                            |
| Add a new metric                    | `ppocr/metrics/__init__.py`                                |
| Add a new data augmentation         | `ppocr/data/imaug/`                                       |
| Create a training config            | `configs/det/det_mv3_db.yml` (as template)                 |
| Understand the model composition    | `ppocr/modeling/architectures/base_model.py`               |
| Understand the training loop        | `tools/program.py` train() function                        |
| Understand config loading           | `tools/program.py` load_config(), merge_config()           |
| See how the OCR pipeline works      | `paddleocr/_pipelines/ocr.py`                              |
| See how the VL pipeline works       | `paddleocr/_pipelines/paddleocr_vl.py`                     |
| See all exported classes            | `paddleocr/__init__.py`                                    |
| Run CLI inference                   | `paddleocr/_cli.py`                                        |
| Export a model                      | `tools/export_model.py` + `ppocr/utils/export_model.py`   |
| Find character dictionaries         | `ppocr/utils/dict/`                                        |
