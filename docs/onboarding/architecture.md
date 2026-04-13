# PaddleOCR Architecture

This document explains the internal architecture of PaddleOCR. It covers
the two-layer design, model composition, the configuration system, and the
data pipeline.

## Two-Layer Architecture Overview

PaddleOCR has two distinct layers of code that serve different purposes:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: Pipeline API  (paddleocr/)                               │
│                                                                     │
│  User-facing Python API and CLI. Clean 2025-era code using ABCs.   │
│  Wraps PaddleX for model management and inference orchestration.   │
│                                                                     │
│  Entry: paddleocr/__init__.py                                       │
│  Key:   PaddleOCR, PPStructureV3, PaddleOCRVL, CLI subcommands    │
├─────────────────────────────────────────────────────────────────────┤
│  PaddleX  (external dependency)                                     │
│                                                                     │
│  Model registry, download, inference engine, pipeline orchestration │
│  Lives outside this repo — treat as a black box                     │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: Core ML Framework  (ppocr/)                               │
│                                                                     │
│  Model definitions, training loop, losses, metrics, data pipeline.  │
│  Original 2020-era code using eval()-based dynamic instantiation.   │
│                                                                     │
│  Entry: tools/train.py, tools/eval.py, tools/export_model.py       │
│  Key:   BaseModel, build_backbone/neck/head, YAML configs          │
└─────────────────────────────────────────────────────────────────────┘
```

**When to use which layer:**

| Task                           | Layer   | Entry point                      |
|--------------------------------|---------|----------------------------------|
| Run inference on an image      | Layer 1 | `PaddleOCR().predict("img.jpg")` |
| Deploy to production           | Layer 1 | `paddleocr ocr --input img.jpg`  |
| Train a model                  | Layer 2 | `python tools/train.py -c ...`   |
| Add a new model component      | Layer 2 | Edit `ppocr/modeling/`           |
| Evaluate a checkpoint          | Layer 2 | `python tools/eval.py -c ...`    |
| Export for deployment           | Layer 2 | `python tools/export_model.py`   |

---

## Layer 1: Pipeline API (`paddleocr/`)

### Entry Point

`paddleocr/__init__.py` exports everything users interact with:

**13 model classes** (each wraps a single PaddleX predictor):

| Class                              | Task                             |
|------------------------------------|----------------------------------|
| `TextDetection`                    | Locate text regions              |
| `TextRecognition`                  | Read text from cropped regions   |
| `TextLineOrientationClassification`| Classify text line rotation      |
| `DocImgOrientationClassification`  | Classify document page rotation  |
| `TextImageUnwarping`               | Correct perspective distortion   |
| `LayoutDetection`                  | Detect document layout regions   |
| `TableStructureRecognition`        | Parse table structure            |
| `TableCellsDetection`             | Detect table cell boundaries     |
| `TableClassification`             | Classify table type (wired/wireless) |
| `FormulaRecognition`              | Convert formula images to LaTeX  |
| `SealTextDetection`               | Detect text in seal stamps       |
| `ChartParsing`                    | Parse chart images               |
| `DocVLM`                          | Vision-Language document parsing |

**10 pipeline classes** (each orchestrates multiple models):

| Class                       | Pipeline                                 |
|-----------------------------|------------------------------------------|
| `PaddleOCR`                 | General OCR (detection + recognition)    |
| `PPStructureV3`             | Document structure analysis              |
| `PaddleOCRVL`              | Vision-Language document parsing         |
| `PPChatOCRv4Doc`           | Chat-based document QA                   |
| `PPDocTranslation`         | Document translation                     |
| `DocPreprocessor`          | Image preprocessing (orientation, unwarp)|
| `DocUnderstanding`         | Document comprehension                   |
| `FormulaRecognitionPipeline`| Formula OCR pipeline                    |
| `SealRecognition`          | Seal stamp text extraction               |
| `TableRecognitionPipelineV2`| Advanced table recognition              |

### Base Classes

All models inherit from `PaddleXPredictorWrapper`
(`paddleocr/_models/base.py:30`):

```python
class PaddleXPredictorWrapper(metaclass=abc.ABCMeta):
    def __init__(self, *, model_name=None, model_dir=None, **common_args):
        self._model_name = model_name or self.default_model_name
        self.paddlex_predictor = self._create_paddlex_predictor()

    def _create_paddlex_predictor(self):
        return create_predictor(model_name=self._model_name, ...)

    def predict(self, *args, **kwargs):
        return list(self.predict_iter(*args, **kwargs))
```

All pipelines inherit from `PaddleXPipelineWrapper`
(`paddleocr/_pipelines/base.py:54`):

```python
class PaddleXPipelineWrapper(metaclass=abc.ABCMeta):
    def __init__(self, *, paddlex_config=None, **common_args):
        self._merged_paddlex_config = self._get_merged_paddlex_config()
        self.paddlex_pipeline = self._create_paddlex_pipeline()

    def _create_paddlex_pipeline(self):
        return create_pipeline(config=self._merged_paddlex_config, ...)
```

The pattern is the same: take user parameters, merge into a PaddleX
configuration, and delegate to PaddleX's `create_predictor` or
`create_pipeline`.

### CLI System

`paddleocr/_cli.py` registers every model and pipeline as a CLI subcommand:

```bash
paddleocr ocr --input image.jpg              # PaddleOCR pipeline
paddleocr pp_structurev3 --input doc.pdf      # PPStructureV3 pipeline
paddleocr doc_parser --input page.png         # PaddleOCRVL pipeline
paddleocr text_detection --input image.jpg    # Single model inference
```

### PaddleOCR-VL (Vision-Language Model)

PaddleOCR-VL is a fundamentally different approach from the traditional
pipeline. Instead of separate detection and recognition models, it uses a
single 0.9B-parameter Vision-Language Model:

```
┌────────────────────────────────────────────────────────────┐
│  PaddleOCR-VL-1.5  (0.9B parameters)                      │
│                                                            │
│  ┌───────────────┐    ┌──────────────────────────────┐    │
│  │ NaViT-style   │    │  ERNIE-4.5-0.3B              │    │
│  │ Visual Encoder │──>│  Language Model               │──> │ Markdown
│  │ (dynamic res) │    │  (text generation)            │    │ Output
│  └───────────────┘    └──────────────────────────────┘    │
│                                                            │
│  Handles: text, tables, formulas, charts, seals            │
│  Languages: 109                                            │
│  Accuracy: 94.5% on OmniDocBench v1.5                     │
└────────────────────────────────────────────────────────────┘
```

Use PaddleOCR-VL when you need end-to-end document parsing. Use the
traditional OCR pipeline when you need fine-grained control over individual
detection/recognition stages.

### How a Prediction Flows Through Layer 1

```
User code:
  ocr = PaddleOCR(lang="en")
  result = ocr.predict("image.jpg")
         │
         v
PaddleOCR.__init__()                   [paddleocr/_pipelines/ocr.py:55]
  │  Resolves model names from lang/version
  │  Builds config overrides dict
  │  Calls super().__init__()
  v
PaddleXPipelineWrapper.__init__()      [paddleocr/_pipelines/base.py:54]
  │  Loads PaddleX pipeline config
  │  Merges user overrides
  │  Calls create_pipeline(config=...)
  v
PaddleX (external)
  │  Downloads models if needed
  │  Initializes sub-models (det, rec, cls)
  │  Returns pipeline object
  v
ocr.predict("image.jpg")
  │  Delegates to paddlex_pipeline.predict()
  v
PaddleX pipeline orchestration:
  │  1. [Optional] Doc orientation classification
  │  2. [Optional] Text image unwarping
  │  3. Text detection (finds text regions)
  │  4. [Optional] Text line orientation classification
  │  5. Text recognition (reads text in each region)
  v
Result: list of dicts with rec_texts, rec_scores, dt_polys, etc.
```

---

## Layer 2: Core ML Framework (`ppocr/`)

This is where model architectures are defined, training happens, and new
algorithms are implemented.

### The Four-Component Model Architecture

Every model in Layer 2 is built from up to four components chained together.
This is defined in `ppocr/modeling/architectures/base_model.py:27-117`:

```
                  in_channels = 3 (RGB)
                        │
                        v
              ┌──────────────────┐
              │    Transform     │  Optional. TPS (Thin Plate Spline)
              │                  │  for text rectification (rec only).
              │  out_channels ───│──> in_channels for next component
              └──────────────────┘
                        │
                        v
              ┌──────────────────┐
              │    Backbone      │  Feature extractor.
              │                  │  MobileNetV3, ResNet, SVTRNet, etc.
              │  out_channels ───│──> in_channels for next component
              └──────────────────┘
                        │
                        v
              ┌──────────────────┐
              │    Neck          │  Feature fusion / sequence encoding.
              │                  │  DBFPN, SequenceEncoder, RNN, etc.
              │  out_channels ───│──> in_channels for next component
              └──────────────────┘
                        │
                        v
              ┌──────────────────┐
              │    Head          │  Task-specific output layer.
              │                  │  DBHead, CTCHead, ClsHead, etc.
              └──────────────────┘
                        │
                        v
                  Predictions
            (logits, probability maps, etc.)
```

**Critical: in_channels threading.** Each component reads `in_channels` from
its config, and must set `self.out_channels` so the next component in the
chain receives the correct value. If `out_channels` is wrong or missing, you
get a shape mismatch error.

Here is the actual code (simplified):

```python
class BaseModel(nn.Layer):
    def __init__(self, config):
        in_channels = config.get("in_channels", 3)
        model_type = config["model_type"]

        # Each component: set in_channels, build, read out_channels
        if config.get("Transform"):
            config["Transform"]["in_channels"] = in_channels
            self.transform = build_transform(config["Transform"])
            in_channels = self.transform.out_channels

        config["Backbone"]["in_channels"] = in_channels
        self.backbone = build_backbone(config["Backbone"], model_type)
        in_channels = self.backbone.out_channels

        if config.get("Neck"):
            config["Neck"]["in_channels"] = in_channels
            self.neck = build_neck(config["Neck"])
            in_channels = self.neck.out_channels

        config["Head"]["in_channels"] = in_channels
        self.head = build_head(config["Head"])
```

### Dynamic Instantiation Pattern

All `build_*` functions follow the same pattern — they pop the `"name"` key
from the config dict and use `eval()` to instantiate the class:

```python
# ppocr/modeling/backbones/__init__.py:145-152
def build_backbone(config, model_type):
    # ... imports gated by model_type ...
    module_name = config.pop("name")
    assert module_name in support_dict
    module_class = eval(module_name)(**config)
    return module_class
```

This pattern is used by `build_backbone`, `build_neck`, `build_head`,
`build_loss`, `build_post_process`, and `build_metric`. It means:

1. **Class names in YAML must exactly match Python class names.** `"DBHead"`
   works. `"Dbhead"` or `"DB_Head"` does not.
2. **The class must be imported** in the `__init__.py` file where `eval()` runs.

### model_type Gating (Backbones Only)

`build_backbone` is unique — it uses `model_type` to control which backbones
are available:

```python
def build_backbone(config, model_type):
    if model_type == "det" or model_type == "table":
        from .det_mobilenet_v3 import MobileNetV3
        # ... detection backbones ...
        support_dict = ["MobileNetV3", "ResNet", "ResNet_vd", ...]
    elif model_type == "rec" or model_type == "cls":
        from .rec_mobilenet_v3 import MobileNetV3
        # ... recognition backbones ...
        support_dict = ["MobileNetV1Enhance", "MobileNetV3", ...]
    elif model_type == "kie":
        # ... KIE backbones ...
    # ...
```

A backbone registered under `det` is **invisible** when `model_type` is `rec`.
Some class names (like `MobileNetV3`) appear in multiple branches but are
imported from different files with different implementations.

### Component Inventory

**Backbones** (`ppocr/modeling/backbones/`):
- Detection: MobileNetV3, ResNet, ResNet_vd, PPLCNet, PPLCNetV3, PPHGNet,
  RepSVTR_det (12 classes across det/table)
- Recognition: MobileNetV3, ResNet, SVTRNet, SVTRv2, PPLCNetV3, DenseNet,
  ViTSTR, DonutSwinModel, PPHGNetV2_B4, and more (28 classes across rec/cls)
- KIE: Kie_backbone, LayoutLMForSer, LayoutXLMForSer, etc. (6 classes)

**Necks** (`ppocr/modeling/necks/`):
- Detection: DBFPN, EASTFPN, SASTFPN, FCEFPN, CSPPAN, RSEFPN
- Recognition: SequenceEncoder (wraps RNN/BiLSTM), SVTR
- Table: TableFPN

**Heads** (`ppocr/modeling/heads/`):
- Detection: DBHead, EASTHead, SASTHead, PSEHead, FCEHead, CT_Head, DRRGHead
- Recognition: CTCHead, AttentionHead, SARHead, MultiHead, ParseQHead,
  ABINetHead, and many more (20+ recognition heads)
- Classification: ClsHead
- Table: TableAttentionHead, SLAHead, TableMasterHead
- KIE: SDMGRHead

**Transforms** (`ppocr/modeling/transforms/`):
- TPS (Thin Plate Spline) — geometric text rectification, recognition only

### DistillationModel

`ppocr/modeling/architectures/distillation_model.py` defines a
teacher-student setup for knowledge distillation. It wraps multiple
BaseModel instances and routes data through teacher and student models
during training.

---

## The Configuration System

PaddleOCR is config-driven. YAML files in `configs/` define every aspect of
a model: architecture, loss, optimizer, data, and evaluation.

### Config File Structure

Here is a complete config annotated section by section, using
`configs/det/det_mv3_db.yml` as the example:

```yaml
# ── Global training parameters ──────────────────────────────────
Global:
  use_gpu: true
  epoch_num: 1200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/db_mv3/
  save_epoch_step: 1200
  eval_batch_step: [0, 2000]     # evaluate every 2000 iterations
  cal_metric_during_train: false
  pretrained_model: ./pretrain_models/MobileNetV3_large_x0_5_pretrained
  checkpoints:                    # resume training from checkpoint
  save_inference_dir:             # where to save exported model

# ── Model architecture ──────────────────────────────────────────
Architecture:
  model_type: det                 # det | rec | cls | kie | table | e2e
  algorithm: DB                   # used for algorithm-specific logic
  Transform:                      # null for detection
  Backbone:
    name: MobileNetV3             # must match Python class name exactly
    scale: 0.5
    model_name: large
  Neck:
    name: DBFPN
    out_channels: 256
  Head:
    name: DBHead
    k: 50                         # DB binarization parameter

# ── Loss function ───────────────────────────────────────────────
Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3

# ── Optimizer ───────────────────────────────────────────────────
Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    learning_rate: 0.001
  regularizer:
    name: L2
    factor: 0

# ── Post-processing (decoding model output) ─────────────────────
PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5

# ── Evaluation metric ──────────────────────────────────────────
Metric:
  name: DetMetric
  main_indicator: hmean           # metric for best checkpoint selection

# ── Training data ──────────────────────────────────────────────
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/icdar2015/text_localization/
    label_file_list:
      - ./train_data/icdar2015/text_localization/train_icdar2015_label.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - DetLabelEncode:
      - IaaAugment: ...           # data augmentation
      - MakeBorderMap: ...        # generate DB target maps
      - MakeShrinkMap: ...
      - NormalizeImage: ...
      - ToCHWImage:
      - KeepKeys:
          keep_keys: [image, threshold_map, threshold_mask, ...]
  loader:
    shuffle: true
    batch_size_per_card: 16
    num_workers: 8

# ── Evaluation data ────────────────────────────────────────────
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/icdar2015/text_localization/
    label_file_list:
      - ./train_data/icdar2015/text_localization/test_icdar2015_label.txt
    transforms:
      - DecodeImage: ...
      - DetLabelEncode:
      - DetResizeForTest:
          image_shape: [736, 1280]
      - NormalizeImage: ...
      - ToCHWImage:
      - KeepKeys:
          keep_keys: [image, shape, polys, ignore_tags]
  loader:
    shuffle: false
    batch_size_per_card: 1
```

### Config Loading and CLI Overrides

Configs are loaded and merged in `tools/program.py`:

```python
# Load from YAML file
config = load_config(file_path)         # program.py:75

# Merge CLI overrides: -o key=value
merge_config(config, opts)              # program.py:88

# CLI: python tools/train.py -c config.yml -o Global.use_gpu=false
```

The `-o` flag uses dot notation for nested keys:
`-o Architecture.Backbone.name=ResNet_vd`

### How Config Maps to Code

```
YAML Config                          Python Code
────────────────                     ──────────────────

Architecture:
  model_type: det       ──────────>  build_model(config["Architecture"])
  algorithm: DB                            │
  Backbone:                                v
    name: MobileNetV3   ──────────>  build_backbone(config, "det")
    scale: 0.5                         eval("MobileNetV3")(scale=0.5)
  Neck:                                    │
    name: DBFPN         ──────────>  build_neck(config)
    out_channels: 256                  eval("DBFPN")(out_channels=256)
  Head:                                    │
    name: DBHead        ──────────>  build_head(config)
    k: 50                              eval("DBHead")(k=50)

Loss:
  name: DBLoss          ──────────>  build_loss(config["Loss"])
                                       eval("DBLoss")(**params)

PostProcess:
  name: DBPostProcess   ──────────>  build_post_process(config["PostProcess"])
                                       eval("DBPostProcess")(**params)

Metric:
  name: DetMetric       ──────────>  build_metric(config["Metric"])
                                       eval("DetMetric")(**params)
```

---

## The Data Pipeline

### Dataset Classes

All datasets live in `ppocr/data/`:

| Class              | Format                    | Use case                 |
|--------------------|---------------------------|--------------------------|
| `SimpleDataSet`    | Image files + label txt   | Most common              |
| `LMDBDataSet`      | LMDB key-value store      | Large-scale training     |
| `MultiScaleDataSet`| Multi-resolution images   | Multi-scale training     |
| `PubTabDataSet`    | Table structure data      | Table recognition        |

**SimpleDataSet label format** — one line per image:

```
image_path\t[{"transcription": "text", "points": [[x1,y1], ...]}]
```

### Transform Chain

Data augmentation is configured as a list of operators in the YAML config
under `Train.dataset.transforms`. Each operator is a class in
`ppocr/data/imaug/`:

```yaml
transforms:
  - DecodeImage:         # Load image from path
      img_mode: BGR
  - DetLabelEncode:      # Parse label annotations
  - IaaAugment:          # Random augmentation (flip, rotate, resize)
  - MakeBorderMap:       # Generate DB border regression target
  - MakeShrinkMap:       # Generate DB shrink map target
  - NormalizeImage:      # Scale to [0,1], apply mean/std
  - ToCHWImage:          # Convert HWC to CHW format
  - KeepKeys:            # Select which fields to pass to the model
```

Operators are instantiated by `create_operators()` in
`ppocr/data/imaug/__init__.py` using the same `eval()` pattern.

### DataLoader Construction

`ppocr/data/__init__.py:build_dataloader` builds a PaddlePaddle DataLoader
from the config:

```
YAML Train/Eval section
         │
         v
build_dataloader(config, "Train", device, logger, seed)
         │
         ├── Instantiates dataset class (SimpleDataSet, etc.)
         ├── Creates transform pipeline from config
         ├── Sets up batch sampler
         └── Returns paddle.io.DataLoader
```

---

## What's Next?

- **[Codebase Map](codebase-map.md)** — Quick reference for where things
  live in the repo
- **[Adding Models](adding-models.md)** — Step-by-step guide for adding
  new components
