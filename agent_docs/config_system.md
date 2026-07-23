# Config System

PaddleOCR has **two separate config systems** — one for training and one for inference. They are unrelated.

| | Training Config | Inference Config |
|---|---|---|
| **Used by** | `tools/train.py`, `tools/eval.py` | `paddleocr` Python API & CLI |
| **Format** | YAML files in `configs/` | PaddleX pipeline config (dict/YAML) |
| **Loading** | `tools/program.py` | PaddleX's `load_pipeline_config()` |

---

## Part 1: Training Config

### Overview

Training configs are YAML files in `configs/`, organized by task: `det/`, `rec/`, `cls/`, `table/`, `e2e/`, `kie/`, `sr/`. Config loading logic lives in `tools/program.py`.

### Running with Configs

```bash
# Basic training
python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml

# Override any value with -o (dot notation for nested keys)
python tools/train.py -c config.yml -o Global.use_gpu=false -o Optimizer.lr.learning_rate=0.001

# Multi-GPU
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c config.yml
```

The `-o` flag parses values as YAML: `true` → bool, `[1,2,3]` → list, `0.001` → float.

### Top-Level Sections

Every config has these top-level keys:

```yaml
Global:        # Device, epochs, checkpoints, character set
Architecture:  # Model definition (algorithm, Backbone, Neck, Head)
Loss:          # Loss function
Optimizer:     # Optimizer, learning rate schedule, regularizer
PostProcess:   # Post-processing (decode predictions to output format)
Train:         # Training dataset and data transforms
Eval:          # Evaluation dataset (optional)
Metric:        # Evaluation metric
```

### Global

Common keys:

```yaml
Global:
  use_gpu: true
  epoch_num: 500
  save_model_dir: ./output/model_name/
  save_epoch_step: 3
  eval_batch_step: [0, 2000]       # [start_iter, interval]
  print_batch_step: 10
  log_smooth_window: 20
  cal_metric_during_train: true

  # Pretrained weights / resume
  pretrained_model: null            # Path or URL
  checkpoints: null                 # Resume from checkpoint

  # Character set (recognition tasks)
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt
  max_text_length: 25
  use_space_char: true

  # Export
  save_inference_dir: ./inference/
  model_name: model_name            # Name for static export

  # AMP (optional)
  use_amp: false
  amp_level: O2                     # O1 or O2
  amp_dtype: float16                # float16 or bfloat16
```

### Architecture

Defines the model as composable components:

```yaml
Architecture:
  model_type: det                    # det, rec, cls, table, e2e, kie, sr
  algorithm: DB                      # Algorithm name (DB, CRNN, SLANet, etc.)
  Transform: null                    # Optional spatial transformer (TPS, STN)
  Backbone:
    name: ResNet_vd
    layers: 50
  Neck:
    name: DBFPN
    out_channels: 256
  Head:
    name: DBHead
    k: 50
```

Components are resolved via `support_dict` registries in `ppocr/modeling/`. To see available backbones, necks, heads — read the `support_dict` in the corresponding `__init__.py` under `ppocr/modeling/backbones/`, `ppocr/modeling/necks/`, `ppocr/modeling/heads/`.

**Distillation** uses a special structure:

```yaml
Architecture:
  name: DistillationModel
  algorithm: Distillation
  model_type: det
  Models:
    Student:
      algorithm: DB
      Backbone: { name: MobileNetV3, scale: 0.5 }
      Neck: { name: DBFPN, out_channels: 96 }
      Head: { name: DBHead, k: 50 }
    Teacher:
      algorithm: DB
      Backbone: { name: ResNet_vd, layers: 18 }
      Neck: { name: DBFPN, out_channels: 256 }
      Head: { name: DBHead, k: 50 }
```

### Optimizer

```yaml
Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  clip_norm_global: 5.0             # Gradient clipping
  lr:
    name: Cosine                    # LR scheduler
    learning_rate: 0.001
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 0.0005
```

Available LR schedulers (defined in `ppocr/optimizer/learning_rate.py`):
- `Const`, `Linear`, `Cosine`, `Step`, `Piecewise`, `MultiStepDecay`
- `LinearWarmupCosine`, `CyclicalCosine`, `OneCycle`, `TwoStepCosine`

### Train / Eval (Dataset + Transforms)

```yaml
Train:
  dataset:
    name: SimpleDataSet              # SimpleDataSet, LMDBDataSet, PubTabDataSet, etc.
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/labels.txt
    ratio_list: [1.0]                # Sampling ratio per file
    transforms:                      # Ordered pipeline
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - ResizeImg:
          image_shape: [3, 32, 100]
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
      - KeepKeys:
          keep_keys: ['image', 'label']
  loader:
    shuffle: true
    batch_size_per_card: 256
    drop_last: true
    num_workers: 8
    use_shared_memory: true
```

Transforms execute in order. Each is `{ TransformName: { params } }`. Available transforms are in `ppocr/data/imaug/`.

### YAML Anchors

Configs use YAML anchors to avoid repeating values:

```yaml
Global:
  max_text_length: &max_text_length 500

Architecture:
  Head:
    max_text_length: *max_text_length

Train:
  dataset:
    transforms:
      - SomeTransform:
          max_text_length: *max_text_length
```

### Config Loading Flow

```
config.yml → load_config() → dict
                                ↓
                         merge_config() ← -o CLI overrides
                                ↓
                         preprocess() → device setup, validation
                                ↓
                    ┌─── build_dataloader(config, "Train")
                    ├─── build_model(config["Architecture"])
                    ├─── build_loss(config["Loss"])
                    ├─── build_optimizer(config["Optimizer"], ...)
                    ├─── build_post_process(config["PostProcess"])
                    └─── build_metric(config["Metric"])
```

Config loading: `tools/program.py` (`load_config`, `merge_config`, `ArgsParser`).
Builder functions: `ppocr/data/__init__.py`, `ppocr/modeling/architectures/__init__.py`, `ppocr/losses/__init__.py`, `ppocr/optimizer/__init__.py`, `ppocr/postprocess/__init__.py`, `ppocr/metrics/__init__.py`.

### Key File Locations (Training)

| What | Where |
|------|-------|
| Config files | `configs/{det,rec,cls,table,e2e,kie,sr}/` |
| Config loading | `tools/program.py` |
| LR schedulers | `ppocr/optimizer/learning_rate.py` |
| Data transforms | `ppocr/data/imaug/` |
| Component registries | `ppocr/modeling/{backbones,necks,heads}/__init__.py` |

---

## Part 2: Inference Config (PaddleX)

### Overview

Inference in PaddleOCR 3.x is powered by PaddleX. Configuration flows through a layered system:

```
PaddleX built-in defaults → paddlex_config parameter → constructor kwargs → CLI args
```

Each layer overrides the previous. Users rarely need to touch PaddleX configs directly — constructor parameters are the primary interface.

### Configuration via Constructor Parameters

```python
from paddleocr import PaddleOCR

# Most common: use constructor kwargs to configure
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    text_det_limit_side_len=1280,
    text_det_thresh=0.3,
    text_rec_score_thresh=0.5,
)

# Runtime options (device, precision, acceleration)
ocr = PaddleOCR(
    device="gpu:0",
    enable_hpi=True,
    use_tensorrt=True,
    precision="fp16",
)
```

### Configuration via paddlex_config

For advanced use, pass a PaddleX config dict or YAML path directly:

```python
# Dict form
ocr = PaddleOCR(paddlex_config={
    "SubModules": {
        "TextDetection": {
            "model_name": "PP-OCRv5_server_det",
            "thresh": 0.25,
        }
    }
})

# Or a config name that PaddleX resolves
ocr = PaddleOCR(paddlex_config="OCR")
```

### PaddleX Config Structure

PaddleX configs use a nested dict with `SubModules` (individual models) and `SubPipelines` (nested pipelines):

```yaml
# Pipeline-level toggles
use_doc_orientation_classify: false
use_doc_unwarping: false

# Individual models
SubModules:
  TextDetection:
    model_name: PP-OCRv5_server_det
    thresh: 0.3
    batch_size: 1
  TextRecognition:
    model_name: ch_PP-OCRv5_server_rec
    score_thresh: 0.5
    batch_size: 8

# Nested sub-pipelines
SubPipelines:
  DocPreprocessor:
    SubModules:
      DocOrientationClassify:
        model_name: PP-LCNet_x1_0_doc_ori
```

### How Constructor Params Map to PaddleX Config

Each pipeline class implements `_get_paddlex_config_overrides()` that maps constructor kwargs to PaddleX config paths using dot notation:

```python
# Example mapping (from paddleocr/_pipelines/ocr.py)
"SubModules.TextDetection.model_name"  ← text_detection_model_name
"SubModules.TextDetection.thresh"      ← text_det_thresh
"SubModules.TextRecognition.score_thresh" ← text_rec_score_thresh
```

The utility `create_config_from_structure()` in `paddleocr/_pipelines/utils.py` converts these dot-notation mappings into nested dicts, which are then deep-merged with the base PaddleX config.

### Exporting Config

To inspect or save the merged config:

```python
ocr = PaddleOCR(text_det_thresh=0.3)
ocr.export_paddlex_config_to_yaml("my_config.yaml")
```

### Common Runtime Options

These apply to all pipelines and models:

| Parameter | Description |
|-----------|-------------|
| `device` | `"gpu:0"`, `"cpu"`, `"npu:0"` |
| `enable_hpi` | High-performance inference |
| `use_tensorrt` | TensorRT acceleration (GPU only) |
| `precision` | `"fp32"`, `"fp16"` |
| `enable_mkldnn` | MKL-DNN acceleration (CPU only) |
| `cpu_threads` | Number of CPU threads |
| `enable_cinn` | CINN compiler optimization |

Handled by `parse_common_args()` and `prepare_common_init_args()` in `paddleocr/_common_args.py`.

### Key File Locations (Inference)

| What | Where |
|------|-------|
| Pipeline base class | `paddleocr/_pipelines/base.py` (`PaddleXPipelineWrapper`) |
| Model base class | `paddleocr/_models/base.py` (`PaddleXPredictorWrapper`) |
| Common args parsing | `paddleocr/_common_args.py` |
| Config structure util | `paddleocr/_pipelines/utils.py` |
| Default constants | `paddleocr/_constants.py` |
| Config override examples | `paddleocr/_pipelines/ocr.py`, `paddleocr/_pipelines/pp_structurev3.py` |
