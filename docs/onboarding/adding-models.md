# Adding Models

Step-by-step guide for adding new components to PaddleOCR's Layer 2 core ML
framework. Read [Architecture](architecture.md) first to understand the
four-component model pattern and the dynamic instantiation system.

## Overview

"Adding a model" in PaddleOCR means adding one or more of these components:

| Component       | Directory                         | Registry file              |
|-----------------|-----------------------------------|----------------------------|
| Backbone        | `ppocr/modeling/backbones/`       | `backbones/__init__.py`    |
| Neck            | `ppocr/modeling/necks/`           | `necks/__init__.py`        |
| Head            | `ppocr/modeling/heads/`           | `heads/__init__.py`        |
| Loss            | `ppocr/losses/`                   | `losses/__init__.py`       |
| Post-processor  | `ppocr/postprocess/`              | `postprocess/__init__.py`  |
| Metric          | `ppocr/metrics/`                  | `metrics/__init__.py`      |

All components follow the same registration pattern. Once registered, they
are selected by name in YAML config files.

---

## The Registration Pattern

Every component type uses the same pattern for dynamic instantiation:

```
YAML config                    __init__.py                    Python class
───────────                    ───────────                    ────────────

Backbone:            ──>    build_backbone(config, model_type)
  name: MobileNetV3              │
  scale: 0.5                     ├── 1. Pop "name" from config
                                 ├── 2. Assert name in support_dict
                                 ├── 3. eval("MobileNetV3")(scale=0.5)
                                 │
                                 v
                            MobileNetV3 instance
                            with self.out_channels set
```

This means:
1. Your class name in Python must **exactly match** the `name` field in YAML
2. Your class must be **imported** in the `__init__.py` where `eval()` runs
3. Your class name must be **listed** in the `support_dict` list
4. All remaining config keys (after `name` is popped) are passed as
   `**kwargs` to your constructor

---

## Adding a New Backbone

### Step 1: Create the Implementation File

Create a file in `ppocr/modeling/backbones/`. Follow the naming convention:
`{task}_{name}.py` (e.g., `det_my_backbone.py` or `rec_my_backbone.py`).

Your backbone must:
- Inherit from `paddle.nn.Layer`
- Accept `in_channels` as a constructor parameter
- Set `self.out_channels` (critical for the in_channels threading chain)

```python
# ppocr/modeling/backbones/det_my_backbone.py

import paddle
import paddle.nn as nn

class MyBackbone(nn.Layer):
    def __init__(self, in_channels, num_features=256, **kwargs):
        super().__init__()
        self.out_channels = num_features  # REQUIRED: next component reads this
        # ... build your layers ...

    def forward(self, x):
        # x shape: (batch, in_channels, H, W)
        # return: feature maps or dict of feature maps
        return x
```

### Step 2: Register in `__init__.py`

Edit `ppocr/modeling/backbones/__init__.py`. You must:
1. Add an import in the correct `model_type` branch
2. Add the class name to the `support_dict` list

```python
def build_backbone(config, model_type):
    if model_type == "det" or model_type == "table":
        from .det_mobilenet_v3 import MobileNetV3
        from .det_my_backbone import MyBackbone    # <-- ADD IMPORT
        # ...
        support_dict = [
            "MobileNetV3",
            "MyBackbone",    # <-- ADD TO LIST
            # ...
        ]
```

**Important: model_type gating.** The backbone is only available for the
model_type branch where you register it. If you register under `det`, it
will NOT be available when `model_type` is `rec`. Register in multiple
branches if needed.

### Step 3: Create a Config File

Create a YAML config in `configs/det/` (or the appropriate task directory):

```yaml
Architecture:
  model_type: det
  algorithm: DB
  Backbone:
    name: MyBackbone           # Must match class name exactly
    num_features: 256          # Passed as kwargs to constructor
  Neck:
    name: DBFPN
    out_channels: 256
  Head:
    name: DBHead
    k: 50
# ... Loss, Optimizer, PostProcess, Metric, Train, Eval sections ...
```

### Step 4: Smoke Test

```bash
# Quick training test (will fail on data if you don't have ICDAR, but
# verifies the model builds correctly)
python tools/train.py -c configs/det/my_config.yml \
    -o Global.epoch_num=1 \
    -o Global.save_model_dir=./output/test/
```

---

## Adding a New Head

Same pattern as backbones, but in `ppocr/modeling/heads/`:

1. Create `ppocr/modeling/heads/det_my_head.py` (or `rec_my_head.py`)
2. Must accept `in_channels` as constructor parameter
3. Import and add to `support_dict` in `ppocr/modeling/heads/__init__.py`

```python
# ppocr/modeling/heads/det_my_head.py
class MyDetHead(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        # ... build layers using in_channels ...
```

Heads do NOT need `out_channels` since they are the last component in the
chain.

Note: `build_head` does NOT use model_type gating — all heads are available
regardless of task type. But by convention, detection heads start with `det_`
and recognition heads with `rec_`.

---

## Adding a New Neck

Same pattern in `ppocr/modeling/necks/`:

1. Create `ppocr/modeling/necks/my_neck.py`
2. Must accept `in_channels` and set `self.out_channels`
3. Import and add to `support_dict` in `ppocr/modeling/necks/__init__.py`

---

## Adding a New Loss Function

In `ppocr/losses/`:

1. Create `ppocr/losses/my_loss.py`
2. Inherit from `paddle.nn.Layer`
3. Implement `forward(self, predicts, batch)` returning a dict with a
   `"loss"` key
4. Import and add to `support_dict` in `ppocr/losses/__init__.py`

```python
# ppocr/losses/my_loss.py
class MyLoss(nn.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.alpha = alpha

    def forward(self, predicts, batch):
        # predicts: model output (dict or tensor)
        # batch: ground truth data from dataloader
        loss = ...
        return {"loss": loss}
```

Config:
```yaml
Loss:
  name: MyLoss
  alpha: 2.0
```

---

## Adding a New Post-Processor

In `ppocr/postprocess/`:

1. Create or add to a file in `ppocr/postprocess/`
2. Implement `__call__(self, preds, label=None, *args, **kwargs)`
3. Import and add to `support_dict` in `ppocr/postprocess/__init__.py`

Post-processors convert raw model output (logits, probability maps) into
human-readable results (text strings, polygon coordinates).

---

## Adding a New Metric

In `ppocr/metrics/`:

1. Create `ppocr/metrics/my_metric.py`
2. Implement `__call__(self, preds, batch)` that accumulates results
3. Implement `get_metric(self)` that returns the final metric dict
4. Import and add to `support_dict` in `ppocr/metrics/__init__.py`

The metric dict must include the key specified by `main_indicator` in the
config (e.g., `hmean` for detection, `acc` for recognition).

---

## Creating a Complete Model Configuration

To create a full config from scratch, use an existing config as a template.
`configs/det/det_mv3_db.yml` is the simplest complete detection config.

Checklist for a new config:

- [ ] `Architecture.model_type` matches where your backbone is registered
- [ ] `Architecture.algorithm` is set (used for algorithm-specific logic in
      the training loop)
- [ ] All component `name` fields exactly match Python class names
- [ ] `Loss.name` matches a registered loss class
- [ ] `PostProcess.name` matches a registered post-processor
- [ ] `Metric.name` and `main_indicator` are set correctly
- [ ] `Train.dataset` and `Eval.dataset` point to valid data
- [ ] Transform pipeline is appropriate for your task

---

## The `extra_input_models` List (Critical Gotcha)

The training loop in `tools/program.py` has a hardcoded list called
`extra_input_models` (around line 267). Algorithms in this list pass extra
data to the model during the forward pass:

```python
extra_input_models = [
    "SRN", "NRTR", "SAR", "SEED", "SVTR", "SVTR_LCNet", "VisionLAN",
    "RobustScanner", "SPIN", "ABINet", "CPPD", "SATRN", "ParseQ",
    # ...
]
```

If your new algorithm needs `data=batch[1:]` in the forward pass (i.e., it
uses ground-truth labels during training beyond just the image), you **must**
add your algorithm name to this list. Otherwise, only the image tensor will be
passed to your model.

Similarly, the training and evaluation functions have algorithm-specific
branches for loss computation and metric evaluation. Check if your algorithm
needs special handling.

---

## In-Channels Threading

The four-component chain threads `in_channels` through each component:

```
in_channels = 3 (RGB input)
       │
       v
[Transform]  reads in_channels=3,  sets out_channels=3
       │
       v
[Backbone]   reads in_channels=3,  sets out_channels=256
       │
       v
[Neck]       reads in_channels=256, sets out_channels=128
       │
       v
[Head]       reads in_channels=128
```

If any component sets `out_channels` incorrectly, the next component
receives the wrong `in_channels` and you get a shape mismatch error during
the forward pass. Always verify `out_channels` is correct.

For backbones that produce multi-scale features (common in detection),
`out_channels` is typically a list (e.g., `[64, 128, 256, 512]`). The neck
must handle this list format.

---

## Common Pitfalls

1. **Class name mismatch**: `name: MyBackBone` in YAML but class is
   `MyBackbone` in Python — the `eval()` call will fail.

2. **Missing model_type registration**: You add a backbone for detection but
   try to use it in a recognition config. Register in the correct branch of
   `build_backbone`.

3. **Missing out_channels**: Your backbone or neck does not set
   `self.out_channels`. The next component in the chain gets `in_channels`
   from the previous component's attribute.

4. **Missing from extra_input_models**: Your model needs ground-truth data
   during training but isn't listed in `tools/program.py:extra_input_models`.
   Only the image tensor gets passed.

5. **Config key naming**: After `name` is popped, all remaining keys are
   passed as `**kwargs`. If your constructor parameter is `num_features` but
   you write `features` in YAML, you get an unexpected keyword argument error.
