# Training and Evaluation

End-to-end guide for training PaddleOCR models and running evaluation. This
covers the training pipeline, data preparation, evaluation, and checkpointing.

## Training Overview

```
┌──────────────┐     ┌─────────────────┐     ┌───────────────────────┐
│              │     │                 │     │                       │
│  YAML Config │────>│  program.       │────>│  Build Components:    │
│  (-c flag)   │     │  preprocess()   │     │  - DataLoader         │
│              │     │                 │     │  - Model (BaseModel)  │
└──────────────┘     └─────────────────┘     │  - Loss               │
                                              │  - Optimizer + LR     │
                                              │  - PostProcess        │
┌──────────────┐     ┌─────────────────┐     │  - Metric             │
│ CLI Override │────>│  merge_config() │────>│                       │
│ (-o flag)    │     │                 │     └───────────┬───────────┘
└──────────────┘     └─────────────────┘                 │
                                                          v
                                              ┌───────────────────────┐
                                              │  program.train()      │
                                              │                       │
                                              │  For each epoch:      │
                                              │    For each batch:    │
                                              │      forward pass     │
                                              │      compute loss     │
                                              │      backward + step  │
                                              │    Periodic eval      │
                                              │    Save checkpoint    │
                                              └───────────────────────┘
```

The entry point is `tools/train.py`. It calls `program.preprocess()` to
load config and set up logging, then `main()` to build all components, and
finally `program.train()` for the training loop.

## Prerequisites

- **PaddlePaddle** with GPU support (for training)
- **Training data** in the expected format (see Data Preparation below)
- **A config file** (start by copying an existing one from `configs/`)

## Running Training

### Single-GPU Training

```bash
python tools/train.py -c configs/det/det_mv3_db.yml
```

### Multi-GPU Training

```bash
python -m paddle.distributed.launch \
    --gpus '0,1,2,3' \
    tools/train.py \
    -c configs/det/det_mv3_db.yml
```

### CLI Config Overrides

Override any config value with `-o`:

```bash
python tools/train.py -c configs/det/det_mv3_db.yml \
    -o Global.use_gpu=true \
    -o Global.epoch_num=100 \
    -o Train.loader.batch_size_per_card=8 \
    -o Global.pretrained_model=./my_pretrain/model \
    -o Architecture.Backbone.name=ResNet_vd
```

Dot notation navigates nested keys. This is implemented in
`tools/program.py:merge_config()`.

## The Training Loop in Detail

### 1. `program.preprocess()` (tools/program.py)

- Parses CLI args (`-c` config path, `-o` overrides, `-p` profiler)
- Loads YAML config via `load_config()`
- Merges CLI overrides via `merge_config()`
- Sets up device (GPU/CPU), logging, and VisualDL writer
- Returns `(config, device, logger, vdl_writer)`

### 2. `train.py:main()` (tools/train.py:46)

Builds all components in this order:

```python
# 1. Build data loaders
train_dataloader = build_dataloader(config, "Train", device, logger, seed)
valid_dataloader = build_dataloader(config, "Eval", device, logger, seed)

# 2. Build post-processor (needed before model for character count)
post_process_class = build_post_process(config["PostProcess"], global_config)

# 3. Build model (may adjust head out_channels based on char count)
model = build_model(config["Architecture"])

# 4. Build loss
loss_class = build_loss(config["Loss"])

# 5. Build optimizer and LR scheduler
optimizer, lr_scheduler = build_optimizer(config["Optimizer"], ...)

# 6. Build metric
eval_class = build_metric(config["Metric"])

# 7. Load pretrained weights or resume from checkpoint
pre_best_model_dict = load_model(config, model, optimizer, ...)

# 8. Start training
program.train(config, train_dataloader, valid_dataloader, device, model,
              loss_class, optimizer, lr_scheduler, post_process_class,
              eval_class, ...)
```

**Why is post-processing built before the model?** For recognition models,
the post-processor loads the character dictionary, and the character count
determines the output dimension of the recognition head. The training script
reads `len(post_process_class.character)` and sets
`config["Architecture"]["Head"]["out_channels"]` accordingly.

### 3. `program.train()` (tools/program.py)

The training loop:

```
For epoch in range(start_epoch, total_epochs):
    For batch_idx, batch in enumerate(train_dataloader):
        images = batch[0]

        # Forward pass (with optional AMP)
        if extra_input:
            preds = model(images, data=batch[1:])
        else:
            preds = model(images)

        # Compute loss
        loss = loss_class(preds, batch)

        # Backward pass
        loss["loss"].backward()
        optimizer.step()
        optimizer.clear_grad()
        lr_scheduler.step()

        # Periodic evaluation
        if global_step % eval_batch_step == 0:
            metric = program.eval(model, valid_dataloader, ...)
            if metric[main_indicator] > best:
                save_model(model, "best_accuracy")
```

## Data Preparation

### SimpleDataSet Format

The most common format. You need:

1. **Image files** in a directory
2. **A label file** (text file, one line per image):

**Detection labels:**
```
img_001.jpg\t[{"transcription": "HELLO", "points": [[100,50],[200,50],[200,80],[100,80]]}, ...]
img_002.jpg\t[{"transcription": "WORLD", "points": [[50,100],[150,100],[150,130],[50,130]]}]
```

**Recognition labels:**
```
img_001.jpg\tHello World
img_002.jpg\t你好世界
```

Fields are separated by `\t` (tab). The data directory and label file are
specified in the config:

```yaml
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/icdar2015/text_localization/
    label_file_list:
      - ./train_data/icdar2015/text_localization/train_label.txt
```

### LMDBDataSet Format

For large-scale training. LMDB stores images and labels in a key-value
database for fast sequential access. Use when your dataset has millions of
samples.

### Data Augmentation Pipeline

The transform chain is configured in `Train.dataset.transforms`. Common
operators:

| Operator           | Purpose                                          |
|--------------------|--------------------------------------------------|
| `DecodeImage`      | Load image from path (BGR or RGB)                |
| `DetLabelEncode`   | Parse detection label JSON into polygon arrays   |
| `CTCLabelEncode`   | Encode text string to character indices           |
| `IaaAugment`       | Random flip, rotation, resize                    |
| `EastRandomCropData`| Random crop for detection training              |
| `MakeBorderMap`    | Generate DB border regression target             |
| `MakeShrinkMap`    | Generate DB shrink map target                    |
| `RecResizeImg`     | Resize recognition images to fixed height        |
| `NormalizeImage`   | Scale to [0,1], subtract mean, divide by std     |
| `ToCHWImage`       | Convert from HWC to CHW tensor format            |
| `KeepKeys`         | Select which data fields to pass to the model    |

```
Data Flow:

Raw image file + label text
         │
         v
    DecodeImage          → dict: {image: ndarray, label: str, ...}
         │
         v
    DetLabelEncode       → dict: {image, polys, ignore_tags, ...}
         │
         v
    IaaAugment           → dict: {image (augmented), polys, ...}
         │
         v
    MakeBorderMap        → dict: {image, polys, threshold_map, ...}
    MakeShrinkMap        → dict: {image, polys, shrink_map, ...}
         │
         v
    NormalizeImage        → dict: {image (float32, normalized), ...}
    ToCHWImage            → dict: {image (CHW format), ...}
         │
         v
    KeepKeys             → tuple: (image, threshold_map, shrink_map, ...)
         │
         v
    DataLoader batches   → tensor batch for model.forward()
```

## Checkpointing and Resuming

### Saving Checkpoints

Controlled by these Global config keys:

```yaml
Global:
  save_model_dir: ./output/db_mv3/     # where to save
  save_epoch_step: 100                   # save every N epochs
```

Checkpoints are saved as:
- `{save_model_dir}/latest.pdparams` — latest model weights
- `{save_model_dir}/latest.pdopt` — optimizer state
- `{save_model_dir}/latest.states` — training state (epoch, best metric)
- `{save_model_dir}/best_accuracy.pdparams` — best model by main_indicator

### Resuming Training

Set the `checkpoints` field to resume from a saved checkpoint:

```yaml
Global:
  checkpoints: ./output/db_mv3/latest
```

Or via CLI:
```bash
python tools/train.py -c config.yml \
    -o Global.checkpoints=./output/db_mv3/latest
```

### Loading a Pretrained Model (Fine-Tuning)

Use `pretrained_model` to load weights without optimizer state:

```yaml
Global:
  pretrained_model: ./pretrain_models/MobileNetV3_large_x0_5_pretrained
```

Difference: `checkpoints` loads both model + optimizer + training state
(for resuming). `pretrained_model` loads only model weights (for
fine-tuning).

## Running Evaluation

### Standalone Evaluation

```bash
python tools/eval.py -c configs/det/det_mv3_db.yml \
    -o Global.checkpoints=./output/db_mv3/best_accuracy
```

### Evaluation During Training

Controlled by `eval_batch_step`:

```yaml
Global:
  eval_batch_step: [0, 2000]    # [start_step, interval]
```

This runs evaluation every 2000 training iterations, starting from iteration
0. If the metric improves, the best checkpoint is saved.

### Understanding Metric Output

**Detection** (DetMetric):
```
precision: 0.8523   ← of detected boxes, % that are correct
recall:    0.7891   ← of ground-truth boxes, % that were found
hmean:     0.8195   ← harmonic mean (THE primary metric)
```

**Recognition** (RecMetric):
```
acc:       0.9234   ← % of text lines recognized exactly right
norm_edit_dis: 0.0412  ← average normalized edit distance
```

The `main_indicator` field in the Metric config determines which value is
used for "best checkpoint" selection.

## Mixed Precision Training (AMP)

Enable AMP for faster training on modern GPUs:

```yaml
Global:
  use_amp: true
  amp_level: O2          # O1 (conservative) or O2 (aggressive)
  amp_dtype: float16      # float16 or bfloat16
  scale_loss: 1.0
  use_dynamic_loss_scaling: false
```

Or via CLI:
```bash
python tools/train.py -c config.yml -o Global.use_amp=true
```

## Logging

### Console Logging

```yaml
Global:
  log_smooth_window: 20    # average loss over this many steps
  print_batch_step: 10     # print every N steps
```

### WandB Integration

PaddleOCR supports Weights & Biases logging via `ppocr/utils/loggers.py`.
Configure in your training script or environment.

## What's Next?

- **[Deployment & Export](deployment-export.md)** — How to export your
  trained model and deploy it to production
- **[Testing & Debugging](testing-debugging.md)** — How to test and debug
  common training issues
