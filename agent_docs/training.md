# Training & Config System

## Running Training

```bash
# Single GPU
python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml

# Multi-GPU distributed
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c <config.yml>
```

## Inference & Export Scripts

```bash
python tools/infer_det.py -c <config.yml> -o Global.infer_img=<image_path>
python tools/infer_rec.py -c <config.yml> -o Global.infer_img=<image_path>
python tools/export_model.py -c <config.yml> -o Global.save_inference_dir=<output_dir>
python tools/eval.py -c <config.yml>
```

## Config YAML Structure

Training configs in `configs/` follow this pattern:

```yaml
Global:        # use_gpu, epoch_num, save_model_dir, pretrained_model
Architecture:  # model_type, algorithm, Backbone, Neck, Head
Loss:          # loss function config
Optimizer:     # optimizer and learning rate schedule
PostProcess:   # post-processing config
Train:         # dataset and data transforms
Eval:          # evaluation dataset config
Metric:        # metric computation
```

Override any value via CLI: `-o Global.use_gpu=false`

## Config Directories

Configs organized by task under `configs/`:
- `det/` — text detection
- `rec/` — text recognition
- `cls/` — text angle classification
- `table/` — table recognition
- `e2e/` — end-to-end detection+recognition
- `kie/` — key information extraction
- `sr/` — super resolution

## Internal Training Framework

The `ppocr/` directory contains the training internals:
- `modeling/` — model architectures (Backbone, Neck, Head)
- `data/` — data loading and augmentation
- `losses/` — loss functions
- `metrics/` — evaluation metrics
- `postprocess/` — post-processing
- `optimizer/` — optimizer and LR schedule builders
