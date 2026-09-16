# CLAUDE.md

This is a clone of the upstream PaddleOCR repo plus **custom work**: finetuning a
PP-OCRv5 text-recognition model on b6 jersey-number data and exporting it to ONNX.
This file documents that custom workflow and its non-obvious gotchas.

## Environments

- **`.venv`** — the main/training env. **Paddle 3.3.1 + CUDA** (RTX 4080). Use for
  training, eval, and running `paddle2onnx` (it also has `onnxruntime` +
  `onnx-graphsurgeon` for ONNX constant-folding). Activate: `source .venv/bin/activate`.
- **`.venv-export`** — isolated, **CPU-only Paddle 3.0.0** env used *only* to export
  the inference model for ONNX (see the ONNX gotcha below). Activate:
  `source .venv-export/bin/activate`.

### Recreating the environments from scratch
```bash
# Main training+convert env — Paddle 3.3.1 + CUDA
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install paddlepaddle-gpu==3.3.1 -r requirements.txt
uv pip install h5py onnxruntime onnx onnx-graphsurgeon paddle2onnx
deactivate

# Export-only env — CPU Paddle 3.0.0 (just for the ONNX inference-model export)
uv venv .venv-export --python 3.10 && source .venv-export/bin/activate
uv pip install paddlepaddle==3.0.0 paddle2onnx==2.1.0 setuptools -r requirements.txt
```
`paddlepaddle-gpu==3.3.1` is matched to this box (RTX 4080, CUDA 11.8 runtime). On
different hardware, install the matching Paddle GPU wheel from PaddlePaddle's install
page — but **keep the version pins**: the 3.3.1 vs 3.0.0 split is exactly what the ONNX
gotcha below depends on.

## The data (`b6-h5/`)

`train.h5` (42,140), `val.h5` (7,036), `test.h5` (2,459). Each sample is an h5 group
`<idx>` with:
- `<idx>/image` — `(H,W,3)` uint8, **RGB**, a cropped jersey-number image
- `<idx>/label` — `(2,)` int64 `[first_digit, second_digit]`; **`second_digit == 10`
  means "no second digit"** (single-digit number). So `[2,2]`→`"22"`, `[2,10]`→`"2"`.

### Convert h5 → PaddleOCR rec format
```bash
source .venv/bin/activate
python3 b6-h5/convert_h5_to_paddleocr.py --h5-dir b6-h5 --out train_data
```
Produces `train_data/rec/{train,val,test}/<idx>.png` + `train_data/{train,val,test}_list.txt`
(tab-separated `relpath\ttext`). **The script swaps RGB→BGR before `cv2.imwrite`** so the
files read back as BGR via PaddleOCR's `DecodeImage(img_mode: BGR)` — the convention the
pretrained model was trained on. Don't remove that swap.

## Finetuning

Config: **`configs/rec/PP-OCRv5/b6_jersey_rec_finetune.yml`** (based on
`multi_language/en_PP-OCRv5_mobile_rec.yaml`).

Key choice: it reuses **`ppocr/utils/dict/ppocrv5_en_dict.txt`** (the same dict as the
English pretrained model), NOT a digit-only dict. This keeps the CTC/NRTR head dims
identical to the pretrained weights (CTC fc = **438** = 436 dict + 1 space + 1 blank;
NRTR embedding = **442**), so finetuning loads the head instead of reinitializing it.
If you ever change the dict, the head will be reinitialized — don't, unless intended.

Pretrained weights live at `pretrain_models/en_PP-OCRv5_mobile_rec.pdparams`
(downloaded from
`https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv5_mobile_rec_pretrained.pdparams`).

```bash
source .venv/bin/activate
python3 tools/train.py -c configs/rec/PP-OCRv5/b6_jersey_rec_finetune.yml
# resume:  -o Global.checkpoints=./output/b6_jersey_rec/latest
# OOM:     -o Train.sampler.first_bs=64   (already lowered to 64 in the config)
```
Outputs to `output/b6_jersey_rec/`; best model = `best_accuracy`, evaluated on
`val_list.txt`. Evaluate on the held-out test set:
```bash
python3 tools/eval.py -c configs/rec/PP-OCRv5/b6_jersey_rec_finetune.yml \
  -o Eval.dataset.label_file_list=[./train_data/test_list.txt] \
     Global.checkpoints=./output/b6_jersey_rec/best_accuracy
```

## ONNX export — IMPORTANT GOTCHA

**Paddle 3.3.1 (`.venv`) cannot export an ONNX-convertible model.** Its PIR exporter
emits ops `linear_v2`, `shape64`, `batch_norm_` (inplace) that the latest `paddle2onnx`
(2.1.0, also the newest on PyPI) has no converters for → `paddle2onnx` crashes with
`SIGABRT` / "unsupported operators". Things that do NOT fix it: `export_with_pir=False`
(hits a Paddle 3.3.1 PIR/old-IR bug), prim decomposition at export, `--enable_dist_prim_all`,
higher opset.

**The fix: export under Paddle 3.0.0 (`.venv-export`)**, which emits classic
`matmul`/`shape`/`batch_norm` ops that paddle2onnx understands. Two steps:

```bash
# 1. Export inference model under Paddle 3.0.0 (CPU is fine)
source .venv-export/bin/activate
python3 tools/export_model.py -c configs/rec/PP-OCRv5/b6_jersey_rec_finetune.yml \
  -o Global.checkpoints=./output/b6_jersey_rec/best_accuracy \
     Global.save_inference_dir=./output/b6_jersey_rec/inference_p30 \
     Global.use_gpu=False

# 2. Convert to ONNX from the MAIN .venv (it has onnxruntime + onnx-graphsurgeon,
#    needed for constant-folding). paddle2onnx parses the saved program, so the
#    Paddle version of the venv running it doesn't matter here.
source .venv/bin/activate
paddle2onnx --model_dir ./output/b6_jersey_rec/inference_p30 \
  --model_filename inference.json --params_filename inference.pdiparams \
  --save_file ./output/b6_jersey_rec/rec.onnx --opset_version 16 --enable_onnx_checker True
```
Folding shrinks it ~55 MB → ~7.9 MB (1121→549 nodes). Result: **`output/b6_jersey_rec/rec.onnx`**.

### Validate the ONNX model
```bash
source .venv/bin/activate
python3 b6-h5/validate_onnx.py 500   # exact-match accuracy vs test_list.txt (~92%)
```

## rec.onnx I/O contract
- **Input**: `[N, 3, 48, W]` float32, **BGR**, normalized `(x/255 - 0.5)/0.5`, width
  padded to 320 (width axis is dynamic).
- **Output**: `[N, T, 438]` softmax probs. CTC-decode: argmax over last axis, collapse
  consecutive repeats, drop index 0 (blank). Charset =
  `["blank"] + ppocrv5_en_dict.txt lines + [" "]`.
- `b6-h5/validate_onnx.py` is the reference pre/post-processing implementation.

## Helper scripts in `b6-h5/`
- `convert_h5_to_paddleocr.py` — h5 → PaddleOCR rec format (re-runnable).
- `export_to_onnx.sh` — one-shot wrapper for the two-venv ONNX export (steps 3a+3b
  below). Run from the repo root: `bash b6-h5/export_to_onnx.sh [CHECKPOINT] [OUT_ONNX]`.
- `validate_onnx.py` — ONNX accuracy check + reference pre/post-processing.

## End-to-end command sequence

```bash
cd <repo root>

# 1. CONVERT  (.venv)
source .venv/bin/activate
python3 b6-h5/convert_h5_to_paddleocr.py --h5-dir b6-h5 --out train_data

# 2. FINETUNE (.venv) — best model = output/b6_jersey_rec/best_accuracy
python3 tools/train.py -c configs/rec/PP-OCRv5/b6_jersey_rec_finetune.yml

# 3. EXPORT to ONNX — one-shot wrapper that runs both venv steps (3a export under
#    .venv-export, 3b paddle2onnx under .venv). Equivalent manual steps follow.
bash b6-h5/export_to_onnx.sh
#   3a (manual) EXPORT inference model  (.venv-export — Paddle 3.0.0, CPU)
#   source .venv-export/bin/activate
#   python3 tools/export_model.py -c configs/rec/PP-OCRv5/b6_jersey_rec_finetune.yml \
#     -o Global.checkpoints=./output/b6_jersey_rec/best_accuracy \
#        Global.save_inference_dir=./output/b6_jersey_rec/inference_p30 \
#        Global.use_gpu=False
#   3b (manual) CONVERT to ONNX  (back to .venv)
#   source .venv/bin/activate
#   paddle2onnx --model_dir ./output/b6_jersey_rec/inference_p30 \
#     --model_filename inference.json --params_filename inference.pdiparams \
#     --save_file ./output/b6_jersey_rec/rec.onnx --opset_version 16 --enable_onnx_checker True

# 4. VALIDATE (optional)
python3 b6-h5/validate_onnx.py 500
```

## Reproducing on another machine (handoff)

The only files custom to this work (everything else is the upstream PaddleOCR clone):
- `configs/rec/PP-OCRv5/b6_jersey_rec_finetune.yml` — the finetune config
- `b6-h5/convert_h5_to_paddleocr.py` — h5 → rec format
- `b6-h5/export_to_onnx.sh` — one-shot ONNX export wrapper
- `b6-h5/validate_onnx.py` — ONNX validation + reference pre/post-processing
- `CLAUDE.md` — this file

Fetched separately, not copied between machines:
- **PaddleOCR repo code** (`tools/`, `ppocr/`, and the dict
  `ppocr/utils/dict/ppocrv5_en_dict.txt`) — all upstream, comes with the clone.
- **`.h5` datasets** — from S3, into `b6-h5/`.
- **Pretrained weights** → `pretrain_models/en_PP-OCRv5_mobile_rec.pdparams` (URL in the
  Finetuning section above).

Do **not** copy between machines (recreate locally): `.venv/`, `.venv-export/`,
`train_data/`, `output/`.
