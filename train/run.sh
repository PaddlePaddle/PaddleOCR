#!/usr/bin/env bash
# Fine-tune latin_PP-OCRv5_mobile_rec on XFUND-de. Run from the repo root.
# Usage: bash train/run.sh [path to XFUND/de]   (default: train_data/XFUND/de)
set -euo pipefail

SRC="${1:-train_data/XFUND/de}"
WEIGHTS=train/pretrained/latin_PP-OCRv5_mobile_rec_pretrained.pdparams
URL=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams

mkdir -p train/pretrained
[ -f "$WEIGHTS" ] || curl -L -o "$WEIGHTS" "$URL"

[ -f train/data/train_list.txt ] || python train/prepare_data.py --src "$SRC" --out train/data

python tools/train.py -c train/latin_de_finetune.yml

python tools/export_model.py -c train/latin_de_finetune.yml \
  -o Global.pretrained_model=./output/latin_de_rec_finetune/best_accuracy \
     Global.save_inference_dir=./output/latin_de_rec_infer
