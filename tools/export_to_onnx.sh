#!/usr/bin/env bash
# Export a finetuned b6 jersey rec checkpoint to ONNX.
#
# This wraps the two-venv dance required by the Paddle 3.3.1 ONNX gotcha (see CLAUDE.md):
#   1. Export the inference model under .venv-export (Paddle 3.0.0), which emits ops
#      paddle2onnx can convert. Paddle 3.3.1 emits linear_v2/shape64/batch_norm_ that
#      crash paddle2onnx, so this step MUST run in .venv-export.
#   2. Run paddle2onnx from .venv (has onnxruntime + onnx-graphsurgeon for folding).
#
# Run from the repo root:
#   bash b6-h5/export_to_onnx.sh [CHECKPOINT] [OUT_ONNX]
# Defaults:
#   CHECKPOINT = ./output/b6_jersey_rec/best_accuracy
#   OUT_ONNX   = ./output/b6_jersey_rec/rec.onnx
set -euo pipefail

CONFIG="configs/rec/PP-OCRv5/b6_jersey_rec_finetune.yml"
CHECKPOINT="${1:-./output/b6_jersey_rec/best_accuracy}"
OUT_ONNX="${2:-./output/b6_jersey_rec/rec.onnx}"
INFER_DIR="$(dirname "$OUT_ONNX")/inference_p30"

if [[ ! -d ".venv-export" || ! -d ".venv" ]]; then
  echo "ERROR: expected .venv and .venv-export in $(pwd). See CLAUDE.md to create them." >&2
  exit 1
fi

echo "==> [1/2] Export inference model under .venv-export (Paddle 3.0.0, CPU)"
source .venv-export/bin/activate
python3 tools/export_model.py -c "$CONFIG" \
  -o Global.checkpoints="$CHECKPOINT" \
     Global.save_inference_dir="$INFER_DIR" \
     Global.use_gpu=False
deactivate

echo "==> [2/2] Convert to ONNX under .venv (with constant folding)"
source .venv/bin/activate
paddle2onnx --model_dir "$INFER_DIR" \
  --model_filename inference.json --params_filename inference.pdiparams \
  --save_file "$OUT_ONNX" --opset_version 16 --enable_onnx_checker True
deactivate

echo "==> Done: $OUT_ONNX"
echo "    Validate with:  source .venv/bin/activate && python3 b6-h5/validate_onnx.py 500"
