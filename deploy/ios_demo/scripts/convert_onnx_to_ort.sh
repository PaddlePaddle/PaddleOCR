#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Convert ONNX models to ORT format for ONNX Runtime.
# See: https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html
#
# Requires: python3
#
# Usage:
#   ./scripts/convert_onnx_to_ort.sh
#   ./scripts/convert_onnx_to_ort.sh --out-dir ./out/ort_bundles
#   ./scripts/convert_onnx_to_ort.sh --input-dir /path/to/models --out-dir /tmp/ort
#   ./scripts/convert_onnx_to_ort.sh --out-dir /tmp/ort --optimization_style Runtime

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DEMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_INPUT_DIR="${IOS_DEMO_ROOT}/PaddleOCRDemo/Models"

INPUT_DIR="${DEFAULT_INPUT_DIR}"
ORT_OUTPUT_DIR=""

usage() {
  cat <<EOF
Usage: ./scripts/convert_onnx_to_ort.sh [OPTIONS] [ORT_CONVERTER_ARGS...]

Options:
  --input-dir <dir>  Root directory to scan for *.onnx (recursive). Default: PaddleOCRDemo/Models.
  --out-dir <dir>    If set, write .ort and ORT config artifacts under this directory, mirroring
                     relative paths from the input tree; also copies inference.yml beside each
                     converted bundle (same layout as under --input-dir) so you can swap in the
                     whole output folder. If omitted, write next to each .onnx in-place.
  -h, --help         Show this help

ORT_CONVERTER_ARGS are forwarded to:
  python3 -m onnxruntime.tools.convert_onnx_models_to_ort
(documentation: https://onnxruntime.ai/ — place this script’s options first, e.g. --out-dir
before --optimization_style.)

Examples:
  ./scripts/convert_onnx_to_ort.sh
  ./scripts/convert_onnx_to_ort.sh --out-dir ./out/ort_only
  ./scripts/convert_onnx_to_ort.sh --input-dir ./PaddleOCRDemo/Models --out-dir ./out/ort_bundles
  ./scripts/convert_onnx_to_ort.sh --out-dir /tmp/ort --optimization_style Runtime
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

# Copy every inference.yml from src_root into dst_root with the same relative path.
copy_inference_yml_mirrored() {
  local src_root="$1"
  local dst_root="$2"
  local yml
  local n=0
  while IFS= read -r -d '' yml; do
    local rel="${yml#"${src_root}/"}"
    local dest="${dst_root}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    cp -f "${yml}" "${dest}"
    n=$((n + 1))
  done < <(find "${src_root}" -name 'inference.yml' -type f -print0)
  echo "Copied ${n} inference.yml into --out-dir (same relative paths as under --input-dir)."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      [[ -n "${2:-}" ]] || die "--input-dir requires a directory"
      INPUT_DIR="$2"
      shift 2
      ;;
    --out-dir)
      [[ -n "${2:-}" ]] || die "--out-dir requires a directory"
      ORT_OUTPUT_DIR="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

CONVERTER_EXTRA=("$@")

command -v python3 >/dev/null || die "python3 is required"
[[ -d "${INPUT_DIR}" ]] || die "input directory not found: ${INPUT_DIR}"
INPUT_ABS="$(cd "${INPUT_DIR}" && pwd)"

if [[ -n "${ORT_OUTPUT_DIR}" ]]; then
  mkdir -p "${ORT_OUTPUT_DIR}" || die "cannot create --out-dir: ${ORT_OUTPUT_DIR}"
  ORT_OUT_ABS="$(cd "${ORT_OUTPUT_DIR}" && pwd)"
  echo "Converting .onnx -> .ort: --input-dir ${INPUT_ABS} -> --out-dir ${ORT_OUT_ABS} ..."
  python3 -m onnxruntime.tools.convert_onnx_models_to_ort \
    --output_dir "${ORT_OUT_ABS}" \
    "${CONVERTER_EXTRA[@]}" \
    -- \
    "${INPUT_ABS}"
  copy_inference_yml_mirrored "${INPUT_ABS}" "${ORT_OUT_ABS}"
else
  echo "Converting .onnx -> .ort in-place under ${INPUT_ABS} ..."
  python3 -m onnxruntime.tools.convert_onnx_models_to_ort \
    "${CONVERTER_EXTRA[@]}" \
    -- \
    "${INPUT_ABS}"
fi
echo "Done."
