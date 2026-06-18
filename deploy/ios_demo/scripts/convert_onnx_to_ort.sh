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
#   ./scripts/convert_onnx_to_ort.sh --output-dir ./out/ort_bundles
#   ./scripts/convert_onnx_to_ort.sh --input-dir /path/to/models --output-dir /tmp/ort
#   ./scripts/convert_onnx_to_ort.sh --optimization-style Runtime

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DEMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_INPUT_DIR="${IOS_DEMO_ROOT}/PaddleOCRDemo/Models"

INPUT_DIR="${DEFAULT_INPUT_DIR}"
ORT_OUTPUT_DIR=""
KEEP_OPERATOR_CONFIG=false
EXPLICIT_OPTIMIZATION_STYLE=false
OPTIMIZATION_STYLE_VALUES=()

usage() {
  cat <<EOF
Usage: ./scripts/convert_onnx_to_ort.sh [OPTIONS] [ORT_CONVERTER_ARGS...]

Options:
  --input-dir <dir>     Root directory to scan for *.onnx (recursive). Default: PaddleOCRDemo/Models.
  --output-dir <dir>    If set, write .ort files under this directory, mirroring relative paths from
                        the input tree; also copies inference.yml beside each converted bundle (same
                        layout as under --input-dir) so you can swap in the whole output folder. If
                        omitted, write next to each .onnx in-place.
  --keep-operator-config
                        Keep required_operators*.config files emitted by the ORT converter (for
                        minimal ORT builds). By default these are removed after a successful run.
  --optimization-style <style>...
                        ORT conversion style(s): Fixed and/or Runtime (forwarded as Python
                        --optimization_style). One or more values after the flag, e.g.
                        --optimization-style Runtime  or  --optimization-style Fixed Runtime
                        The flag may appear more than once; values accumulate.
                        Alias: --optimization_style
                        Default when omitted: Fixed (single .ort per ONNX; upstream defaults to both).
  -h, --help            Show this help

ORT_CONVERTER_ARGS are forwarded to:
  python3 -m onnxruntime.tools.convert_onnx_models_to_ort
(documentation: https://onnxruntime.ai/ — place this script’s options first, e.g. --output-dir
before other converter flags such as --enable_type_reduction.)

Examples:
  ./scripts/convert_onnx_to_ort.sh
  ./scripts/convert_onnx_to_ort.sh --output-dir ./out/ort_only
  ./scripts/convert_onnx_to_ort.sh --input-dir ./PaddleOCRDemo/Models --output-dir ./out/ort_bundles
  ./scripts/convert_onnx_to_ort.sh --output-dir /tmp/ort --optimization-style Runtime
  ./scripts/convert_onnx_to_ort.sh --optimization-style Fixed Runtime --output-dir ./out/both
  ./scripts/convert_onnx_to_ort.sh --output-dir ./out/ort_only --keep-operator-config
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

# Remove ORT converter operator-config files.
prune_operator_config_files() {
  local root="$1"
  local n=0
  while IFS= read -r -d '' f; do
    rm -f "${f}"
    n=$((n + 1))
  done < <(find "${root}" -name 'required_operators*.config' -type f -print0 2>/dev/null || true)
  if [[ "$n" -gt 0 ]]; then
    echo "Removed ${n} required_operators*.config file(s) (minimal-build manifests; use --keep-operator-config to retain)."
  fi
}

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
  echo "Copied ${n} inference.yml into --output-dir (same relative paths as under --input-dir)."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      [[ -n "${2:-}" ]] || die "--input-dir requires a directory"
      INPUT_DIR="$2"
      shift 2
      ;;
    --output-dir)
      [[ -n "${2:-}" ]] || die "--output-dir requires a directory"
      ORT_OUTPUT_DIR="$2"
      shift 2
      ;;
    --keep-operator-config)
      KEEP_OPERATOR_CONFIG=true
      shift
      ;;
    --optimization-style|--optimization_style)
      shift
      EXPLICIT_OPTIMIZATION_STYLE=true
      chunk=()
      while [[ $# -gt 0 ]]; do
        case "$1" in
          Fixed|Runtime)
            chunk+=("$1")
            shift
            ;;
          *)
            break
            ;;
        esac
      done
      [[ ${#chunk[@]} -gt 0 ]] || die "--optimization-style requires one or more of: Fixed Runtime"
      OPTIMIZATION_STYLE_VALUES+=("${chunk[@]}")
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

if [[ "$EXPLICIT_OPTIMIZATION_STYLE" == false ]]; then
  OPTIMIZATION_STYLE_VALUES=(Fixed)
fi

ORT_OPTIMIZATION_ARGS=(--optimization_style "${OPTIMIZATION_STYLE_VALUES[@]}")

command -v python3 >/dev/null || die "python3 is required"
[[ -d "${INPUT_DIR}" ]] || die "input directory not found: ${INPUT_DIR}"
INPUT_ABS="$(cd "${INPUT_DIR}" && pwd)"

if [[ -n "${ORT_OUTPUT_DIR}" ]]; then
  mkdir -p "${ORT_OUTPUT_DIR}" || die "cannot create --output-dir: ${ORT_OUTPUT_DIR}"
  ORT_OUT_ABS="$(cd "${ORT_OUTPUT_DIR}" && pwd)"
  echo "Converting .onnx -> .ort: --input-dir ${INPUT_ABS} -> --output-dir ${ORT_OUT_ABS} ..."
  python3 -m onnxruntime.tools.convert_onnx_models_to_ort \
    --output_dir "${ORT_OUT_ABS}" \
    "${ORT_OPTIMIZATION_ARGS[@]}" \
    "${CONVERTER_EXTRA[@]}" \
    -- \
    "${INPUT_ABS}"
  copy_inference_yml_mirrored "${INPUT_ABS}" "${ORT_OUT_ABS}"
  if [[ "$KEEP_OPERATOR_CONFIG" == false ]]; then
    prune_operator_config_files "${ORT_OUT_ABS}"
  fi
else
  echo "Converting .onnx -> .ort in-place under ${INPUT_ABS} ..."
  python3 -m onnxruntime.tools.convert_onnx_models_to_ort \
    "${ORT_OPTIMIZATION_ARGS[@]}" \
    "${CONVERTER_EXTRA[@]}" \
    -- \
    "${INPUT_ABS}"
  if [[ "$KEEP_OPERATOR_CONFIG" == false ]]; then
    prune_operator_config_files "${INPUT_ABS}"
  fi
fi
echo "Done."
