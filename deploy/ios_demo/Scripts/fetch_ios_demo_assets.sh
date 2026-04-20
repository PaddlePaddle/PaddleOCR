#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Fetch ONNX models, demo sample images, and validation fixtures for the iOS
# demo.
#
# Requires: bash, curl, tar

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DEMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------- Models (App bundle; ONNX for detection + recognition) --------
BASE_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
ALLOWED_PRESETS=(PP-OCRv6_small PP-OCRv6_tiny)
DEFAULT_PRESET="PP-OCRv6_small"
DEST_DET="${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/det"
DEST_REC="${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/rec"

# -------- Demo sample images (App bundle; shown in the UI image picker) --------
SAMPLE_IMAGE_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
SAMPLE_IMAGE_FILE="general_ocr_002.png"
DEST_SAMPLES="${IOS_DEMO_ROOT}/PaddleOCRDemo/Resources/SampleImages"

# -------- Validation fixtures (test bundle; input to Scripts/run_validation.sh) --------
FIXTURE_IMAGE_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
FIXTURE_IMAGE_FILE="general_ocr_002.png"
DEST_FIXTURES="${IOS_DEMO_ROOT}/PaddleOCRDemoTests/Fixtures"

WORKDIR="${IOS_DEMO_ROOT}/.fetch_ios_demo_assets_work"

usage() {
  cat <<EOF
Usage: ./Scripts/fetch_ios_demo_assets.sh [OPTIONS] [preset]

Installs three independent asset groups for the iOS demo:
  - Models:     ONNX detection + recognition bundles (App bundle)
  - Samples:    UI demo images shown in the image picker (App bundle)
  - Fixtures:   validation inputs consumed by Scripts/run_validation.sh (test bundle)

Options:
  --models-only       Install models only
  --samples-only      Install demo sample images only
  --fixtures-only     Install validation fixtures only
  -h, --help          Show this help

Positional:
  preset              Model preset; one of: ${ALLOWED_PRESETS[*]} (default: ${DEFAULT_PRESET})
EOF
}

run_models=1
run_samples=1
run_fixtures=1
preset="${DEFAULT_PRESET}"
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-only)
      run_samples=0
      run_fixtures=0
      shift
      ;;
    --samples-only)
      run_models=0
      run_fixtures=0
      shift
      ;;
    --fixtures-only)
      run_models=0
      run_samples=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      exit 1
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ ${#args[@]} -gt 1 ]]; then
  echo "error: at most one model preset argument is allowed" >&2
  exit 1
fi
if [[ ${#args[@]} -eq 1 ]]; then
  preset="${args[0]}"
fi

if [[ "${run_models}" -eq 0 && "${run_samples}" -eq 0 && "${run_fixtures}" -eq 0 ]]; then
  echo "error: nothing to do (all target groups skipped)" >&2
  exit 1
fi

die() {
  echo "error: $*" >&2
  exit 1
}

preset_allowed() {
  local v="$1"
  local x
  for x in "${ALLOWED_PRESETS[@]}"; do
    if [[ "$x" == "$v" ]]; then
      return 0
    fi
  done
  return 1
}

command -v curl >/dev/null || die "curl is required"
command -v tar >/dev/null || die "tar is required"

if [[ "${run_models}" -eq 1 ]]; then
  if ! preset_allowed "${preset}"; then
    die "unsupported model preset: ${preset} (allowed: ${ALLOWED_PRESETS[*]})"
  fi

  DET_TAR="${preset}_det_onnx.tar"
  REC_TAR="${preset}_rec_onnx.tar"

  mkdir -p "${WORKDIR}"

  fetch() {
    local name="$1"
    local url="${BASE_URL}/${name}"
    local out="${WORKDIR}/${name}"
    if [[ -f "${out}" ]]; then
      echo "Using cached ${out}"
    else
      echo "Downloading ${url}"
      curl -fL --retry 3 --retry-delay 2 -o "${out}" "${url}"
    fi
  }

  extract() {
    local tar_path="$1"
    tar -xf "${tar_path}" -C "${WORKDIR}"
  }

  install_extracted_dir() {
    local src_dir="$1"
    local dest_dir="$2"
    local label="$3"
    [[ -d "${src_dir}" ]] || die "expected directory ${src_dir} (${label})"
    mkdir -p "$(dirname "${dest_dir}")"
    rm -rf "${dest_dir}"
    mv "${src_dir}" "${dest_dir}"
  }

  echo "=== ONNX models (preset: ${preset}) ==="
  echo "Work directory: ${WORKDIR}"
  fetch "${DET_TAR}"
  fetch "${REC_TAR}"

  extract "${WORKDIR}/${DET_TAR}"
  extract "${WORKDIR}/${REC_TAR}"

  DET_SRC="${WORKDIR}/$(basename "${DET_TAR}" .tar)"
  REC_SRC="${WORKDIR}/$(basename "${REC_TAR}" .tar)"

  install_extracted_dir "${DET_SRC}" "${DEST_DET}" "det"
  install_extracted_dir "${REC_SRC}" "${DEST_REC}" "rec"

  echo "Installed:"
  echo "  ${DEST_DET}"
  echo "  ${DEST_REC}"
  echo ""
fi

if [[ "${run_samples}" -eq 1 ]]; then
  echo "=== Demo sample image ==="
  mkdir -p "${DEST_SAMPLES}"
  sample_path="${DEST_SAMPLES}/${SAMPLE_IMAGE_FILE}"
  if [[ -f "${sample_path}" ]]; then
    echo "Sample already present: ${sample_path}"
  else
    echo "Downloading ${SAMPLE_IMAGE_URL}"
    curl -fL --retry 3 --retry-delay 2 -o "${sample_path}" "${SAMPLE_IMAGE_URL}"
    echo "Installed ${sample_path}"
  fi
  echo ""
fi

if [[ "${run_fixtures}" -eq 1 ]]; then
  echo "=== Validation fixture ==="
  mkdir -p "${DEST_FIXTURES}"
  fixture_path="${DEST_FIXTURES}/${FIXTURE_IMAGE_FILE}"
  if [[ -f "${fixture_path}" ]]; then
    echo "Fixture already present: ${fixture_path}"
  else
    echo "Downloading ${FIXTURE_IMAGE_URL}"
    curl -fL --retry 3 --retry-delay 2 -o "${fixture_path}" "${FIXTURE_IMAGE_URL}"
    echo "Installed ${fixture_path}"
  fi
  echo ""
fi

echo "Done."
echo "Ensure Xcode copies PaddleOCRDemo/Models/ and PaddleOCRDemo/Resources/SampleImages/ into the app target,"
echo "and PaddleOCRDemoTests/Fixtures/ into the test target (folder references / Copy Bundle Resources)."
