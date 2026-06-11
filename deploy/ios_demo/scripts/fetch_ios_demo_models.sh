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
# Fetch ONNX detection + recognition model bundles for the iOS demo.
#
# Requires: bash, curl, tar

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DEMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
ALLOWED_PRESETS=(PP-OCRv6_small PP-OCRv6_tiny PP-OCRv5_mobile)
DEFAULT_PRESET="PP-OCRv6_small"
DEST_DET="${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/det"
DEST_REC="${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/rec"

WORKDIR="${IOS_DEMO_ROOT}/.fetch_ios_demo_models_work"

usage() {
  cat <<EOF
Usage: ./scripts/fetch_ios_demo_models.sh [-h|--help] [preset]

Downloads ONNX bundles into:
  ${DEST_DET}
  ${DEST_REC}

Options:
  -h, --help    Show this help

Positional:
  preset        Model preset; one of: ${ALLOWED_PRESETS[*]} (default: ${DEFAULT_PRESET})
EOF
}

preset="${DEFAULT_PRESET}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

args=()
while [[ $# -gt 0 ]]; do
  args+=("$1")
  shift
done

if [[ ${#args[@]} -gt 1 ]]; then
  echo "error: at most one model preset argument is allowed" >&2
  exit 1
fi
if [[ ${#args[@]} -eq 1 ]]; then
  preset="${args[0]}"
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

if ! preset_allowed "${preset}"; then
  die "unsupported model preset: ${preset} (allowed: ${ALLOWED_PRESETS[*]})"
fi

DET_TAR="${preset}_det_onnx_infer.tar"
REC_TAR="${preset}_rec_onnx_infer.tar"

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
