#!/usr/bin/env bash
# Fetch ONNX models and demo sample images.
#
# Requires: bash, curl, tar
#
# Usage:
#   ./deploy/ios_demo/fetch_ios_demo_assets.sh
#   ./deploy/ios_demo/fetch_ios_demo_assets.sh PP-OCRv6_small
#   ./deploy/ios_demo/fetch_ios_demo_assets.sh --models-only
#   ./deploy/ios_demo/fetch_ios_demo_assets.sh --samples-only
#
# Help:
#   ./deploy/ios_demo/fetch_ios_demo_assets.sh --help

set -euo pipefail

BASE_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
SAMPLE_IMAGE_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
SAMPLE_IMAGE_FILE="general_ocr_002.png"

ALLOWED_VARIANTS=(PP-OCRv6_small PP-OCRv6_tiny)
DEFAULT_VARIANT="PP-OCRv6_small"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ONNX bundles and sample images live under PaddleOCRDemo/ (Xcode app sources and bundle resources).
DEST_DET="${SCRIPT_DIR}/PaddleOCRDemo/Models/det"
DEST_REC="${SCRIPT_DIR}/PaddleOCRDemo/Models/rec"
DEST_SAMPLES="${SCRIPT_DIR}/PaddleOCRDemo/Resources/SampleImages"
WORKDIR="${SCRIPT_DIR}/.fetch_ios_demo_assets_work"

run_models=1
run_samples=1
variant="${DEFAULT_VARIANT}"
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-only)
      run_samples=0
      shift
      ;;
    --samples-only)
      run_models=0
      shift
      ;;
    -h|--help)
      sed -n '1,18p' "$0"
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
  echo "error: at most one model variant argument is allowed" >&2
  exit 1
fi
if [[ ${#args[@]} -eq 1 ]]; then
  variant="${args[0]}"
fi

if [[ "${run_models}" -eq 0 && "${run_samples}" -eq 0 ]]; then
  echo "error: nothing to do (both models and samples skipped)" >&2
  exit 1
fi

die() {
  echo "error: $*" >&2
  exit 1
}

variant_allowed() {
  local v="$1"
  local x
  for x in "${ALLOWED_VARIANTS[@]}"; do
    if [[ "$x" == "$v" ]]; then
      return 0
    fi
  done
  return 1
}

command -v curl >/dev/null || die "curl is required"
command -v tar >/dev/null || die "tar is required"

if [[ "${run_models}" -eq 1 ]]; then
  if ! variant_allowed "${variant}"; then
    die "unsupported OCR variant: ${variant} (allowed: ${ALLOWED_VARIANTS[*]})"
  fi

  DET_TAR="${variant}_det_onnx.tar"
  REC_TAR="${variant}_rec_onnx.tar"

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

  echo "=== ONNX models (variant: ${variant}) ==="
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
  local_path="${DEST_SAMPLES}/${SAMPLE_IMAGE_FILE}"
  if [[ -f "${local_path}" ]]; then
    echo "Sample already present: ${local_path}"
  else
    echo "Downloading ${SAMPLE_IMAGE_URL}"
    curl -fL --retry 3 --retry-delay 2 -o "${local_path}" "${SAMPLE_IMAGE_URL}"
    echo "Installed ${local_path}"
  fi
  echo ""
fi

echo "Done."
echo "Ensure Xcode copies PaddleOCRDemo/Models/ and PaddleOCRDemo/Resources/SampleImages/ into the app target (folder references / Copy Bundle Resources)."
