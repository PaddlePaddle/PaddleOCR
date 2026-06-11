#!/usr/bin/env bash

set -euo pipefail

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

HPS_PIPELINE_NAME="${HPS_PIPELINE_NAME:-PaddleOCR-VL-1.6}"
HPS_PADDLEX_VERSION="${HPS_PADDLEX_VERSION:-3.6}"
HPS_SDK_VERSION="${HPS_SDK_VERSION:-v${HPS_PADDLEX_VERSION}}"
HPS_SDK_DIR="${HPS_SDK_DIR:-paddlex_hps_${HPS_PIPELINE_NAME}_sdk}"
HPS_VLM_URL="${HPS_VLM_URL:-http://paddleocr-vlm-server:8080}"

SDK_ARCHIVE="${HPS_SDK_DIR}.tar.gz"
SDK_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/${HPS_SDK_VERSION}/${SDK_ARCHIVE}"
PIPELINE_CONFIG="${HPS_SDK_DIR}/server/pipeline_config.yaml"

_extract_vlm_name() {
    grep -A5 'module_name: vl_recognition' "$1" \
        | grep 'model_name:' \
        | head -1 \
        | awk '{print $2}'
}

_set_env_value() {
    local key="$1"
    local value="$2"
    local file=".env"

    if [ ! -f "$file" ]; then
        return
    fi

    if grep -q "^${key}=" "$file"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$file"
    else
        echo "${key}=${value}" >>"$file"
    fi
}

echo "Preparing high-stability serving SDK for ${HPS_PIPELINE_NAME}"
echo "  SDK archive: ${SDK_ARCHIVE}"

wget "${SDK_URL}"
tar -xf "${SDK_ARCHIVE}"

if grep -qE '^\s*backend: native\s*$' "${PIPELINE_CONFIG}"; then
    sed -i \
        -e "s|^\( *\)backend: native$|\1backend: vllm-server\n\1server_url: ${HPS_VLM_URL%/}/v1|" \
        "${PIPELINE_CONFIG}"
fi

VLM_NAME="$(_extract_vlm_name "${PIPELINE_CONFIG}")"
if [ -z "${VLM_NAME}" ]; then
    echo "Failed to read VLM name from ${PIPELINE_CONFIG}" >&2
    exit 1
fi
echo "  VLM name: ${VLM_NAME}"

_set_env_value "HPS_PIPELINE_NAME" "${HPS_PIPELINE_NAME}"
_set_env_value "HPS_SDK_DIR" "${HPS_SDK_DIR}"
_set_env_value "HPS_VLM_URL" "${HPS_VLM_URL}"

echo "High-stability serving SDK prepared at ${HPS_SDK_DIR}/"
