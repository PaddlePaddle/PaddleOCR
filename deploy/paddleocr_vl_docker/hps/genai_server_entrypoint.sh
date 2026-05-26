#!/usr/bin/env sh

set -eu

CONFIG="${PIPELINE_CONFIG:-/config/pipeline_config.yaml}"

VLM_NAME=$(
    grep -A5 'module_name: vl_recognition' "$CONFIG" \
        | grep 'model_name:' \
        | head -1 \
        | awk '{print $2}'
)

if [ -z "$VLM_NAME" ]; then
    echo "Failed to read VLM name from ${CONFIG}" >&2
    exit 1
fi

exec paddleocr genai_server \
    --model_name "$VLM_NAME" \
    --host 0.0.0.0 \
    --port 8080 \
    --backend vllm
