#!/usr/bin/env bash

build_for_offline='false'
tag_suffix='latest'
paddleocr_version='>=3.3.1,<3.4'

while [[ $# -gt 0 ]]; do
    case $1 in
        --offline)
            build_for_offline='true'
            tag_suffix='latest-offline'
            shift
            ;;
        --ppocr-version)
            paddleocr_version="==$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

docker build \
    -f pipeline.Dockerfile \
    -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:${tag_suffix}" \
    --build-arg BUILD_FOR_OFFLINE="${build_for_offline}" \
    --build-arg PADDLEOCR_VERSION="${paddleocr_version}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .

docker build \
    -f vlm.Dockerfile \
    -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:${tag_suffix}" \
    --build-arg BUILD_FOR_OFFLINE="${build_for_offline}" \
    --build-arg PADDLEOCR_VERSION="${paddleocr_version}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .
