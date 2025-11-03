#!/usr/bin/env bash

build_for_offline='false'
paddleocr_version='>=3.3.2,<3.4'
build_for_sm120='false'
fastdeploy_version='2.2.1'
tag_suffix='latest'
vlm_base_image_tag_suffix='latest'

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
        --sm120)
            build_for_sm120='true'
            vlm_base_image_tag_suffix='latest-sm120'
            shift
            ;;
        --fd-version)
            fastdeploy_version="==$2"
            shift
            shift
            ;;
        --tag-suffix)
            tag_suffix="$2"
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
    --build-arg BUILD_FOR_SM120="${build_for_sm120}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .

docker build \
    -f vllm.Dockerfile \
    -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:${tag_suffix}" \
    --build-arg BUILD_FOR_OFFLINE="${build_for_offline}" \
    --build-arg PADDLEOCR_VERSION="${paddleocr_version}" \
    --build-arg BASE_IMAGE_TAG_SUFFIX="${vlm_base_image_tag_suffix}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .

docker build \
    -f fastdeploy.Dockerfile \
    -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:${tag_suffix}" \
    --build-arg BUILD_FOR_OFFLINE="${build_for_offline}" \
    --build-arg PADDLEOCR_VERSION="${paddleocr_version}" \
    --build-arg FASTDEPLOY_VERSION="${fastdeploy_version}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .
