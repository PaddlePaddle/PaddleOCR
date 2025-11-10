#!/usr/bin/env bash

build_for_offline='false'
paddleocr_version='>=3.3.2,<3.4'
tag_suffix='latest'
base_image='python:3.10'

while [[ $# -gt 0 ]]; do
    case $1 in
        --device-type)
            device_type="$2"
            shift
            shift
            if [ "${device_type}" != 'gpu' ] && [ "${device_type}" != 'gpu-sm120' ] && [ "${device_type}" != 'dcu' ] && [ "${device_type}" != 'xpu' ]; then
                echo "Unknown device type: ${device_type}" >&2
                exit 2
            fi
            ;;
        --offline)
            build_for_offline='true'
            shift
            ;;
        --ppocr-version)
            paddleocr_version="==$2"
            shift
            shift
            ;;
        --tag-suffix)
            tag_suffix="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [ "${device_type}" != 'gpu' ]; then
    tag_suffix="${tag_suffix}-${device_type}"
fi

if [ "${build_for_offline}" = 'true' ]; then
    tag_suffix="${tag_suffix}-offline"
fi

if [ "${device_type}" = 'dcu' ]; then
    base_image='ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle-dcu:dtk24.04.1-kylinv10-gcc82'
elif [ "${device_type}" = 'xpu' ]; then
    base_image='ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310'
fi

docker build \
    -f pipeline.Dockerfile \
    -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:${tag_suffix}" \
    --build-arg BASE_IMAGE="${base_image}" \
    --build-arg DEVICE_TYPE="${device_type}" \
    --build-arg BUILD_FOR_OFFLINE="${build_for_offline}" \
    --build-arg PADDLEOCR_VERSION="${paddleocr_version}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .
