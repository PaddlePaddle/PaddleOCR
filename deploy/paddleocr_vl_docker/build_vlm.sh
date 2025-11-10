#!/usr/bin/env bash

device_type=''
backend=''
build_for_offline='false'
paddleocr_version='>=3.3.2,<3.4'
tag_suffix='latest'
base_image=''

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
        --backend)
            backend="$2"
            shift
            shift
            if [ "${backend}" != 'vllm' ] && [ "${backend}" != 'fastdeploy' ]; then
                echo "Unknown backend: ${backend}" >&2
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

if [ "${backend}" = 'vllm' ]; then
    if [ "${device_type}" = 'gpu' ]; then
        base_image='ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest'
    elif [ "${device_type}" = 'gpu-sm120' ]; then
        base_image='ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest-sm120'
    elif [ "${device_type}" = 'dcu' ]; then
        base_image='image.sourcefind.cn:5000/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk25.04.2-py3.10'
    elif [ "${device_type}" = 'xpu' ]; then
        base_image=''
    else
        base_image=''
    fi
elif [ "${backend}" = 'fastdeploy' ]; then
    if [ "${device_type}" = 'gpu' ]; then
        base_image='ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.6:2.3.0'
    elif [ "${device_type}" = 'gpu-sm120' ]; then
        base_image=''
    elif [ "${device_type}" = 'dcu' ]; then
        base_image=''
    elif [ "${device_type}" = 'xpu' ]; then
        base_image='ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:2.3.0'
    else
        base_image=''
    fi
else
    echo "Unknown backend: ${backend}" >&2
    exit 1
fi

if [ -z "${base_image}" ]; then
    echo "Backend '${backend}' does not support device type '${device_type}'" >&2
    exit 2
fi

docker build \
    -f vlm.Dockerfile \
    -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-${backend}-server:${tag_suffix}" \
    --build-arg BASE_IMAGE="${base_image}" \
    --build-arg BUILD_FOR_OFFLINE="${build_for_offline}" \
    --build-arg PADDLEOCR_VERSION="${paddleocr_version}" \
    --build-arg BACKEND="${backend}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .
