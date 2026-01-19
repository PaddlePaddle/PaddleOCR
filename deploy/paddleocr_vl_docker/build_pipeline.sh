#!/usr/bin/env bash

build_for_offline='false'
paddleocr_version='>=3.3.2,<3.4'
tag_suffix='latest'
dockerfile=pipeline.Dockerfile

while [[ $# -gt 0 ]]; do
    case $1 in
        --device-type)
            [ -z "$2" ] && {
                echo "`--device-type` requires a value" >&2
                exit 2
            }
            device_type="$2"
            shift
            shift
            case "${device_type}" in
                gpu|gpu-sm120|dcu|xpu|metax-gpu)
                    ;;
                *)
                    echo "Unknown device type: ${device_type}" >&2
                    exit 2
                    ;;
            esac
            ;;
        --offline)
            build_for_offline='true'
            shift
            ;;
        --ppocr-version)
            [ -z "$2" ] && {
                echo "`--ppocr-version` requires a value" >&2
                exit 2
            }
            paddleocr_version="==$2"
            shift
            shift
            ;;
        --tag-suffix)
            [ -z "$2" ] && {
                echo "`--tag-suffix` requires a value" >&2
                exit 2
            }
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

dockerfile="accelerators/${device_type}/pipeline.Dockerfile"

docker build \
    -f "${dockerfile}" \
    -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:${tag_suffix}" \
    --build-arg BUILD_FOR_OFFLINE="${build_for_offline}" \
    --build-arg PADDLEOCR_VERSION="${paddleocr_version}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .
