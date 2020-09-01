#!/bin/bash
set -e

OCR_MODEL_URL="https://paddleocr.bj.bcebos.com/deploy/lite/ocr_v1_for_cpu.tar.gz"
PADDLE_LITE_LIB_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/paddle_lite_libs_v2_6_0.tar.gz"
OPENCV3_FRAMEWORK_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/opencv3.framework.tar.gz"

download_and_extract() {
    local url="$1"
    local dst_dir="$2"
    local tempdir=$(mktemp -d)

    echo "Downloading ${url} ..."
    curl -L ${url} > ${tempdir}/temp.tar.gz
    echo "Download ${url} done "

    if [ ! -d ${dst_dir} ];then
        mkdir -p ${dst_dir}
    fi

    echo "Extracting ..."
    tar -zxvf ${tempdir}/temp.tar.gz -C ${dst_dir}
    echo "Extract done "

    rm -rf ${tempdir}
}

echo -e "[Download ios ocr demo denpendancy]\n"
download_and_extract "${OCR_MODEL_URL}" "./ocr_demo/models"
download_and_extract "${PADDLE_LITE_LIB_URL}" "./ocr_demo"
download_and_extract "${OPENCV3_FRAMEWORK_URL}" "./ocr_demo"
echo -e "[done]\n"
