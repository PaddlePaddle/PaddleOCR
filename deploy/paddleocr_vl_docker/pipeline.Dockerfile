ARG BASE_IMAGE="python:3.10"

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

ENV PIP_NO_CACHE_DIR=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ARG DEVICE_TYPE="gpu"

RUN if [ "${DEVICE_TYPE}" != 'dcu' ]; then \
        apt-get update \
            && apt-get install -y libgl1 \
            && rm -rf /var/lib/apt/lists/*; \
    fi

RUN python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

RUN if [ "${DEVICE_TYPE}" = 'gpu' ]; then \
        python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/; \
    elif [ "${DEVICE_TYPE}" = 'gpu-sm120' ]; then \
        python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/; \
    elif [ "${DEVICE_TYPE}" = 'dcu' ]; then \
        python -m pip install paddlepaddle-dcu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/; \
    elif [ "${DEVICE_TYPE}" = 'xpu' ]; then \
        python -m pip install paddlepaddle-xpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/; \
    else \
        false; \
    fi

ARG PADDLEOCR_VERSION=">=3.3.2,<3.4"
RUN python -m pip install "paddleocr[doc-parser]${PADDLEOCR_VERSION}" \
    && paddlex --install serving

RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr
ENV HOME=/home/paddleocr
WORKDIR /home/paddleocr

USER paddleocr

ARG BUILD_FOR_OFFLINE=false
RUN if [ "${BUILD_FOR_OFFLINE}" = 'true' ]; then \
        mkdir -p "${HOME}/.paddlex/official_models" \
        && cd "${HOME}/.paddlex/official_models" \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar \
            https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar \
            https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV2_infer.tar \
            https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PaddleOCR-VL_infer.tar \
        && tar -xf UVDoc_infer.tar \
        && mv UVDoc_infer UVDoc \
        && tar -xf PP-LCNet_x1_0_doc_ori_infer.tar \
        && mv PP-LCNet_x1_0_doc_ori_infer PP-LCNet_x1_0_doc_ori \
        && tar -xf PP-DocLayoutV2_infer.tar \
        && mv PP-DocLayoutV2_infer PP-DocLayoutV2 \
        && tar -xf PaddleOCR-VL_infer.tar \
        && mv PaddleOCR-VL_infer PaddleOCR-VL \
        && rm -f UVDoc_infer.tar PP-LCNet_x1_0_doc_ori_infer.tar PP-DocLayoutV2_infer.tar PaddleOCR-VL_infer.tar \
        && mkdir -p "${HOME}/.paddlex/fonts" \
        && wget -P "${HOME}/.paddlex/fonts" https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/PingFang-SC-Regular.ttf; \
    fi

COPY --chown=paddleocr:paddleocr pipeline_config_vllm.yaml /home/paddleocr
COPY --chown=paddleocr:paddleocr pipeline_config_fastdeploy.yaml /home/paddleocr

EXPOSE 8080

CMD ["paddlex", "--serve", "--pipeline", "/home/paddleocr/pipeline_config_vllm.yaml"]
