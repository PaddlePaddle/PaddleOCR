ARG BACKEND=vllm
ARG VLLM_BASE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:paddlex3.8-vllm0.19.0


FROM ${VLLM_BASE_IMAGE} AS base-vllm


FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.6:2.3.0 AS base-fastdeploy


FROM base-${BACKEND}

ARG PADDLEOCR_VERSION=">=3.8.0,<3.9.0"
ARG PADDLEX_VERSION=">=3.8.0,<3.9.0"
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "paddleocr${PADDLEOCR_VERSION}" "paddlex${PADDLEX_VERSION}"

RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr
ENV HOME=/home/paddleocr
WORKDIR /home/paddleocr

USER paddleocr

ARG BUILD_FOR_OFFLINE=false
RUN if [ "${BUILD_FOR_OFFLINE}" = 'true' ]; then \
        mkdir -p "${HOME}/.paddlex/official_models" \
        && cd "${HOME}/.paddlex/official_models" \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PaddleOCR-VL-1.8_infer.tar \
        && tar -xf PaddleOCR-VL-1.8_infer.tar \
        && mv PaddleOCR-VL-1.8_infer PaddleOCR-VL-1.8 \
        && rm -f PaddleOCR-VL-1.8_infer.tar; \
    fi

ARG BACKEND
ENV BACKEND=${BACKEND}
CMD ["/bin/bash", "-c", "paddleocr genai_server --model_name PaddleOCR-VL-1.8-0.9B --host 0.0.0.0 --port 8080 --backend ${BACKEND}"]
