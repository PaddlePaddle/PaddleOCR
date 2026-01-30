# TODO: Allow regular users

ARG BACKEND=fastdeploy


FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-fastdeploy-metax-gpu:2.3.0 AS base-fastdeploy


FROM base-${BACKEND}

ARG PADDLEOCR_VERSION=">=3.4.0,<3.5"
ARG PADDLEX_VERSION=">=3.4.0,<3.5"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "paddleocr${PADDLEOCR_VERSION}" "paddlex${PADDLEX_VERSION}"

RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr
ENV HOME=/home/paddleocr
WORKDIR /home/paddleocr

USER paddleocr

ENV MACA_PATH=/opt/maca

RUN "${MACA_PATH}/tools/cu-bridge/tools/pre_make"

ENV CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"

ENV LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}"

ENV MACA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# TODO: Set these env vars only in FastDeploy image
ENV PADDLE_XCCL_BACKEND="metax_gpu"
ENV FLAGS_weight_only_linear_arch=80
ENV FD_MOE_BACKEND="cutlass"
ENV FD_ENC_DEC_BLOCK_NUM=2

ARG BUILD_FOR_OFFLINE=false
RUN if [ "${BUILD_FOR_OFFLINE}" = 'true' ]; then \
        mkdir -p "${HOME}/.paddlex/official_models" \
        && cd "${HOME}/.paddlex/official_models" \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PaddleOCR-VL-1.5_infer.tar \
        && tar -xf PaddleOCR-VL-1.5_infer.tar \
        && mv PaddleOCR-VL-1.5_infer PaddleOCR-VL-1.5 \
        && rm -f PaddleOCR-VL-1.5_infer.tar; \
    fi

ARG BACKEND
ENV BACKEND=${BACKEND}
CMD ["/bin/bash", "-c", "paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8080 --backend ${BACKEND}"]
