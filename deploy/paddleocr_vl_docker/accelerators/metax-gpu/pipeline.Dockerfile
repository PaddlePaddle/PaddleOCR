# TODO: Allow regular users

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-paddle-metax-gpu:3.3.0

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fontconfig \
        fonts-dejavu-core \
        fonts-liberation \
        fonts-noto-cjk \
        fonts-wqy-microhei \
        fonts-freefont-ttf \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

ARG PADDLEOCR_VERSION=">=3.4.0,<3.5"
ARG PADDLEX_VERSION=">=3.4.0,<3.5"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "paddleocr[doc-parser]${PADDLEOCR_VERSION}" "paddlex[serving]${PADDLEX_VERSION}"

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
ENV PADDLE_XCCL_BACKEND="metax_gpu"
ENV FLAGS_weight_only_linear_arch=80
ENV FD_MOE_BACKEND="cutlass"
ENV FD_ENC_DEC_BLOCK_NUM=2

ARG BUILD_FOR_OFFLINE=false
RUN if [ "${BUILD_FOR_OFFLINE}" = 'true' ]; then \
        mkdir -p "${HOME}/.paddlex/official_models" \
        && cd "${HOME}/.paddlex/official_models" \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar \
            https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar \
            https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV3_infer.tar \
            https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PaddleOCR-VL-1.5_infer.tar \
        && tar -xf UVDoc_infer.tar \
        && mv UVDoc_infer UVDoc \
        && tar -xf PP-LCNet_x1_0_doc_ori_infer.tar \
        && mv PP-LCNet_x1_0_doc_ori_infer PP-LCNet_x1_0_doc_ori \
        && tar -xf PP-DocLayoutV3_infer.tar \
        && mv PP-DocLayoutV3_infer PP-DocLayoutV3 \
        && tar -xf PaddleOCR-VL-1.5_infer.tar \
        && mv PaddleOCR-VL-1.5_infer PaddleOCR-VL-1.5 \
        && rm -f UVDoc_infer.tar PP-LCNet_x1_0_doc_ori_infer.tar PP-DocLayoutV3_infer.tar PaddleOCR-VL-1.5_infer.tar \
        && mkdir -p "${HOME}/.paddlex/fonts" \
        && wget -P "${HOME}/.paddlex/fonts" https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/PingFang-SC-Regular.ttf; \
    fi

COPY --chown=paddleocr:paddleocr pipeline_config_vllm.yaml /home/paddleocr
COPY --chown=paddleocr:paddleocr pipeline_config_fastdeploy.yaml /home/paddleocr

EXPOSE 8080

CMD ["paddlex", "--serve", "--pipeline", "/home/paddleocr/pipeline_config_vllm.yaml", "--device", "metax_gpu"]
