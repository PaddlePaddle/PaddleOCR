# TODO: Allow regular users

ARG BACKEND=fastdeploy


FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:paddle-ocr-vl-1107 AS base-fastdeploy

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install fastdeploy_iluvatar_gpu==2.4.0.dev0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/


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

# TODO: Set these env vars only in FastDeploy image
ENV PADDLE_XCCL_BACKEND=iluvatar_gpu
ENV FD_SAMPLING_CLASS=rejection
ENV LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1

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
