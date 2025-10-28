FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest

ARG PADDLEOCR_VERSION=">=3.3.1,<3.4"
RUN python -m pip install "paddleocr${PADDLEOCR_VERSION}"
RUN paddleocr install_genai_server_deps vllm

RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr
ENV HOME=/home/paddleocr
WORKDIR /home/paddleocr

USER paddleocr

ARG BUILD_FOR_OFFLINE=false
RUN if [ "${BUILD_FOR_OFFLINE}" = 'true' ]; then \
        mkdir -p "${HOME}/.paddlex/official_models" \
        && cd "${HOME}/.paddlex/official_models" \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PaddleOCR-VL_infer.tar \
        && tar -xf PaddleOCR-VL_infer.tar \
        && mv PaddleOCR-VL_infer PaddleOCR-VL \
        && rm -f PaddleOCR-VL_infer.tar; \
    fi

CMD ["paddleocr", "genai_server", "--model_name", "PaddleOCR-VL-0.9B", "--host", "0.0.0.0", "--port", "8080", "--backend", "vllm"]
