# TODO: Allow regular users

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle-dcu:dtk24.04.1-kylinv10-gcc82

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN yum install -y \
        fontconfig \
        dejavu-sans-fonts \
        dejavu-serif-fonts \
        liberation-fonts \
        liberation-mono-fonts \
        liberation-sans-fonts \
        liberation-serif-fonts \
        google-noto-cjk-fonts \
        wqy-microhei-fonts \
        gnu-free-fonts-common \
        gnu-free-mono-fonts \
        gnu-free-sans-fonts \
        gnu-free-serif-fonts \
    && fc-cache -fv \
    && yum clean all

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl \
    && python -m pip install paddlepaddle-dcu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/

ARG PADDLEOCR_VERSION=">=3.4.0,<3.5"
ARG PADDLEX_VERSION=">=3.4.0,<3.5"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "paddleocr[doc-parser]${PADDLEOCR_VERSION}" "paddlex[serving]${PADDLEX_VERSION}"

RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr
ENV HOME=/home/paddleocr
WORKDIR /home/paddleocr

USER paddleocr

ENV HYHAL_PATH=/opt/hyhal
ENV DTKROOT=/opt/
ENV AMDGPU_TARGETS="gfx906;gfx926;gfx928"
ENV ROCM_PATH=/opt/dtk-24.04.1
ENV HIP_PATH=/opt/dtk-24.04.1/hip
ENV MIOPEN_FIND_MODE=3

ENV PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/hip/bin/hipify:${HYHAL_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${HYHAL_PATH}/lib:${HYHAL_PATH}/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH}"
ENV C_INCLUDE_PATH="${ROCM_PATH}/include:${HYHAL_PATH}/include:${ROCM_PATH}/llvm/include"
ENV CPLUS_INCLUDE_PATH="${ROCM_PATH}/include:${HYHAL_PATH}/include:${ROCM_PATH}/llvm/include"

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

CMD ["paddlex", "--serve", "--pipeline", "/home/paddleocr/pipeline_config_vllm.yaml", "--device", "dcu"]
