# TODO: Allow regular users

ARG TARGETARCH


FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-x86_64-gcc84 AS base-amd64

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:$LD_LIBRARY_PATH


FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-aarch64-gcc84 AS base-arm64

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch_64:$LD_LIBRARY_PATH


FROM base-${TARGETARCH} AS base

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

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    && python -m pip install paddle-custom-npu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/npu/

ARG PADDLEOCR_VERSION=">=3.4.0,<3.5"
ARG PADDLEX_VERSION=">=3.4.0,<3.5"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "paddleocr[doc-parser]${PADDLEOCR_VERSION}" "paddlex[serving]${PADDLEX_VERSION}"

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install numpy==1.26.4 opencv-contrib-python==4.11.0.86

RUN groupadd -g 1001 paddleocr \
    && useradd -m -s /bin/bash -u 1001 -g 1001 paddleocr
ENV HOME=/home/paddleocr
WORKDIR /home/paddleocr

USER paddleocr

ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

ENV ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
ENV LD_LIBRARY_PATH=${ATB_HOME_PATH}/lib:${ATB_HOME_PATH}/examples:${ATB_HOME_PATH}/tests/atbopstest:${LD_LIBRARY_PATH}
ENV PATH=${ATB_HOME_PATH}/bin:${PATH}

ENV ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
ENV ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0
ENV ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0
ENV ATB_OPSRUNNER_SETUP_CACHE_ENABLE=1
ENV ATB_OPSRUNNER_KERNEL_CACHE_TYPE=3
ENV ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
ENV ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5
ENV ATB_OPSRUNNER_KERNEL_CACHE_TILING_SIZE=10240
ENV ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1
ENV ATB_WORKSPACE_MEM_ALLOC_GLOBAL=0
ENV ATB_COMPARE_TILING_EVERY_KERNEL=0
ENV ATB_HOST_TILING_BUFFER_BLOCK_NUM=128
ENV ATB_DEVICE_TILING_BUFFER_BLOCK_NUM=32
ENV ATB_SHARE_MEMORY_NAME_SUFFIX=""
ENV ATB_LAUNCH_KERNEL_WITH_TILING=1
ENV ATB_MATMUL_SHUFFLE_K_ENABLE=1
ENV ATB_RUNNER_POOL_SIZE=64

ENV ASDOPS_HOME_PATH=${ATB_HOME_PATH}
ENV ASDOPS_MATMUL_PP_FLAG=1
ENV ASDOPS_LOG_LEVEL=ERROR
ENV ASDOPS_LOG_TO_STDOUT=0
ENV ASDOPS_LOG_TO_FILE=1
ENV ASDOPS_LOG_TO_FILE_FLUSH=0
ENV ASDOPS_LOG_TO_BOOST_TYPE=atb
ENV ASDOPS_LOG_PATH=${HOME}
ENV ASDOPS_TILING_PARSE_CACHE_DISABLE=0
ENV LCCL_DETERMINISTIC=0

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

CMD ["paddlex", "--serve", "--pipeline", "/home/paddleocr/pipeline_config_vllm.yaml", "--device", "npu"]
