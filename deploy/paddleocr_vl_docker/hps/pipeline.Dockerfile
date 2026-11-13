ARG BASE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.8-gpu
ARG DEVICE_TYPE=gpu

FROM ${BASE_IMAGE}

WORKDIR /app
ARG HPS_SDK_DIR=paddlex_hps_PaddleOCR-VL-1.8_sdk
COPY ${HPS_SDK_DIR}/server .

ARG DEVICE_TYPE
ENV PADDLEX_HPS_DEVICE_TYPE=${DEVICE_TYPE}
CMD ["/bin/bash", "server.sh"]
