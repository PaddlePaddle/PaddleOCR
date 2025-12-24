FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.3-gpu
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY paddlex_hps_PaddleOCR-VL_sdk/server .
ENV PADDLEX_HPS_DEVICE_TYPE=gpu
CMD ["/bin/bash", "server.sh"]
