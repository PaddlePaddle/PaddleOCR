# Dockerfile for PaddleOCR on Linux aarch64 (ARM64)
#
# Builds ultra-infer from source with the ONNX Runtime backend, then
# installs PaddleOCR with HPI enabled so inference is routed through
# ONNX Runtime on aarch64.
#
# Build (takes ~10-15 min on Apple Silicon):
#   docker build --platform linux/arm64 -t paddleocr-aarch64 \
#       -f deploy/docker/aarch64/Dockerfile .
#
# Run tests:
#   docker run --platform linux/arm64 --rm paddleocr-aarch64
#
# See: https://github.com/PaddlePaddle/PaddleOCR/issues/17590

ARG PYTHON_VERSION=3.12

# ── Stage 1: build ultra-infer wheel for aarch64 ────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS ultra-infer-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        patchelf \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir wheel numpy setuptools

# Clone PaddleX (ultra-infer lives at deploy/ultra-infer/)
RUN git clone --depth 1 https://github.com/PaddlePaddle/PaddleX.git /build/PaddleX

WORKDIR /build/PaddleX/deploy/ultra-infer/python

# Build ultra-infer with ORT backend only (the safe path on aarch64).
# CMake auto-detects aarch64 and downloads the correct prebuilt
# onnxruntime, paddle2onnx, opencv, patchelf, and protobuf binaries.
ENV WITH_GPU=OFF
ENV ENABLE_ORT_BACKEND=ON
ENV ENABLE_OPENVINO_BACKEND=OFF
ENV ENABLE_TRT_BACKEND=OFF
ENV ENABLE_PADDLE_BACKEND=OFF
ENV ENABLE_VISION=OFF
ENV ENABLE_TEXT=OFF
ENV ENABLE_BENCHMARK=OFF
ENV CC=gcc
ENV CXX=g++

RUN python setup.py build && python setup.py bdist_wheel

# ── Stage 2: runtime image ──────────────────────────────────────────
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Prevent PaddlePaddle PIR executor from crashing on aarch64
ENV FLAGS_enable_pir_in_executor=0
ENV FLAGS_enable_pir_api=0
ENV FLAGS_use_mkldnn=0
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

WORKDIR /opt/paddleocr

# Install PaddlePaddle (needed by PaddleX for model I/O, but its
# native inference kernels are bypassed via HPI + ONNX Runtime).
RUN pip install --no-cache-dir \
    "paddlepaddle>=3.0.0" \
    -f https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install the aarch64 ultra-infer wheel built in stage 1,
# plus onnxruntime and paddle2onnx (used by PaddleX for model conversion).
COPY --from=ultra-infer-builder /build/PaddleX/deploy/ultra-infer/python/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/ultra_infer*.whl onnxruntime paddle2onnx \
    && rm -f /tmp/*.whl

# Install PaddleOCR from local source
COPY . /opt/paddleocr
ARG PADDLEOCR_VERSION=3.4.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${PADDLEOCR_VERSION}
RUN pip install --no-cache-dir ".[all]"

# Apply aarch64 patch to PaddleX's HPI backend selection.
# This adds aarch64 to suggest_inference_backend_and_config().
COPY deploy/docker/aarch64/patch_paddlex_hpi.py /tmp/patch_paddlex_hpi.py
RUN python /tmp/patch_paddlex_hpi.py && rm /tmp/patch_paddlex_hpi.py

COPY deploy/docker/aarch64/test_aarch64.py /opt/paddleocr/test_aarch64.py

CMD ["python", "test_aarch64.py"]
