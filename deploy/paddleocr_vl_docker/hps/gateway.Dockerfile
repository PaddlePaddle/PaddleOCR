FROM python:3.10-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy gateway application
COPY gateway .

# Install Python dependencies
RUN --mount=type=bind,source=paddlex_hps_PaddleOCR-VL_sdk/client,target=/tmp/sdk \
    python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -r /tmp/sdk/requirements.txt \
    && python -m pip install --no-cache-dir /tmp/sdk/paddlex_hps_client-*.whl

# Configuration via environment variables
ENV HPS_TRITON_URL=paddleocr-vl-tritonserver:8001
ENV HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=16
ENV HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=64
ENV HPS_INFERENCE_TIMEOUT=600
ENV HPS_LOG_LEVEL=INFO
ENV UVICORN_WORKERS=4

EXPOSE 8080

# Run with multiple workers for better concurrency
# Note: Using shell form to allow environment variable expansion
CMD uvicorn --host 0.0.0.0 --port 8080 --workers ${UVICORN_WORKERS} app:app
