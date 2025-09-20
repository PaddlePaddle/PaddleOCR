FROM python:3.11-slim

# System deps for Pillow, OpenCV-lite bits, and Paddle OCR runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 ca-certificates wget \
 && rm -rf /var/lib/apt/lists/*

# Faster pip, no cache
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/

# Install PaddleOCR and friends (CPU)
# NOTE: PaddleOCR brings its own Paddle dependencies for manylinux CPU wheels.
# If you hit issues, install paddlepaddle-cpu explicitly:
#   pip install "paddlepaddle==3.1.1" -i https://mirror.baidu.com/pypi/simple
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py /app/
EXPOSE 8000

# Simple healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
  wget -qO- http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
