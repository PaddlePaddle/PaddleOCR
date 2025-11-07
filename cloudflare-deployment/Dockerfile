# Optimized Dockerfile for PaddleOCR on Cloudflare Containers
# Target: 8GB RAM, 4 vCPU, linux/amd64
# Python 3.11 for best performance and compatibility

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Paddle inference optimizations for CPU
    CPU_NUM=4 \
    # Reduce memory footprint
    PADDLEOCR_MODEL_DIR=/models \
    # Use system locale
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Install system dependencies required by OpenCV and PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Create models directory
RUN mkdir -p /models

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Use CPU version of PaddlePaddle for smaller image size
RUN pip install --no-cache-dir -r requirements.txt && \
    # Verify installation
    python -c "import paddle; print(f'PaddlePaddle version: {paddle.__version__}')" && \
    python -c "import paddleocr; print('PaddleOCR imported successfully')"

# Copy application code
COPY app.py .
COPY config.py .

# Expose port (required for local dev, accessible in production)
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run the application
CMD ["python", "app.py"]
