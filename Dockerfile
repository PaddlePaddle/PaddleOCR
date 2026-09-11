FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OCR_TEMP_DIR=/tmp/company_ocr_temp \
    AUTH_MODE=dual \
    TEMP_FILE_TTL_MINUTES=15

# Install system dependencies (Poppler, OpenCV runtime libraries, ZBar)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    libzbar0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications and install
COPY service_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY audit_logger.py image_quality.py micr_reader.py qr_decoder.py verifier.py extractors.py ocr_engine.py queue_manager.py security.py main.py /app/

# Create temp working directory
RUN mkdir -p /tmp/company_ocr_temp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
