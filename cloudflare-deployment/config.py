"""
Configuration for PaddleOCR Cloudflare Deployment
"""
import os

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))  # Single worker for container

# PaddleOCR configuration
OCR_CONFIG = {
    # Language support - default to Chinese + English, can be changed
    "lang": os.getenv("OCR_LANG", "ch"),  # ch, en, french, german, korean, japan, etc.

    # OCR version - use latest PP-OCRv4 or PP-OCRv5
    "ocr_version": os.getenv("OCR_VERSION", "PP-OCRv4"),

    # Use lightweight models for faster inference and lower memory
    "use_angle_cls": os.getenv("USE_ANGLE_CLS", "true").lower() == "true",

    # CPU optimization
    "use_gpu": False,  # No GPU in Cloudflare Containers (yet)
    "enable_mkldnn": True,  # Intel MKL-DNN for faster CPU inference
    "cpu_threads": 4,  # Match container vCPU count

    # Memory optimization
    "use_space_char": True,
    "drop_score": 0.5,  # Minimum confidence score

    # Performance tuning
    "det_limit_side_len": 960,  # Max image dimension
    "rec_batch_num": 6,  # Batch size for recognition

    # Model directories (models will auto-download on first run)
    "det_model_dir": os.getenv("DET_MODEL_DIR"),
    "rec_model_dir": os.getenv("REC_MODEL_DIR"),
    "cls_model_dir": os.getenv("CLS_MODEL_DIR"),
}

# API configuration
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB default
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
