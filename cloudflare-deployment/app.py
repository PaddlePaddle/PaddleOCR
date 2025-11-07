"""
FastAPI application for PaddleOCR on Cloudflare Containers
Optimized for 8GB RAM, 4 vCPU CPU-only inference
"""
import asyncio
import base64
import io
import logging
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import numpy as np

from paddleocr import PaddleOCR
from config import OCR_CONFIG, HOST, PORT, WORKERS, MAX_IMAGE_SIZE, ALLOWED_EXTENSIONS, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PaddleOCR API",
    description="OCR service powered by PaddleOCR on Cloudflare Containers",
    version="3.3.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR instance (initialized on startup)
ocr_engine: Optional[PaddleOCR] = None


# Request/Response Models
class OCRRequest(BaseModel):
    """Request model for base64 encoded images"""
    image: str = Field(..., description="Base64 encoded image")
    lang: Optional[str] = Field(None, description="Language code (ch, en, etc.)")

class OCRResult(BaseModel):
    """Single OCR detection result"""
    text: str
    confidence: float
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

class OCRResponse(BaseModel):
    """OCR response model"""
    success: bool
    results: List[OCRResult]
    processing_time_ms: float
    image_size: Optional[tuple] = None
    error: Optional[str] = None


# Startup event - Initialize OCR engine
@app.on_event("startup")
async def startup_event():
    """Initialize PaddleOCR on application startup"""
    global ocr_engine

    logger.info("Initializing PaddleOCR engine...")
    logger.info(f"Configuration: {OCR_CONFIG}")

    try:
        start_time = time.time()

        # Initialize OCR with config
        ocr_engine = PaddleOCR(**OCR_CONFIG)

        # Warm up with a dummy image to load models into memory
        logger.info("Warming up OCR engine with dummy inference...")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = ocr_engine.ocr(dummy_img, cls=OCR_CONFIG.get("use_angle_cls", True))

        init_time = time.time() - start_time
        logger.info(f"PaddleOCR initialized successfully in {init_time:.2f}s")
        logger.info("Ready to process requests")

    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR: {e}", exc_info=True)
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PaddleOCR API",
        "version": "3.3.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ocr_file": "/ocr (POST multipart/form-data)",
            "ocr_base64": "/ocr/base64 (POST JSON)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")

    return {
        "status": "healthy",
        "ocr_ready": True,
        "config": {
            "lang": OCR_CONFIG["lang"],
            "ocr_version": OCR_CONFIG["ocr_version"],
            "cpu_threads": OCR_CONFIG["cpu_threads"]
        }
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr_from_file(
    file: UploadFile = File(...),
    lang: Optional[str] = None
):
    """
    Perform OCR on uploaded image file

    Args:
        file: Image file (JPEG, PNG, BMP, TIFF, WebP)
        lang: Optional language override (ch, en, french, etc.)

    Returns:
        OCRResponse with detected text, bboxes, and confidence scores
    """
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Read file
    try:
        contents = await file.read()

        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_IMAGE_SIZE / 1024 / 1024}MB"
            )

        # Load image
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image)

    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Perform OCR
    return await perform_ocr(img_array, image.size, lang)


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_from_base64(request: OCRRequest):
    """
    Perform OCR on base64 encoded image

    Args:
        request: OCRRequest with base64 encoded image

    Returns:
        OCRResponse with detected text, bboxes, and confidence scores
    """
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")

    try:
        # Decode base64 image
        # Handle both with and without data URI prefix
        image_data = request.image
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        image_bytes = base64.b64decode(image_data)

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Max size: {MAX_IMAGE_SIZE / 1024 / 1024}MB"
            )

        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)

    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    # Perform OCR
    return await perform_ocr(img_array, image.size, request.lang)


async def perform_ocr(
    img_array: np.ndarray,
    image_size: tuple,
    lang_override: Optional[str] = None
) -> OCRResponse:
    """
    Perform OCR inference on image array

    Args:
        img_array: NumPy array of image
        image_size: Original image size (width, height)
        lang_override: Optional language override

    Returns:
        OCRResponse with results
    """
    start_time = time.time()

    try:
        # Run OCR in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ocr_engine.ocr(
                img_array,
                cls=OCR_CONFIG.get("use_angle_cls", True)
            )
        )

        processing_time = (time.time() - start_time) * 1000

        # Parse results
        ocr_results = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    bbox, (text, confidence) = line
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=float(confidence),
                        bbox=[[int(x), int(y)] for x, y in bbox]
                    ))

        logger.info(
            f"OCR completed: {len(ocr_results)} detections in {processing_time:.2f}ms"
        )

        return OCRResponse(
            success=True,
            results=ocr_results,
            processing_time_ms=processing_time,
            image_size=image_size
        )

    except Exception as e:
        logger.error(f"OCR inference failed: {e}", exc_info=True)
        return OCRResponse(
            success=False,
            results=[],
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        log_level=LOG_LEVEL.lower()
    )
