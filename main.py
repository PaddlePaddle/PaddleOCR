"""
main.py
Company-Server OCR FastAPI Microservice:
- Endpoints:
  - POST /ocr/{doc_type} (Async job queue with optional ?sync=true)
  - GET /ocr/jobs/{job_id} (Polling endpoint)
  - POST /auth/token (Mint short-lived JWTs)
  - GET /health (Health check)
- End-to-end document verification, PII minimisation, QR/MICR decoding,
  image quality pre-checks, temp file lifecycle management, and audit logging.
"""

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger("company_server_ocr")


from fastapi import (

    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from audit_logger import log_audit_event
import document_store
from extractors import (
    PII_ALLOWLIST,
    extract_document_fields,
    extract_document_fields_raw,
    sanitize_extracted_fields,
)
from image_quality import evaluate_image_quality
from micr_reader import extract_micr_from_cheque
from ocr_engine import (
    OCREngine,
    OCRDocumentResult,
    check_pdf_text_layer,
    get_languages_for_doc_type,
    get_ocr_engine_info,
    render_pdf_pages_to_images,
    render_thumbnail,
    validate_ocr_engine_configuration,
)
from qr_decoder import decode_qr_from_image, parse_qr_payload, reconcile_ocr_and_qr
from queue_manager import JobRecord, job_queue
from security import (
    authenticate_request,
    create_access_token,
    get_auth_mode,
    is_auth_enabled,
    validate_security_configuration,
    verify_client_credentials,
)
from verifier import (
    check_doc_type_mismatch,
    detect_document_type,
    determine_document_status,
    load_valid_bank_codes,
    perform_cross_check,
    validate_document_checksums,
)

# Configuration & Constants
TEMP_DIR = os.getenv("OCR_TEMP_DIR", os.path.join(tempfile.gettempdir(), "company_ocr_temp"))
TEMP_FILE_TTL_MINUTES = int(os.getenv("TEMP_FILE_TTL_MINUTES", "15"))

os.makedirs(TEMP_DIR, exist_ok=True)
ocr_engine = OCREngine()


# ==============================================================================
# Orphaned Temp File Sweeper
# ==============================================================================

def sweep_orphaned_temp_files(max_age_minutes: int = TEMP_FILE_TTL_MINUTES):
    """Clean up any leftover temporary files older than max_age_minutes."""
    now = time.time()
    cutoff = now - (max_age_minutes * 60)
    try:
        for entry in os.scandir(TEMP_DIR):
            if entry.is_file():
                try:
                    stat = entry.stat()
                    if stat.st_mtime < cutoff:
                        os.remove(entry.path)
                except Exception:
                    pass
    except Exception:
        pass


async def periodic_temp_cleaner():
    """Periodic background task to ensure temp files are cleaned even on crashes."""
    while True:
        try:
            await asyncio.sleep(300)  # Sweep every 5 minutes
            sweep_orphaned_temp_files()
        except asyncio.CancelledError:
            break
        except Exception:
            pass


_startup_security_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_security_error
    try:
        # Startup: fail fast if security configuration is invalid
        validate_security_configuration()
        # Startup: fail fast if bank codes configuration is invalid
        load_valid_bank_codes()
        # Startup: fail fast if configured OCR engine is invalid or unavailable
        validate_ocr_engine_configuration()
        _startup_security_error = None
    except Exception as ex:
        _startup_security_error = str(ex)
        raise

    # Sweep leftover temp files, start job queue and background sweep task
    sweep_orphaned_temp_files()
    await job_queue.start()
    cleaner_task = asyncio.create_task(periodic_temp_cleaner())
    yield
    # Shutdown: stop worker and cleaner
    cleaner_task.cancel()
    await job_queue.stop()



app = FastAPI(
    title="Company-Server OCR",
    description="FastAPI OCR verification and PII minimisation service built on PaddleOCR",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_startup_security_check(request: Request, call_next):
    if _startup_security_error is not None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": f"Service unavailable: startup security check failed: {_startup_security_error}"
            },
        )
    return await call_next(request)


# ==============================================================================
# Core OCR Processing Logic
# ==============================================================================

def execute_ocr_pipeline(
    file_path: str,
    doc_type: str,
    expected_data: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the end-to-end OCR pipeline synchronously:
    1. Image quality pre-check
    2. OCR Engine pass (PaddleOCR)
    3. Document type mismatch check
    4. QR code decoding & reconciliation
    5. MICR reading (if cancelled cheque)
    6. Field extraction & PII minimisation
    7. Checksums & cross-check verification
    8. Structured audit log entry (strictly no PII)
    """
    start_time = time.time()
    doc_type = doc_type.lower().strip()

    # 1. Quality Pre-check (evaluate first page or image)
    quality_issues: List[str] = []
    first_page_img: Optional[Image.Image] = None

    try:
        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            with Image.open(file_path) as img:
                first_page_img = img.copy()
        elif file_path.lower().endswith(".pdf"):
            # Render first page for quality check if poppler/PyMuPDF available
            try:
                import fitz
                doc = fitz.open(file_path)
                if len(doc) > 0:
                    page = doc[0]
                    pix = page.get_pixmap()
                    first_page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            except Exception:
                pass
    except Exception:
        pass

    if first_page_img:
        q_result = evaluate_image_quality(first_page_img)
        if not q_result.is_acceptable:
            duration_ms = (time.time() - start_time) * 1000
            log_audit_event(
                doc_type=doc_type,
                status="low_confidence",
                confidence=0.0,
                job_id=job_id,
                customer_id=customer_id,
                reason="image_quality",
                duration_ms=duration_ms,
                extra_meta={"quality_issues": q_result.issues},
            )
            return {
                "status": "low_confidence",
                "reason": "image_quality",
                "doc_type": doc_type,
                "confidence": 0.0,
                "field_confidences": {},
                "quality_details": {
                    "issues": q_result.issues,
                    "blur_score": q_result.blur_score,
                },
                "extracted_fields": {},
            }

    # 2. OCR Extraction
    languages = get_languages_for_doc_type(doc_type)
    try:
        doc_res: OCRDocumentResult = ocr_engine.process_file(file_path, languages=languages)
    except Exception as ex:
        duration_ms = (time.time() - start_time) * 1000
        logger.error("OCR engine crashed for job %s: %s", job_id, ex, exc_info=True)
        log_audit_event(
            doc_type=doc_type,
            status="error",
            confidence=0.0,
            job_id=job_id,
            customer_id=customer_id,
            reason="ocr_engine_returned_no_text",
            duration_ms=duration_ms,
            extra_meta={"engine_error": str(ex)},
        )
        return {
            "status": "error",
            "reason": "ocr_engine_returned_no_text",
            "doc_type": doc_type,
            "confidence": 0.0,
            "field_confidences": {},
            "extracted_fields": {},
            "message": f"OCR engine execution failed: {str(ex)}",
        }

    # Hard check: empty/whitespace text, zero confidence, zero lines, or explicit engine_error
    has_text = bool(doc_res.full_text and doc_res.full_text.strip())
    has_lines = any(bool(p.lines) for p in doc_res.pages) if doc_res.pages else False
    has_engine_error = bool(getattr(doc_res, "engine_error", None))

    if has_engine_error or not has_text or doc_res.average_confidence == 0.0 or not has_lines:
        duration_ms = (time.time() - start_time) * 1000
        failure_reason = getattr(doc_res, "engine_error", None) or "OCR engine returned no text or zero confidence"
        log_audit_event(
            doc_type=doc_type,
            status="error",
            confidence=0.0,
            job_id=job_id,
            customer_id=customer_id,
            reason="ocr_engine_returned_no_text",
            duration_ms=duration_ms,
            extra_meta={"engine_error": failure_reason},
        )
        return {
            "status": "error",
            "reason": "ocr_engine_returned_no_text",
            "doc_type": doc_type,
            "confidence": 0.0,
            "field_confidences": {},
            "extracted_fields": {},
            "message": failure_reason,
        }

    # 3. Document-Type Mismatch Detection
    is_mismatch, detected_type = check_doc_type_mismatch(doc_type, doc_res.full_text)
    if is_mismatch:
        duration_ms = (time.time() - start_time) * 1000
        log_audit_event(
            doc_type=doc_type,
            status="error",
            confidence=0.0,
            job_id=job_id,
            customer_id=customer_id,
            reason="doc_type_mismatch",
            duration_ms=duration_ms,
        )
        return {
            "status": "error",
            "reason": "doc_type_mismatch",
            "doc_type": doc_type,
            "detected_type": detected_type,
            "confidence": 0.0,
            "field_confidences": {},
            "extracted_fields": {},
            "message": f"Uploaded document appears to be '{detected_type}' rather than requested '{doc_type}'",
        }

    # 4. Raw Field Extraction (with unmasked & raw fields available for verification)
    raw_fields, field_confidences = extract_document_fields_raw(doc_type, doc_res)

    # 5. QR Code Decoding & Reconciliation (Aadhaar, Udyam, FSSAI)
    qr_disagreements: List[Dict[str, Any]] = []
    if doc_type in ("aadhaar", "udyam", "fssai"):
        qr_texts: List[str] = []
        if first_page_img:
            qr_texts = decode_qr_from_image(first_page_img)

        for qr_text in qr_texts:
            qr_data = parse_qr_payload(doc_type, qr_text)
            reconciled_fields, disagreements = reconcile_ocr_and_qr(raw_fields, qr_data)
            raw_fields = reconciled_fields
            qr_disagreements.extend(disagreements)

    # 6. MICR Line Reading (Cancelled Cheque)
    if doc_type == "cancelled_cheque":
        ocr_func = None
        if first_page_img:
            ocr_func = lambda img: (ocr_engine.process_image(img).full_text, None)
        micr_data = extract_micr_from_cheque(
            cheque_img=first_page_img or Image.new("RGB", (100, 100)),
            ocr_func=ocr_func,
            ocr_full_text=doc_res.full_text,
            full_page_fields=raw_fields,
        )
        raw_fields["micr_line"] = micr_data.get("micr_line")
        raw_fields["micr_confidence"] = micr_data.get("micr_confidence", "low")
        if micr_data.get("micr_code"):
            raw_fields["micr_code"] = micr_data["micr_code"]
        if micr_data.get("account_number_masked"):
            raw_fields["micr_account_number_masked"] = micr_data["account_number_masked"]
        if micr_data.get("tran_code"):
            raw_fields["tran_code"] = micr_data["tran_code"]
        if micr_data.get("cheque_number") and not raw_fields.get("cheque_number"):
            raw_fields["cheque_number"] = micr_data["cheque_number"]
        if "micr_match" in micr_data:
            raw_fields["micr_match"] = micr_data["micr_match"]
        if micr_data.get("micr_disagreements"):
            raw_fields["micr_disagreements"] = micr_data["micr_disagreements"]

    # 7. Checksums & Format Validation on RAW fields (raw_aadhaar is present for Verhoeff validation)
    is_valid_format, invalid_reason = validate_document_checksums(doc_type, raw_fields)
    overall_status = "success"
    status_reason = None

    if not is_valid_format:
        overall_status = "low_confidence"
        status_reason = invalid_reason

    # 8. Cross-check against expected fields on RAW fields (unmasked employee_name is present)
    cross_check_results = None
    if expected_data:
        cross_check_results = perform_cross_check(raw_fields, expected_data)

    # 9. PII Minimisation: Sanitize fields strictly AFTER verification and cross-checking
    sanitized_fields, sanitized_confidences = sanitize_extracted_fields(doc_type, raw_fields, field_confidences)

    # Overall confidence calculation
    if sanitized_confidences:
        calculated_conf = sum(sanitized_confidences.values()) / len(sanitized_confidences)
    elif field_confidences:
        calculated_conf = sum(field_confidences.values()) / len(field_confidences)
    else:
        calculated_conf = doc_res.average_confidence

    duration_ms = (time.time() - start_time) * 1000

    # 10. Structured Audit Logging (STRICTLY NO PII)
    log_audit_event(
        doc_type=doc_type,
        status=overall_status,
        confidence=calculated_conf,
        job_id=job_id,
        customer_id=customer_id,
        reason=status_reason,
        duration_ms=duration_ms,
        extra_meta={
            "pages_count": len(doc_res.pages),
            "qr_detected": bool(qr_disagreements or (doc_type in ("aadhaar", "udyam", "fssai"))),
            "micr_detected": doc_type == "cancelled_cheque",
        },
    )

    response_payload = {
        "status": overall_status,
        "doc_type": doc_type,
        "confidence": round(calculated_conf, 4),
        "field_confidences": sanitized_confidences,
        "extracted_fields": sanitized_fields,
    }

    if status_reason:
        response_payload["reason"] = status_reason
    if qr_disagreements:
        response_payload["qr_disagreements"] = qr_disagreements
    if cross_check_results:
        response_payload["cross_check"] = cross_check_results

    return response_payload



# ==============================================================================
# Endpoints
# ==============================================================================

@app.post("/ocr/{doc_type}")
async def process_ocr_endpoint(
    doc_type: str,
    file: UploadFile = File(...),
    expected: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    customer_id: Optional[str] = Form(None),
    sync: bool = Query(False, description="Set True for immediate synchronous execution"),
    auth: dict = Depends(authenticate_request),
):
    """
    POST /ocr/{doc_type}
    Enqueues OCR job and returns a job_id (async queue pattern).
    Set ?sync=true for immediate synchronous execution.
    """
    # Parse optional expected JSON data
    expected_data = None
    if expected:
        try:
            expected_data = json.loads(expected)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Expected data must be a valid JSON string",
            )

    # Save uploaded file to temp directory
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=TEMP_DIR, suffix=suffix)
    temp_path = temp_file.name

    try:
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
    except Exception as ex:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded file: {str(ex)}",
        )

    # If synchronous mode requested
    if sync:
        try:
            result = execute_ocr_pipeline(
                file_path=temp_path,
                doc_type=doc_type,
                expected_data=expected_data,
                customer_id=customer_id,
            )
            return JSONResponse(content=result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Async Job-Queue pattern
    job = job_queue.create_job(
        doc_type=doc_type,
        webhook_url=webhook_url,
        customer_id=customer_id,
    )

    async def run_job():
        try:
            return execute_ocr_pipeline(
                file_path=temp_path,
                doc_type=doc_type,
                expected_data=expected_data,
                job_id=job.job_id,
                customer_id=customer_id,
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    await job_queue.enqueue(job.job_id, run_job)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job.job_id,
            "status": "pending",
            "doc_type": doc_type,
            "created_at": job.created_at,
            "poll_url": f"/ocr/jobs/{job.job_id}",
        },
    )


@app.get("/ocr/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    auth: dict = Depends(authenticate_request),
):
    """Poll the status and result of an async OCR job."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )

    return {
        "job_id": job.job_id,
        "doc_type": job.doc_type,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "result": job.result,
        "error": job.error,
    }


@app.post("/auth/token")
async def issue_token(
    client_id: str = Form(...),
    client_secret: str = Form(...),
):
    """
    Mint short-lived JWT Bearer token for registered clients.
    Rejects with 401 Unauthorized if client_id or client_secret is invalid.
    """
    if not verify_client_credentials(client_id, client_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=client_id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_minutes": 30,
    }



@app.get("/health")
@app.get("/api/health")
async def health():
    """Health check endpoint reporting operational status, active OCR engine, and auth state."""
    engine_info = get_ocr_engine_info()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "ocr_engine": engine_info["active_engine"],
        "ocr_backend": engine_info["active_engine"],
        "ocr_engine_status": engine_info["status"],
        "auth_enabled": is_auth_enabled(),
        "auth_mode": get_auth_mode(),
    }


@app.get("/auth-status")
@app.get("/api/auth-status")
async def auth_status_endpoint():
    """
    Public unauthenticated endpoint returning current authentication configuration.
    Enables frontend client to dynamically adjust its authentication gating at runtime.
    """
    return {
        "auth_enabled": is_auth_enabled(),
        "auth_mode": get_auth_mode(),
    }


@app.get("/engine-info")
@app.get("/api/engine-info")
async def engine_info_endpoint():
    """Returns detailed information about active and configured OCR engine."""
    return get_ocr_engine_info()



# ==============================================================================
# Web Application & Dashboard REST APIs
# ==============================================================================

SUPPORTED_DOC_TYPES = [
    {"id": "auto", "name": "Auto Detect", "category": "General"},
    {"id": "pan", "name": "PAN Card", "category": "Identity"},
    {"id": "aadhaar", "name": "Aadhaar Card", "category": "Identity"},
    {"id": "cancelled_cheque", "name": "Cancelled Cheque", "category": "Financial"},
    {"id": "bank_statement", "name": "Bank Statement", "category": "Financial"},
    {"id": "salary_slip", "name": "Salary Slip", "category": "Financial"},
    {"id": "utility_bill", "name": "Utility Bill", "category": "Utility"},
    {"id": "udyam", "name": "Udyam Certificate", "category": "Business"},
    {"id": "fssai", "name": "FSSAI License", "category": "Business"},
    {"id": "shop_establishment", "name": "Shop & Establishment", "category": "Business"},
    {"id": "passport", "name": "Passport", "category": "Identity"},
    {"id": "voter_id", "name": "Voter ID", "category": "Identity"},
    {"id": "driving_licence", "name": "Driving Licence", "category": "Identity"},
    {"id": "itr", "name": "ITR Ack", "category": "Tax"},
    {"id": "gst_certificate", "name": "GST Certificate", "category": "Business"},
    {"id": "certificate_of_incorporation", "name": "Certificate of Incorporation", "category": "Business"},
    {"id": "partnership_deed", "name": "Partnership Deed", "category": "Legal"},
    {"id": "rent_agreement", "name": "Rent Agreement", "category": "Legal"},
    {"id": "form_16", "name": "Form 16", "category": "Tax"},
    {"id": "bank_passbook", "name": "Bank Passbook", "category": "Financial"},
    {"id": "property_tax_receipt", "name": "Property Tax Receipt", "category": "Tax"},
    {"id": "iec_certificate", "name": "IEC Certificate", "category": "Business"},
    {"id": "income_certificate", "name": "Income Certificate", "category": "Certificate"},
]


@app.get("/api/supported-types")
async def get_supported_types():
    """Returns list of supported document types for UI dropdowns and badges."""
    return SUPPORTED_DOC_TYPES


@app.get("/api/stats")
async def get_dashboard_stats(
    auth: dict = Depends(authenticate_request),
):
    """Retrieve aggregate document statistics for the dashboard."""
    return document_store.get_stats()


@app.get("/api/documents")
async def list_documents_endpoint(
    search: Optional[str] = Query(None),
    doc_type: Optional[str] = Query(None),
    ocr_required: Optional[bool] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth: dict = Depends(authenticate_request),
):
    """Retrieve paginated list of uploaded documents with search and filtering."""
    items = document_store.list_documents(
        status_filter=status,
        ocr_required_filter=ocr_required,
        doc_type_filter=doc_type,
        search=search,
        limit=limit,
        offset=offset,
    )
    all_filtered = document_store.list_documents(
        status_filter=status,
        ocr_required_filter=ocr_required,
        doc_type_filter=doc_type,
        search=search,
        limit=10000,
        offset=0,
    )
    return {
        "items": items,
        "total": len(all_filtered),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/documents/{doc_id}")
async def get_document_endpoint(
    doc_id: str,
    auth: dict = Depends(authenticate_request),
):
    """Retrieve full details, extracted fields, and verification results of a document."""
    doc = document_store.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )
    return doc


@app.get("/api/documents/{doc_id}/file")
async def get_document_file_endpoint(
    doc_id: str,
    download: bool = Query(False),
    auth: dict = Depends(authenticate_request),
):
    """Serve the original uploaded document for in-browser PDF or image preview (inline) or download."""
    file_path = document_store.get_document_file_path(doc_id)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File for document '{doc_id}' not found",
        )

    ext = os.path.splitext(file_path)[1].lower()
    media_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    doc = document_store.get_document(doc_id)
    filename = doc.get("filename") if doc else os.path.basename(file_path)
    disposition = "attachment" if download else "inline"
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename,
        content_disposition_type=disposition,
    )


@app.get("/api/documents/{doc_id}/preview")
async def get_document_preview_endpoint(
    doc_id: str,
    auth: dict = Depends(authenticate_request),
):
    """Serve the thumbnail image preview for the document."""
    preview_path = document_store.get_document_preview_path(doc_id)
    if not preview_path or not os.path.exists(preview_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preview for document '{doc_id}' not found",
        )
    return FileResponse(preview_path, media_type="image/png")


@app.delete("/api/documents/{doc_id}")
async def delete_document_endpoint(
    doc_id: str,
    auth: dict = Depends(authenticate_request),
):
    """Permanently delete a document, its stored files, and metadata record."""
    deleted = document_store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )
    return {"success": True, "id": doc_id, "message": "Document deleted successfully"}


@app.post("/api/upload")
async def upload_document_endpoint(
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(None),
    expected_data: Optional[str] = Form(None),
    auth: dict = Depends(authenticate_request),
):
    """
    Direct browser upload endpoint:
    - Automatically tests if PDF has selectable text layer (>= 50 chars via pdftotext)
    - If text layer exists: extracts directly (ocr_required=false, text_source='embedded_pdf_text')
    - If scanned/image: runs PaddleOCR (ocr_required=true, text_source='paddle_ocr')
    - Classifies doc_type if not manually supplied (auto-detect)
    - Extracts fields, performs checksum/format verification, minimises PII
    - Generates document thumbnail
    - Stores document and analysis results locally in uploads/
    - Returns rich document record
    """
    filename = file.filename or "document.bin"
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: PDF, PNG, JPG, JPEG, WEBP",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    if len(file_bytes) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 25 MB limit",
        )

    temp_path = os.path.join(TEMP_DIR, f"web_upload_{uuid.uuid4()}{ext}")
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    try:
        # 1. Text Layer Detection & OCR
        requested_type = (doc_type or "").strip().lower()
        init_langs = get_languages_for_doc_type(requested_type) if requested_type and requested_type != "auto" else None
        try:
            if ext == ".pdf":
                doc_res = ocr_engine.process_pdf(temp_path, languages=init_langs)
            else:
                doc_res = ocr_engine.process_file(temp_path, languages=init_langs)
        except Exception as ex:
            logger.error("OCR extraction exception in /api/ingest: %s", ex, exc_info=True)
            doc_res = OCRDocumentResult(
                pages=[],
                full_text="",
                average_confidence=0.0,
                ocr_required=True,
                text_source="none",
                engine_error=f"ocr_extraction_failed: {str(ex)}",
            )

        ocr_required = getattr(doc_res, "ocr_required", True)
        text_source = getattr(doc_res, "text_source", "none")

        has_text = bool(doc_res.full_text and doc_res.full_text.strip())
        has_lines = any(bool(p.lines) for p in doc_res.pages) if doc_res.pages else False
        has_engine_error = bool(getattr(doc_res, "engine_error", None))

        raw_fields: Dict[str, Any] = {}
        sanitized_fields: Dict[str, Any] = {}
        field_confs: Dict[str, float] = {}
        checksum_valid = True
        checksum_reason: Optional[str] = None
        cross_check_results = None

        if has_engine_error or not has_text or doc_res.average_confidence == 0.0 or not has_lines:
            failure_detail = getattr(doc_res, "engine_error", None) or "OCR engine returned no text or zero confidence: document image may be blank, corrupt, or unreadable"
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=failure_detail,
            )

        # 2. Document Type Classification
        if not requested_type or requested_type == "auto":
            detected = detect_document_type(doc_res.full_text)
            resolved_type = detected if detected else "unknown"
            # If resolved_type uses Devanagari passes and the initial pass was English-only,
            # and OCR was actually required, execute the bilingual pass now
            auto_langs = get_languages_for_doc_type(resolved_type)
            if auto_langs and len(auto_langs) > 1:
                dev_chars_now = len(re.findall(r"[\u0900-\u097F]", doc_res.full_text))
                if doc_res.ocr_required or dev_chars_now < 10:
                    try:
                        if ext == ".pdf":
                            doc_res = ocr_engine.process_pdf(temp_path, languages=auto_langs)
                        else:
                            doc_res = ocr_engine.process_file(temp_path, languages=auto_langs)
                    except Exception as ex:
                        logger.warning("Secondary Devanagari pass in /api/upload failed: %s", ex)
            elif resolved_type == "unknown" and doc_res.ocr_required:
                try:
                    fallback_langs = ["en", "hi", "mr"]
                    if ext == ".pdf":
                        retry_res = ocr_engine.process_pdf(temp_path, languages=fallback_langs)
                    else:
                        retry_res = ocr_engine.process_file(temp_path, languages=fallback_langs)
                    retry_detected = detect_document_type(retry_res.full_text)
                    if retry_detected:
                        doc_res = retry_res
                        resolved_type = retry_detected
                    elif len(re.findall(r"[\u0900-\u097F]", retry_res.full_text)) > len(re.findall(r"[\u0900-\u097F]", doc_res.full_text)):
                        doc_res = retry_res
                except Exception as ex:
                    logger.warning("Regional fallback pass in /api/upload failed: %s", ex)
        else:
            resolved_type = requested_type

        first_page_img: Optional[Image.Image] = None
        if doc_res.pages and doc_res.pages[0].image:
            first_page_img = doc_res.pages[0].image
        elif ext == ".pdf":
            pdf_imgs = render_pdf_pages_to_images(temp_path)
            if pdf_imgs:
                first_page_img = pdf_imgs[0]
        else:
            try:
                with Image.open(temp_path) as img:
                    first_page_img = img.convert("RGB").copy()
            except Exception:
                pass

        # 3. Field Extraction & Verification
        if resolved_type != "unknown":
            try:
                raw_fields, field_confs = extract_document_fields_raw(resolved_type, doc_res)
                if resolved_type == "cancelled_cheque":
                    ocr_func = None
                    if first_page_img:
                        ocr_func = lambda img: (ocr_engine.process_image(img).full_text, None)
                    micr_data = extract_micr_from_cheque(
                        cheque_img=first_page_img or Image.new("RGB", (100, 100)),
                        ocr_func=ocr_func,
                        ocr_full_text=doc_res.full_text,
                        full_page_fields=raw_fields,
                    )
                    raw_fields["micr_line"] = micr_data.get("micr_line")
                    raw_fields["micr_confidence"] = micr_data.get("micr_confidence", "low")
                    if micr_data.get("micr_code"):
                        raw_fields["micr_code"] = micr_data["micr_code"]
                    if micr_data.get("account_number_masked"):
                        raw_fields["micr_account_number_masked"] = micr_data["account_number_masked"]
                    if micr_data.get("tran_code"):
                        raw_fields["tran_code"] = micr_data["tran_code"]
                    if micr_data.get("cheque_number") and not raw_fields.get("cheque_number"):
                        raw_fields["cheque_number"] = micr_data["cheque_number"]
                    if "micr_match" in micr_data:
                        raw_fields["micr_match"] = micr_data["micr_match"]
                    if micr_data.get("micr_disagreements"):
                        raw_fields["micr_disagreements"] = micr_data["micr_disagreements"]

                checksum_valid, checksum_reason = validate_document_checksums(resolved_type, raw_fields)
                # 4. Optional Cross-check on raw unmasked fields
                if expected_data:
                    try:
                        expected_dict = json.loads(expected_data)
                        if isinstance(expected_dict, dict):
                            cross_check_results = perform_cross_check(raw_fields, expected_dict)
                    except Exception:
                        pass
                sanitized_fields, field_confs = sanitize_extracted_fields(resolved_type, raw_fields, field_confs)
            except Exception as ex:
                checksum_valid = False
                checksum_reason = f"Extraction error: {str(ex)}"
        else:
            sanitized_fields = {"document_type": "Unknown / Unclassified"}

        # 5. Determine Overall Status
        doc_status = determine_document_status(
            checksum_valid=checksum_valid,
            average_confidence=doc_res.average_confidence,
            vault_mode=True,
        )

        # 6. Generate Thumbnail Preview
        thumb_bytes = render_thumbnail(temp_path, is_pdf=(ext == ".pdf"))

        # 7. Package Result Data (Strictly sanitized fields, no raw PII)
        result_payload = {
            "doc_type": resolved_type,
            "ocr_required": ocr_required,
            "text_source": text_source,
            "status": doc_status,
            "confidence": doc_res.average_confidence,
            "pages": len(doc_res.pages),
            "reason": checksum_reason,
            "extracted_fields": sanitized_fields,
            "field_confidences": field_confs,
            "extracted_text": doc_res.full_text,
            "checksum_valid": checksum_valid,
            "checksum_reason": checksum_reason,
            "cross_check": cross_check_results,
        }

        # 8. Persist to Document Store
        saved_record = document_store.save_document(
            file_bytes=file_bytes,
            filename=filename,
            result_data=result_payload,
            thumbnail_bytes=thumb_bytes,
        )

        return saved_record
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# Mount frontend single page application if built
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

