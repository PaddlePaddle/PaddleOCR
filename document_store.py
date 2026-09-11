"""
document_store.py
Document storage abstraction for Document OCR Web Application:
- Local storage directories: uploads/original/, uploads/processed/, uploads/results/
- Thread-safe metadata index in uploads/documents.json
- Querying, filtering, statistics, and retrieval
- Designed for easy future transition to S3 / Supabase / GCS
"""

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

STORAGE_ROOT = os.getenv("DOCUMENT_STORAGE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads"))
ORIGINAL_DIR = os.path.join(STORAGE_ROOT, "original")
PROCESSED_DIR = os.path.join(STORAGE_ROOT, "processed")
RESULTS_DIR = os.path.join(STORAGE_ROOT, "results")
INDEX_FILE = os.path.join(STORAGE_ROOT, "documents.json")

_lock = threading.Lock()


def init_storage():
    """Ensure all required storage directories and index file exist."""
    os.makedirs(ORIGINAL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(INDEX_FILE):
        with _lock:
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)


# Initialize on import
init_storage()


def _load_index() -> List[Dict[str, Any]]:
    """Load metadata list from documents.json."""
    if not os.path.exists(INDEX_FILE):
        return []
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_index(items: List[Dict[str, Any]]):
    """Save metadata list to documents.json."""
    tmp_path = INDEX_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    os.replace(tmp_path, INDEX_FILE)


def sanitize_filename(filename: str) -> str:
    """Strip dangerous path characters from filename."""
    base = os.path.basename(filename)
    return "".join(c for c in base if c.isalnum() or c in "._- ") or "document.bin"


def save_document(
    file_bytes: bytes,
    filename: str,
    result_data: Dict[str, Any],
    thumbnail_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Persist an uploaded document, its analysis result, and optional thumbnail.
    Updates the local document index.
    """
    doc_id = str(uuid.uuid4())
    clean_name = sanitize_filename(filename)
    ext = os.path.splitext(clean_name)[1].lower() or ".bin"
    stored_filename = f"{doc_id}_{clean_name}"
    original_file_path = os.path.join(ORIGINAL_DIR, stored_filename)

    # 1. Write original file
    with open(original_file_path, "wb") as f:
        f.write(file_bytes)

    # 2. Write thumbnail if provided
    preview_filename = None
    if thumbnail_bytes:
        preview_filename = f"{doc_id}_thumb.png"
        preview_file_path = os.path.join(PROCESSED_DIR, preview_filename)
        with open(preview_file_path, "wb") as f:
            f.write(thumbnail_bytes)

    # 3. Format full document record
    doc_type = result_data.get("doc_type") or result_data.get("document_type") or "unknown"
    doc_type_clean = doc_type.lower().strip()
    status = result_data.get("status", "completed")
    if status in ("error", "failed"):
        human_doc_type = "OCR Error" if status == "error" else "Processing Failed"
    else:
        human_doc_type = doc_type_clean.replace("_", " ").title() if doc_type_clean != "unknown" else "Unknown Document"


    doc_record = {
        "id": doc_id,
        "filename": clean_name,
        "file_path": original_file_path,
        "file_size": len(file_bytes),
        "file_type": ext,
        "doc_type": doc_type_clean,
        "document_type": human_doc_type,
        "ocr_required": result_data.get("ocr_required", True),
        "text_source": result_data.get("text_source", "paddle_ocr"),
        "status": result_data.get("status", "completed"),
        "confidence": result_data.get("confidence", 1.0),
        "pages": result_data.get("pages", 1),
        "reason": result_data.get("reason"),
        "checksum_valid": result_data.get("checksum_valid", False if result_data.get("status") == "warning" else True),
        "checksum_reason": result_data.get("checksum_reason", result_data.get("reason")),
        "cross_check": result_data.get("cross_check"),
        "extracted_fields": result_data.get("extracted_fields", {}),
        "fields": result_data.get("extracted_fields", result_data.get("fields", {})),
        "field_confidences": result_data.get("field_confidences", {}),
        "extracted_text": result_data.get("extracted_text", ""),
        "has_preview": preview_filename is not None,
        "preview_url": f"/api/documents/{doc_id}/preview" if preview_filename else None,
        "file_url": f"/api/documents/{doc_id}/file",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # 4. Save result JSON
    result_path = os.path.join(RESULTS_DIR, f"{doc_id}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(doc_record, f, indent=2)

    # 5. Update index
    with _lock:
        items = _load_index()
        # Keep recent items at front
        items.insert(0, doc_record)
        _save_index(items)

    return doc_record


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full document details by ID."""
    result_path = os.path.join(RESULTS_DIR, f"{doc_id}.json")
    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def list_documents(
    status_filter: Optional[str] = None,
    ocr_required_filter: Optional[bool] = None,
    doc_type_filter: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Query stored documents with filters."""
    items = _load_index()

    filtered = []
    for item in items:
        if status_filter and status_filter.lower() != "all":
            if item.get("status", "").lower() != status_filter.lower():
                continue
        if ocr_required_filter is not None:
            if item.get("ocr_required") != ocr_required_filter:
                continue
        if doc_type_filter and doc_type_filter.lower() != "all":
            if item.get("doc_type", "").lower() != doc_type_filter.lower():
                continue
        if search:
            query = search.lower()
            name_match = query in item.get("filename", "").lower()
            type_match = query in item.get("document_type", "").lower()
            text_match = query in item.get("extracted_text", "").lower()
            if not (name_match or type_match or text_match):
                continue
        filtered.append(item)

    return filtered[offset : offset + limit]


def delete_document(doc_id: str) -> bool:
    """Delete a document, its files, and index entry."""
    with _lock:
        items = _load_index()
        target = next((item for item in items if item.get("id") == doc_id), None)
        if not target:
            return False

        # Remove files
        if target.get("file_path") and os.path.exists(target["file_path"]):
            try:
                os.remove(target["file_path"])
            except Exception:
                pass

        preview_file = os.path.join(PROCESSED_DIR, f"{doc_id}_thumb.png")
        if os.path.exists(preview_file):
            try:
                os.remove(preview_file)
            except Exception:
                pass

        result_path = os.path.join(RESULTS_DIR, f"{doc_id}.json")
        if os.path.exists(result_path):
            try:
                os.remove(result_path)
            except Exception:
                pass

        # Update index
        updated = [item for item in items if item.get("id") != doc_id]
        _save_index(updated)
        return True


def get_document_file_path(doc_id: str) -> Optional[str]:
    """Return local path to the original uploaded document file."""
    doc = get_document(doc_id)
    if doc and doc.get("file_path") and os.path.exists(doc["file_path"]):
        return doc["file_path"]
    return None


def get_document_preview_path(doc_id: str) -> Optional[str]:
    """Return local path to the rendered thumbnail."""
    preview_path = os.path.join(PROCESSED_DIR, f"{doc_id}_thumb.png")
    if os.path.exists(preview_path):
        return preview_path
    # Fallback: if original is an image, return original
    file_path = get_document_file_path(doc_id)
    if file_path and file_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        return file_path
    return None


def get_stats() -> Dict[str, int]:
    """Compute dashboard statistics."""
    items = _load_index()
    total = len(items)
    ocr_processed = sum(1 for d in items if d.get("ocr_required") is True)
    ocr_not_required = sum(1 for d in items if d.get("ocr_required") is False)
    failed = sum(1 for d in items if d.get("status") in ("error", "failed"))
    completed = sum(1 for d in items if d.get("status") in ("success", "completed", "low_confidence", "warning"))

    return {
        "total": total,
        "ocr_processed": ocr_processed,
        "ocr_not_required": ocr_not_required,
        "completed": completed,
        "failed": failed,
    }
