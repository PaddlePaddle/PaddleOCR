"""
audit_logger.py
Structured Audit Logging for Company-Server OCR service.
Ensures zero PII leakage: only logs metadata (doc_type, status, confidence,
customer_id, job_id, duration_ms) and asserts field values never enter logs.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

AUDIT_LOGGER_NAME = "company_server_ocr.audit"
audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)
audit_logger.setLevel(logging.INFO)

# In-memory buffer for testing audit log assertions
_audit_log_buffer = []


class JsonAuditFormatter(logging.Formatter):
    """Formats log records as strict JSON without PII."""

    def format(self, record: logging.LogRecord) -> str:
        data = getattr(record, "audit_data", None)
        if data is None:
            data = {"message": record.getMessage()}
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data["level"] = record.levelname
        return json.dumps(data)


# Configure a dedicated handler for the audit logger
_handler = logging.StreamHandler()
_handler.setFormatter(JsonAuditFormatter())
audit_logger.handlers = [_handler]
audit_logger.propagate = False


def log_audit_event(
    doc_type: str,
    status: str,
    confidence: Optional[float],
    job_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    reason: Optional[str] = None,
    duration_ms: Optional[float] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Log structured audit event. STRICTLY NO PII ALLOWED.
    Only metadata and operational metrics are accepted.
    """
    audit_data = {
        "event": "ocr_audit",
        "doc_type": doc_type,
        "status": status,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "job_id": job_id,
        "customer_id": customer_id,
        "reason": reason,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
    }
    if extra_meta:
        # Only allow safe metadata keys
        safe_keys = {"pages_count", "source", "qr_detected", "micr_detected", "quality_issues"}
        for k, v in extra_meta.items():
            if k in safe_keys:
                audit_data[k] = v

    record = logging.LogRecord(
        name=AUDIT_LOGGER_NAME,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    record.audit_data = audit_data
    audit_logger.handle(record)
    _audit_log_buffer.append(audit_data)
    if len(_audit_log_buffer) > 1000:
        del _audit_log_buffer[:200]
    return audit_data


def get_audit_log_buffer():
    """Retrieve in-memory audit logs for test assertions."""
    return list(_audit_log_buffer)


def clear_audit_log_buffer():
    """Clear in-memory audit logs."""
    _audit_log_buffer.clear()
