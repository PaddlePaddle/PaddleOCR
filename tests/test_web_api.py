"""
tests/test_web_api.py
Unit and integration tests for the Document OCR Web Application REST APIs.
Tests:
- /api/supported-types
- /api/stats
- /api/upload (text-based PDF -> ocr_required=False, image -> ocr_required=True)
- /api/documents (listing, filtering, search)
- /api/documents/{id} (retrieval)
- /api/documents/{id}/file (file serving)
- /api/documents/{id}/preview (thumbnail serving)
- DELETE /api/documents/{id}
- Authentication regression: 401 on unauthenticated access to /api/upload and dashboard endpoints
- Query parameter authentication for browser media loading (?token=...)
"""

import io
import json
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from main import app
import document_store
from security import create_access_token


@pytest.fixture(autouse=True)
def reset_app_security(monkeypatch):
    import main
    main._startup_security_error = None
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.setenv("AUTH_MODE", "disabled")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("REGISTERED_CLIENTS_JSON", raising=False)
    monkeypatch.delenv("REGISTERED_CLIENTS_FILE", raising=False)
    yield
    main._startup_security_error = None


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_public_endpoints_remain_open(client):
    """Liveness probe, engine metadata, and supported document catalog remain public."""
    assert client.get("/health").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert client.get("/engine-info").status_code == 200
    assert client.get("/api/engine-info").status_code == 200

    resp = client.get("/api/supported-types")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 14
    ids = [item["id"] for item in data]
    assert "auto" in ids
    assert "pan" in ids
    assert "aadhaar" in ids
    assert "bank_statement" in ids


def test_unauthenticated_requests_succeed_by_default(client):
    """By default (AUTH_MODE=disabled), requests succeed without authentication headers or tokens."""
    # 1. Stats and listings
    stats_res = client.get("/api/stats")
    assert stats_res.status_code == 200
    assert "total" in stats_res.json()

    docs_res = client.get("/api/documents")
    assert docs_res.status_code == 200
    assert "items" in docs_res.json()

    # 2. Non-existent document returns 404 Not Found, NEVER 401 Unauthorized
    assert client.get("/api/documents/non-existent-doc-id").status_code == 404
    assert client.get("/api/documents/non-existent-doc-id/file").status_code == 404
    assert client.get("/api/documents/non-existent-doc-id/preview").status_code == 404
    assert client.delete("/api/documents/non-existent-doc-id").status_code == 404


def test_auth_status_endpoint_reports_correct_state(client, monkeypatch):
    """Verify /api/auth-status reports runtime auth configuration to frontend."""
    # 1. Default: auth disabled
    res = client.get("/api/auth-status")
    assert res.status_code == 200
    data = res.json()
    assert data["auth_enabled"] is False
    assert data["auth_mode"] == "disabled"

    # Also test /auth-status alias and /health field
    alias_res = client.get("/auth-status")
    assert alias_res.status_code == 200
    assert alias_res.json()["auth_enabled"] is False

    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["auth_enabled"] is False

    # 2. When AUTH_ENABLED=true
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-at-least-32-chars-long-123456")
    monkeypatch.setenv("REGISTERED_CLIENTS_JSON", json.dumps({"test-client": "secret123"}))

    with TestClient(app) as test_client:
        auth_res = test_client.get("/api/auth-status")
        assert auth_res.status_code == 200
        auth_data = auth_res.json()
        assert auth_data["auth_enabled"] is True
        assert auth_data["auth_mode"] == "jwt"


def test_authenticated_flow_login_then_upload_document_success(monkeypatch):
    """
    Regression test for frontend-backend auth flow:
    When auth is enabled, verifies end-to-end flow:
    1. Unauthenticated upload fails with 401
    2. Client logs in via /auth/token and obtains access token
    3. Authenticated upload with Bearer token succeeds with 200 OK
    """
    secret = "test-secret-key-at-least-32-chars-long-123456"
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setenv("REGISTERED_CLIENTS_JSON", json.dumps({"frontend-app": "app-secret-password-123"}))

    with TestClient(app) as test_client:
        img = Image.new("RGB", (300, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), "INCOME TAX DEPARTMENT", fill=(0, 0, 0))
        draw.text((10, 50), "ABCDE1234F", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # Step 1: Unauthenticated request to /api/upload MUST fail with 401
        unauth_res = test_client.post(
            "/api/upload",
            files={"file": ("pan_test.png", buf, "image/png")},
            data={"doc_type": "auto"},
        )
        assert unauth_res.status_code == 401

        # Step 2: Login via /auth/token using client credentials
        login_res = test_client.post(
            "/auth/token",
            data={
                "client_id": "frontend-app",
                "client_secret": "app-secret-password-123",
            },
        )
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        assert token is not None

        # Step 3: Call /api/upload with Bearer token in Authorization header
        buf.seek(0)
        auth_upload_res = test_client.post(
            "/api/upload",
            files={"file": ("pan_test.png", buf, "image/png")},
            data={"doc_type": "auto"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert auth_upload_res.status_code == 200
        doc = auth_upload_res.json()
        assert doc["id"] is not None
        assert doc["filename"] == "pan_test.png"

        # Step 4: Clean up document
        del_res = test_client.delete(f"/api/documents/{doc['id']}", headers={"Authorization": f"Bearer {token}"})
        assert del_res.status_code == 200


def test_get_stats(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    stats = response.json()
    assert "total" in stats
    assert "ocr_processed" in stats
    assert "ocr_not_required" in stats


def test_upload_image_document(client, tmp_path):
    # Create an image with readable text in memory
    img = Image.new("RGB", (400, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "INCOME TAX DEPARTMENT", fill=(0, 0, 0))
    draw.text((20, 60), "PERMANENT ACCOUNT NUMBER", fill=(0, 0, 0))
    draw.text((20, 90), "ABCDE1234F", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/upload",
        files={"file": ("test_pan_card.png", buf, "image/png")},
        data={"doc_type": "auto"},
    )
    assert response.status_code == 200
    doc = response.json()
    assert doc["id"] is not None
    assert doc["filename"] == "test_pan_card.png"
    assert doc["ocr_required"] is True
    assert doc["text_source"] in ("rapid_ocr", "paddle_ocr")
    assert doc["file_type"] == ".png"
    doc_id = doc["id"]

    # Verify listing includes it
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    items = list_res.json()["items"]
    assert any(i["id"] == doc_id for i in items)

    # Verify fetching by ID
    get_res = client.get(f"/api/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == doc_id

    # Verify file endpoint directly without auth headers
    file_res = client.get(f"/api/documents/{doc_id}/file")
    assert file_res.status_code == 200
    assert file_res.headers["content-type"] == "image/png"

    # Verify preview endpoint directly without auth headers
    preview_res = client.get(f"/api/documents/{doc_id}/preview")
    assert preview_res.status_code == 200
    assert preview_res.headers["content-type"] == "image/png"

    # Verify deletion
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # After deletion, 404
    get_after = client.get(f"/api/documents/{doc_id}")
    assert get_after.status_code == 404


def test_upload_blank_image_rejected(client):
    # Blank/unreadable images must be rejected with HTTP 422 and not saved
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/upload",
        files={"file": ("blank_card.png", buf, "image/png")},
        data={"doc_type": "auto"},
    )
    assert response.status_code == 422
    assert "no text or zero confidence" in response.json()["detail"]

    # Verify blank document was NOT persisted to repository
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    items = list_res.json()["items"]
    assert not any(i.get("filename") == "blank_card.png" for i in items)


def test_upload_real_pdf_demo(client):
    demo_pdf_path = "/home/vighnesh/Downloads/Demo_PAN_Card.pdf"
    if not os.path.exists(demo_pdf_path):
        pytest.skip("Demo_PAN_Card.pdf not available in Downloads")

    with open(demo_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    response = client.post(
        "/api/upload",
        files={"file": ("Demo_PAN_Card.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"doc_type": "auto"},
    )
    assert response.status_code == 200
    doc = response.json()
    assert doc["id"] is not None
    # Embedded text layer was detected, so OCR was not required!
    assert doc["ocr_required"] is False
    assert doc["text_source"] in ("pdf_text_layer", "embedded_pdf_text")
    assert doc["doc_type"] == "pan"
    assert "ABCDE1234F" in str(doc.get("extracted_fields", {}))

    # Clean up
    client.delete(f"/api/documents/{doc['id']}")


def test_upload_cross_check_ordering_with_masked_fields(client):
    """
    Regression test for cross-check execution order bug:
    Ensures that /api/upload runs perform_cross_check against raw_fields BEFORE
    sanitize_extracted_fields. If it ran against sanitized fields, cross-checking
    an applicant's raw name against 'employee_name_masked' (e.g. 'RAHUL S*****')
    would fail, but running against raw fields must succeed with matched: True.
    """
    # Create a synthetic salary slip image with clear unmasked employee name
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "PAYSLIP / SALARY SLIP", fill=(0, 0, 0))
    draw.text((20, 70), "Employer: ACME GLOBAL SOLUTIONS PVT LTD", fill=(0, 0, 0))
    draw.text((20, 110), "Employee Name: RAHUL SHARMA", fill=(0, 0, 0))
    draw.text((20, 150), "Pay Period: July 2026", fill=(0, 0, 0))
    draw.text((20, 190), "Net Pay: Rs. 85,000", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Submit with expected_data containing the raw unmasked name
    expected_payload = json.dumps({"name": "RAHUL SHARMA"})
    response = client.post(
        "/api/upload",
        files={"file": ("test_salary_slip.png", buf, "image/png")},
        data={"doc_type": "salary_slip", "expected_data": expected_payload},
    )
    assert response.status_code == 200
    doc = response.json()
    doc_id = doc["id"]

    try:
        # Cross-check MUST succeed against raw unmasked name
        cross_check = doc.get("cross_check")
        assert cross_check is not None, "cross_check result missing from response"
        assert "name" in cross_check, f"cross_check missing 'name' field: {cross_check}"
        assert cross_check["name"]["matched"] is True, f"Cross-check failed: {cross_check['name']}"
        assert cross_check["name"]["score"] >= 0.95

        # PII minimisation: the response extracted_fields must ONLY contain masked fields
        ext_fields = doc.get("extracted_fields", {})
        assert "employee_name_masked" in ext_fields
        assert ext_fields["employee_name_masked"] == "RAHUL S*****"
        assert "employee_name" not in ext_fields
        assert "raw_employee_name" not in ext_fields
        assert not any(k.startswith("raw_") for k in ext_fields.keys())
        assert "RAHUL SHARMA" not in str(ext_fields)
    finally:
        # Clean up
        client.delete(f"/api/documents/{doc_id}")
