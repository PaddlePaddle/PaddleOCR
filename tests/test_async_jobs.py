"""
tests/test_async_jobs.py
Tests for async job queue, polling, JWT/API-key authentication,
client credential validation on /auth/token, fail-fast startup checks,
image quality pre-checks, and document mismatch error handling.
"""

import asyncio
import io
import json
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure test secrets are explicitly configured
os.environ["JWT_SECRET"] = "test-secret-key-for-unit-tests-only-32bytes"
os.environ["API_KEY"] = "test-static-api-key-2026"
os.environ["AUTH_MODE"] = "dual"
os.environ["AUTH_ENABLED"] = "true"

from main import app
from security import (
    clear_registered_clients,
    create_access_token,
    register_client,
    validate_security_configuration,
)


@pytest.fixture(autouse=True)
def setup_test_auth():
    """Setup test secrets and register standard test client."""
    import main
    main._startup_security_error = None
    os.environ["AUTH_ENABLED"] = "true"
    os.environ["AUTH_MODE"] = "dual"
    os.environ["JWT_SECRET"] = "test-secret-key-for-unit-tests-only-32bytes"
    os.environ["API_KEY"] = "test-static-api-key-2026"
    clear_registered_clients()
    register_client("n8n-node", "super-secret-n8n-token-credential")
    yield
    clear_registered_clients()
    main._startup_security_error = None
    os.environ.pop("AUTH_ENABLED", None)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_header():
    token = create_access_token(subject="test-user")
    return {"Authorization": f"Bearer {token}"}


def create_test_image_bytes(width=300, height=300, color=(255, 255, 255)):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_sharp_test_image_bytes(width=300, height=300):
    from PIL import ImageDraw
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(0, width, 10):
        draw.line([(i, 0), (i, height)], fill=(0, 0, 0), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()



def test_auth_rejection(client):
    """Endpoints must reject unauthenticated requests."""
    files = {"file": ("test.png", create_test_image_bytes(), "image/png")}
    resp = client.post("/ocr/pan", files=files)
    assert resp.status_code == 401


def test_auth_static_api_key(client):
    """Static API Key should be accepted when AUTH_MODE enables it."""
    headers = {"X-API-Key": "test-static-api-key-2026"}
    files = {"file": ("test.png", create_test_image_bytes(), "image/png")}
    resp = client.post("/ocr/pan?sync=true", files=files, headers=headers)
    assert resp.status_code in (200, 422)


def test_auth_token_minting_success_and_rejection(client):
    """
    POST /auth/token must:
    1. Mint token when valid registered client credentials are provided.
    2. Reject with 401 when wrong secret or unregistered client_id is given.
    """
    # 1. Success Path: registered client + correct secret
    resp = client.post(
        "/auth/token",
        data={
            "client_id": "n8n-node",
            "client_secret": "super-secret-n8n-token-credential",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in_minutes"] == 30

    # 2. Rejection Path: registered client + invalid secret -> 401
    bad_resp = client.post(
        "/auth/token",
        data={
            "client_id": "n8n-node",
            "client_secret": "incorrect-password",
        },
    )
    assert bad_resp.status_code == 401
    assert "Invalid client credentials" in bad_resp.json()["detail"]

    # 3. Rejection Path: unregistered client -> 401
    unregistered_resp = client.post(
        "/auth/token",
        data={
            "client_id": "unknown-intruder",
            "client_secret": "any-secret",
        },
    )
    assert unregistered_resp.status_code == 401

    # 4. Rejection Path: missing secret -> 422 Unprocessable Entity
    missing_secret_resp = client.post(
        "/auth/token",
        data={"client_id": "n8n-node"},
    )
    assert missing_secret_resp.status_code == 422


def test_service_refuses_to_start_without_jwt_secret(monkeypatch):
    """Assert service fails fast at startup if JWT_SECRET is unset in production/jwt mode."""
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.delenv("JWT_SECRET", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        validate_security_configuration()
    assert "Missing required environment variable 'JWT_SECRET'" in str(exc_info.value)


def test_service_refuses_to_start_in_api_key_mode_without_api_key(monkeypatch):
    """Assert service fails fast if API_KEY is unset in api_key mode."""
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        validate_security_configuration()
    assert "Missing required environment variable 'API_KEY'" in str(exc_info.value)


def test_service_refuses_to_start_without_registered_clients(monkeypatch):
    """
    Task 2: In jwt/dual mode, if no registered clients are configured,
    startup validation must fail fast.
    """
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-unit-tests-only-32bytes")
    monkeypatch.delenv("REGISTERED_CLIENTS_JSON", raising=False)
    monkeypatch.delenv("REGISTERED_CLIENTS_FILE", raising=False)
    clear_registered_clients()

    with pytest.raises(RuntimeError) as exc_info:
        validate_security_configuration()
    assert "No client credentials registered" in str(exc_info.value)


def test_client_credentials_loaded_from_external_file(monkeypatch, tmp_path):
    """
    Task 2: Confirm credentials are not hardcoded in source and load properly
    from external configured file, failing if the file is missing.
    """
    clear_registered_clients()
    creds_file = tmp_path / "clients.json"
    creds_file.write_text(json.dumps({"external-node": "external-secret-123"}))

    monkeypatch.setenv("REGISTERED_CLIENTS_FILE", str(creds_file))
    from security import is_client_store_configured, load_registered_clients, verify_client_credentials

    load_registered_clients()
    assert is_client_store_configured() is True
    assert verify_client_credentials("external-node", "external-secret-123") is True
    assert verify_client_credentials("external-node", "wrong-secret") is False

    # Missing file fails fast
    clear_registered_clients()
    monkeypatch.setenv("REGISTERED_CLIENTS_FILE", "/non/existent/clients.json")
    with pytest.raises(RuntimeError) as exc_info:
        load_registered_clients()
    assert "does not exist" in str(exc_info.value)


def test_integration_startup_failure_blocks_serving_requests(monkeypatch):
    """
    Task 3: Confirm that startup security check failure actually prevents
    the application from serving requests, returning 503 rather than unauthenticated traffic.
    """
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    clear_registered_clients()

    # 1. With context manager: Starlette lifespan raises RuntimeError during startup
    with pytest.raises(RuntimeError):
        with TestClient(app) as test_client:
            test_client.get("/health")

    # 2. Even if server exceptions are suppressed, middleware strictly intercepts requests
    test_client = TestClient(app, raise_server_exceptions=False)
    resp = test_client.get("/health")
    assert resp.status_code == 503
    assert "startup security check failed" in resp.json()["detail"]


def test_service_starts_cleanly_when_auth_disabled(monkeypatch):
    """When AUTH_MODE=disabled, the service starts cleanly with zero secrets configured and serves requests."""
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.setenv("AUTH_MODE", "disabled")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("REGISTERED_CLIENTS_JSON", raising=False)
    monkeypatch.delenv("REGISTERED_CLIENTS_FILE", raising=False)
    clear_registered_clients()

    with TestClient(app) as test_client:
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("healthy", "ok")

        # Also verify unauthenticated requests to protected endpoints succeed
        assert test_client.get("/api/stats").status_code == 200



def test_async_job_enqueue_and_poll(client, auth_header):
    """POST /ocr/{doc_type} enqueues job and GET /ocr/jobs/{job_id} retrieves result."""
    demo_pdf_path = "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Aadhaar_Card.pdf"
    if not os.path.exists(demo_pdf_path):
        pytest.skip("Demo file not found")

    with open(demo_pdf_path, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("Demo_Aadhaar_Card.pdf", file_bytes, "application/pdf")}
    resp = client.post("/ocr/aadhaar", files=files, headers=auth_header)
    assert resp.status_code == 202
    job_data = resp.json()
    assert "job_id" in job_data
    assert job_data["status"] == "pending"

    job_id = job_data["job_id"]

    # Poll job status
    poll_resp = client.get(f"/ocr/jobs/{job_id}", headers=auth_header)
    assert poll_resp.status_code == 200
    pdata = poll_resp.json()
    assert pdata["job_id"] == job_id
    assert pdata["doc_type"] == "aadhaar"


def test_sync_mode_execution(client, auth_header, monkeypatch):
    """?sync=true should return final result immediately."""
    demo_pdf_path = "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Cancelled_Cheque.pdf"
    if not os.path.exists(demo_pdf_path):
        pytest.skip("Demo file not found")

    # Inject DEMO bank code for synthetic demo cheque
    import verifier
    monkeypatch.setattr(verifier, "VALID_BANK_CODES", verifier.VALID_BANK_CODES | {"DEMO"})

    with open(demo_pdf_path, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("Demo_Cancelled_Cheque.pdf", file_bytes, "application/pdf")}
    resp = client.post("/ocr/cancelled_cheque?sync=true", files=files, headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_type"] == "cancelled_cheque"
    assert "extracted_fields" in data
    assert "cheque_number" in data["extracted_fields"]
    assert "ifsc" in data["extracted_fields"]
    assert data["status"] == "success"


def test_doc_type_mismatch_rejection(client, auth_header):
    """Uploading driving licence when doc_type=aadhaar must return error with reason doc_type_mismatch."""
    demo_dl_path = "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Driving_Licence.pdf"
    if not os.path.exists(demo_dl_path):
        pytest.skip("Demo file not found")

    with open(demo_dl_path, "rb") as f:
        file_bytes = f.read()

    # Upload Driving Licence as doc_type=aadhaar
    files = {"file": ("Demo_Driving_Licence.pdf", file_bytes, "application/pdf")}
    resp = client.post("/ocr/aadhaar?sync=true", files=files, headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert data["reason"] == "doc_type_mismatch"
    assert data["detected_type"] == "driving_licence"


def test_image_quality_precheck_blur_and_resolution(client, auth_header):
    """Tiny or severely blurry image must trigger low_confidence image_quality error."""
    tiny_img = create_test_image_bytes(width=50, height=50)
    files = {"file": ("tiny.png", tiny_img, "image/png")}

    resp = client.post("/ocr/pan?sync=true", files=files, headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "low_confidence"
    assert data["reason"] == "image_quality"
    assert "quality_details" in data
    assert any("low_resolution" in issue for issue in data["quality_details"]["issues"])


def test_ocr_empty_text_returns_hard_failure(client, auth_header, monkeypatch):
    """When OCR returns empty/whitespace text, pipeline must fail hard with reason=ocr_engine_returned_no_text."""
    import main
    from ocr_engine import OCRDocumentResult, OCRPageResult

    dummy_img = create_sharp_test_image_bytes(width=200, height=200)
    files = {"file": ("blank.png", dummy_img, "image/png")}

    # Mock OCR engine returning empty result
    empty_res = OCRDocumentResult(
        pages=[OCRPageResult(page_num=1, full_text="", lines=[], average_confidence=0.0)],
        full_text="",
        average_confidence=0.0,
        ocr_required=True,
        text_source="rapid_ocr",
    )
    monkeypatch.setattr(main.ocr_engine, "process_file", lambda path: empty_res)

    resp = client.post("/ocr/pan?sync=true", files=files, headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert data["reason"] == "ocr_engine_returned_no_text"
    assert data["extracted_fields"] == {}


def test_ocr_zero_confidence_returns_hard_failure(client, auth_header, monkeypatch):
    """When OCR engine returns 0.0 confidence, pipeline must fail hard with reason=ocr_engine_returned_no_text."""
    import main
    from ocr_engine import OCRDocumentResult, OCRPageResult, OCRLine

    dummy_img = create_sharp_test_image_bytes(width=200, height=200)
    files = {"file": ("zero_conf.png", dummy_img, "image/png")}

    zero_conf_res = OCRDocumentResult(
        pages=[OCRPageResult(page_num=1, full_text="SAMPLE TEXT", lines=[OCRLine("SAMPLE TEXT", 0.0)], average_confidence=0.0)],
        full_text="SAMPLE TEXT",
        average_confidence=0.0,
        ocr_required=True,
        text_source="rapid_ocr",
    )
    monkeypatch.setattr(main.ocr_engine, "process_file", lambda path: zero_conf_res)

    resp = client.post("/ocr/pan?sync=true", files=files, headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert data["reason"] == "ocr_engine_returned_no_text"
    assert data["extracted_fields"] == {}


def test_ocr_exception_returns_hard_failure(client, auth_header, monkeypatch):
    """When OCR engine raises an unhandled exception, pipeline must return error with reason=ocr_engine_returned_no_text."""
    import main

    dummy_img = create_sharp_test_image_bytes(width=200, height=200)
    files = {"file": ("crash.png", dummy_img, "image/png")}

    def crashing_ocr(path):
        raise RuntimeError("ONNX Runtime Engine execution aborted")

    monkeypatch.setattr(main.ocr_engine, "process_file", crashing_ocr)

    resp = client.post("/ocr/pan?sync=true", files=files, headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert data["reason"] == "ocr_engine_returned_no_text"
    assert "failed" in data.get("message", "").lower() or "aborted" in data.get("message", "").lower()



def test_engine_info_and_health_endpoints(client):
    """Health and engine-info endpoints must report active and configured engine details."""
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    hdata = health_resp.json()
    assert "ocr_engine" in hdata
    assert hdata["ocr_engine_status"] == "ready"

    info_resp = client.get("/engine-info")
    assert info_resp.status_code == 200
    idata = info_resp.json()
    assert idata["active_engine"] in ("rapidocr", "paddleocr")
    assert idata["configured_engine"] in ("auto", "rapidocr", "paddleocr")
    assert idata["status"] == "ready"
    assert idata["pdf_fallback"] == "pdftotext"


def test_explicit_ocr_engine_selection_and_validation():
    """OCR_ENGINE selection fails loud if unavailable engine is forced or setting is invalid."""
    import ocr_engine

    # Invalid engine name must raise ValueError
    with pytest.raises(ValueError):
        ocr_engine.init_ocr_engine("unsupported_engine")

    # Forcing rapidocr works
    ocr_engine.init_ocr_engine("rapidocr")
    assert ocr_engine.get_active_ocr_engine() == "rapidocr"

    # Forcing paddleocr on Python 3.14 (where paddle binary is unavailable) fails loud
    with pytest.raises(RuntimeError):
        ocr_engine.init_ocr_engine("paddleocr")

    # Reset back to auto
    ocr_engine.init_ocr_engine("auto")
    assert ocr_engine.get_active_ocr_engine() == "rapidocr"


def test_digital_pdf_fallback_honestly_labeled(monkeypatch):
    """Digital PDF with text layer must be honestly labeled text_source='pdf_text_layer' even with no neural OCR."""
    from ocr_engine import OCREngine

    demo_pdf_path = "/home/vighnesh/Downloads/More_Demo_OCR_Test_Documents/Demo_Aadhaar_Card.pdf"
    if not os.path.exists(demo_pdf_path):
        pytest.skip("Demo file not found")

    engine = OCREngine()
    # Mock neural OCR instance as None to simulate neural engine completely absent
    monkeypatch.setattr(engine, "paddle", None)

    doc_res = engine.process_pdf(demo_pdf_path)
    assert doc_res.ocr_required is False
    assert doc_res.text_source == "pdf_text_layer"
    assert doc_res.average_confidence >= 0.95
    assert len(doc_res.full_text.strip()) > 50
    assert doc_res.engine_error is None


def test_async_blank_image_fails_loud(client, auth_header):
    """
    Submitting an unreadable/blank image via the async job path must:
    1. Accept the job (202 Accepted) with status='pending'.
    2. Process through the queue and surface a failed/error state with
       reason='ocr_engine_returned_no_text' and zero confidence — not a silent success.
    """
    import time
    from PIL import ImageDraw

    # Sharp border to pass quality check, but interior is completely blank
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(5, 5), (595, 395)], outline=(0, 0, 0), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("blank_card.png", buf.getvalue(), "image/png")}
    resp = client.post("/ocr/pan", files=files, headers=auth_header)
    assert resp.status_code == 202
    job_data = resp.json()
    job_id = job_data["job_id"]

    # Poll until complete
    poll_result = None
    for _ in range(30):
        time.sleep(0.1)
        poll_resp = client.get(f"/ocr/jobs/{job_id}", headers=auth_header)
        assert poll_resp.status_code == 200
        pdata = poll_resp.json()
        if pdata["status"] not in ("pending", "processing"):
            poll_result = pdata
            break

    assert poll_result is not None, "Job did not complete within polling timeout"
    assert poll_result["status"] in ("error", "failed")
    res = poll_result.get("result") or {}
    assert res.get("status") == "error"
    assert res.get("reason") == "ocr_engine_returned_no_text"
    assert res.get("confidence") == 0.0
    assert res.get("extracted_fields") == {}


def test_sync_and_async_pipeline_parity_new_document_types(client, auth_header):
    """
    Verify that sync (/api/upload) and async (/ocr/{doc_type} -> /ocr/jobs/{job_id})
    paths produce identical extracted_fields and validation status for the new document types,
    specifically testing:
    1. bank_passbook (masked-field handling: raw account/holder stripped)
    2. partnership_deed (list-valued field: partner_names)
    """
    import time
    from PIL import ImageDraw

    # Helper to poll async job
    def poll_async_job(job_id: str) -> dict:
        for _ in range(50):
            time.sleep(0.1)
            poll_resp = client.get(f"/ocr/jobs/{job_id}", headers=auth_header)
            assert poll_resp.status_code == 200
            pdata = poll_resp.json()
            if pdata["status"] not in ("pending", "processing"):
                return pdata
        raise TimeoutError(f"Job {job_id} did not complete in time")

    # --- 1. Bank Passbook ---
    img_pb = Image.new("RGB", (800, 400), color=(255, 255, 255))
    draw_pb = ImageDraw.Draw(img_pb)
    draw_pb.text((20, 30), "STATE BANK OF INDIA - PASSBOOK", fill=(0, 0, 0))
    draw_pb.text((20, 70), "Account Holder: RAMESH SHARMA", fill=(0, 0, 0))
    draw_pb.text((20, 110), "Account No: 12345678901", fill=(0, 0, 0))
    draw_pb.text((20, 150), "IFSC: SBIN0001234", fill=(0, 0, 0))
    buf_pb = io.BytesIO()
    img_pb.save(buf_pb, format="PNG")
    pb_bytes = buf_pb.getvalue()

    # Sync /api/upload
    resp_sync_pb = client.post(
        "/api/upload",
        files={"file": ("passbook.png", pb_bytes, "image/png")},
        data={"doc_type": "bank_passbook"},
        headers=auth_header,
    )
    assert resp_sync_pb.status_code == 200
    sync_pb_data = resp_sync_pb.json()

    # Async /ocr/bank_passbook
    resp_async_pb = client.post(
        "/ocr/bank_passbook",
        files={"file": ("passbook.png", pb_bytes, "image/png")},
        headers=auth_header,
    )
    assert resp_async_pb.status_code == 202
    job_id_pb = resp_async_pb.json()["job_id"]
    async_pb_result = poll_async_job(job_id_pb)
    assert async_pb_result["status"] == "completed"

    async_pb_fields = async_pb_result["result"]["extracted_fields"]
    assert sync_pb_data["extracted_fields"] == async_pb_fields
    assert "ifsc" in async_pb_fields
    assert async_pb_fields["ifsc"] == "SBIN0001234"
    assert async_pb_fields["account_number_masked"] == "XXXXXXX8901"
    assert async_pb_fields["account_holder_name_masked"] == "RAMESH S*****"
    # Both paths strictly omit raw PII fields
    assert "raw_account_number" not in sync_pb_data["extracted_fields"]
    assert "raw_account_number" not in async_pb_fields
    assert "raw_account_holder_name" not in sync_pb_data["extracted_fields"]
    assert "raw_account_holder_name" not in async_pb_fields

    # --- 2. Partnership Deed ---
    img_pd = Image.new("RGB", (900, 500), color=(255, 255, 255))
    draw_pd = ImageDraw.Draw(img_pd)
    draw_pd.text((20, 30), "DEED OF PARTNERSHIP", fill=(0, 0, 0))
    draw_pd.text((20, 70), "Firm Name: M/S APEX VENTURES", fill=(0, 0, 0))
    draw_pd.text((20, 110), "Party of the First Part: VIKRAM MALHOTRA", fill=(0, 0, 0))
    draw_pd.text((20, 150), "Party of the Second Part: ROHAN DESHMUKH", fill=(0, 0, 0))
    draw_pd.text((20, 190), "Date of Deed: 15/05/2024", fill=(0, 0, 0))
    draw_pd.text((20, 230), "Profit Sharing Ratio: 50:50", fill=(0, 0, 0))
    buf_pd = io.BytesIO()
    img_pd.save(buf_pd, format="PNG")
    pd_bytes = buf_pd.getvalue()

    # Sync /api/upload
    resp_sync_pd = client.post(
        "/api/upload",
        files={"file": ("partnership_deed.png", pd_bytes, "image/png")},
        data={"doc_type": "partnership_deed"},
        headers=auth_header,
    )
    assert resp_sync_pd.status_code == 200
    sync_pd_data = resp_sync_pd.json()

    # Async /ocr/partnership_deed
    resp_async_pd = client.post(
        "/ocr/partnership_deed",
        files={"file": ("partnership_deed.png", pd_bytes, "image/png")},
        headers=auth_header,
    )
    assert resp_async_pd.status_code == 202
    job_id_pd = resp_async_pd.json()["job_id"]
    async_pd_result = poll_async_job(job_id_pd)
    assert async_pd_result["status"] == "completed"

    async_pd_fields = async_pd_result["result"]["extracted_fields"]
    assert sync_pd_data["extracted_fields"] == async_pd_fields
    assert "partner_names" not in async_pd_fields
    assert isinstance(async_pd_fields["partner_names_masked"], list)
    assert async_pd_fields["partner_names_masked"] == ["VIKRAM M*******", "ROHAN D*******"]
    assert async_pd_fields["profit_sharing_ratio"] == "50:50"



