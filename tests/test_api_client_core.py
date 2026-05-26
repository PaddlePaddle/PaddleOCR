import pytest

from paddleocr._api_client._core import (
    default_payload,
    job_id_for_task,
    resolve_document_model,
    resolve_document_options,
    resolve_ocr_model,
    validate_input_source,
)
from paddleocr._api_client.errors import InvalidRequestError
from paddleocr._api_client.models import (
    Model,
    OCROptions,
    PaddleOCRVLOptions,
    PPStructureV3Options,
)
from paddleocr._api_client.results import Job


def test_core_resolves_models_and_default_payloads():
    assert resolve_ocr_model("PP-OCRv5") is Model.PP_OCRV5
    assert resolve_document_model("PaddleOCR-VL-1.6") is Model.PADDLE_OCR_VL_16

    assert default_payload(Model.PP_OCRV5) == OCROptions().to_payload()
    assert default_payload(Model.PADDLE_OCR_VL_16) == PaddleOCRVLOptions().to_payload()

    with pytest.raises(InvalidRequestError):
        resolve_ocr_model("PP-StructureV3")

    with pytest.raises(InvalidRequestError):
        resolve_document_model("unknown-model")


def test_core_selects_document_options_by_model():
    assert isinstance(
        resolve_document_options(Model.PP_STRUCTURE_V3, None),
        PPStructureV3Options,
    )
    assert isinstance(
        resolve_document_options(Model.PADDLE_OCR_VL_16, None),
        PaddleOCRVLOptions,
    )

    with pytest.raises(InvalidRequestError):
        resolve_document_options(Model.PP_STRUCTURE_V3, PaddleOCRVLOptions())

    with pytest.raises(InvalidRequestError):
        resolve_document_options(Model.PADDLE_OCR_VL_16, PPStructureV3Options())


def test_core_validates_input_source_and_job_task():
    validate_input_source("https://example.test/file.pdf", None)
    validate_input_source(None, "/tmp/file.pdf")

    with pytest.raises(InvalidRequestError):
        validate_input_source(None, None)

    with pytest.raises(InvalidRequestError):
        validate_input_source("https://example.test/file.pdf", "/tmp/file.pdf")

    job = Job(job_id="job-1", model=Model.PP_OCRV5.value, task="ocr")
    assert job_id_for_task(job, "ocr") == "job-1"
    assert job_id_for_task("job-2", "ocr") == "job-2"

    with pytest.raises(InvalidRequestError):
        job_id_for_task(job, "document_parsing")
