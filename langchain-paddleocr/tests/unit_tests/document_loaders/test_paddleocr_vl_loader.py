from __future__ import annotations

from typing import Any

import pytest

from langchain_paddleocr import PaddleOCRVLLoader


def test_snake_to_camel_conversion_and_additional_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure that snake_case parameters and additional_params are converted to
    camelCase."""

    captured_service_params: dict[str, Any] = {}

    # Patch the loader to expose internal _service_params for inspection.
    original_init = PaddleOCRVLLoader.__init__

    def _wrapped_init(self: PaddleOCRVLLoader, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        nonlocal captured_service_params
        captured_service_params = getattr(self, "_service_params")

    monkeypatch.setattr(PaddleOCRVLLoader, "__init__", _wrapped_init)

    _ = PaddleOCRVLLoader(
        file_path="dummy.pdf",
        api_url="http://example.com",
        use_doc_orientation_classify=True,
        layout_unclip_ratio=(0.1, 0.9),
        prompt_label="ocr",
        additional_params={"customOption": 1, "anotherFlag": True},
    )

    # Keys from constructor
    assert captured_service_params["useDocOrientationClassify"] is True
    assert captured_service_params["layoutUnclipRatio"] == (0.1, 0.9)
    assert captured_service_params["promptLabel"] == "ocr"

    # Keys from additional_params
    assert captured_service_params["customOption"] == 1
    assert captured_service_params["anotherFlag"] is True


def test_file_type_normalization_and_inference(
    tmp_path_factory: pytest.TempdirFactory,
) -> None:
    """Ensure file_type normalization and type inference from file extension
    work as expected."""
    pdf_file = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file.write_bytes(b"%PDF-1.4")

    image_file = tmp_path_factory.mktemp("img") / "sample.png"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    loader_pdf_hint = PaddleOCRVLLoader(
        file_path=str(pdf_file),
        api_url="http://example.com",
        file_type="pdf",
    )
    assert loader_pdf_hint.file_type == 0

    loader_image_hint = PaddleOCRVLLoader(
        file_path=str(image_file),
        api_url="http://example.com",
        file_type="image",
    )
    assert loader_image_hint.file_type == 1


def test_lazy_load_raises_for_unreadable_file() -> None:
    """Ensure lazy_load raises when a file cannot be read."""
    loader = PaddleOCRVLLoader(
        file_path="nonexistent-file.pdf",
        api_url="http://example.com",
    )
    with pytest.raises(ValueError):
        list(loader.lazy_load())
