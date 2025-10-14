import pytest

from paddleocr import SealRecognition
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def ocr_engine() -> SealRecognition:
    return SealRecognition()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "seal.png",
    ],
)
def test_predict(ocr_engine: SealRecognition, image_path: str) -> None:
    """
    Test PaddleOCR's seal recognition functionality.

    Args:
        ocr_engine: An instance of `SealRecognition`.
        image_path: Path to the image to be processed.
    """
    result = ocr_engine.predict(str(image_path))

    check_simple_inference_result(result)
    res = result[0]["seal_res_list"][0]
    assert len(res["dt_polys"]) > 0
    assert isinstance(res["rec_texts"], list)
    assert len(res["rec_texts"]) > 0
    for text in res["rec_texts"]:
        assert isinstance(text, str)


@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False, "use_doc_unwarping": False},
        {"use_layout_detection": False},
        {"layout_det_res": None},
        {"layout_threshold": 0.5},
        {"layout_nms": False},
        {"layout_unclip_ratio": 1.0},
        {"layout_merge_bboxes_mode": "large"},
        {"seal_det_limit_side_len": 736},
        {"seal_det_limit_type": "min"},
        {"seal_det_thresh": 0.5},
        {"seal_det_box_thresh": 0.6},
        {"seal_det_unclip_ratio": 0.5},
        {"seal_rec_score_thresh": 0.05},
    ],
)
def test_predict_params(
    monkeypatch,
    ocr_engine: SealRecognition,
    params: dict,
) -> None:
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        ocr_engine,
        "paddlex_pipeline",
        "dummy_path",
        params,
    )


def test_paddlex_version_warning(monkeypatch) -> None:
    """
    Test that a warning is issued for PaddleX versions with the multi-page PDF bug.
    """
    import warnings

    # Mock paddlex to simulate version 3.2.0
    class MockPaddleX:
        __version__ = "3.2.0"

    monkeypatch.setitem(__import__("sys").modules, "paddlex", MockPaddleX())

    # Test that a warning is issued during initialization
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Import seal_recognition here to trigger the warning
        from paddleocr._pipelines.seal_recognition import SealRecognition

        # Create instance which should trigger the version check
        try:
            _ = SealRecognition()
        except Exception:
            # Initialization might fail due to missing PaddleX modules, but that's OK
            # We're just testing if the warning is issued
            pass

        # Check that a warning was issued
        assert len(w) >= 1
        assert any(
            "PaddleX version" in str(warning.message) and "multi-page PDFs" in str(warning.message)
            for warning in w
        ), f"Expected warning about PaddleX version bug, got: {[str(warning.message) for warning in w]}"

