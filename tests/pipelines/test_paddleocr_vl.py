import pytest

from paddleocr import PaddleOCRVL
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def paddleocr_vl_pipeline():
    return PaddleOCRVL()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "doc_with_formula.png",
    ],
)
def test_predict(paddleocr_vl_pipeline, image_path):
    result = paddleocr_vl_pipeline.predict(str(image_path))

    check_simple_inference_result(result)
    res = result[0]
    layout_det_res = res["layout_det_res"]
    assert len(layout_det_res["boxes"]) > 0


@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False},
        {"use_doc_unwarping": False},
        {"use_layout_detection": False},
        {"use_chart_recognition": False},
        {"layout_threshold": 0.88},
        {"layout_threshold": [0.45, 0.4]},
        {"layout_threshold": {0: 0.45, 2: 0.48, 7: 0.4}},
        {"layout_nms": False},
        {"layout_unclip_ratio": 1.1},
        {"layout_unclip_ratio": [1.2, 1.5]},
        {"layout_unclip_ratio": {0: 1.2, 2: 1.5, 7: 1.8}},
        {"layout_merge_bboxes_mode": "large"},
        {"layout_merge_bboxes_mode": {0: "large", 2: "small", 7: "union"}},
        {"use_queues": False},
        {"use_layout_detection": False, "prompt_label": "table"},
        {"format_block_content": False},
    ],
)
def test_predict_params(
    monkeypatch,
    paddleocr_vl_pipeline,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        paddleocr_vl_pipeline,
        "paddlex_pipeline",
        "dummy_path",
        params,
    )


# TODO: Test constructor and other methods
