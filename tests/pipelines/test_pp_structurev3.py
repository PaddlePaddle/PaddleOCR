import pytest

from paddleocr import PPStructureV3
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def pp_structurev3_pipeline():
    return PPStructureV3()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "doc_with_formula.png",
    ],
)
def test_visual_predict(pp_structurev3_pipeline, image_path):
    result = pp_structurev3_pipeline.predict(str(image_path))

    check_simple_inference_result(result)
    res = result[0]
    overall_ocr_res = res["overall_ocr_res"]
    assert len(overall_ocr_res["dt_polys"]) > 0
    assert len(overall_ocr_res["rec_texts"]) > 0
    assert len(overall_ocr_res["rec_polys"]) > 0
    assert len(overall_ocr_res["rec_boxes"]) > 0


@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False},
        {"use_doc_unwarping": False},
        {"use_general_ocr": False},
        {"use_table_recognition": False},
        {"use_formula_recognition": False},
        {"layout_threshold": 0.88},
        {"layout_threshold": [0.45, 0.4]},
        {"layout_threshold": {0: 0.45, 2: 0.48, 7: 0.4}},
        {"layout_nms": False},
        {"layout_unclip_ratio": 1.1},
        {"layout_unclip_ratio": [1.2, 1.5]},
        {"layout_unclip_ratio": {0: 1.2, 2: 1.5, 7: 1.8}},
        {"layout_merge_bboxes_mode": "large"},
        {"layout_merge_bboxes_mode": {0: "large", 2: "small", 7: "union"}},
        {"text_det_limit_side_len": 640, "text_det_limit_type": "min"},
        {"text_det_thresh": 0.5},
        {"text_det_box_thresh": 0.3},
        {"text_det_unclip_ratio": 3.0},
        {"text_rec_score_thresh": 0.5},
    ],
)
def test_predict_params(
    monkeypatch,
    pp_structurev3_pipeline,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        pp_structurev3_pipeline,
        "paddlex_pipeline",
        "dummy_path",
        params,
    )


# TODO: Test constructor and other methods
