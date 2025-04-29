import pytest

from paddleocr import TableRecognitionPipelineV2
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def table_recognition_v2_pipeline():
    return TableRecognitionPipelineV2()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "table.jpg",
    ],
)
def test_visual_predict(table_recognition_v2_pipeline, image_path):
    result = table_recognition_v2_pipeline.predict(
        str(image_path), use_doc_orientation_classify=False, use_doc_unwarping=False
    )

    check_simple_inference_result(result)
    res = result[0]
    assert len(res["table_res_list"]) > 0
    assert isinstance(res["table_res_list"][0], dict)
    assert len(res["table_res_list"][0]["cell_box_list"]) > 0
    assert isinstance(res["table_res_list"][0]["pred_html"], str)
    assert isinstance(res["table_res_list"][0]["table_ocr_pred"], dict)


@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False},
        {"use_doc_unwarping": False},
        {"use_layout_detection": False},
        {"use_ocr_model": False},
        {"text_det_limit_side_len": 640, "text_det_limit_type": "min"},
        {"text_det_thresh": 0.5},
        {"text_det_box_thresh": 0.3},
        {"text_det_unclip_ratio": 3.0},
        {"text_rec_score_thresh": 0.5},
    ],
)
def test_predict_params(
    monkeypatch,
    table_recognition_v2_pipeline,
    params,
):
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        table_recognition_v2_pipeline,
        "paddlex_pipeline",
        "dummy_path",
        params,
    )


# TODO: Test constructor and other methods
