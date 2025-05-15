import pytest

from paddleocr import PPChatOCRv4Doc
from ..testing_utils import TEST_DATA_DIR


@pytest.fixture(scope="module")
def pp_chatocrv4_doc_pipeline():
    return PPChatOCRv4Doc()


@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "doc_with_formula.png",
    ],
)
def test_visual_predict(pp_chatocrv4_doc_pipeline, image_path):
    result = pp_chatocrv4_doc_pipeline.visual_predict(str(image_path))

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    assert isinstance(res, dict)
    assert res.keys() == {"visual_info", "layout_parsing_result"}
    assert isinstance(res["visual_info"], dict)
    assert isinstance(res["layout_parsing_result"], dict)


@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False},
        {"use_doc_unwarping": False},
        {"use_table_recognition": False},
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
    pp_chatocrv4_doc_pipeline,
    params,
):
    def _dummy_visual_predict(input, **params):
        yield {"visual_info": {}, "layout_parsing_result": params}

    monkeypatch.setattr(
        pp_chatocrv4_doc_pipeline.paddlex_pipeline,
        "visual_predict",
        _dummy_visual_predict,
    )

    result = pp_chatocrv4_doc_pipeline.visual_predict(
        input,
        **params,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    res = res["layout_parsing_result"]
    for k, v in params.items():
        assert res[k] == v


# TODO: Test constructor and other methods
