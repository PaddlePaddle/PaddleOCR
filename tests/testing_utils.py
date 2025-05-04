from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "test_files"


def check_simple_inference_result(result, *, expected_length=1):
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == expected_length
    for res in result:
        assert isinstance(res, dict)


def check_wrapper_simple_inference_param_forwarding(
    monkeypatch,
    wrapper,
    wrapped_obj_attr_name,
    input,
    params,
):
    def _dummy_predict(input, **params):
        yield params

    monkeypatch.setattr(
        getattr(wrapper, wrapped_obj_attr_name), "predict", _dummy_predict
    )

    result = getattr(wrapper, "predict")(
        input,
        **params,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    for k, v in params.items():
        assert result[0][k] == v
