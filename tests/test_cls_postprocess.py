import os
import sys
import pytest
import paddle
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from ppocr.postprocess.cls_postprocess import ClsPostProcess


# Fixtures for common test inputs
@pytest.fixture
def preds_tensor():
    return paddle.to_tensor(np.array([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]]))


@pytest.fixture
def label_list():
    return {0: "class0", 1: "class1", 2: "class2"}


# Parameterize tests to cover multiple scenarios
@pytest.mark.parametrize(
    "label_list, expected",
    [
        ({0: "class0", 1: "class1", 2: "class2"}, [("class1", 0.7), ("class2", 0.4)]),
        (None, [(1, 0.7), (2, 0.4)]),
    ],
)
def test_cls_post_process_with_and_without_label_list(
    preds_tensor, label_list, expected
):
    post_process = ClsPostProcess(label_list=label_list)
    result = post_process(preds_tensor)
    assert isinstance(result, list), "Result should be a list"
    assert result == expected, f"Expected {expected}, got {result}"


# Test with a key in the prediction dictionary
def test_cls_post_process_with_key(preds_tensor, label_list):
    preds_dict = {"key": preds_tensor}
    post_process = ClsPostProcess(label_list=label_list, key="key")
    result = post_process(preds_dict)
    expected = [("class1", 0.7), ("class2", 0.4)]
    assert isinstance(result, list), "Result should be a list"
    assert result == expected, f"Expected {expected}, got {result}"


# Test with label input
def test_cls_post_process_with_label(preds_tensor, label_list):
    labels = [2, 0]
    post_process = ClsPostProcess(label_list=label_list)
    result, label_result = post_process(preds_tensor, labels)
    expected_result = [("class1", 0.7), ("class2", 0.4)]
    expected_label_result = [("class2", 1.0), ("class0", 1.0)]
    assert isinstance(result, list), "Result should be a list"
    assert result == expected_result, f"Expected {expected_result}, got {result}"
    assert isinstance(label_result, list), "Label result should be a list"
    assert (
        label_result == expected_label_result
    ), f"Expected {expected_label_result}, got {label_result}"
