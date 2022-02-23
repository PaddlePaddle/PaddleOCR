import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.utils._encode import _unique
from sklearn.utils._encode import _encode
from sklearn.utils._encode import _check_unknown


@pytest.mark.parametrize(
    "values, expected",
    [
        (np.array([2, 1, 3, 1, 3], dtype="int64"), np.array([1, 2, 3], dtype="int64")),
        (
            np.array(["b", "a", "c", "a", "c"], dtype=object),
            np.array(["a", "b", "c"], dtype=object),
        ),
        (np.array(["b", "a", "c", "a", "c"]), np.array(["a", "b", "c"])),
    ],
    ids=["int64", "object", "str"],
)
def test_encode_util(values, expected):
    uniques = _unique(values)
    assert_array_equal(uniques, expected)
    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, np.array([1, 0, 2, 0, 2]))


def test_encode_with_check_unknown():
    # test for the check_unknown parameter of _encode()
    uniques = np.array([1, 2, 3])
    values = np.array([1, 2, 3, 4])

    # Default is True, raise error
    with pytest.raises(ValueError, match="y contains previously unseen labels"):
        _encode(values, uniques=uniques, check_unknown=True)

    # dont raise error if False
    _encode(values, uniques=uniques, check_unknown=False)

    # parameter is ignored for object dtype
    uniques = np.array(["a", "b", "c"], dtype=object)
    values = np.array(["a", "b", "c", "d"], dtype=object)
    with pytest.raises(ValueError, match="y contains previously unseen labels"):
        _encode(values, uniques=uniques, check_unknown=False)


def _assert_check_unknown(values, uniques, expected_diff, expected_mask):
    diff = _check_unknown(values, uniques)
    assert_array_equal(diff, expected_diff)

    diff, valid_mask = _check_unknown(values, uniques, return_mask=True)
    assert_array_equal(diff, expected_diff)
    assert_array_equal(valid_mask, expected_mask)


@pytest.mark.parametrize(
    "values, uniques, expected_diff, expected_mask",
    [
        (np.array([1, 2, 3, 4]), np.array([1, 2, 3]), [4], [True, True, True, False]),
        (np.array([2, 1, 4, 5]), np.array([2, 5, 1]), [4], [True, True, False, True]),
        (np.array([2, 1, np.nan]), np.array([2, 5, 1]), [np.nan], [True, True, False]),
        (
            np.array([2, 1, 4, np.nan]),
            np.array([2, 5, 1, np.nan]),
            [4],
            [True, True, False, True],
        ),
        (
            np.array([2, 1, 4, np.nan]),
            np.array([2, 5, 1]),
            [4, np.nan],
            [True, True, False, False],
        ),
        (
            np.array([2, 1, 4, 5]),
            np.array([2, 5, 1, np.nan]),
            [4],
            [True, True, False, True],
        ),
        (
            np.array(["a", "b", "c", "d"], dtype=object),
            np.array(["a", "b", "c"], dtype=object),
            np.array(["d"], dtype=object),
            [True, True, True, False],
        ),
        (
            np.array(["d", "c", "a", "b"], dtype=object),
            np.array(["a", "c", "b"], dtype=object),
            np.array(["d"], dtype=object),
            [False, True, True, True],
        ),
        (
            np.array(["a", "b", "c", "d"]),
            np.array(["a", "b", "c"]),
            np.array(["d"]),
            [True, True, True, False],
        ),
        (
            np.array(["d", "c", "a", "b"]),
            np.array(["a", "c", "b"]),
            np.array(["d"]),
            [False, True, True, True],
        ),
    ],
)
def test_check_unknown(values, uniques, expected_diff, expected_mask):
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)


@pytest.mark.parametrize("missing_value", [None, np.nan, float("nan")])
@pytest.mark.parametrize("pickle_uniques", [True, False])
def test_check_unknown_missing_values(missing_value, pickle_uniques):
    # check for check_unknown with missing values with object dtypes
    values = np.array(["d", "c", "a", "b", missing_value], dtype=object)
    uniques = np.array(["c", "a", "b", missing_value], dtype=object)
    if pickle_uniques:
        uniques = pickle.loads(pickle.dumps(uniques))

    expected_diff = ["d"]
    expected_mask = [False, True, True, True, True]
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)

    values = np.array(["d", "c", "a", "b", missing_value], dtype=object)
    uniques = np.array(["c", "a", "b"], dtype=object)
    if pickle_uniques:
        uniques = pickle.loads(pickle.dumps(uniques))

    expected_diff = ["d", missing_value]

    expected_mask = [False, True, True, True, False]
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)

    values = np.array(["a", missing_value], dtype=object)
    uniques = np.array(["a", "b", "z"], dtype=object)
    if pickle_uniques:
        uniques = pickle.loads(pickle.dumps(uniques))

    expected_diff = [missing_value]
    expected_mask = [True, False]
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)


@pytest.mark.parametrize("missing_value", [np.nan, None, float("nan")])
@pytest.mark.parametrize("pickle_uniques", [True, False])
def test_unique_util_missing_values_objects(missing_value, pickle_uniques):
    # check for _unique and _encode with missing values with object dtypes
    values = np.array(["a", "c", "c", missing_value, "b"], dtype=object)
    expected_uniques = np.array(["a", "b", "c", missing_value], dtype=object)

    uniques = _unique(values)

    if missing_value is None:
        assert_array_equal(uniques, expected_uniques)
    else:  # missing_value == np.nan
        assert_array_equal(uniques[:-1], expected_uniques[:-1])
        assert np.isnan(uniques[-1])

    if pickle_uniques:
        uniques = pickle.loads(pickle.dumps(uniques))

    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, np.array([0, 2, 2, 3, 1]))


def test_unique_util_missing_values_numeric():
    # Check missing values in numerical values
    values = np.array([3, 1, np.nan, 5, 3, np.nan], dtype=float)
    expected_uniques = np.array([1, 3, 5, np.nan], dtype=float)
    expected_inverse = np.array([1, 0, 3, 2, 1, 3])

    uniques = _unique(values)
    assert_array_equal(uniques, expected_uniques)

    uniques, inverse = _unique(values, return_inverse=True)
    assert_array_equal(uniques, expected_uniques)
    assert_array_equal(inverse, expected_inverse)

    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, expected_inverse)


def test_unique_util_with_all_missing_values():
    # test for all types of missing values for object dtype
    values = np.array([np.nan, "a", "c", "c", None, float("nan"), None], dtype=object)

    uniques = _unique(values)
    assert_array_equal(uniques[:-1], ["a", "c", None])
    # last value is nan
    assert np.isnan(uniques[-1])

    expected_inverse = [3, 0, 1, 1, 2, 3, 2]
    _, inverse = _unique(values, return_inverse=True)
    assert_array_equal(inverse, expected_inverse)


def test_check_unknown_with_both_missing_values():
    # test for both types of missing values for object dtype
    values = np.array([np.nan, "a", "c", "c", None, np.nan, None], dtype=object)

    diff = _check_unknown(values, known_values=np.array(["a", "c"], dtype=object))
    assert diff[0] is None
    assert np.isnan(diff[1])

    diff, valid_mask = _check_unknown(
        values, known_values=np.array(["a", "c"], dtype=object), return_mask=True
    )

    assert diff[0] is None
    assert np.isnan(diff[1])
    assert_array_equal(valid_mask, [False, True, True, True, False, False, False])
