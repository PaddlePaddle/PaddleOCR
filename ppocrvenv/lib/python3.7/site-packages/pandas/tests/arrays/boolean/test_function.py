import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "ufunc", [np.add, np.logical_or, np.logical_and, np.logical_xor]
)
def test_ufuncs_binary(ufunc):
    # two BooleanArrays
    a = pd.array([True, False, None], dtype="boolean")
    result = ufunc(a, a)
    expected = pd.array(ufunc(a._data, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    result = ufunc(s, a)
    expected = pd.Series(ufunc(a._data, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_series_equal(result, expected)

    # Boolean with numpy array
    arr = np.array([True, True, False])
    result = ufunc(a, arr)
    expected = pd.array(ufunc(a._data, arr), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(arr, a)
    expected = pd.array(ufunc(arr, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    # BooleanArray with scalar
    result = ufunc(a, True)
    expected = pd.array(ufunc(a._data, True), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(True, a)
    expected = pd.array(ufunc(True, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    # not handled types
    msg = r"operand type\(s\) all returned NotImplemented from __array_ufunc__"
    with pytest.raises(TypeError, match=msg):
        ufunc(a, "test")


@pytest.mark.parametrize("ufunc", [np.logical_not])
def test_ufuncs_unary(ufunc):
    a = pd.array([True, False, None], dtype="boolean")
    result = ufunc(a)
    expected = pd.array(ufunc(a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    result = ufunc(s)
    expected = pd.Series(ufunc(a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("values", [[True, False], [True, None]])
def test_ufunc_reduce_raises(values):
    a = pd.array(values, dtype="boolean")
    msg = "The 'reduce' method is not supported"
    with pytest.raises(NotImplementedError, match=msg):
        np.add.reduce(a)


def test_value_counts_na():
    arr = pd.array([True, False, pd.NA], dtype="boolean")
    result = arr.value_counts(dropna=False)
    expected = pd.Series([1, 1, 1], index=[True, False, pd.NA], dtype="Int64")
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([1, 1], index=[True, False], dtype="Int64")
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize():
    s = pd.Series([True, False, pd.NA], dtype="boolean")
    result = s.value_counts(normalize=True)
    expected = pd.Series([1, 1], index=[True, False], dtype="Float64") / 2
    tm.assert_series_equal(result, expected)


def test_diff():
    a = pd.array(
        [True, True, False, False, True, None, True, None, False], dtype="boolean"
    )
    result = pd.core.algorithms.diff(a, 1)
    expected = pd.array(
        [None, False, True, False, True, None, None, None, None], dtype="boolean"
    )
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    result = s.diff()
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)
