"""
Additional tests for PandasArray that aren't covered by
the interface tests.
"""
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import PandasDtype

import pandas as pd
import pandas._testing as tm
from pandas.arrays import PandasArray


@pytest.fixture(
    params=[
        np.array(["a", "b"], dtype=object),
        np.array([0, 1], dtype=float),
        np.array([0, 1], dtype=int),
        np.array([0, 1 + 2j], dtype=complex),
        np.array([True, False], dtype=bool),
        np.array([0, 1], dtype="datetime64[ns]"),
        np.array([0, 1], dtype="timedelta64[ns]"),
    ]
)
def any_numpy_array(request):
    """
    Parametrized fixture for NumPy arrays with different dtypes.

    This excludes string and bytes.
    """
    return request.param


# ----------------------------------------------------------------------------
# PandasDtype


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("bool", True),
        ("int", True),
        ("uint", True),
        ("float", True),
        ("complex", True),
        ("str", False),
        ("bytes", False),
        ("datetime64[ns]", False),
        ("object", False),
        ("void", False),
    ],
)
def test_is_numeric(dtype, expected):
    dtype = PandasDtype(dtype)
    assert dtype._is_numeric is expected


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("bool", True),
        ("int", False),
        ("uint", False),
        ("float", False),
        ("complex", False),
        ("str", False),
        ("bytes", False),
        ("datetime64[ns]", False),
        ("object", False),
        ("void", False),
    ],
)
def test_is_boolean(dtype, expected):
    dtype = PandasDtype(dtype)
    assert dtype._is_boolean is expected


def test_repr():
    dtype = PandasDtype(np.dtype("int64"))
    assert repr(dtype) == "PandasDtype('int64')"


def test_constructor_from_string():
    result = PandasDtype.construct_from_string("int64")
    expected = PandasDtype(np.dtype("int64"))
    assert result == expected


def test_dtype_univalent(any_numpy_dtype):
    dtype = PandasDtype(any_numpy_dtype)

    result = PandasDtype(dtype)
    assert result == dtype


# ----------------------------------------------------------------------------
# Construction


def test_constructor_no_coercion():
    with pytest.raises(ValueError, match="NumPy array"):
        PandasArray([1, 2, 3])


def test_series_constructor_with_copy():
    ndarray = np.array([1, 2, 3])
    ser = pd.Series(PandasArray(ndarray), copy=True)

    assert ser.values is not ndarray


def test_series_constructor_with_astype():
    ndarray = np.array([1, 2, 3])
    result = pd.Series(PandasArray(ndarray), dtype="float64")
    expected = pd.Series([1.0, 2.0, 3.0], dtype="float64")
    tm.assert_series_equal(result, expected)


def test_from_sequence_dtype():
    arr = np.array([1, 2, 3], dtype="int64")
    result = PandasArray._from_sequence(arr, dtype="uint64")
    expected = PandasArray(np.array([1, 2, 3], dtype="uint64"))
    tm.assert_extension_array_equal(result, expected)


def test_constructor_copy():
    arr = np.array([0, 1])
    result = PandasArray(arr, copy=True)

    assert np.shares_memory(result._ndarray, arr) is False


def test_constructor_with_data(any_numpy_array):
    nparr = any_numpy_array
    arr = PandasArray(nparr)
    assert arr.dtype.numpy_dtype == nparr.dtype


# ----------------------------------------------------------------------------
# Conversion


def test_to_numpy():
    arr = PandasArray(np.array([1, 2, 3]))
    result = arr.to_numpy()
    assert result is arr._ndarray

    result = arr.to_numpy(copy=True)
    assert result is not arr._ndarray

    result = arr.to_numpy(dtype="f8")
    expected = np.array([1, 2, 3], dtype="f8")
    tm.assert_numpy_array_equal(result, expected)


# ----------------------------------------------------------------------------
# Setitem


def test_setitem_series():
    ser = pd.Series([1, 2, 3])
    ser.array[0] = 10
    expected = pd.Series([10, 2, 3])
    tm.assert_series_equal(ser, expected)


def test_setitem(any_numpy_array):
    nparr = any_numpy_array
    arr = PandasArray(nparr, copy=True)

    arr[0] = arr[1]
    nparr[0] = nparr[1]

    tm.assert_numpy_array_equal(arr.to_numpy(), nparr)


# ----------------------------------------------------------------------------
# Reductions


def test_bad_reduce_raises():
    arr = np.array([1, 2, 3], dtype="int64")
    arr = PandasArray(arr)
    msg = "cannot perform not_a_method with type int"
    with pytest.raises(TypeError, match=msg):
        arr._reduce(msg)


def test_validate_reduction_keyword_args():
    arr = PandasArray(np.array([1, 2, 3]))
    msg = "the 'keepdims' parameter is not supported .*all"
    with pytest.raises(ValueError, match=msg):
        arr.all(keepdims=True)


# ----------------------------------------------------------------------------
# Ops


def test_ufunc():
    arr = PandasArray(np.array([-1.0, 0.0, 1.0]))
    result = np.abs(arr)
    expected = PandasArray(np.abs(arr._ndarray))
    tm.assert_extension_array_equal(result, expected)

    r1, r2 = np.divmod(arr, np.add(arr, 2))
    e1, e2 = np.divmod(arr._ndarray, np.add(arr._ndarray, 2))
    e1 = PandasArray(e1)
    e2 = PandasArray(e2)
    tm.assert_extension_array_equal(r1, e1)
    tm.assert_extension_array_equal(r2, e2)


def test_basic_binop():
    # Just a basic smoke test. The EA interface tests exercise this
    # more thoroughly.
    x = PandasArray(np.array([1, 2, 3]))
    result = x + x
    expected = PandasArray(np.array([2, 4, 6]))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [None, object])
def test_setitem_object_typecode(dtype):
    arr = PandasArray(np.array(["a", "b", "c"], dtype=dtype))
    arr[0] = "t"
    expected = PandasArray(np.array(["t", "b", "c"], dtype=dtype))
    tm.assert_extension_array_equal(arr, expected)


def test_setitem_no_coercion():
    # https://github.com/pandas-dev/pandas/issues/28150
    arr = PandasArray(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="int"):
        arr[0] = "a"

    # With a value that we do coerce, check that we coerce the value
    #  and not the underlying array.
    arr[0] = 2.5
    assert isinstance(arr[0], (int, np.integer)), type(arr[0])


def test_setitem_preserves_views():
    # GH#28150, see also extension test of the same name
    arr = PandasArray(np.array([1, 2, 3]))
    view1 = arr.view()
    view2 = arr[:]
    view3 = np.asarray(arr)

    arr[0] = 9
    assert view1[0] == 9
    assert view2[0] == 9
    assert view3[0] == 9

    arr[-1] = 2.5
    view1[-1] = 5
    assert arr[-1] == 5
