import sys

import numpy as np
import pytest

from pandas.compat import (
    IS64,
    PYPY,
)

from pandas.core.dtypes.common import (
    is_categorical_dtype,
    is_object_dtype,
)

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
)


@pytest.mark.parametrize(
    "op_name, op",
    [
        ("add", "+"),
        ("sub", "-"),
        ("mul", "*"),
        ("mod", "%"),
        ("pow", "**"),
        ("truediv", "/"),
        ("floordiv", "//"),
    ],
)
@pytest.mark.parametrize("klass", [Series, DataFrame])
def test_binary_ops_docstring(klass, op_name, op):
    # not using the all_arithmetic_functions fixture with _get_opstr
    # as _get_opstr is used internally in the dynamic implementation of the docstring
    operand1 = klass.__name__.lower()
    operand2 = "other"
    expected_str = " ".join([operand1, op, operand2])
    assert expected_str in getattr(klass, op_name).__doc__

    # reverse version of the binary ops
    expected_str = " ".join([operand2, op, operand1])
    assert expected_str in getattr(klass, "r" + op_name).__doc__


def test_ndarray_compat_properties(index_or_series_obj):
    obj = index_or_series_obj

    # Check that we work.
    for p in ["shape", "dtype", "T", "nbytes"]:
        assert getattr(obj, p, None) is not None

    # deprecated properties
    for p in ["strides", "itemsize", "base", "data"]:
        assert not hasattr(obj, p)

    msg = "can only convert an array of size 1 to a Python scalar"
    with pytest.raises(ValueError, match=msg):
        obj.item()  # len > 1

    assert obj.ndim == 1
    assert obj.size == len(obj)

    assert Index([1]).item() == 1
    assert Series([1]).item() == 1


@pytest.mark.skipif(PYPY, reason="not relevant for PyPy")
def test_memory_usage(index_or_series_obj):
    obj = index_or_series_obj

    res = obj.memory_usage()
    res_deep = obj.memory_usage(deep=True)

    is_object = is_object_dtype(obj) or (
        isinstance(obj, Series) and is_object_dtype(obj.index)
    )
    is_categorical = is_categorical_dtype(obj.dtype) or (
        isinstance(obj, Series) and is_categorical_dtype(obj.index.dtype)
    )

    if len(obj) == 0:
        if isinstance(obj, Index):
            expected = 0
        else:
            expected = 108 if IS64 else 64
        assert res_deep == res == expected
    elif is_object or is_categorical:
        # only deep will pick them up
        assert res_deep > res
    else:
        assert res == res_deep

    # sys.getsizeof will call the .memory_usage with
    # deep=True, and add on some GC overhead
    diff = res_deep - sys.getsizeof(obj)
    assert abs(diff) < 100


def test_memory_usage_components_series(series_with_simple_index):
    series = series_with_simple_index
    total_usage = series.memory_usage(index=True)
    non_index_usage = series.memory_usage(index=False)
    index_usage = series.index.memory_usage()
    assert total_usage == non_index_usage + index_usage


def test_memory_usage_components_narrow_series(narrow_series):
    series = narrow_series
    total_usage = series.memory_usage(index=True)
    non_index_usage = series.memory_usage(index=False)
    index_usage = series.index.memory_usage()
    assert total_usage == non_index_usage + index_usage


def test_searchsorted(index_or_series_obj):
    # numpy.searchsorted calls obj.searchsorted under the hood.
    # See gh-12238
    obj = index_or_series_obj

    if isinstance(obj, pd.MultiIndex):
        # See gh-14833
        pytest.skip("np.searchsorted doesn't work on pd.MultiIndex")

    max_obj = max(obj, default=0)
    index = np.searchsorted(obj, max_obj)
    assert 0 <= index <= len(obj)

    index = np.searchsorted(obj, max_obj, sorter=range(len(obj)))
    assert 0 <= index <= len(obj)


def test_access_by_position(index):

    if len(index) == 0:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(index, pd.MultiIndex):
        pytest.skip("Can't instantiate Series from MultiIndex")

    series = Series(index)
    assert index[0] == series.iloc[0]
    assert index[5] == series.iloc[5]
    assert index[-1] == series.iloc[-1]

    size = len(index)
    assert index[-1] == index[size - 1]

    msg = f"index {size} is out of bounds for axis 0 with size {size}"
    with pytest.raises(IndexError, match=msg):
        index[size]
    msg = "single positional indexer is out-of-bounds"
    with pytest.raises(IndexError, match=msg):
        series.iloc[size]
