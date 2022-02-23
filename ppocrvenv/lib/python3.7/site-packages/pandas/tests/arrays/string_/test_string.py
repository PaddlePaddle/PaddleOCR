"""
This module tests the functionality of StringArray and ArrowStringArray.
Tests for the str accessors are in pandas/tests/strings/test_string_array.py
"""
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_dtype_equal

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import ArrowStringArray


@pytest.fixture
def dtype(string_storage):
    return pd.StringDtype(storage=string_storage)


@pytest.fixture
def cls(dtype):
    return dtype.construct_array_type()


def test_repr(dtype):
    df = pd.DataFrame({"A": pd.array(["a", pd.NA, "b"], dtype=dtype)})
    expected = "      A\n0     a\n1  <NA>\n2     b"
    assert repr(df) == expected

    expected = "0       a\n1    <NA>\n2       b\nName: A, dtype: string"
    assert repr(df.A) == expected

    arr_name = "ArrowStringArray" if dtype.storage == "pyarrow" else "StringArray"
    expected = f"<{arr_name}>\n['a', <NA>, 'b']\nLength: 3, dtype: string"
    assert repr(df.A.array) == expected


def test_none_to_nan(cls):
    a = cls._from_sequence(["a", None, "b"])
    assert a[1] is not None
    assert a[1] is pd.NA


def test_setitem_validates(cls):
    arr = cls._from_sequence(["a", "b"])

    if cls is pd.arrays.StringArray:
        msg = "Cannot set non-string value '10' into a StringArray."
    else:
        msg = "Scalar must be NA or str"
    with pytest.raises(ValueError, match=msg):
        arr[0] = 10

    if cls is pd.arrays.StringArray:
        msg = "Must provide strings."
    else:
        msg = "Scalar must be NA or str"
    with pytest.raises(ValueError, match=msg):
        arr[:] = np.array([1, 2])


def test_setitem_with_scalar_string(dtype):
    # is_float_dtype considers some strings, like 'd', to be floats
    # which can cause issues.
    arr = pd.array(["a", "c"], dtype=dtype)
    arr[0] = "d"
    expected = pd.array(["d", "c"], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)


def test_astype_roundtrip(dtype, request):
    if dtype.storage == "pyarrow":
        reason = "ValueError: Could not convert object to NumPy datetime"
        mark = pytest.mark.xfail(reason=reason, raises=ValueError)
        request.node.add_marker(mark)
    else:
        mark = pytest.mark.xfail(
            reason="GH#36153 casting from StringArray to dt64 fails", raises=ValueError
        )
        request.node.add_marker(mark)

    ser = pd.Series(pd.date_range("2000", periods=12))
    ser[0] = None

    casted = ser.astype(dtype)
    assert is_dtype_equal(casted.dtype, dtype)

    result = casted.astype("datetime64[ns]")
    tm.assert_series_equal(result, ser)


def test_add(dtype, request):
    if dtype.storage == "pyarrow":
        reason = (
            "unsupported operand type(s) for +: 'ArrowStringArray' and "
            "'ArrowStringArray'"
        )
        mark = pytest.mark.xfail(raises=TypeError, reason=reason)
        request.node.add_marker(mark)

    a = pd.Series(["a", "b", "c", None, None], dtype=dtype)
    b = pd.Series(["x", "y", None, "z", None], dtype=dtype)

    result = a + b
    expected = pd.Series(["ax", "by", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = a.add(b)
    tm.assert_series_equal(result, expected)

    result = a.radd(b)
    expected = pd.Series(["xa", "yb", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = a.add(b, fill_value="-")
    expected = pd.Series(["ax", "by", "c-", "-z", None], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_add_2d(dtype, request):
    if dtype.storage == "pyarrow":
        reason = "Failed: DID NOT RAISE <class 'ValueError'>"
        mark = pytest.mark.xfail(raises=None, reason=reason)
        request.node.add_marker(mark)

    a = pd.array(["a", "b", "c"], dtype=dtype)
    b = np.array([["a", "b", "c"]], dtype=object)
    with pytest.raises(ValueError, match="3 != 1"):
        a + b

    s = pd.Series(a)
    with pytest.raises(ValueError, match="3 != 1"):
        s + b


def test_add_sequence(dtype, request):
    if dtype.storage == "pyarrow":
        reason = "unsupported operand type(s) for +: 'ArrowStringArray' and 'list'"
        mark = pytest.mark.xfail(raises=TypeError, reason=reason)
        request.node.add_marker(mark)

    a = pd.array(["a", "b", None, None], dtype=dtype)
    other = ["x", None, "y", None]

    result = a + other
    expected = pd.array(["ax", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = other + a
    expected = pd.array(["xa", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_mul(dtype, request):
    if dtype.storage == "pyarrow":
        reason = "unsupported operand type(s) for *: 'ArrowStringArray' and 'int'"
        mark = pytest.mark.xfail(raises=TypeError, reason=reason)
        request.node.add_marker(mark)

    a = pd.array(["a", "b", None], dtype=dtype)
    result = a * 2
    expected = pd.array(["aa", "bb", None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = 2 * a
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.xfail(reason="GH-28527")
def test_add_strings(dtype):
    arr = pd.array(["a", "b", "c", "d"], dtype=dtype)
    df = pd.DataFrame([["t", "u", "v", "w"]])
    assert arr.__add__(df) is NotImplemented

    result = arr + df
    expected = pd.DataFrame([["at", "bu", "cv", "dw"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)

    result = df + arr
    expected = pd.DataFrame([["ta", "ub", "vc", "wd"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(reason="GH-28527")
def test_add_frame(dtype):
    arr = pd.array(["a", "b", np.nan, np.nan], dtype=dtype)
    df = pd.DataFrame([["x", np.nan, "y", np.nan]])

    assert arr.__add__(df) is NotImplemented

    result = arr + df
    expected = pd.DataFrame([["ax", np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)

    result = df + arr
    expected = pd.DataFrame([["xa", np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)


def test_comparison_methods_scalar(all_compare_operators, dtype):
    op_name = all_compare_operators
    a = pd.array(["a", None, "c"], dtype=dtype)
    other = "a"
    result = getattr(a, op_name)(other)
    expected = np.array([getattr(item, op_name)(other) for item in a], dtype=object)
    expected = pd.array(expected, dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_scalar_pd_na(all_compare_operators, dtype):
    op_name = all_compare_operators
    a = pd.array(["a", None, "c"], dtype=dtype)
    result = getattr(a, op_name)(pd.NA)
    expected = pd.array([None, None, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_scalar_not_string(all_compare_operators, dtype, request):
    if all_compare_operators not in ["__eq__", "__ne__"]:
        reason = "comparison op not supported between instances of 'str' and 'int'"
        mark = pytest.mark.xfail(raises=TypeError, reason=reason)
        request.node.add_marker(mark)

    op_name = all_compare_operators
    a = pd.array(["a", None, "c"], dtype=dtype)
    other = 42
    result = getattr(a, op_name)(other)
    expected_data = {"__eq__": [False, None, False], "__ne__": [True, None, True]}[
        op_name
    ]
    expected = pd.array(expected_data, dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_array(all_compare_operators, dtype, request):
    if dtype.storage == "pyarrow":
        mark = pytest.mark.xfail(
            raises=AssertionError, reason="left is not an ExtensionArray"
        )
        request.node.add_marker(mark)

    op_name = all_compare_operators

    a = pd.array(["a", None, "c"], dtype=dtype)
    other = [None, None, "c"]
    result = getattr(a, op_name)(other)
    expected = np.empty_like(a, dtype="object")
    expected[-1] = getattr(other[-1], op_name)(a[-1])
    expected = pd.array(expected, dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    result = getattr(a, op_name)(pd.NA)
    expected = pd.array([None, None, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_constructor_raises(cls):
    if cls is pd.arrays.StringArray:
        msg = "StringArray requires a sequence of strings or pandas.NA"
    else:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowStringArray"

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", "b"], dtype="S1"))

    with pytest.raises(ValueError, match=msg):
        cls(np.array([]))

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", np.nan], dtype=object))

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", None], dtype=object))

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", pd.NaT], dtype=object))


@pytest.mark.parametrize("copy", [True, False])
def test_from_sequence_no_mutate(copy, cls, request):
    if cls is ArrowStringArray and copy is False:
        mark = pytest.mark.xfail(
            raises=AssertionError, reason="numpy array are different"
        )
        request.node.add_marker(mark)

    nan_arr = np.array(["a", np.nan], dtype=object)
    na_arr = np.array(["a", pd.NA], dtype=object)

    result = cls._from_sequence(nan_arr, copy=copy)

    if cls is ArrowStringArray:
        import pyarrow as pa

        expected = cls(pa.array(na_arr, type=pa.string(), from_pandas=True))
    else:
        expected = cls(na_arr)

    tm.assert_extension_array_equal(result, expected)

    expected = nan_arr if copy else na_arr
    tm.assert_numpy_array_equal(nan_arr, expected)


def test_astype_int(dtype):
    arr = pd.array(["1", "2", "3"], dtype=dtype)
    result = arr.astype("int64")
    expected = np.array([1, 2, 3], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)
    msg = r"int\(\) argument must be a string, a bytes-like object or a( real)? number"
    with pytest.raises(TypeError, match=msg):
        arr.astype("int64")


def test_astype_nullable_int(dtype):
    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)

    result = arr.astype("Int64")
    expected = pd.array([1, pd.NA, 3], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


def test_astype_float(dtype, any_float_allowed_nullable_dtype):
    # Don't compare arrays (37974)
    ser = pd.Series(["1.1", pd.NA, "3.3"], dtype=dtype)
    result = ser.astype(any_float_allowed_nullable_dtype)
    expected = pd.Series([1.1, np.nan, 3.3], dtype=any_float_allowed_nullable_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.xfail(reason="Not implemented StringArray.sum")
def test_reduce(skipna, dtype):
    arr = pd.Series(["a", "b", "c"], dtype=dtype)
    result = arr.sum(skipna=skipna)
    assert result == "abc"


@pytest.mark.parametrize("method", ["min", "max"])
@pytest.mark.parametrize("skipna", [True, False])
def test_min_max(method, skipna, dtype, request):
    if dtype.storage == "pyarrow":
        reason = "'ArrowStringArray' object has no attribute 'max'"
        mark = pytest.mark.xfail(raises=AttributeError, reason=reason)
        request.node.add_marker(mark)

    arr = pd.Series(["a", "b", "c", None], dtype=dtype)
    result = getattr(arr, method)(skipna=skipna)
    if skipna:
        expected = "a" if method == "min" else "c"
        assert result == expected
    else:
        assert result is pd.NA


@pytest.mark.parametrize("method", ["min", "max"])
@pytest.mark.parametrize("box", [pd.Series, pd.array])
def test_min_max_numpy(method, box, dtype, request):
    if dtype.storage == "pyarrow":
        if box is pd.array:
            raises = TypeError
            reason = "'<=' not supported between instances of 'str' and 'NoneType'"
        else:
            raises = AttributeError
            reason = "'ArrowStringArray' object has no attribute 'max'"
        mark = pytest.mark.xfail(raises=raises, reason=reason)
        request.node.add_marker(mark)

    arr = box(["a", "b", "c", None], dtype=dtype)
    result = getattr(np, method)(arr)
    expected = "a" if method == "min" else "c"
    assert result == expected


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.xfail(reason="Not implemented StringArray.sum")
def test_reduce_missing(skipna, dtype):
    arr = pd.Series([None, "a", None, "b", "c", None], dtype=dtype)
    result = arr.sum(skipna=skipna)
    if skipna:
        assert result == "abc"
    else:
        assert pd.isna(result)


def test_fillna_args(dtype, request):
    # GH 37987

    if dtype.storage == "pyarrow":
        reason = (
            "Regex pattern \"Cannot set non-string value '1' into "
            "a StringArray.\" does not match 'Scalar must be NA or str'"
        )
        mark = pytest.mark.xfail(raises=AssertionError, reason=reason)
        request.node.add_marker(mark)

    arr = pd.array(["a", pd.NA], dtype=dtype)

    res = arr.fillna(value="b")
    expected = pd.array(["a", "b"], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)

    res = arr.fillna(value=np.str_("b"))
    expected = pd.array(["a", "b"], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)

    msg = "Cannot set non-string value '1' into a StringArray."
    with pytest.raises(ValueError, match=msg):
        arr.fillna(value=1)


@td.skip_if_no("pyarrow")
def test_arrow_array(dtype):
    # protocol added in 0.15.0
    import pyarrow as pa

    data = pd.array(["a", "b", "c"], dtype=dtype)
    arr = pa.array(data)
    expected = pa.array(list(data), type=pa.string(), from_pandas=True)
    if dtype.storage == "pyarrow":
        expected = pa.chunked_array(expected)

    assert arr.equals(expected)


@td.skip_if_no("pyarrow")
def test_arrow_roundtrip(dtype, string_storage2):
    # roundtrip possible from arrow 1.0.0
    import pyarrow as pa

    data = pd.array(["a", "b", None], dtype=dtype)
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    assert table.field("a").type == "string"
    with pd.option_context("string_storage", string_storage2):
        result = table.to_pandas()
    assert isinstance(result["a"].dtype, pd.StringDtype)
    expected = df.astype(f"string[{string_storage2}]")
    tm.assert_frame_equal(result, expected)
    # ensure the missing value is represented by NA and not np.nan or None
    assert result.loc[2, "a"] is pd.NA


@td.skip_if_no("pyarrow")
def test_arrow_load_from_zero_chunks(dtype, string_storage2):
    # GH-41040
    import pyarrow as pa

    data = pd.array([], dtype=dtype)
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    assert table.field("a").type == "string"
    # Instantiate the same table with no chunks at all
    table = pa.table([pa.chunked_array([], type=pa.string())], schema=table.schema)
    with pd.option_context("string_storage", string_storage2):
        result = table.to_pandas()
    assert isinstance(result["a"].dtype, pd.StringDtype)
    expected = df.astype(f"string[{string_storage2}]")
    tm.assert_frame_equal(result, expected)


def test_value_counts_na(dtype):
    arr = pd.array(["a", "b", "a", pd.NA], dtype=dtype)
    result = arr.value_counts(dropna=False)
    expected = pd.Series([2, 1, 1], index=["a", "b", pd.NA], dtype="Int64")
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([2, 1], index=["a", "b"], dtype="Int64")
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize(dtype):
    s = pd.Series(["a", "b", "a", pd.NA], dtype=dtype)
    result = s.value_counts(normalize=True)
    expected = pd.Series([2, 1], index=["a", "b"], dtype="Float64") / 3
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "values, expected",
    [
        (["a", "b", "c"], np.array([False, False, False])),
        (["a", "b", None], np.array([False, False, True])),
    ],
)
def test_use_inf_as_na(values, expected, dtype):
    # https://github.com/pandas-dev/pandas/issues/33655
    values = pd.array(values, dtype=dtype)
    with pd.option_context("mode.use_inf_as_na", True):
        result = values.isna()
        tm.assert_numpy_array_equal(result, expected)

        result = pd.Series(values).isna()
        expected = pd.Series(expected)
        tm.assert_series_equal(result, expected)

        result = pd.DataFrame(values).isna()
        expected = pd.DataFrame(expected)
        tm.assert_frame_equal(result, expected)


def test_memory_usage(dtype):
    # GH 33963

    if dtype.storage == "pyarrow":
        pytest.skip("not applicable")

    series = pd.Series(["a", "b", "c"], dtype=dtype)

    assert 0 < series.nbytes <= series.memory_usage() < series.memory_usage(deep=True)


@pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
def test_astype_from_float_dtype(float_dtype, dtype):
    # https://github.com/pandas-dev/pandas/issues/36451
    s = pd.Series([0.1], dtype=float_dtype)
    result = s.astype(dtype)
    expected = pd.Series(["0.1"], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_to_numpy_returns_pdna_default(dtype):
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    result = np.array(arr)
    expected = np.array(["a", pd.NA, "b"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_na_value(dtype, nulls_fixture):
    na_value = nulls_fixture
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    result = arr.to_numpy(na_value=na_value)
    expected = np.array(["a", na_value, "b"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_isin(dtype, request):
    s = pd.Series(["a", "b", None], dtype=dtype)

    result = s.isin(["a", "c"])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", pd.NA])
    expected = pd.Series([True, False, True])
    tm.assert_series_equal(result, expected)

    result = s.isin([])
    expected = pd.Series([False, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", pd.Timestamp.now()])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)
