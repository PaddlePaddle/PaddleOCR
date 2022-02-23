import numpy as np
import pytest

from pandas.core.dtypes.common import (
    is_datetime64_dtype,
    is_timedelta64_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import (
    CategoricalIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    IntervalArray,
    PandasArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)


class TestToIterable:
    # test that we convert an iterable to python types

    dtypes = [
        ("int8", int),
        ("int16", int),
        ("int32", int),
        ("int64", int),
        ("uint8", int),
        ("uint16", int),
        ("uint32", int),
        ("uint64", int),
        ("float16", float),
        ("float32", float),
        ("float64", float),
        ("datetime64[ns]", Timestamp),
        ("datetime64[ns, US/Eastern]", Timestamp),
        ("timedelta64[ns]", Timedelta),
    ]

    @pytest.mark.parametrize("dtype, rdtype", dtypes)
    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    def test_iterable(self, index_or_series, method, dtype, rdtype):
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        typ = index_or_series
        s = typ([1], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        "dtype, rdtype, obj",
        [
            ("object", object, "a"),
            ("object", int, 1),
            ("category", object, "a"),
            ("category", int, 1),
        ],
    )
    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    def test_iterable_object_and_category(
        self, index_or_series, method, dtype, rdtype, obj
    ):
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        typ = index_or_series
        s = typ([obj], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize("dtype, rdtype", dtypes)
    def test_iterable_items(self, dtype, rdtype):
        # gh-13258
        # test if items yields the correct boxed scalars
        # this only applies to series
        s = Series([1], dtype=dtype)
        _, result = list(s.items())[0]
        assert isinstance(result, rdtype)

        _, result = list(s.items())[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        "dtype, rdtype", dtypes + [("object", int), ("category", int)]
    )
    def test_iterable_map(self, index_or_series, dtype, rdtype):
        # gh-13236
        # coerce iteration to underlying python / pandas types
        typ = index_or_series
        s = typ([1], dtype=dtype)
        result = s.map(type)[0]
        if not isinstance(rdtype, tuple):
            rdtype = (rdtype,)
        assert result in rdtype

    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    def test_categorial_datetimelike(self, method):
        i = CategoricalIndex([Timestamp("1999-12-31"), Timestamp("2000-12-31")])

        result = method(i)[0]
        assert isinstance(result, Timestamp)

    def test_iter_box(self):
        vals = [Timestamp("2011-01-01"), Timestamp("2011-01-02")]
        s = Series(vals)
        assert s.dtype == "datetime64[ns]"
        for res, exp in zip(s, vals):
            assert isinstance(res, Timestamp)
            assert res.tz is None
            assert res == exp

        vals = [
            Timestamp("2011-01-01", tz="US/Eastern"),
            Timestamp("2011-01-02", tz="US/Eastern"),
        ]
        s = Series(vals)

        assert s.dtype == "datetime64[ns, US/Eastern]"
        for res, exp in zip(s, vals):
            assert isinstance(res, Timestamp)
            assert res.tz == exp.tz
            assert res == exp

        # timedelta
        vals = [Timedelta("1 days"), Timedelta("2 days")]
        s = Series(vals)
        assert s.dtype == "timedelta64[ns]"
        for res, exp in zip(s, vals):
            assert isinstance(res, Timedelta)
            assert res == exp

        # period
        vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
        s = Series(vals)
        assert s.dtype == "Period[M]"
        for res, exp in zip(s, vals):
            assert isinstance(res, pd.Period)
            assert res.freq == "M"
            assert res == exp


@pytest.mark.parametrize(
    "arr, expected_type, dtype",
    [
        (np.array([0, 1], dtype=np.int64), np.ndarray, "int64"),
        (np.array(["a", "b"]), np.ndarray, "object"),
        (pd.Categorical(["a", "b"]), pd.Categorical, "category"),
        (
            pd.DatetimeIndex(["2017", "2018"], tz="US/Central"),
            DatetimeArray,
            "datetime64[ns, US/Central]",
        ),
        (
            pd.PeriodIndex([2018, 2019], freq="A"),
            PeriodArray,
            pd.core.dtypes.dtypes.PeriodDtype("A-DEC"),
        ),
        (pd.IntervalIndex.from_breaks([0, 1, 2]), IntervalArray, "interval"),
        # This test is currently failing for datetime64[ns] and timedelta64[ns].
        # The NumPy type system is sufficient for representing these types, so
        # we just use NumPy for Series / DataFrame columns of these types (so
        # we get consolidation and so on).
        # However, DatetimeIndex and TimedeltaIndex use the DateLikeArray
        # abstraction to for code reuse.
        # At the moment, we've judged that allowing this test to fail is more
        # practical that overriding Series._values to special case
        # Series[M8[ns]] and Series[m8[ns]] to return a DateLikeArray.
        pytest.param(
            pd.DatetimeIndex(["2017", "2018"]),
            np.ndarray,
            "datetime64[ns]",
            marks=[pytest.mark.xfail(reason="datetime _values")],
        ),
        pytest.param(
            pd.TimedeltaIndex([10 ** 10]),
            np.ndarray,
            "m8[ns]",
            marks=[pytest.mark.xfail(reason="timedelta _values")],
        ),
    ],
)
def test_values_consistent(arr, expected_type, dtype):
    l_values = Series(arr)._values
    r_values = pd.Index(arr)._values
    assert type(l_values) is expected_type
    assert type(l_values) is type(r_values)

    tm.assert_equal(l_values, r_values)


@pytest.mark.parametrize("arr", [np.array([1, 2, 3])])
def test_numpy_array(arr):
    ser = Series(arr)
    result = ser.array
    expected = PandasArray(arr)
    tm.assert_extension_array_equal(result, expected)


def test_numpy_array_all_dtypes(any_numpy_dtype):
    ser = Series(dtype=any_numpy_dtype)
    result = ser.array
    if is_datetime64_dtype(any_numpy_dtype):
        assert isinstance(result, DatetimeArray)
    elif is_timedelta64_dtype(any_numpy_dtype):
        assert isinstance(result, TimedeltaArray)
    else:
        assert isinstance(result, PandasArray)


@pytest.mark.parametrize(
    "arr, attr",
    [
        (pd.Categorical(["a", "b"]), "_codes"),
        (pd.core.arrays.period_array(["2000", "2001"], freq="D"), "_data"),
        (pd.array([0, np.nan], dtype="Int64"), "_data"),
        (IntervalArray.from_breaks([0, 1]), "_left"),
        (SparseArray([0, 1]), "_sparse_values"),
        (DatetimeArray(np.array([1, 2], dtype="datetime64[ns]")), "_data"),
        # tz-aware Datetime
        (
            DatetimeArray(
                np.array(
                    ["2000-01-01T12:00:00", "2000-01-02T12:00:00"], dtype="M8[ns]"
                ),
                dtype=DatetimeTZDtype(tz="US/Central"),
            ),
            "_data",
        ),
    ],
)
def test_array(arr, attr, index_or_series):
    box = index_or_series
    if arr.dtype.name in ("Int64", "Sparse[int64, 0]") and box is pd.Index:
        pytest.skip(f"No index type for {arr.dtype}")
    result = box(arr, copy=False).array

    if attr:
        arr = getattr(arr, attr)
        result = getattr(result, attr)

    assert result is arr


def test_array_multiindex_raises():
    idx = pd.MultiIndex.from_product([["A"], ["a", "b"]])
    msg = "MultiIndex has no single backing array"
    with pytest.raises(ValueError, match=msg):
        idx.array


@pytest.mark.parametrize(
    "arr, expected",
    [
        (np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64)),
        (pd.Categorical(["a", "b"]), np.array(["a", "b"], dtype=object)),
        (
            pd.core.arrays.period_array(["2000", "2001"], freq="D"),
            np.array([pd.Period("2000", freq="D"), pd.Period("2001", freq="D")]),
        ),
        (pd.array([0, np.nan], dtype="Int64"), np.array([0, pd.NA], dtype=object)),
        (
            IntervalArray.from_breaks([0, 1, 2]),
            np.array([pd.Interval(0, 1), pd.Interval(1, 2)], dtype=object),
        ),
        (SparseArray([0, 1]), np.array([0, 1], dtype=np.int64)),
        # tz-naive datetime
        (
            DatetimeArray(np.array(["2000", "2001"], dtype="M8[ns]")),
            np.array(["2000", "2001"], dtype="M8[ns]"),
        ),
        # tz-aware stays tz`-aware
        (
            DatetimeArray(
                np.array(
                    ["2000-01-01T06:00:00", "2000-01-02T06:00:00"], dtype="M8[ns]"
                ),
                dtype=DatetimeTZDtype(tz="US/Central"),
            ),
            np.array(
                [
                    Timestamp("2000-01-01", tz="US/Central"),
                    Timestamp("2000-01-02", tz="US/Central"),
                ]
            ),
        ),
        # Timedelta
        (
            TimedeltaArray(np.array([0, 3600000000000], dtype="i8"), freq="H"),
            np.array([0, 3600000000000], dtype="m8[ns]"),
        ),
        # GH#26406 tz is preserved in Categorical[dt64tz]
        (
            pd.Categorical(date_range("2016-01-01", periods=2, tz="US/Pacific")),
            np.array(
                [
                    Timestamp("2016-01-01", tz="US/Pacific"),
                    Timestamp("2016-01-02", tz="US/Pacific"),
                ]
            ),
        ),
    ],
)
def test_to_numpy(arr, expected, index_or_series_or_array, request):
    box = index_or_series_or_array
    thing = box(arr)

    if arr.dtype.name in ("Int64", "Sparse[int64, 0]") and box is pd.Index:
        pytest.skip(f"No index type for {arr.dtype}")

    if arr.dtype.name == "int64" and box is pd.array:
        mark = pytest.mark.xfail(reason="thing is Int64 and to_numpy() returns object")
        request.node.add_marker(mark)

    result = thing.to_numpy()
    tm.assert_numpy_array_equal(result, expected)

    result = np.asarray(thing)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("as_series", [True, False])
@pytest.mark.parametrize(
    "arr", [np.array([1, 2, 3], dtype="int64"), np.array(["a", "b", "c"], dtype=object)]
)
def test_to_numpy_copy(arr, as_series):
    obj = pd.Index(arr, copy=False)
    if as_series:
        obj = Series(obj.values, copy=False)

    # no copy by default
    result = obj.to_numpy()
    assert np.shares_memory(arr, result) is True

    result = obj.to_numpy(copy=False)
    assert np.shares_memory(arr, result) is True

    # copy=True
    result = obj.to_numpy(copy=True)
    assert np.shares_memory(arr, result) is False


@pytest.mark.parametrize("as_series", [True, False])
def test_to_numpy_dtype(as_series):
    tz = "US/Eastern"
    obj = pd.DatetimeIndex(["2000", "2001"], tz=tz)
    if as_series:
        obj = Series(obj)

    # preserve tz by default
    result = obj.to_numpy()
    expected = np.array(
        [Timestamp("2000", tz=tz), Timestamp("2001", tz=tz)], dtype=object
    )
    tm.assert_numpy_array_equal(result, expected)

    result = obj.to_numpy(dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    result = obj.to_numpy(dtype="M8[ns]")
    expected = np.array(["2000-01-01T05", "2001-01-01T05"], dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "values, dtype, na_value, expected",
    [
        ([1, 2, None], "float64", 0, [1.0, 2.0, 0.0]),
        (
            [Timestamp("2000"), Timestamp("2000"), pd.NaT],
            None,
            Timestamp("2000"),
            [np.datetime64("2000-01-01T00:00:00.000000000")] * 3,
        ),
    ],
)
def test_to_numpy_na_value_numpy_dtype(
    index_or_series, values, dtype, na_value, expected
):
    obj = index_or_series(values)
    result = obj.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array(expected)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_kwargs_raises():
    # numpy
    s = Series([1, 2, 3])
    msg = r"to_numpy\(\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg):
        s.to_numpy(foo=True)

    # extension
    s = Series([1, 2, 3], dtype="Int64")
    with pytest.raises(TypeError, match=msg):
        s.to_numpy(foo=True)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [1, 2, None]},
        {"a": np.array([1, 2, 3]), "b": np.array([1, 2, np.nan])},
        {"a": pd.array([1, 2, 3]), "b": pd.array([1, 2, None])},
    ],
)
@pytest.mark.parametrize("dtype, na_value", [(float, np.nan), (object, None)])
def test_to_numpy_dataframe_na_value(data, dtype, na_value):
    # https://github.com/pandas-dev/pandas/issues/33820
    df = pd.DataFrame(data)
    result = df.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array([[1, 1], [2, 2], [3, na_value]], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            {"a": pd.array([1, 2, None])},
            np.array([[1.0], [2.0], [np.nan]], dtype=float),
        ),
        (
            {"a": [1, 2, 3], "b": [1, 2, 3]},
            np.array([[1, 1], [2, 2], [3, 3]], dtype=float),
        ),
    ],
)
def test_to_numpy_dataframe_single_block(data, expected):
    # https://github.com/pandas-dev/pandas/issues/33820
    df = pd.DataFrame(data)
    result = df.to_numpy(dtype=float, na_value=np.nan)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_dataframe_single_block_no_mutate():
    # https://github.com/pandas-dev/pandas/issues/33820
    result = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
    expected = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
    result.to_numpy(na_value=0.0)
    tm.assert_frame_equal(result, expected)


class TestAsArray:
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_asarray_object_dt64(self, tz):
        ser = Series(date_range("2000", periods=2, tz=tz))

        with tm.assert_produces_warning(None):
            # Future behavior (for tzaware case) with no warning
            result = np.asarray(ser, dtype=object)

        expected = np.array(
            [Timestamp("2000-01-01", tz=tz), Timestamp("2000-01-02", tz=tz)]
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_naive(self):
        # This shouldn't produce a warning.
        ser = Series(date_range("2000", periods=2))
        expected = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        result = np.asarray(ser)

        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_aware(self):
        tz = "US/Central"
        ser = Series(date_range("2000", periods=2, tz=tz))
        expected = np.array(["2000-01-01T06", "2000-01-02T06"], dtype="M8[ns]")
        result = np.asarray(ser, dtype="datetime64[ns]")

        tm.assert_numpy_array_equal(result, expected)

        # Old behavior with no warning
        result = np.asarray(ser, dtype="M8[ns]")

        tm.assert_numpy_array_equal(result, expected)
