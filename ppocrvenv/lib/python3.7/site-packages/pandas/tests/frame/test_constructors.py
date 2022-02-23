from collections import (
    OrderedDict,
    abc,
)
from datetime import (
    date,
    datetime,
    timedelta,
)
import functools
import itertools
import re
import warnings

import numpy as np
import numpy.ma as ma
import numpy.ma.mrecords as mrecords
import pytest
import pytz

from pandas.compat import np_version_under1p19
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
    PandasDtype,
    PeriodDtype,
)

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    MultiIndex,
    Period,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.arrays import (
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
)

MIXED_FLOAT_DTYPES = ["float16", "float32", "float64"]
MIXED_INT_DTYPES = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
]


class TestDataFrameConstructors:
    def test_constructor_dict_with_tzaware_scalar(self):
        # GH#42505
        dt = Timestamp("2019-11-03 01:00:00-0700").tz_convert("America/Los_Angeles")

        df = DataFrame({"dt": dt}, index=[0])
        expected = DataFrame({"dt": [dt]})
        tm.assert_frame_equal(df, expected)

        # Non-homogeneous
        df = DataFrame({"dt": dt, "value": [1]})
        expected = DataFrame({"dt": [dt], "value": [1]})
        tm.assert_frame_equal(df, expected)

    def test_construct_ndarray_with_nas_and_int_dtype(self):
        # GH#26919 match Series by not casting np.nan to meaningless int
        arr = np.array([[1, np.nan], [2, 3]])
        df = DataFrame(arr, dtype="i8")
        assert df.values.dtype == arr.dtype
        assert isna(df.iloc[0, 1])

        # check this matches Series behavior
        ser = Series(arr[0], dtype="i8", name=0)
        expected = df.iloc[0]
        tm.assert_series_equal(ser, expected)

    def test_construct_from_list_of_datetimes(self):
        df = DataFrame([datetime.now(), datetime.now()])
        assert df[0].dtype == np.dtype("M8[ns]")

    def test_constructor_from_tzaware_datetimeindex(self):
        # don't cast a DatetimeIndex WITH a tz, leave as object
        # GH#6032
        naive = DatetimeIndex(["2013-1-1 13:00", "2013-1-2 14:00"], name="B")
        idx = naive.tz_localize("US/Pacific")

        expected = Series(np.array(idx.tolist(), dtype="object"), name="B")
        assert expected.dtype == idx.dtype

        # convert index to series
        result = Series(idx)
        tm.assert_series_equal(result, expected)

    def test_array_of_dt64_nat_with_td64dtype_raises(self, frame_or_series):
        # GH#39462
        nat = np.datetime64("NaT", "ns")
        arr = np.array([nat], dtype=object)
        if frame_or_series is DataFrame:
            arr = arr.reshape(1, 1)

        msg = "|".join(
            [
                "Could not convert object to NumPy timedelta",
                "Invalid type for timedelta scalar: <class 'numpy.datetime64'>",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            frame_or_series(arr, dtype="m8[ns]")

    @pytest.mark.parametrize("kind", ["m", "M"])
    def test_datetimelike_values_with_object_dtype(self, kind, frame_or_series):
        # with dtype=object, we should cast dt64 values to Timestamps, not pydatetimes
        if kind == "M":
            dtype = "M8[ns]"
            scalar_type = Timestamp
        else:
            dtype = "m8[ns]"
            scalar_type = Timedelta

        arr = np.arange(6, dtype="i8").view(dtype).reshape(3, 2)
        if frame_or_series is Series:
            arr = arr[:, 0]

        obj = frame_or_series(arr, dtype=object)
        assert obj._mgr.arrays[0].dtype == object
        assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)

        # go through a different path in internals.construction
        obj = frame_or_series(frame_or_series(arr), dtype=object)
        assert obj._mgr.arrays[0].dtype == object
        assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)

        obj = frame_or_series(frame_or_series(arr), dtype=PandasDtype(object))
        assert obj._mgr.arrays[0].dtype == object
        assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)

        if frame_or_series is DataFrame:
            # other paths through internals.construction
            sers = [Series(x) for x in arr]
            obj = frame_or_series(sers, dtype=object)
            assert obj._mgr.arrays[0].dtype == object
            assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)

    def test_series_with_name_not_matching_column(self):
        # GH#9232
        x = Series(range(5), name=1)
        y = Series(range(5), name=0)

        result = DataFrame(x, columns=[0])
        expected = DataFrame([], columns=[0])
        tm.assert_frame_equal(result, expected)

        result = DataFrame(y, columns=[1])
        expected = DataFrame([], columns=[1])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: DataFrame(),
            lambda: DataFrame(None),
            lambda: DataFrame({}),
            lambda: DataFrame(()),
            lambda: DataFrame([]),
            lambda: DataFrame(_ for _ in []),
            lambda: DataFrame(range(0)),
            lambda: DataFrame(data=None),
            lambda: DataFrame(data={}),
            lambda: DataFrame(data=()),
            lambda: DataFrame(data=[]),
            lambda: DataFrame(data=(_ for _ in [])),
            lambda: DataFrame(data=range(0)),
        ],
    )
    def test_empty_constructor(self, constructor):
        expected = DataFrame()
        result = constructor()
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "emptylike,expected_index,expected_columns",
        [
            ([[]], RangeIndex(1), RangeIndex(0)),
            ([[], []], RangeIndex(2), RangeIndex(0)),
            ([(_ for _ in [])], RangeIndex(1), RangeIndex(0)),
        ],
    )
    def test_emptylike_constructor(self, emptylike, expected_index, expected_columns):
        expected = DataFrame(index=expected_index, columns=expected_columns)
        result = DataFrame(emptylike)
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed(self, float_string_frame):
        index, data = tm.getMixedTypeDict()

        # TODO(wesm), incomplete test?
        indexed_frame = DataFrame(data, index=index)  # noqa
        unindexed_frame = DataFrame(data)  # noqa

        assert float_string_frame["foo"].dtype == np.object_

    def test_constructor_cast_failure(self):
        msg = "either all columns will be cast to that dtype, or a TypeError will"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            foo = DataFrame({"a": ["a", "b", "c"]}, dtype=np.float64)
        assert foo["a"].dtype == object

        # GH 3010, constructing with odd arrays
        df = DataFrame(np.ones((4, 2)))

        # this is ok
        df["foo"] = np.ones((4, 2)).tolist()

        # this is not ok
        msg = "|".join(
            [
                "Wrong number of items passed 2, placement implies 1",
                "Expected a 1D array, got an array with shape \\(4, 2\\)",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            df["test"] = np.ones((4, 2))

        # this is ok
        df["foo2"] = np.ones((4, 2)).tolist()

    def test_constructor_dtype_copy(self):
        orig_df = DataFrame({"col1": [1.0], "col2": [2.0], "col3": [3.0]})

        new_df = DataFrame(orig_df, dtype=float, copy=True)

        new_df["col1"] = 200.0
        assert orig_df["col1"][0] == 1.0

    def test_constructor_dtype_nocast_view_dataframe(self):
        df = DataFrame([[1, 2]])
        should_be_view = DataFrame(df, dtype=df[0].dtype)
        should_be_view[0][0] = 99
        assert df.values[0, 0] == 99

    @td.skip_array_manager_invalid_test  # TODO(ArrayManager) keep view on 2D array?
    def test_constructor_dtype_nocast_view_2d_array(self):
        df = DataFrame([[1, 2]])
        should_be_view = DataFrame(df.values, dtype=df[0].dtype)
        should_be_view[0][0] = 97
        assert df.values[0, 0] == 97

    def test_constructor_dtype_list_data(self):
        df = DataFrame([[1, "2"], [None, "a"]], dtype=object)
        assert df.loc[1, 0] is None
        assert df.loc[0, 1] == "2"

    @pytest.mark.skipif(np_version_under1p19, reason="NumPy change.")
    def test_constructor_list_of_2d_raises(self):
        # https://github.com/pandas-dev/pandas/issues/32289
        a = DataFrame()
        b = np.empty((0, 0))
        with pytest.raises(ValueError, match=r"shape=\(1, 0, 0\)"):
            DataFrame([a])

        with pytest.raises(ValueError, match=r"shape=\(1, 0, 0\)"):
            DataFrame([b])

        a = DataFrame({"A": [1, 2]})
        with pytest.raises(ValueError, match=r"shape=\(2, 2, 1\)"):
            DataFrame([a, a])

    def test_constructor_mixed_dtypes(self):
        def _make_mixed_dtypes_df(typ, ad=None):

            if typ == "int":
                dtypes = MIXED_INT_DTYPES
                arrays = [np.array(np.random.rand(10), dtype=d) for d in dtypes]
            elif typ == "float":
                dtypes = MIXED_FLOAT_DTYPES
                arrays = [
                    np.array(np.random.randint(10, size=10), dtype=d) for d in dtypes
                ]

            for d, a in zip(dtypes, arrays):
                assert a.dtype == d
            if ad is None:
                ad = {}
            ad.update({d: a for d, a in zip(dtypes, arrays)})
            return DataFrame(ad)

        def _check_mixed_dtypes(df, dtypes=None):
            if dtypes is None:
                dtypes = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
            for d in dtypes:
                if d in df:
                    assert df.dtypes[d] == d

        # mixed floating and integer coexist in the same frame
        df = _make_mixed_dtypes_df("float")
        _check_mixed_dtypes(df)

        # add lots of types
        df = _make_mixed_dtypes_df("float", {"A": 1, "B": "foo", "C": "bar"})
        _check_mixed_dtypes(df)

        # GH 622
        df = _make_mixed_dtypes_df("int")
        _check_mixed_dtypes(df)

    def test_constructor_complex_dtypes(self):
        # GH10952
        a = np.random.rand(10).astype(np.complex64)
        b = np.random.rand(10).astype(np.complex128)

        df = DataFrame({"a": a, "b": b})
        assert a.dtype == df.a.dtype
        assert b.dtype == df.b.dtype

    def test_constructor_dtype_str_na_values(self, string_dtype):
        # https://github.com/pandas-dev/pandas/issues/21083
        df = DataFrame({"A": ["x", None]}, dtype=string_dtype)
        result = df.isna()
        expected = DataFrame({"A": [False, True]})
        tm.assert_frame_equal(result, expected)
        assert df.iloc[1, 0] is None

        df = DataFrame({"A": ["x", np.nan]}, dtype=string_dtype)
        assert np.isnan(df.iloc[1, 0])

    def test_constructor_rec(self, float_frame):
        rec = float_frame.to_records(index=False)
        rec.dtype.names = list(rec.dtype.names)[::-1]

        index = float_frame.index

        df = DataFrame(rec)
        tm.assert_index_equal(df.columns, Index(rec.dtype.names))

        df2 = DataFrame(rec, index=index)
        tm.assert_index_equal(df2.columns, Index(rec.dtype.names))
        tm.assert_index_equal(df2.index, index)

        # case with columns != the ones we would infer from the data
        rng = np.arange(len(rec))[::-1]
        df3 = DataFrame(rec, index=rng, columns=["C", "B"])
        expected = DataFrame(rec, index=rng).reindex(columns=["C", "B"])
        tm.assert_frame_equal(df3, expected)

    def test_constructor_bool(self):
        df = DataFrame({0: np.ones(10, dtype=bool), 1: np.zeros(10, dtype=bool)})
        assert df.values.dtype == np.bool_

    def test_constructor_overflow_int64(self):
        # see gh-14881
        values = np.array([2 ** 64 - i for i in range(1, 10)], dtype=np.uint64)

        result = DataFrame({"a": values})
        assert result["a"].dtype == np.uint64

        # see gh-2355
        data_scores = [
            (6311132704823138710, 273),
            (2685045978526272070, 23),
            (8921811264899370420, 45),
            (17019687244989530680, 270),
            (9930107427299601010, 273),
        ]
        dtype = [("uid", "u8"), ("score", "u8")]
        data = np.zeros((len(data_scores),), dtype=dtype)
        data[:] = data_scores
        df_crawls = DataFrame(data)
        assert df_crawls["uid"].dtype == np.uint64

    @pytest.mark.parametrize(
        "values",
        [
            np.array([2 ** 64], dtype=object),
            np.array([2 ** 65]),
            [2 ** 64 + 1],
            np.array([-(2 ** 63) - 4], dtype=object),
            np.array([-(2 ** 64) - 1]),
            [-(2 ** 65) - 2],
        ],
    )
    def test_constructor_int_overflow(self, values):
        # see gh-18584
        value = values[0]
        result = DataFrame(values)

        assert result[0].dtype == object
        assert result[0][0] == value

    def test_constructor_ordereddict(self):
        import random

        nitems = 100
        nums = list(range(nitems))
        random.shuffle(nums)
        expected = [f"A{i:d}" for i in nums]
        df = DataFrame(OrderedDict(zip(expected, [[0]] * nitems)))
        assert expected == list(df.columns)

    def test_constructor_dict(self):
        datetime_series = tm.makeTimeSeries(nper=30)
        # test expects index shifted by 5
        datetime_series_short = tm.makeTimeSeries(nper=30)[5:]

        frame = DataFrame({"col1": datetime_series, "col2": datetime_series_short})

        # col2 is padded with NaN
        assert len(datetime_series) == 30
        assert len(datetime_series_short) == 25

        tm.assert_series_equal(frame["col1"], datetime_series.rename("col1"))

        exp = Series(
            np.concatenate([[np.nan] * 5, datetime_series_short.values]),
            index=datetime_series.index,
            name="col2",
        )
        tm.assert_series_equal(exp, frame["col2"])

        frame = DataFrame(
            {"col1": datetime_series, "col2": datetime_series_short},
            columns=["col2", "col3", "col4"],
        )

        assert len(frame) == len(datetime_series_short)
        assert "col1" not in frame
        assert isna(frame["col3"]).all()

        # Corner cases
        assert len(DataFrame()) == 0

        # mix dict and array, wrong size - no spec for which error should raise
        # first
        msg = "Mixing dicts with non-Series may lead to ambiguous ordering."
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})

    def test_constructor_dict_length1(self):
        # Length-one dict micro-optimization
        frame = DataFrame({"A": {"1": 1, "2": 2}})
        tm.assert_index_equal(frame.index, Index(["1", "2"]))

    def test_constructor_dict_with_index(self):
        # empty dict plus index
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx)
        assert frame.index is idx

    def test_constructor_dict_with_index_and_columns(self):
        # empty dict with index and columns
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx, columns=idx)
        assert frame.index is idx
        assert frame.columns is idx
        assert len(frame._series) == 3

    def test_constructor_dict_of_empty_lists(self):
        # with dict of empty list and Series
        frame = DataFrame({"A": [], "B": []}, columns=["A", "B"])
        tm.assert_index_equal(frame.index, RangeIndex(0), exact=True)

    def test_constructor_dict_with_none(self):
        # GH 14381
        # Dict with None value
        frame_none = DataFrame({"a": None}, index=[0])
        frame_none_list = DataFrame({"a": [None]}, index=[0])
        assert frame_none._get_value(0, "a") is None
        assert frame_none_list._get_value(0, "a") is None
        tm.assert_frame_equal(frame_none, frame_none_list)

    def test_constructor_dict_errors(self):
        # GH10856
        # dict with scalar values should raise error, even if columns passed
        msg = "If using all scalar values, you must pass an index"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7})

        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7}, columns=["a"])

    @pytest.mark.parametrize("scalar", [2, np.nan, None, "D"])
    def test_constructor_invalid_items_unused(self, scalar):
        # No error if invalid (scalar) value is in fact not used:
        result = DataFrame({"a": scalar}, columns=["b"])
        expected = DataFrame(columns=["b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [2, np.nan, None, float("nan")])
    def test_constructor_dict_nan_key(self, value):
        # GH 18455
        cols = [1, value, 3]
        idx = ["a", value]
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values(1).sort_values("a", axis=1)
        expected = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx).sort_values("a", axis=1)
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [np.nan, None, float("nan")])
    def test_constructor_dict_nan_tuple_key(self, value):
        # GH 18455
        cols = Index([(11, 21), (value, 22), (13, value)])
        idx = Index([("a", value), (value, 2)])
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values((11, 21)).sort_values(("a", value), axis=1)
        expected = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx).sort_values(("a", value), axis=1)
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_order_insertion(self):
        datetime_series = tm.makeTimeSeries(nper=30)
        datetime_series_short = tm.makeTimeSeries(nper=25)

        # GH19018
        # initialization ordering: by insertion order if python>= 3.6
        d = {"b": datetime_series_short, "a": datetime_series}
        frame = DataFrame(data=d)
        expected = DataFrame(data=d, columns=list("ba"))
        tm.assert_frame_equal(frame, expected)

    def test_constructor_dict_nan_key_and_columns(self):
        # GH 16894
        result = DataFrame({np.nan: [1, 2], 2: [2, 3]}, columns=[np.nan, 2])
        expected = DataFrame([[1, 2], [2, 3]], columns=[np.nan, 2])
        tm.assert_frame_equal(result, expected)

    def test_constructor_multi_index(self):
        # GH 4078
        # construction error with mi and all-nan frame
        tuples = [(2, 3), (3, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

        tuples = [(3, 3), (2, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

    def test_constructor_2d_index(self):
        # GH 25416
        # handling of 2d index in construction
        df = DataFrame([[1]], columns=[[1]], index=[1, 2])
        expected = DataFrame(
            [1, 1],
            index=pd.Int64Index([1, 2], dtype="int64"),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
        )
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[1]], columns=[[1]], index=[[1, 2]])
        expected = DataFrame(
            [1, 1],
            index=MultiIndex(levels=[[1, 2]], codes=[[0, 1]]),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
        )
        tm.assert_frame_equal(df, expected)

    def test_constructor_error_msgs(self):
        msg = "Empty data passed with indices specified."
        # passing an empty array with columns specified.
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.empty(0), columns=list("abc"))

        msg = "Mixing dicts with non-Series may lead to ambiguous ordering."
        # mix dict and array, wrong size
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})

        # wrong size ndarray, GH 3105
        msg = r"Shape of passed values is \(4, 3\), indices imply \(3, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=["foo", "bar", "baz"],
                index=date_range("2000-01-01", periods=3),
            )

        arr = np.array([[4, 5, 6]])
        msg = r"Shape of passed values is \(1, 3\), indices imply \(1, 4\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(0, 4), data=arr)

        arr = np.array([4, 5, 6])
        msg = r"Shape of passed values is \(3, 1\), indices imply \(1, 4\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(0, 4), data=arr)

        # higher dim raise exception
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            DataFrame(np.zeros((3, 3, 3)), columns=["A", "B", "C"], index=[1])

        # wrong size axis labels
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.random.rand(2, 3), columns=["A", "B", "C"], index=[1])

        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.random.rand(2, 3), columns=["A", "B"], index=[1, 2])

        # gh-26429
        msg = "2 columns passed, passed data had 10 columns"
        with pytest.raises(ValueError, match=msg):
            DataFrame((range(10), range(10, 20)), columns=("ones", "twos"))

        msg = "If using all scalar values, you must pass an index"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": False, "b": True})

    def test_constructor_subclass_dict(self, dict_subclass):
        # Test for passing dict subclass to constructor
        data = {
            "col1": dict_subclass((x, 10.0 * x) for x in range(10)),
            "col2": dict_subclass((x, 20.0 * x) for x in range(10)),
        }
        df = DataFrame(data)
        refdf = DataFrame({col: dict(val.items()) for col, val in data.items()})
        tm.assert_frame_equal(refdf, df)

        data = dict_subclass(data.items())
        df = DataFrame(data)
        tm.assert_frame_equal(refdf, df)

    def test_constructor_defaultdict(self, float_frame):
        # try with defaultdict
        from collections import defaultdict

        data = {}
        float_frame["B"][:10] = np.nan
        for k, v in float_frame.items():
            dct = defaultdict(dict)
            dct.update(v.to_dict())
            data[k] = dct
        frame = DataFrame(data)
        expected = frame.reindex(index=float_frame.index)
        tm.assert_frame_equal(float_frame, expected)

    def test_constructor_dict_block(self):
        expected = np.array([[4.0, 3.0, 2.0, 1.0]])
        df = DataFrame(
            {"d": [4.0], "c": [3.0], "b": [2.0], "a": [1.0]},
            columns=["d", "c", "b", "a"],
        )
        tm.assert_numpy_array_equal(df.values, expected)

    def test_constructor_dict_cast(self):
        # cast float tests
        test_data = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}
        frame = DataFrame(test_data, dtype=float)
        assert len(frame) == 3
        assert frame["B"].dtype == np.float64
        assert frame["A"].dtype == np.float64

        frame = DataFrame(test_data)
        assert len(frame) == 3
        assert frame["B"].dtype == np.object_
        assert frame["A"].dtype == np.float64

    def test_constructor_dict_cast2(self):
        # can't cast to float
        test_data = {
            "A": dict(zip(range(20), tm.makeStringIndex(20))),
            "B": dict(zip(range(15), np.random.randn(15))),
        }
        msg = "either all columns will be cast to that dtype, or a TypeError will"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            frame = DataFrame(test_data, dtype=float)

        assert len(frame) == 20
        assert frame["A"].dtype == np.object_
        assert frame["B"].dtype == np.float64

    def test_constructor_dict_dont_upcast(self):
        d = {"Col1": {"Row1": "A String", "Row2": np.nan}}
        df = DataFrame(d)
        assert isinstance(df["Col1"]["Row2"], float)

    def test_constructor_dict_dont_upcast2(self):
        dm = DataFrame([[1, 2], ["a", "b"]], index=[1, 2], columns=[1, 2])
        assert isinstance(dm[1][1], int)

    def test_constructor_dict_of_tuples(self):
        # GH #1491
        data = {"a": (1, 2, 3), "b": (4, 5, 6)}

        result = DataFrame(data)
        expected = DataFrame({k: list(v) for k, v in data.items()})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_dict_of_ranges(self):
        # GH 26356
        data = {"a": range(3), "b": range(3, 6)}

        result = DataFrame(data)
        expected = DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_iterators(self):
        # GH 26349
        data = {"a": iter(range(3)), "b": reversed(range(3))}

        result = DataFrame(data)
        expected = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_generators(self):
        # GH 26349
        data = {"a": (i for i in (range(3))), "b": (i for i in reversed(range(3)))}
        result = DataFrame(data)
        expected = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_multiindex(self):
        def check(result, expected):
            return tm.assert_frame_equal(
                result,
                expected,
                check_dtype=True,
                check_index_type=True,
                check_column_type=True,
                check_names=True,
            )

        d = {
            ("a", "a"): {("i", "i"): 0, ("i", "j"): 1, ("j", "i"): 2},
            ("b", "a"): {("i", "i"): 6, ("i", "j"): 5, ("j", "i"): 4},
            ("b", "c"): {("i", "i"): 7, ("i", "j"): 8, ("j", "i"): 9},
        }
        _d = sorted(d.items())
        df = DataFrame(d)
        expected = DataFrame(
            [x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])
        ).T
        expected.index = MultiIndex.from_tuples(expected.index)
        check(df, expected)

        d["z"] = {"y": 123.0, ("i", "i"): 111, ("i", "j"): 111, ("j", "i"): 111}
        _d.insert(0, ("z", d["z"]))
        expected = DataFrame(
            [x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)
        ).T
        expected.index = Index(expected.index, tupleize_cols=False)
        df = DataFrame(d)
        df = df.reindex(columns=expected.columns, index=expected.index)
        check(df, expected)

    def test_constructor_dict_datetime64_index(self):
        # GH 10160
        dates_as_str = ["1984-02-19", "1988-11-06", "1989-12-03", "1990-03-15"]

        def create_data(constructor):
            return {i: {constructor(s): 2 * i} for i, s in enumerate(dates_as_str)}

        data_datetime64 = create_data(np.datetime64)
        data_datetime = create_data(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        data_Timestamp = create_data(Timestamp)

        expected = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timestamp(dt) for dt in dates_as_str],
        )

        result_datetime64 = DataFrame(data_datetime64)
        result_datetime = DataFrame(data_datetime)
        result_Timestamp = DataFrame(data_Timestamp)
        tm.assert_frame_equal(result_datetime64, expected)
        tm.assert_frame_equal(result_datetime, expected)
        tm.assert_frame_equal(result_Timestamp, expected)

    def test_constructor_dict_timedelta64_index(self):
        # GH 10160
        td_as_int = [1, 2, 3, 4]

        def create_data(constructor):
            return {i: {constructor(s): 2 * i} for i, s in enumerate(td_as_int)}

        data_timedelta64 = create_data(lambda x: np.timedelta64(x, "D"))
        data_timedelta = create_data(lambda x: timedelta(days=x))
        data_Timedelta = create_data(lambda x: Timedelta(x, "D"))

        expected = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, "D") for td in td_as_int],
        )

        result_timedelta64 = DataFrame(data_timedelta64)
        result_timedelta = DataFrame(data_timedelta)
        result_Timedelta = DataFrame(data_Timedelta)
        tm.assert_frame_equal(result_timedelta64, expected)
        tm.assert_frame_equal(result_timedelta, expected)
        tm.assert_frame_equal(result_Timedelta, expected)

    def test_constructor_period_dict(self):
        # PeriodIndex
        a = pd.PeriodIndex(["2012-01", "NaT", "2012-04"], freq="M")
        b = pd.PeriodIndex(["2012-02-01", "2012-03-01", "NaT"], freq="D")
        df = DataFrame({"a": a, "b": b})
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype

        # list of periods
        df = DataFrame({"a": a.astype(object).tolist(), "b": b.astype(object).tolist()})
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype

    def test_constructor_dict_extension_scalar(self, ea_scalar_and_dtype):
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df = DataFrame({"a": ea_scalar}, index=[0])
        assert df["a"].dtype == ea_dtype

        expected = DataFrame(index=[0], columns=["a"], data=ea_scalar)

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "data,dtype",
        [
            (Period("2020-01"), PeriodDtype("M")),
            (Interval(left=0, right=5), IntervalDtype("int64", "right")),
            (
                Timestamp("2011-01-01", tz="US/Eastern"),
                DatetimeTZDtype(tz="US/Eastern"),
            ),
        ],
    )
    def test_constructor_extension_scalar_data(self, data, dtype):
        # GH 34832
        df = DataFrame(index=[0, 1], columns=["a", "b"], data=data)

        assert df["a"].dtype == dtype
        assert df["b"].dtype == dtype

        arr = pd.array([data] * 2, dtype=dtype)
        expected = DataFrame({"a": arr, "b": arr})

        tm.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self):
        rng = pd.period_range("1/1/2000", periods=5)
        df = DataFrame(np.random.randn(10, 5), columns=rng)

        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)

        result = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)

        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)

        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def _check_basic_constructor(self, empty):
        # mat: 2d matrix with shape (3, 2) to input. empty - makes sized
        # objects
        mat = empty((2, 3), dtype=float)
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3

        # 1-D input
        frame = DataFrame(empty((3,)), columns=["A"], index=[1, 2, 3])
        assert len(frame.index) == 3
        assert len(frame.columns) == 1

        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64)
        if empty is np.ones:
            # passing dtype casts
            assert frame.values.dtype == np.int64
        else:
            # i.e. ma.masked_all
            # Since we have NaNs, refuse to cast to int dtype, which would take NaN
            #  to meaningless integers.  This matches Series behavior.  GH#26919
            assert frame.isna().all().all()
            assert frame.values.dtype == np.float64
            assert isna(frame.values).all()

        # wrong size axis labels
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=["A", "B", "C"], index=[1])
        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=["A", "B"], index=[1, 2])

        # higher dim raise exception
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            DataFrame(empty((3, 3, 3)), columns=["A", "B", "C"], index=[1])

        # automatic labeling
        frame = DataFrame(mat)
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)

        frame = DataFrame(mat, index=[1, 2])
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)

        frame = DataFrame(mat, columns=["A", "B", "C"])
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)

        # 0-length axis
        frame = DataFrame(empty((0, 3)))
        assert len(frame.index) == 0

        frame = DataFrame(empty((3, 0)))
        assert len(frame.columns) == 0

    def test_constructor_ndarray(self):
        self._check_basic_constructor(np.ones)

        frame = DataFrame(["foo", "bar"], index=[0, 1], columns=["A"])
        assert len(frame) == 2

    def test_constructor_maskedarray(self):
        self._check_basic_constructor(ma.masked_all)

        # Check non-masked values
        mat = ma.masked_all((2, 3), dtype=float)
        mat[0, 0] = 1.0
        mat[1, 2] = 2.0
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert 1.0 == frame["A"][1]
        assert 2.0 == frame["C"][2]

        # what is this even checking??
        mat = ma.masked_all((2, 3), dtype=float)
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert np.all(~np.asarray(frame == frame))

    def test_constructor_maskedarray_nonfloat(self):
        # masked int promoted to float
        mat = ma.masked_all((2, 3), dtype=int)
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))

        # cast type
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.float64)
        assert frame.values.dtype == np.float64

        # Check non-masked values
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert 1 == frame["A"][1]
        assert 2 == frame["C"][2]

        # masked np.datetime64 stays (use NaT as null)
        mat = ma.masked_all((2, 3), dtype="M8[ns]")
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert isna(frame).values.all()

        # cast type
        msg = r"datetime64\[ns\] values and dtype=int64"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message="elementwise comparison failed",
                )
                frame = DataFrame(
                    mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64
                )
        assert frame.values.dtype == np.int64

        # Check non-masked values
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert 1 == frame["A"].view("i8")[1]
        assert 2 == frame["C"].view("i8")[2]

        # masked bool promoted to object
        mat = ma.masked_all((2, 3), dtype=bool)
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))

        # cast type
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=object)
        assert frame.values.dtype == object

        # Check non-masked values
        mat2 = ma.copy(mat)
        mat2[0, 0] = True
        mat2[1, 2] = False
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert frame["A"][1] is True
        assert frame["C"][2] is False

    def test_constructor_maskedarray_hardened(self):
        # Check numpy masked arrays with hard masks -- from GH24574
        mat_hard = ma.masked_all((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)
        # Check case where mask is hard but no data are masked
        mat_hard = ma.ones((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        expected = DataFrame(
            {"A": [1.0, 1.0], "B": [1.0, 1.0]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_maskedrecarray_dtype(self):
        # Ensure constructor honors dtype
        data = np.ma.array(
            np.ma.zeros(5, dtype=[("date", "<f8"), ("price", "<f8")]), mask=[False] * 5
        )
        data = data.view(mrecords.mrecarray)

        with tm.assert_produces_warning(FutureWarning):
            # Support for MaskedRecords deprecated
            result = DataFrame(data, dtype=int)

        expected = DataFrame(np.zeros((5, 2), dtype=int), columns=["date", "price"])
        tm.assert_frame_equal(result, expected)

        # GH#40363 check that the alternative suggested in the deprecation
        #  warning behaves as expected
        alt = DataFrame({name: data[name] for name in data.dtype.names}, dtype=int)
        tm.assert_frame_equal(result, alt)

    @pytest.mark.slow
    def test_constructor_mrecarray(self):
        # Ensure mrecarray produces frame identical to dict of masked arrays
        # from GH3479

        assert_fr_equal = functools.partial(
            tm.assert_frame_equal, check_index_type=True, check_column_type=True
        )
        arrays = [
            ("float", np.array([1.5, 2.0])),
            ("int", np.array([1, 2])),
            ("str", np.array(["abc", "def"])),
        ]
        for name, arr in arrays[:]:
            arrays.append(
                ("masked1_" + name, np.ma.masked_array(arr, mask=[False, True]))
            )
        arrays.append(("masked_all", np.ma.masked_all((2,))))
        arrays.append(("masked_none", np.ma.masked_array([1.0, 2.5], mask=False)))

        # call assert_frame_equal for all selections of 3 arrays
        for comb in itertools.combinations(arrays, 3):
            names, data = zip(*comb)
            mrecs = mrecords.fromarrays(data, names=names)

            # fill the comb
            comb = {k: (v.filled() if hasattr(v, "filled") else v) for k, v in comb}

            with tm.assert_produces_warning(FutureWarning):
                # Support for MaskedRecords deprecated
                result = DataFrame(mrecs)
            expected = DataFrame(comb, columns=names)
            assert_fr_equal(result, expected)

            # specify columns
            with tm.assert_produces_warning(FutureWarning):
                # Support for MaskedRecords deprecated
                result = DataFrame(mrecs, columns=names[::-1])
            expected = DataFrame(comb, columns=names[::-1])
            assert_fr_equal(result, expected)

            # specify index
            with tm.assert_produces_warning(FutureWarning):
                # Support for MaskedRecords deprecated
                result = DataFrame(mrecs, index=[1, 2])
            expected = DataFrame(comb, columns=names, index=[1, 2])
            assert_fr_equal(result, expected)

    def test_constructor_corner_shape(self):
        df = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    @pytest.mark.parametrize(
        "data, index, columns, dtype, expected",
        [
            (None, list(range(10)), ["a", "b"], object, np.object_),
            (None, None, ["a", "b"], "int64", np.dtype("int64")),
            (None, list(range(10)), ["a", "b"], int, np.dtype("float64")),
            ({}, None, ["foo", "bar"], None, np.object_),
            ({"b": 1}, list(range(10)), list("abc"), int, np.dtype("float64")),
        ],
    )
    def test_constructor_dtype(self, data, index, columns, dtype, expected):
        df = DataFrame(data, index, columns, dtype)
        assert df.values.dtype == expected

    @pytest.mark.parametrize(
        "data,input_dtype,expected_dtype",
        (
            ([True, False, None], "boolean", pd.BooleanDtype),
            ([1.0, 2.0, None], "Float64", pd.Float64Dtype),
            ([1, 2, None], "Int64", pd.Int64Dtype),
            (["a", "b", "c"], "string", pd.StringDtype),
        ),
    )
    def test_constructor_dtype_nullable_extension_arrays(
        self, data, input_dtype, expected_dtype
    ):
        df = DataFrame({"a": data}, dtype=input_dtype)
        assert df["a"].dtype == expected_dtype()

    def test_constructor_scalar_inference(self):
        data = {"int": 1, "bool": True, "float": 3.0, "complex": 4j, "object": "foo"}
        df = DataFrame(data, index=np.arange(10))

        assert df["int"].dtype == np.int64
        assert df["bool"].dtype == np.bool_
        assert df["float"].dtype == np.float64
        assert df["complex"].dtype == np.complex128
        assert df["object"].dtype == np.object_

    def test_constructor_arrays_and_scalars(self):
        df = DataFrame({"a": np.random.randn(10), "b": True})
        exp = DataFrame({"a": df["a"].values, "b": [True] * 10})

        tm.assert_frame_equal(df, exp)
        with pytest.raises(ValueError, match="must pass an index"):
            DataFrame({"a": False, "b": True})

    def test_constructor_DataFrame(self, float_frame):
        df = DataFrame(float_frame)
        tm.assert_frame_equal(df, float_frame)

        df_casted = DataFrame(float_frame, dtype=np.int64)
        assert df_casted.values.dtype == np.int64

    def test_constructor_more(self, float_frame):
        # used to be in test_matrix.py
        arr = np.random.randn(10)
        dm = DataFrame(arr, columns=["A"], index=np.arange(10))
        assert dm.values.ndim == 2

        arr = np.random.randn(0)
        dm = DataFrame(arr)
        assert dm.values.ndim == 2
        assert dm.values.ndim == 2

        # no data specified
        dm = DataFrame(columns=["A", "B"], index=np.arange(10))
        assert dm.values.shape == (10, 2)

        dm = DataFrame(columns=["A", "B"])
        assert dm.values.shape == (0, 2)

        dm = DataFrame(index=np.arange(10))
        assert dm.values.shape == (10, 0)

        # can't cast
        mat = np.array(["foo", "bar"], dtype=object).reshape(2, 1)
        msg = "could not convert string to float: 'foo'"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, index=[0, 1], columns=[0], dtype=float)

        dm = DataFrame(DataFrame(float_frame._series))
        tm.assert_frame_equal(dm, float_frame)

        # int cast
        dm = DataFrame(
            {"A": np.ones(10, dtype=int), "B": np.ones(10, dtype=np.float64)},
            index=np.arange(10),
        )

        assert len(dm.columns) == 2
        assert dm.values.dtype == np.float64

    def test_constructor_empty_list(self):
        df = DataFrame([], index=[])
        expected = DataFrame(index=[])
        tm.assert_frame_equal(df, expected)

        # GH 9939
        df = DataFrame([], columns=["A", "B"])
        expected = DataFrame({}, columns=["A", "B"])
        tm.assert_frame_equal(df, expected)

        # Empty generator: list(empty_gen()) == []
        def empty_gen():
            return
            yield

        df = DataFrame(empty_gen(), columns=["A", "B"])
        tm.assert_frame_equal(df, expected)

    def test_constructor_list_of_lists(self):
        # GH #484
        df = DataFrame(data=[[1, "a"], [2, "b"]], columns=["num", "str"])
        assert is_integer_dtype(df["num"])
        assert df["str"].dtype == np.object_

        # GH 4851
        # list of 0-dim ndarrays
        expected = DataFrame({0: np.arange(10)})
        data = [np.array(x) for x in range(10)]
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_like_data_nested_list_column(self):
        # GH 32173
        arrays = [list("abcd"), list("cdef")]
        result = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

        mi = MultiIndex.from_arrays(arrays)
        expected = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=mi)

        tm.assert_frame_equal(result, expected)

    def test_constructor_wrong_length_nested_list_column(self):
        # GH 32173
        arrays = [list("abc"), list("cde")]

        msg = "3 columns passed, passed data had 4"
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    def test_constructor_unequal_length_nested_list_column(self):
        # GH 32173
        arrays = [list("abcd"), list("cde")]

        # exception raised inside MultiIndex constructor
        msg = "all arrays must be same length"
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    @pytest.mark.parametrize(
        "data",
        [
            [[Timestamp("2021-01-01")]],
            [{"x": Timestamp("2021-01-01")}],
            {"x": [Timestamp("2021-01-01")]},
            {"x": Timestamp("2021-01-01")},
        ],
    )
    def test_constructor_one_element_data_list(self, data):
        # GH#42810
        result = DataFrame(data, index=[0, 1, 2], columns=["x"])
        expected = DataFrame({"x": [Timestamp("2021-01-01")] * 3})
        tm.assert_frame_equal(result, expected)

    def test_constructor_sequence_like(self):
        # GH 3783
        # collections.Sequence like

        class DummyContainer(abc.Sequence):
            def __init__(self, lst):
                self._lst = lst

            def __getitem__(self, n):
                return self._lst.__getitem__(n)

            def __len__(self, n):
                return self._lst.__len__()

        lst_containers = [DummyContainer([1, "a"]), DummyContainer([2, "b"])]
        columns = ["num", "str"]
        result = DataFrame(lst_containers, columns=columns)
        expected = DataFrame([[1, "a"], [2, "b"]], columns=columns)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_stdlib_array(self):
        # GH 4297
        # support Array
        import array

        result = DataFrame({"A": array.array("i", range(10))})
        expected = DataFrame({"A": list(range(10))})
        tm.assert_frame_equal(result, expected, check_dtype=False)

        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([array.array("i", range(10)), array.array("i", range(10))])
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_range(self):
        # GH26342
        result = DataFrame(range(10))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_ranges(self):
        result = DataFrame([range(10), range(10)])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterable(self):
        # GH 21987
        class Iter:
            def __iter__(self):
                for i in range(10):
                    yield [1, 2, 3]

        expected = DataFrame([[1, 2, 3]] * 10)
        result = DataFrame(Iter())
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterator(self):
        result = DataFrame(iter(range(10)))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_iterators(self):
        result = DataFrame([iter(range(10)), iter(range(10))])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_generator(self):
        # related #2305

        gen1 = (i for i in range(10))
        gen2 = (i for i in range(10))

        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([gen1, gen2])
        tm.assert_frame_equal(result, expected)

        gen = ([i, "a"] for i in range(10))
        result = DataFrame(gen)
        expected = DataFrame({0: range(10), 1: "a"})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_list_of_dicts(self):

        result = DataFrame([{}])
        expected = DataFrame(index=[0])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dict_type", [dict, OrderedDict])
    def test_constructor_ordered_dict_preserve_order(self, dict_type):
        # see gh-13304
        expected = DataFrame([[2, 1]], columns=["b", "a"])

        data = dict_type()
        data["b"] = [2]
        data["a"] = [1]

        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

        data = dict_type()
        data["b"] = 2
        data["a"] = 1

        result = DataFrame([data])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dict_type", [dict, OrderedDict])
    def test_constructor_ordered_dict_conflicting_orders(self, dict_type):
        # the first dict element sets the ordering for the DataFrame,
        # even if there are conflicting orders from subsequent ones
        row_one = dict_type()
        row_one["b"] = 2
        row_one["a"] = 1

        row_two = dict_type()
        row_two["a"] = 1
        row_two["b"] = 2

        row_three = {"b": 2, "a": 1}

        expected = DataFrame([[2, 1], [2, 1]], columns=["b", "a"])
        result = DataFrame([row_one, row_two])
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2, 1], [2, 1], [2, 1]], columns=["b", "a"])
        result = DataFrame([row_one, row_two, row_three])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_series_aligned_index(self):
        series = [Series(i, index=["b", "a", "c"], name=str(i)) for i in range(3)]
        result = DataFrame(series)
        expected = DataFrame(
            {"b": [0, 1, 2], "a": [0, 1, 2], "c": [0, 1, 2]},
            columns=["b", "a", "c"],
            index=["0", "1", "2"],
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_derived_dicts(self):
        class CustomDict(dict):
            pass

        d = {"a": 1.5, "b": 3}

        data_custom = [CustomDict(d)]
        data = [d]

        result_custom = DataFrame(data_custom)
        result = DataFrame(data)
        tm.assert_frame_equal(result, result_custom)

    def test_constructor_ragged(self):
        data = {"A": np.random.randn(10), "B": np.random.randn(8)}
        with pytest.raises(ValueError, match="All arrays must be of the same length"):
            DataFrame(data)

    def test_constructor_scalar(self):
        idx = Index(range(3))
        df = DataFrame({"a": 0}, index=idx)
        expected = DataFrame({"a": [0, 0, 0]}, index=idx)
        tm.assert_frame_equal(df, expected, check_dtype=False)

    def test_constructor_Series_copy_bug(self, float_frame):
        df = DataFrame(float_frame["A"], index=float_frame.index, columns=["A"])
        df.copy()

    def test_constructor_mixed_dict_and_Series(self):
        data = {}
        data["A"] = {"foo": 1, "bar": 2, "baz": 3}
        data["B"] = Series([4, 3, 2, 1], index=["bar", "qux", "baz", "foo"])

        result = DataFrame(data)
        assert result.index.is_monotonic

        # ordering ambiguous, raise exception
        with pytest.raises(ValueError, match="ambiguous ordering"):
            DataFrame({"A": ["a", "b"], "B": {"a": "a", "b": "b"}})

        # this is OK though
        result = DataFrame({"A": ["a", "b"], "B": Series(["a", "b"], index=["a", "b"])})
        expected = DataFrame({"A": ["a", "b"], "B": ["a", "b"]}, index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed_type_rows(self):
        # Issue 25075
        data = [[1, 2], (3, 4)]
        result = DataFrame(data)
        expected = DataFrame([[1, 2], [3, 4]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "tuples,lists",
        [
            ((), []),
            ((()), []),
            (((), ()), [(), ()]),
            (((), ()), [[], []]),
            (([], []), [[], []]),
            (([1], [2]), [[1], [2]]),  # GH 32776
            (([1, 2, 3], [4, 5, 6]), [[1, 2, 3], [4, 5, 6]]),
        ],
    )
    def test_constructor_tuple(self, tuples, lists):
        # GH 25691
        result = DataFrame(tuples)
        expected = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self):
        result = DataFrame({"A": [(1, 2), (3, 4)]})
        expected = DataFrame({"A": Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self):
        # GH11181
        from collections import namedtuple

        named_tuple = namedtuple("Pandas", list("ab"))
        tuples = [named_tuple(1, 3), named_tuple(2, 4)]
        expected = DataFrame({"a": [1, 2], "b": [3, 4]})
        result = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)

        # with columns
        expected = DataFrame({"y": [1, 2], "z": [3, 4]})
        result = DataFrame(tuples, columns=["y", "z"])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self):
        # GH21910
        from dataclasses import make_dataclass

        Point = make_dataclass("Point", [("x", int), ("y", int)])

        data = [Point(0, 3), Point(1, 3)]
        expected = DataFrame({"x": [0, 1], "y": [3, 3]})
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self):
        # GH21910
        from dataclasses import make_dataclass

        # varying types
        Point = make_dataclass("Point", [("x", int), ("y", int)])
        HLine = make_dataclass("HLine", [("x0", int), ("x1", int), ("y", int)])

        data = [Point(0, 3), HLine(1, 3, 3)]

        expected = DataFrame(
            {"x": [0, np.nan], "y": [3, 3], "x0": [np.nan, 1], "x1": [np.nan, 3]}
        )
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self):
        # GH21910
        from dataclasses import make_dataclass

        Point = make_dataclass("Point", [("x", int), ("y", int)])

        # expect TypeError
        msg = "asdict() should be called on dataclass instances"
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {"x": 1, "y": 0}])

    def test_constructor_list_of_dict_order(self):
        # GH10056
        data = [
            {"First": 1, "Second": 4, "Third": 7, "Fourth": 10},
            {"Second": 5, "First": 2, "Fourth": 11, "Third": 8},
            {"Second": 6, "First": 3, "Fourth": 12, "Third": 9, "YYY": 14, "XXX": 13},
        ]
        expected = DataFrame(
            {
                "First": [1, 2, 3],
                "Second": [4, 5, 6],
                "Third": [7, 8, 9],
                "Fourth": [10, 11, 12],
                "YYY": [None, None, 14],
                "XXX": [None, None, 13],
            }
        )
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named(self):
        a = Series([1, 2, 3], index=["a", "b", "c"], name="x")
        df = DataFrame(a)
        assert df.columns[0] == "x"
        tm.assert_index_equal(df.index, a.index)

        # ndarray like
        arr = np.random.randn(10)
        s = Series(arr, name="x")
        df = DataFrame(s)
        expected = DataFrame({"x": s})
        tm.assert_frame_equal(df, expected)

        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = DataFrame({0: s})
        tm.assert_frame_equal(df, expected)

        msg = r"Shape of passed values is \(10, 1\), indices imply \(10, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])

        # #2234
        a = Series([], name="x", dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == "x"

        # series with name and w/o
        s1 = Series(arr, name="x")
        df = DataFrame([s1, arr]).T
        expected = DataFrame({"x": s1, "Unnamed 0": arr}, columns=["x", "Unnamed 0"])
        tm.assert_frame_equal(df, expected)

        # this is a bit non-intuitive here; the series collapse down to arrays
        df = DataFrame([arr, s1]).T
        expected = DataFrame({1: s1, 0: arr}, columns=[0, 1])
        tm.assert_frame_equal(df, expected)

    def test_constructor_Series_named_and_columns(self):
        # GH 9232 validation

        s0 = Series(range(5), name=0)
        s1 = Series(range(5), name=1)

        # matching name and column gives standard frame
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())

        # non-matching produces empty frame
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self):
        # name
        s1 = Series([1, 2, 3], index=["a", "b", "c"], name="x")

        # no name
        s2 = Series([1, 2, 3], index=["a", "b", "c"])

        other_index = Index(["a", "b"])

        df1 = DataFrame(s1, index=other_index)
        exp1 = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == "x"
        tm.assert_frame_equal(df1, exp1)

        df2 = DataFrame(s2, index=other_index)
        exp2 = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        "name_in1,name_in2,name_in3,name_out",
        [
            ("idx", "idx", "idx", "idx"),
            ("idx", "idx", None, None),
            ("idx", None, None, None),
            ("idx1", "idx2", None, None),
            ("idx1", "idx1", "idx2", None),
            ("idx1", "idx2", "idx3", None),
            (None, None, None, None),
        ],
    )
    def test_constructor_index_names(self, name_in1, name_in2, name_in3, name_out):
        # GH13475
        indices = [
            Index(["a", "b", "c"], name=name_in1),
            Index(["b", "c", "d"], name=name_in2),
            Index(["c", "d", "e"], name=name_in3),
        ]
        series = {
            c: Series([0, 1, 2], index=i) for i, c in zip(indices, ["x", "y", "z"])
        }
        result = DataFrame(series)

        exp_ind = Index(["a", "b", "c", "d", "e"], name=name_out)
        expected = DataFrame(
            {
                "x": [0, 1, 2, np.nan, np.nan],
                "y": [np.nan, 0, 1, 2, np.nan],
                "z": [np.nan, np.nan, 0, 1, 2],
            },
            index=exp_ind,
        )

        tm.assert_frame_equal(result, expected)

    def test_constructor_manager_resize(self, float_frame):
        index = list(float_frame.index[:5])
        columns = list(float_frame.columns[:3])

        result = DataFrame(float_frame._mgr, index=index, columns=columns)
        tm.assert_index_equal(result.index, Index(index))
        tm.assert_index_equal(result.columns, Index(columns))

    def test_constructor_mix_series_nonseries(self, float_frame):
        df = DataFrame(
            {"A": float_frame["A"], "B": list(float_frame["B"])}, columns=["A", "B"]
        )
        tm.assert_frame_equal(df, float_frame.loc[:, ["A", "B"]])

        msg = "does not match index length"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": float_frame["A"], "B": list(float_frame["B"])[:-2]})

    def test_constructor_miscast_na_int_dtype(self):
        df = DataFrame([[np.nan, 1], [1, 0]], dtype=np.int64)
        expected = DataFrame([[np.nan, 1], [1, 0]])
        tm.assert_frame_equal(df, expected)

    def test_constructor_column_duplicates(self):
        # it works! #2079
        df = DataFrame([[8, 5]], columns=["a", "a"])
        edf = DataFrame([[8, 5]])
        edf.columns = ["a", "a"]

        tm.assert_frame_equal(df, edf)

        idf = DataFrame.from_records([(8, 5)], columns=["a", "a"])

        tm.assert_frame_equal(idf, edf)

    def test_constructor_empty_with_string_dtype(self):
        # GH 9428
        expected = DataFrame(index=[0, 1], columns=[0, 1], dtype=object)

        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=str)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=np.str_)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=np.unicode_)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype="U5")
        tm.assert_frame_equal(df, expected)

    def test_constructor_empty_with_string_extension(self, nullable_string_dtype):
        # GH 34915
        expected = DataFrame(index=[], columns=["c1"], dtype=nullable_string_dtype)
        df = DataFrame(columns=["c1"], dtype=nullable_string_dtype)
        tm.assert_frame_equal(df, expected)

    def test_constructor_single_value(self):
        # expecting single value upcasting here
        df = DataFrame(0.0, index=[1, 2, 3], columns=["a", "b", "c"])
        tm.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("float64"), df.index, df.columns)
        )

        df = DataFrame(0, index=[1, 2, 3], columns=["a", "b", "c"])
        tm.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("int64"), df.index, df.columns)
        )

        df = DataFrame("a", index=[1, 2], columns=["a", "c"])
        tm.assert_frame_equal(
            df,
            DataFrame(
                np.array([["a", "a"], ["a", "a"]], dtype=object),
                index=[1, 2],
                columns=["a", "c"],
            ),
        )

        msg = "DataFrame constructor not properly called!"
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", [1, 2])
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", columns=["a", "c"])

        msg = "incompatible data and dtype"
        with pytest.raises(TypeError, match=msg):
            DataFrame("a", [1, 2], ["a", "c"], float)

    def test_constructor_with_datetimes(self):
        intname = np.dtype(np.int_).name
        floatname = np.dtype(np.float_).name
        datetime64name = np.dtype("M8[ns]").name
        objectname = np.dtype(np.object_).name

        # single item
        df = DataFrame(
            {
                "A": 1,
                "B": "foo",
                "C": "bar",
                "D": Timestamp("20010101"),
                "E": datetime(2001, 1, 2, 0, 0),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("int64")]
            + [np.dtype(objectname)] * 2
            + [np.dtype(datetime64name)] * 2,
            index=list("ABCDE"),
        )
        tm.assert_series_equal(result, expected)

        # check with ndarray construction ndim==0 (e.g. we are passing a ndim 0
        # ndarray with a dtype specified)
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                floatname: np.array(1.0, dtype=floatname),
                intname: np.array(1, dtype=intname),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")]
            + [np.dtype("int64")]
            + [np.dtype("object")]
            + [np.dtype("float64")]
            + [np.dtype(intname)],
            index=["a", "b", "c", floatname, intname],
        )
        tm.assert_series_equal(result, expected)

        # check with ndarray construction ndim>0
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                floatname: np.array([1.0] * 10, dtype=floatname),
                intname: np.array([1] * 10, dtype=intname),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")]
            + [np.dtype("int64")]
            + [np.dtype("object")]
            + [np.dtype("float64")]
            + [np.dtype(intname)],
            index=["a", "b", "c", floatname, intname],
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes1(self):

        # GH 2809
        ind = date_range(start="2000-01-01", freq="D", periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        datetime_s = Series(datetimes)
        assert datetime_s.dtype == "M8[ns]"

    def test_constructor_with_datetimes2(self):
        # GH 2810
        ind = date_range(start="2000-01-01", freq="D", periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        dates = [ts.date() for ts in ind]
        df = DataFrame(datetimes, columns=["datetimes"])
        df["dates"] = dates
        result = df.dtypes
        expected = Series(
            [np.dtype("datetime64[ns]"), np.dtype("object")],
            index=["datetimes", "dates"],
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes3(self):
        # GH 7594
        # don't coerce tz-aware
        tz = pytz.timezone("US/Eastern")
        dt = tz.localize(datetime(2012, 1, 1))

        df = DataFrame({"End Date": dt}, index=[0])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[ns, US/Eastern]"})
        )

        df = DataFrame([{"End Date": dt}])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[ns, US/Eastern]"})
        )

    def test_constructor_with_datetimes4(self):
        # tz-aware (UTC and other tz's)
        # GH 8411
        dr = date_range("20130101", periods=3)
        df = DataFrame({"value": dr})
        assert df.iat[0, 0].tz is None
        dr = date_range("20130101", periods=3, tz="UTC")
        df = DataFrame({"value": dr})
        assert str(df.iat[0, 0].tz) == "UTC"
        dr = date_range("20130101", periods=3, tz="US/Eastern")
        df = DataFrame({"value": dr})
        assert str(df.iat[0, 0].tz) == "US/Eastern"

    def test_constructor_with_datetimes5(self):
        # GH 7822
        # preserver an index with a tz on dict construction
        i = date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")

        expected = DataFrame({"a": i.to_series().reset_index(drop=True)})
        df = DataFrame()
        df["a"] = i
        tm.assert_frame_equal(df, expected)

        df = DataFrame({"a": i})
        tm.assert_frame_equal(df, expected)

    def test_constructor_with_datetimes6(self):
        # multiples
        i = date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")
        i_no_tz = date_range("1/1/2011", periods=5, freq="10s")
        df = DataFrame({"a": i, "b": i_no_tz})
        expected = DataFrame({"a": i.to_series().reset_index(drop=True), "b": i_no_tz})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([None, None, None, None, datetime.now(), None]),
            np.array([None, None, datetime.now(), None]),
            [[np.datetime64("NaT")], [None]],
            [[np.datetime64("NaT")], [pd.NaT]],
            [[None], [np.datetime64("NaT")]],
            [[None], [pd.NaT]],
            [[pd.NaT], [np.datetime64("NaT")]],
            [[pd.NaT], [None]],
        ],
    )
    def test_constructor_datetimes_with_nulls(self, arr):
        # gh-15869, GH#11220
        result = DataFrame(arr).dtypes
        expected = Series([np.dtype("datetime64[ns]")])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[M]",
            "datetime64[D]",
            "datetime64[h]",
            "datetime64[m]",
            "datetime64[s]",
            "datetime64[ms]",
            "datetime64[us]",
            "datetime64[ns]",
        ],
    )
    def test_constructor_datetimes_non_ns(self, order, dtype):
        na = np.array(
            [
                ["2015-01-01", "2015-01-02", "2015-01-03"],
                ["2017-01-01", "2017-01-02", "2017-02-03"],
            ],
            dtype=dtype,
            order=order,
        )
        df = DataFrame(na)
        expected = DataFrame(
            [
                ["2015-01-01", "2015-01-02", "2015-01-03"],
                ["2017-01-01", "2017-01-02", "2017-02-03"],
            ]
        )
        expected = expected.astype(dtype=dtype)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "dtype",
        [
            "timedelta64[D]",
            "timedelta64[h]",
            "timedelta64[m]",
            "timedelta64[s]",
            "timedelta64[ms]",
            "timedelta64[us]",
            "timedelta64[ns]",
        ],
    )
    def test_constructor_timedelta_non_ns(self, order, dtype):
        na = np.array(
            [
                [np.timedelta64(1, "D"), np.timedelta64(2, "D")],
                [np.timedelta64(4, "D"), np.timedelta64(5, "D")],
            ],
            dtype=dtype,
            order=order,
        )
        df = DataFrame(na).astype("timedelta64[ns]")
        expected = DataFrame(
            [
                [Timedelta(1, "D"), Timedelta(2, "D")],
                [Timedelta(4, "D"), Timedelta(5, "D")],
            ],
        )
        tm.assert_frame_equal(df, expected)

    def test_constructor_for_list_with_dtypes(self):
        # test list of lists/ndarrays
        df = DataFrame([np.arange(5) for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype("int")] * 5)
        tm.assert_series_equal(result, expected)

        df = DataFrame([np.array(np.arange(5), dtype="int32") for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype("int32")] * 5)
        tm.assert_series_equal(result, expected)

        # overflow issue? (we always expected int64 upcasting here)
        df = DataFrame({"a": [2 ** 31, 2 ** 31 + 1]})
        assert df.dtypes.iloc[0] == np.dtype("int64")

        # GH #2751 (construction with no index specified), make sure we cast to
        # platform values
        df = DataFrame([1, 2])
        assert df.dtypes.iloc[0] == np.dtype("int64")

        df = DataFrame([1.0, 2.0])
        assert df.dtypes.iloc[0] == np.dtype("float64")

        df = DataFrame({"a": [1, 2]})
        assert df.dtypes.iloc[0] == np.dtype("int64")

        df = DataFrame({"a": [1.0, 2.0]})
        assert df.dtypes.iloc[0] == np.dtype("float64")

        df = DataFrame({"a": 1}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype("int64")

        df = DataFrame({"a": 1.0}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype("float64")

        # with object list
        df = DataFrame(
            {
                "a": [1, 2, 4, 7],
                "b": [1.2, 2.3, 5.1, 6.3],
                "c": list("abcd"),
                "d": [datetime(2000, 1, 1) for i in range(4)],
                "e": [1.0, 2, 4.0, 7],
            }
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype("int64"),
                np.dtype("float64"),
                np.dtype("object"),
                np.dtype("datetime64[ns]"),
                np.dtype("float64"),
            ],
            index=list("abcde"),
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_frame_copy(self, float_frame):
        cop = DataFrame(float_frame, copy=True)
        cop["A"] = 5
        assert (cop["A"] == 5).all()
        assert not (float_frame["A"] == 5).all()

    # TODO(ArrayManager) keep view on 2D array?
    @td.skip_array_manager_not_yet_implemented
    def test_constructor_ndarray_copy(self, float_frame):
        df = DataFrame(float_frame.values)

        float_frame.values[5] = 5
        assert (df.values[5] == 5).all()

        df = DataFrame(float_frame.values, copy=True)
        float_frame.values[6] = 6
        assert not (df.values[6] == 6).all()

    # TODO(ArrayManager) keep view on Series?
    @td.skip_array_manager_not_yet_implemented
    def test_constructor_series_copy(self, float_frame):
        series = float_frame._series

        df = DataFrame({"A": series["A"]}, copy=True)
        df["A"][:] = 5

        assert not (series["A"] == 5).all()

    def test_constructor_with_nas(self):
        # GH 5016
        # na's in indices

        def check(df):
            for i in range(len(df.columns)):
                df.iloc[:, i]

            indexer = np.arange(len(df.columns))[isna(df.columns)]

            # No NaN found -> error
            if len(indexer) == 0:
                with pytest.raises(KeyError, match="^nan$"):
                    df.loc[:, np.nan]
            # single nan should result in Series
            elif len(indexer) == 1:
                tm.assert_series_equal(df.iloc[:, indexer[0]], df.loc[:, np.nan])
            # multiple nans should result in DataFrame
            else:
                tm.assert_frame_equal(df.iloc[:, indexer], df.loc[:, np.nan])

        df = DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, np.nan])
        check(df)

        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1.1, 2.2, np.nan])
        check(df)

        df = DataFrame([[0, 1, 2, 3], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan])
        check(df)

        df = DataFrame(
            [[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]
        )
        check(df)

        # GH 21428 (non-unique columns)
        df = DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1, 2, 2])
        check(df)

    def test_constructor_lists_to_object_dtype(self):
        # from #1074
        d = DataFrame({"a": [np.nan, False]})
        assert d["a"].dtype == np.object_
        assert not d["a"][1]

    def test_constructor_ndarray_categorical_dtype(self):
        cat = Categorical(["A", "B", "C"])
        arr = np.array(cat).reshape(-1, 1)
        arr = np.broadcast_to(arr, (3, 4))

        result = DataFrame(arr, dtype=cat.dtype)

        expected = DataFrame({0: cat, 1: cat, 2: cat, 3: cat})
        tm.assert_frame_equal(result, expected)

    def test_constructor_categorical(self):

        # GH8626

        # dict creation
        df = DataFrame({"A": list("abc")}, dtype="category")
        expected = Series(list("abc"), dtype="category", name="A")
        tm.assert_series_equal(df["A"], expected)

        # to_frame
        s = Series(list("abc"), dtype="category")
        result = s.to_frame()
        expected = Series(list("abc"), dtype="category", name=0)
        tm.assert_series_equal(result[0], expected)
        result = s.to_frame(name="foo")
        expected = Series(list("abc"), dtype="category", name="foo")
        tm.assert_series_equal(result["foo"], expected)

        # list-like creation
        df = DataFrame(list("abc"), dtype="category")
        expected = Series(list("abc"), dtype="category", name=0)
        tm.assert_series_equal(df[0], expected)

    def test_construct_from_1item_list_of_categorical(self):
        # ndim != 1
        msg = "will be changed to match the behavior"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df = DataFrame([Categorical(list("abc"))])
        expected = DataFrame({0: Series(list("abc"), dtype="category")})
        tm.assert_frame_equal(df, expected)

    def test_construct_from_list_of_categoricals(self):
        msg = "will be changed to match the behavior"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df = DataFrame([Categorical(list("abc")), Categorical(list("abd"))])
        expected = DataFrame(
            {
                0: Series(list("abc"), dtype="category"),
                1: Series(list("abd"), dtype="category"),
            },
            columns=[0, 1],
        )
        tm.assert_frame_equal(df, expected)

    def test_from_nested_listlike_mixed_types(self):
        # mixed
        msg = "will be changed to match the behavior"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df = DataFrame([Categorical(list("abc")), list("def")])
        expected = DataFrame(
            {0: Series(list("abc"), dtype="category"), 1: list("def")}, columns=[0, 1]
        )
        tm.assert_frame_equal(df, expected)

    def test_construct_from_listlikes_mismatched_lengths(self):
        # invalid (shape)
        msg = "|".join(
            [
                r"Shape of passed values is \(6, 2\), indices imply \(3, 2\)",
                "Passed arrays should have the same length as the rows Index",
            ]
        )
        msg2 = "will be changed to match the behavior"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                DataFrame([Categorical(list("abc")), Categorical(list("abdefg"))])

    def test_constructor_categorical_series(self):

        items = [1, 2, 3, 1]
        exp = Series(items).astype("category")
        res = Series(items, dtype="category")
        tm.assert_series_equal(res, exp)

        items = ["a", "b", "c", "a"]
        exp = Series(items).astype("category")
        res = Series(items, dtype="category")
        tm.assert_series_equal(res, exp)

        # insert into frame with different index
        # GH 8076
        index = date_range("20000101", periods=3)
        expected = Series(
            Categorical(values=[np.nan, np.nan, np.nan], categories=["a", "b", "c"])
        )
        expected.index = index

        expected = DataFrame({"x": expected})
        df = DataFrame({"x": Series(["a", "b", "c"], dtype="category")}, index=index)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "dtype",
        tm.ALL_INT_DTYPES
        + tm.ALL_EA_INT_DTYPES
        + tm.FLOAT_DTYPES
        + tm.COMPLEX_DTYPES
        + tm.DATETIME64_DTYPES
        + tm.TIMEDELTA64_DTYPES
        + tm.BOOL_DTYPES,
    )
    def test_check_dtype_empty_numeric_column(self, dtype):
        # GH24386: Ensure dtypes are set correctly for an empty DataFrame.
        # Empty DataFrame is generated via dictionary data with non-overlapping columns.
        data = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)

        assert data.b.dtype == dtype

    # TODO(ArrayManager) astype to bytes dtypes does not yet give object dtype
    @td.skip_array_manager_not_yet_implemented
    @pytest.mark.parametrize(
        "dtype", tm.STRING_DTYPES + tm.BYTES_DTYPES + tm.OBJECT_DTYPES
    )
    def test_check_dtype_empty_string_column(self, dtype):
        # GH24386: Ensure dtypes are set correctly for an empty DataFrame.
        # Empty DataFrame is generated via dictionary data with non-overlapping columns.
        data = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)

        assert data.b.dtype.name == "object"

    def test_to_frame_with_falsey_names(self):
        # GH 16114
        result = Series(name=0, dtype=object).to_frame().dtypes
        expected = Series({0: object})
        tm.assert_series_equal(result, expected)

        result = DataFrame(Series(name=0, dtype=object)).dtypes
        tm.assert_series_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize("dtype", [None, "uint8", "category"])
    def test_constructor_range_dtype(self, dtype):
        expected = DataFrame({"A": [0, 1, 2, 3, 4]}, dtype=dtype or "int64")

        # GH 26342
        result = DataFrame(range(5), columns=["A"], dtype=dtype)
        tm.assert_frame_equal(result, expected)

        # GH 16804
        result = DataFrame({"A": range(5)}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_frame_from_list_subclass(self):
        # GH21226
        class List(list):
            pass

        expected = DataFrame([[1, 2, 3], [4, 5, 6]])
        result = DataFrame(List([List([1, 2, 3]), List([4, 5, 6])]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "extension_arr",
        [
            Categorical(list("aabbc")),
            SparseArray([1, np.nan, np.nan, np.nan]),
            IntervalArray([Interval(0, 1), Interval(1, 5)]),
            PeriodArray(pd.period_range(start="1/1/2017", end="1/1/2018", freq="M")),
        ],
    )
    def test_constructor_with_extension_array(self, extension_arr):
        # GH11363
        expected = DataFrame(Series(extension_arr))
        result = DataFrame(extension_arr)
        tm.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self):
        # GH 10863
        v = date.today()
        tup = v, v
        result = DataFrame({tup: Series(range(3), index=range(3))}, columns=[tup])
        expected = DataFrame([0, 1, 2], columns=Index(Series([tup])))
        tm.assert_frame_equal(result, expected)

    def test_construct_with_two_categoricalindex_series(self):
        # GH 14600
        s1 = Series([39, 6, 4], index=CategoricalIndex(["female", "male", "unknown"]))
        s2 = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(["f", "female", "m", "male", "unknown"]),
        )
        result = DataFrame([s1, s2])
        expected = DataFrame(
            np.array(
                [[np.nan, 39.0, np.nan, 6.0, 4.0], [2.0, 152.0, 2.0, 242.0, 150.0]]
            ),
            columns=["f", "female", "m", "male", "unknown"],
        )
        tm.assert_frame_equal(result, expected)

    def test_from_M8_structured(self):
        dates = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        arr = np.array(dates, dtype=[("Date", "M8[us]"), ("Forecasting", "M8[us]")])
        df = DataFrame(arr)

        assert df["Date"][0] == dates[0][0]
        assert df["Forecasting"][0] == dates[0][1]

        s = Series(arr["Date"])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def test_from_datetime_subclass(self):
        # GH21142 Verify whether Datetime subclasses are also of dtype datetime
        class DatetimeSubclass(datetime):
            pass

        data = DataFrame({"datetime": [DatetimeSubclass(2020, 1, 1, 1, 1)]})
        assert data.datetime.dtype == "datetime64[ns]"

    def test_with_mismatched_index_length_raises(self):
        # GH#33437
        dti = date_range("2016-01-01", periods=3, tz="US/Pacific")
        msg = "Shape of passed values|Passed arrays should have the same length"
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))

    def test_frame_ctor_datetime64_column(self):
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        dates = np.asarray(rng)

        df = DataFrame({"A": np.random.randn(len(rng)), "B": dates})
        assert np.issubdtype(df["B"].dtype, np.dtype("M8[ns]"))

    def test_dataframe_constructor_infer_multiindex(self):
        index_lists = [["a", "a", "b", "b"], ["x", "y", "x", "y"]]

        multi = DataFrame(
            np.random.randn(4, 4),
            index=[np.array(x) for x in index_lists],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)

        multi = DataFrame(np.random.randn(4, 4), columns=index_lists)
        assert isinstance(multi.columns, MultiIndex)

    @pytest.mark.parametrize(
        "input_vals",
        [
            ([1, 2]),
            (["1", "2"]),
            (list(date_range("1/1/2011", periods=2, freq="H"))),
            (list(date_range("1/1/2011", periods=2, freq="H", tz="US/Eastern"))),
            ([Interval(left=0, right=5)]),
        ],
    )
    def test_constructor_list_str(self, input_vals, string_dtype):
        # GH#16605
        # Ensure that data elements are converted to strings when
        # dtype is str, 'str', or 'U'

        result = DataFrame({"A": input_vals}, dtype=string_dtype)
        expected = DataFrame({"A": input_vals}).astype({"A": string_dtype})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype):

        result = DataFrame({"A": [1.0, 2.0, None]}, dtype=string_dtype)
        expected = DataFrame({"A": ["1.0", "2.0", None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("copy", [False, True])
    @td.skip_array_manager_not_yet_implemented
    def test_dict_nocopy(self, copy, any_nullable_numeric_dtype, any_numpy_dtype):
        a = np.array([1, 2], dtype=any_numpy_dtype)
        b = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ["S", "U"]:
            # These get cast, making the checks below more cumbersome
            return

        c = pd.array([1, 2], dtype=any_nullable_numeric_dtype)
        df = DataFrame({"a": a, "b": b, "c": c}, copy=copy)

        def get_base(obj):
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                # i.e. DatetimeArray, TimedeltaArray
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views():
            # written to work for either BlockManager or ArrayManager
            assert sum(x is c for x in df._mgr.arrays) == 1
            assert (
                sum(
                    get_base(x) is a
                    for x in df._mgr.arrays
                    if isinstance(x.dtype, np.dtype)
                )
                == 1
            )
            assert (
                sum(
                    get_base(x) is b
                    for x in df._mgr.arrays
                    if isinstance(x.dtype, np.dtype)
                )
                == 1
            )

        if not copy:
            # constructor preserves views
            check_views()

        df.iloc[0, 0] = 0
        df.iloc[0, 1] = 0
        if not copy:
            # Check that the underlying data behind df["c"] is still `c`
            #  after setting with iloc.  Since we don't know which entry in
            #  df._mgr.arrays corresponds to df["c"], we just check that exactly
            #  one of these arrays is `c`.  GH#38939
            assert sum(x is c for x in df._mgr.arrays) == 1
            # TODO: we can call check_views if we stop consolidating
            #  in setitem_with_indexer

        # FIXME: until GH#35417, iloc.setitem into EA values does not preserve
        #  view, so we have to check in the other direction
        # df.iloc[0, 2] = 0
        # if not copy:
        #     check_views()
        c[0] = 0

        if copy:
            if a.dtype.kind == "M":
                assert a[0] == a.dtype.type(1, "ns")
                assert b[0] == b.dtype.type(3, "ns")
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            # FIXME: enable after GH#35417
            # assert c[0] == 1
            assert df.iloc[0, 2] == 1
        else:
            # TODO: we can call check_views if we stop consolidating
            #  in setitem_with_indexer
            # FIXME: enable after GH#35417
            # assert b[0] == 0
            assert df.iloc[0, 2] == 0

    def test_from_series_with_name_with_columns(self):
        # GH 7893
        result = DataFrame(Series(1, name="foo"), columns=["bar"])
        expected = DataFrame(columns=["bar"])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self):
        # GH 14467
        result = DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=[["A", "A", "A"], ["a", "b", "c"]]
        )
        expected = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("A", "c")]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_object_array_of_periods_or_intervals(self):
        # Period analogue to GH#26825
        pi = pd.period_range("2016-04-05", periods=3)
        data = pi._data.astype(object).reshape(1, -1)
        df = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == pi.dtype).all()
        assert (df == pi).all().all()

        ii = pd.IntervalIndex.from_breaks([3, 4, 5, 6])
        data2 = ii._data.astype(object).reshape(1, -1)
        df2 = DataFrame(data2)
        assert df2.shape == (1, 3)
        assert (df2.dtypes == ii.dtype).all()
        assert (df2 == ii).all().all()

        # mixed
        data3 = np.r_[data, data2, data, data2].T
        df3 = DataFrame(data3)
        expected = DataFrame({0: pi, 1: ii, 2: pi, 3: ii})
        tm.assert_frame_equal(df3, expected)


class TestDataFrameConstructorWithDtypeCoercion:
    def test_floating_values_integer_dtype(self):
        # GH#40110 make DataFrame behavior with arraylike floating data and
        #  inty dtype match Series behavior

        arr = np.random.randn(10, 5)

        msg = "if they cannot be cast losslessly"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            DataFrame(arr, dtype="i8")

        with tm.assert_produces_warning(None):
            # if they can be cast losslessly, no warning
            DataFrame(arr.round(), dtype="i8")

        # with NaNs, we already have the correct behavior, so no warning
        arr[0, 0] = np.nan
        with tm.assert_produces_warning(None):
            DataFrame(arr, dtype="i8")


class TestDataFrameConstructorWithDatetimeTZ:
    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_construction_preserves_tzaware_dtypes(self, tz):
        # after GH#7822
        # these retain the timezones on dict construction
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        dr_tz = dr.tz_localize(tz)
        df = DataFrame({"A": "foo", "B": dr_tz}, index=dr)
        tz_expected = DatetimeTZDtype("ns", dr_tz.tzinfo)
        assert df["B"].dtype == tz_expected

        # GH#2810 (with timezones)
        datetimes_naive = [ts.to_pydatetime() for ts in dr]
        datetimes_with_tz = [ts.to_pydatetime() for ts in dr_tz]
        df = DataFrame({"dr": dr})
        df["dr_tz"] = dr_tz
        df["datetimes_naive"] = datetimes_naive
        df["datetimes_with_tz"] = datetimes_with_tz
        result = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype(tz=tz),
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype(tz=tz),
            ],
            index=["dr", "dr_tz", "datetimes_naive", "datetimes_with_tz"],
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("pydt", [True, False])
    def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture, pydt):
        # GH#25843, GH#41555, GH#33401
        tz = tz_aware_fixture
        ts = Timestamp("2019", tz=tz)
        if pydt:
            ts = ts.to_pydatetime()
        ts_naive = Timestamp("2019")

        with tm.assert_produces_warning(FutureWarning):
            result = DataFrame({0: [ts]}, dtype="datetime64[ns]")

        expected = DataFrame({0: [ts_naive]})
        tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            result = DataFrame({0: ts}, index=[0], dtype="datetime64[ns]")
        tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            result = DataFrame([ts], dtype="datetime64[ns]")
        tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            result = DataFrame(np.array([ts], dtype=object), dtype="datetime64[ns]")
        tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning):
            result = DataFrame(ts, index=[0], columns=[0], dtype="datetime64[ns]")
        tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            df = DataFrame([Series([ts])], dtype="datetime64[ns]")
        tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            df = DataFrame([[ts]], columns=[0], dtype="datetime64[ns]")
        tm.assert_equal(df, expected)

    def test_from_dict(self):

        # 8260
        # support datetime64 with tz

        idx = Index(date_range("20130101", periods=3, tz="US/Eastern"), name="foo")
        dr = date_range("20130110", periods=3)

        # construction
        df = DataFrame({"A": idx, "B": dr})
        assert df["A"].dtype, "M8[ns, US/Eastern"
        assert df["A"].name == "A"
        tm.assert_series_equal(df["A"], Series(idx, name="A"))
        tm.assert_series_equal(df["B"], Series(dr, name="B"))

    def test_from_index(self):

        # from index
        idx2 = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2["foo"], Series(idx2, name="foo"))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2["foo"], Series(idx2, name="foo"))

        idx2 = date_range("20130101", periods=3, tz="US/Eastern")
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2[0], Series(idx2, name=0))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2[0], Series(idx2, name=0))

    def test_frame_dict_constructor_datetime64_1680(self):
        dr = date_range("1/1/2012", periods=10)
        s = Series(dr, index=dr)

        # it works!
        DataFrame({"a": "foo", "b": s}, index=dr)
        DataFrame({"a": "foo", "b": s.values}, index=dr)

    def test_frame_datetime64_mixed_index_ctor_1681(self):
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        ts = Series(dr)

        # it works!
        d = DataFrame({"A": "foo", "B": ts}, index=dr)
        assert d["B"].isna().all()

    def test_frame_timeseries_column(self):
        # GH19157
        dr = date_range(start="20130101T10:00:00", periods=3, freq="T", tz="US/Eastern")
        result = DataFrame(dr, columns=["timestamps"])
        expected = DataFrame(
            {
                "timestamps": [
                    Timestamp("20130101T10:00:00", tz="US/Eastern"),
                    Timestamp("20130101T10:01:00", tz="US/Eastern"),
                    Timestamp("20130101T10:02:00", tz="US/Eastern"),
                ]
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_nested_dict_construction(self):
        # GH22227
        columns = ["Nevada", "Ohio"]
        pop = {
            "Nevada": {2001: 2.4, 2002: 2.9},
            "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
        }
        result = DataFrame(pop, index=[2001, 2002, 2003], columns=columns)
        expected = DataFrame(
            [(2.4, 1.7), (2.9, 3.6), (np.nan, np.nan)],
            columns=columns,
            index=Index([2001, 2002, 2003]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_tzaware_object_array(self):
        # GH#26825 2D object array of tzaware timestamps should not raise
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        data = dti._data.astype(object).reshape(1, -1)
        df = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == dti.dtype).all()
        assert (df == dti).all().all()

    def test_from_tzaware_mixed_object_array(self):
        # GH#26825
        arr = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T
        res = DataFrame(arr, columns=["A", "B", "C"])

        expected_dtypes = [
            "datetime64[ns]",
            "datetime64[ns, US/Eastern]",
            "datetime64[ns, CET]",
        ]
        assert (res.dtypes == expected_dtypes).all()

    def test_from_2d_ndarray_with_dtype(self):
        # GH#12513
        array_dim2 = np.arange(10).reshape((5, 2))
        df = DataFrame(array_dim2, dtype="datetime64[ns, UTC]")

        expected = DataFrame(array_dim2).astype("datetime64[ns, UTC]")
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("typ", [set, frozenset])
    def test_construction_from_set_raises(self, typ):
        # https://github.com/pandas-dev/pandas/issues/32582
        values = typ({1, 2, 3})
        msg = f"'{typ.__name__}' type is unordered"
        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": values})

        with pytest.raises(TypeError, match=msg):
            Series(values)

    def test_construction_from_ndarray_datetimelike(self):
        # ensure the underlying arrays are properly wrapped as EA when
        # constructed from 2D ndarray
        arr = np.arange(0, 12, dtype="datetime64[ns]").reshape(4, 3)
        df = DataFrame(arr)
        assert all(isinstance(arr, DatetimeArray) for arr in df._mgr.arrays)


def get1(obj):
    if isinstance(obj, Series):
        return obj.iloc[0]
    else:
        return obj.iloc[0, 0]


class TestFromScalar:
    @pytest.fixture(params=[list, dict, None])
    def constructor(self, request, frame_or_series):
        box = request.param

        extra = {"index": range(2)}
        if frame_or_series is DataFrame:
            extra["columns"] = ["A"]

        if box is None:
            return functools.partial(frame_or_series, **extra)

        elif box is dict:
            if frame_or_series is Series:
                return lambda x, **kwargs: frame_or_series(
                    {0: x, 1: x}, **extra, **kwargs
                )
            else:
                return lambda x, **kwargs: frame_or_series({"A": x}, **extra, **kwargs)
        else:
            if frame_or_series is Series:
                return lambda x, **kwargs: frame_or_series([x, x], **extra, **kwargs)
            else:
                return lambda x, **kwargs: frame_or_series(
                    {"A": [x, x]}, **extra, **kwargs
                )

    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_from_nat_scalar(self, dtype, constructor):
        obj = constructor(pd.NaT, dtype=dtype)
        assert np.all(obj.dtypes == dtype)
        assert np.all(obj.isna())

    def test_from_timedelta_scalar_preserves_nanos(self, constructor):
        td = Timedelta(1)

        obj = constructor(td, dtype="m8[ns]")
        assert get1(obj) == td

    def test_from_timestamp_scalar_preserves_nanos(self, constructor):
        ts = Timestamp.now() + Timedelta(1)

        obj = constructor(ts, dtype="M8[ns]")
        assert get1(obj) == ts

    def test_from_timedelta64_scalar_object(self, constructor):

        td = Timedelta(1)
        td64 = td.to_timedelta64()

        obj = constructor(td64, dtype=object)
        assert isinstance(get1(obj), np.timedelta64)

    @pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
    def test_from_scalar_datetimelike_mismatched(self, constructor, cls, request):
        node = request.node
        params = node.callspec.params
        if params["frame_or_series"] is DataFrame and params["constructor"] is dict:
            mark = pytest.mark.xfail(
                reason="DataFrame incorrectly allows mismatched datetimelike"
            )
            node.add_marker(mark)
        scalar = cls("NaT", "ns")
        dtype = {np.datetime64: "m8[ns]", np.timedelta64: "M8[ns]"}[cls]

        msg = "Cannot cast"
        if cls is np.datetime64:
            msg = "|".join(
                [
                    r"dtype datetime64\[ns\] cannot be converted to timedelta64\[ns\]",
                    "Cannot cast",
                ]
            )

        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

        scalar = cls(4, "ns")
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

    @pytest.mark.parametrize("cls", [datetime, np.datetime64])
    def test_from_out_of_bounds_datetime(self, constructor, cls):
        scalar = datetime(9999, 1, 1)
        if cls is np.datetime64:
            scalar = np.datetime64(scalar, "D")
        result = constructor(scalar)

        assert type(get1(result)) is cls

    @pytest.mark.parametrize("cls", [timedelta, np.timedelta64])
    def test_from_out_of_bounds_timedelta(self, constructor, cls):
        scalar = datetime(9999, 1, 1) - datetime(1970, 1, 1)
        if cls is np.timedelta64:
            scalar = np.timedelta64(scalar, "D")
        result = constructor(scalar)

        assert type(get1(result)) is cls

    def test_tzaware_data_tznaive_dtype(self, constructor):
        tz = "US/Eastern"
        ts = Timestamp("2019", tz=tz)
        ts_naive = Timestamp("2019")

        with tm.assert_produces_warning(
            FutureWarning, match="Data is timezone-aware", check_stacklevel=False
        ):
            result = constructor(ts, dtype="M8[ns]")

        assert np.all(result.dtypes == "M8[ns]")
        assert np.all(result == ts_naive)
