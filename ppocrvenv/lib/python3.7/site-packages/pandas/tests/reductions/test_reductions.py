from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
    isna,
    timedelta_range,
    to_timedelta,
)
import pandas._testing as tm
from pandas.core import nanops


def get_objs():
    indexes = [
        tm.makeBoolIndex(10, name="a"),
        tm.makeIntIndex(10, name="a"),
        tm.makeFloatIndex(10, name="a"),
        tm.makeDateIndex(10, name="a"),
        tm.makeDateIndex(10, name="a").tz_localize(tz="US/Eastern"),
        tm.makePeriodIndex(10, name="a"),
        tm.makeStringIndex(10, name="a"),
        tm.makeUnicodeIndex(10, name="a"),
    ]

    arr = np.random.randn(10)
    series = [Series(arr, index=idx, name="a") for idx in indexes]

    objs = indexes + series
    return objs


objs = get_objs()


class TestReductions:
    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize("obj", objs)
    def test_ops(self, opname, obj):
        result = getattr(obj, opname)()
        if not isinstance(obj, PeriodIndex):
            expected = getattr(obj.values, opname)()
        else:
            expected = Period(ordinal=getattr(obj.asi8, opname)(), freq=obj.freq)

        if getattr(obj, "tz", None) is not None:
            # We need to de-localize before comparing to the numpy-produced result
            expected = expected.astype("M8[ns]").astype("int64")
            assert result.value == expected
        else:
            assert result == expected

    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize(
        "dtype, val",
        [
            ("object", 2.0),
            ("float64", 2.0),
            ("datetime64[ns]", datetime(2011, 11, 1)),
            ("Int64", 2),
            ("boolean", True),
        ],
    )
    def test_nanminmax(self, opname, dtype, val, index_or_series):
        # GH#7261
        klass = index_or_series

        if dtype in ["Int64", "boolean"] and klass == Index:
            pytest.skip("EAs can't yet be stored in an index")

        def check_missing(res):
            if dtype == "datetime64[ns]":
                return res is NaT
            elif dtype == "Int64":
                return res is pd.NA
            else:
                return isna(res)

        obj = klass([None], dtype=dtype)
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))

        obj = klass([], dtype=dtype)
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))

        if dtype == "object":
            # generic test with object only works for empty / all NaN
            return

        obj = klass([None, val], dtype=dtype)
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))

        obj = klass([None, val, None], dtype=dtype)
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))

    @pytest.mark.parametrize("opname", ["max", "min"])
    def test_nanargminmax(self, opname, index_or_series):
        # GH#7261
        klass = index_or_series
        arg_op = "arg" + opname if klass is Index else "idx" + opname

        obj = klass([NaT, datetime(2011, 11, 1)])
        assert getattr(obj, arg_op)() == 1
        result = getattr(obj, arg_op)(skipna=False)
        if klass is Series:
            assert np.isnan(result)
        else:
            assert result == -1

        obj = klass([NaT, datetime(2011, 11, 1), NaT])
        # check DatetimeIndex non-monotonic path
        assert getattr(obj, arg_op)() == 1
        result = getattr(obj, arg_op)(skipna=False)
        if klass is Series:
            assert np.isnan(result)
        else:
            assert result == -1

    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize("dtype", ["M8[ns]", "datetime64[ns, UTC]"])
    def test_nanops_empty_object(self, opname, index_or_series, dtype):
        klass = index_or_series
        arg_op = "arg" + opname if klass is Index else "idx" + opname

        obj = klass([], dtype=dtype)

        assert getattr(obj, opname)() is NaT
        assert getattr(obj, opname)(skipna=False) is NaT

        with pytest.raises(ValueError, match="empty sequence"):
            getattr(obj, arg_op)()
        with pytest.raises(ValueError, match="empty sequence"):
            getattr(obj, arg_op)(skipna=False)

    def test_argminmax(self):
        obj = Index(np.arange(5, dtype="int64"))
        assert obj.argmin() == 0
        assert obj.argmax() == 4

        obj = Index([np.nan, 1, np.nan, 2])
        assert obj.argmin() == 1
        assert obj.argmax() == 3
        assert obj.argmin(skipna=False) == -1
        assert obj.argmax(skipna=False) == -1

        obj = Index([np.nan])
        assert obj.argmin() == -1
        assert obj.argmax() == -1
        assert obj.argmin(skipna=False) == -1
        assert obj.argmax(skipna=False) == -1

        obj = Index([NaT, datetime(2011, 11, 1), datetime(2011, 11, 2), NaT])
        assert obj.argmin() == 1
        assert obj.argmax() == 2
        assert obj.argmin(skipna=False) == -1
        assert obj.argmax(skipna=False) == -1

        obj = Index([NaT])
        assert obj.argmin() == -1
        assert obj.argmax() == -1
        assert obj.argmin(skipna=False) == -1
        assert obj.argmax(skipna=False) == -1

    @pytest.mark.parametrize("op, expected_col", [["max", "a"], ["min", "b"]])
    def test_same_tz_min_max_axis_1(self, op, expected_col):
        # GH 10390
        df = DataFrame(
            date_range("2016-01-01 00:00:00", periods=3, tz="UTC"), columns=["a"]
        )
        df["b"] = df.a.subtract(Timedelta(seconds=3600))
        result = getattr(df, op)(axis=1)
        expected = df[expected_col].rename(None)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", ["maximum", "minimum"])
    def test_numpy_reduction_with_tz_aware_dtype(self, tz_aware_fixture, func):
        # GH 15552
        tz = tz_aware_fixture
        arg = pd.to_datetime(["2019"]).tz_localize(tz)
        expected = Series(arg)
        result = getattr(np, func)(expected, expected)
        tm.assert_series_equal(result, expected)


class TestIndexReductions:
    # Note: the name TestIndexReductions indicates these tests
    #  were moved from a Index-specific test file, _not_ that these tests are
    #  intended long-term to be Index-specific

    @pytest.mark.parametrize(
        "start,stop,step",
        [
            (0, 400, 3),
            (500, 0, -6),
            (-(10 ** 6), 10 ** 6, 4),
            (10 ** 6, -(10 ** 6), -4),
            (0, 10, 20),
        ],
    )
    def test_max_min_range(self, start, stop, step):
        # GH#17607
        idx = RangeIndex(start, stop, step)
        expected = idx._int64index.max()
        result = idx.max()
        assert result == expected

        # skipna should be irrelevant since RangeIndex should never have NAs
        result2 = idx.max(skipna=False)
        assert result2 == expected

        expected = idx._int64index.min()
        result = idx.min()
        assert result == expected

        # skipna should be irrelevant since RangeIndex should never have NAs
        result2 = idx.min(skipna=False)
        assert result2 == expected

        # empty
        idx = RangeIndex(start, stop, -step)
        assert isna(idx.max())
        assert isna(idx.min())

    def test_minmax_timedelta64(self):

        # monotonic
        idx1 = TimedeltaIndex(["1 days", "2 days", "3 days"])
        assert idx1.is_monotonic

        # non-monotonic
        idx2 = TimedeltaIndex(["1 days", np.nan, "3 days", "NaT"])
        assert not idx2.is_monotonic

        for idx in [idx1, idx2]:
            assert idx.min() == Timedelta("1 days")
            assert idx.max() == Timedelta("3 days")
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax_timedelta_empty_or_na(self, op):
        # Return NaT
        obj = TimedeltaIndex([])
        assert getattr(obj, op)() is NaT

        obj = TimedeltaIndex([NaT])
        assert getattr(obj, op)() is NaT

        obj = TimedeltaIndex([NaT, NaT, NaT])
        assert getattr(obj, op)() is NaT

    def test_numpy_minmax_timedelta64(self):
        td = timedelta_range("16815 days", "16820 days", freq="D")

        assert np.min(td) == Timedelta("16815 days")
        assert np.max(td) == Timedelta("16820 days")

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(td, out=0)

        assert np.argmin(td) == 0
        assert np.argmax(td) == 5

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(td, out=0)

    def test_timedelta_ops(self):
        # GH#4984
        # make sure ops return Timedelta
        s = Series(
            [Timestamp("20130101") + timedelta(seconds=i * i) for i in range(10)]
        )
        td = s.diff()

        result = td.mean()
        expected = to_timedelta(timedelta(seconds=9))
        assert result == expected

        result = td.to_frame().mean()
        assert result[0] == expected

        result = td.quantile(0.1)
        expected = Timedelta(np.timedelta64(2600, "ms"))
        assert result == expected

        result = td.median()
        expected = to_timedelta("00:00:09")
        assert result == expected

        result = td.to_frame().median()
        assert result[0] == expected

        # GH#6462
        # consistency in returned values for sum
        result = td.sum()
        expected = to_timedelta("00:01:21")
        assert result == expected

        result = td.to_frame().sum()
        assert result[0] == expected

        # std
        result = td.std()
        expected = to_timedelta(Series(td.dropna().values).std())
        assert result == expected

        result = td.to_frame().std()
        assert result[0] == expected

        # GH#10040
        # make sure NaT is properly handled by median()
        s = Series([Timestamp("2015-02-03"), Timestamp("2015-02-07")])
        assert s.diff().median() == timedelta(days=4)

        s = Series(
            [Timestamp("2015-02-03"), Timestamp("2015-02-07"), Timestamp("2015-02-15")]
        )
        assert s.diff().median() == timedelta(days=6)

    @pytest.mark.parametrize("opname", ["skew", "kurt", "sem", "prod", "var"])
    def test_invalid_td64_reductions(self, opname):
        s = Series(
            [Timestamp("20130101") + timedelta(seconds=i * i) for i in range(10)]
        )
        td = s.diff()

        msg = "|".join(
            [
                f"reduction operation '{opname}' not allowed for this dtype",
                rf"cannot perform {opname} with type timedelta64\[ns\]",
                f"'TimedeltaArray' does not implement reduction '{opname}'",
            ]
        )

        with pytest.raises(TypeError, match=msg):
            getattr(td, opname)()

        with pytest.raises(TypeError, match=msg):
            getattr(td.to_frame(), opname)(numeric_only=False)

    def test_minmax_tz(self, tz_naive_fixture):
        tz = tz_naive_fixture
        # monotonic
        idx1 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], tz=tz)
        assert idx1.is_monotonic

        # non-monotonic
        idx2 = DatetimeIndex(
            ["2011-01-01", NaT, "2011-01-03", "2011-01-02", NaT], tz=tz
        )
        assert not idx2.is_monotonic

        for idx in [idx1, idx2]:
            assert idx.min() == Timestamp("2011-01-01", tz=tz)
            assert idx.max() == Timestamp("2011-01-03", tz=tz)
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax_nat_datetime64(self, op):
        # Return NaT
        obj = DatetimeIndex([])
        assert isna(getattr(obj, op)())

        obj = DatetimeIndex([NaT])
        assert isna(getattr(obj, op)())

        obj = DatetimeIndex([NaT, NaT, NaT])
        assert isna(getattr(obj, op)())

    def test_numpy_minmax_integer(self):
        # GH#26125
        idx = Index([1, 2, 3])

        expected = idx.values.max()
        result = np.max(idx)
        assert result == expected

        expected = idx.values.min()
        result = np.min(idx)
        assert result == expected

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)

        expected = idx.values.argmax()
        result = np.argmax(idx)
        assert result == expected

        expected = idx.values.argmin()
        result = np.argmin(idx)
        assert result == expected

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(idx, out=0)

    def test_numpy_minmax_range(self):
        # GH#26125
        idx = RangeIndex(0, 10, 3)

        expected = idx._int64index.max()
        result = np.max(idx)
        assert result == expected

        expected = idx._int64index.min()
        result = np.min(idx)
        assert result == expected

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)

        # No need to test again argmax/argmin compat since the implementation
        # is the same as basic integer index

    def test_numpy_minmax_datetime64(self):
        dr = date_range(start="2016-01-15", end="2016-01-20")

        assert np.min(dr) == Timestamp("2016-01-15 00:00:00")
        assert np.max(dr) == Timestamp("2016-01-20 00:00:00")

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(dr, out=0)

        with pytest.raises(ValueError, match=errmsg):
            np.max(dr, out=0)

        assert np.argmin(dr) == 0
        assert np.argmax(dr) == 5

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(dr, out=0)

        with pytest.raises(ValueError, match=errmsg):
            np.argmax(dr, out=0)

    def test_minmax_period(self):

        # monotonic
        idx1 = PeriodIndex([NaT, "2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        assert not idx1.is_monotonic
        assert idx1[1:].is_monotonic

        # non-monotonic
        idx2 = PeriodIndex(
            ["2011-01-01", NaT, "2011-01-03", "2011-01-02", NaT], freq="D"
        )
        assert not idx2.is_monotonic

        for idx in [idx1, idx2]:
            assert idx.min() == Period("2011-01-01", freq="D")
            assert idx.max() == Period("2011-01-03", freq="D")
        assert idx1.argmin() == 1
        assert idx2.argmin() == 0
        assert idx1.argmax() == 3
        assert idx2.argmax() == 2

        for op in ["min", "max"]:
            # Return NaT
            obj = PeriodIndex([], freq="M")
            result = getattr(obj, op)()
            assert result is NaT

            obj = PeriodIndex([NaT], freq="M")
            result = getattr(obj, op)()
            assert result is NaT

            obj = PeriodIndex([NaT, NaT, NaT], freq="M")
            result = getattr(obj, op)()
            assert result is NaT

    def test_numpy_minmax_period(self):
        pr = pd.period_range(start="2016-01-15", end="2016-01-20")

        assert np.min(pr) == Period("2016-01-15", freq="D")
        assert np.max(pr) == Period("2016-01-20", freq="D")

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(pr, out=0)

        assert np.argmin(pr) == 0
        assert np.argmax(pr) == 5

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(pr, out=0)

    def test_min_max_categorical(self):

        ci = pd.CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)
        msg = (
            r"Categorical is not ordered for operation min\n"
            r"you can use .as_ordered\(\) to change the Categorical to an ordered one\n"
        )
        with pytest.raises(TypeError, match=msg):
            ci.min()
        msg = (
            r"Categorical is not ordered for operation max\n"
            r"you can use .as_ordered\(\) to change the Categorical to an ordered one\n"
        )
        with pytest.raises(TypeError, match=msg):
            ci.max()

        ci = pd.CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=True)
        assert ci.min() == "c"
        assert ci.max() == "b"


class TestSeriesReductions:
    # Note: the name TestSeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def test_sum_inf(self):
        s = Series(np.random.randn(10))
        s2 = s.copy()

        s[5:8] = np.inf
        s2[5:8] = np.nan

        assert np.isinf(s.sum())

        arr = np.random.randn(100, 100).astype("f4")
        arr[:, 2] = np.inf

        with pd.option_context("mode.use_inf_as_na", True):
            tm.assert_almost_equal(s.sum(), s2.sum())

        res = nanops.nansum(arr, axis=1)
        assert np.isinf(res).all()

    @pytest.mark.parametrize("dtype", ["float64", "Int64", "boolean", "object"])
    @pytest.mark.parametrize("use_bottleneck", [True, False])
    @pytest.mark.parametrize("method, unit", [("sum", 0.0), ("prod", 1.0)])
    def test_empty(self, method, unit, use_bottleneck, dtype):
        with pd.option_context("use_bottleneck", use_bottleneck):
            # GH#9422 / GH#18921
            # Entirely empty
            s = Series([], dtype=dtype)
            # NA by default
            result = getattr(s, method)()
            assert result == unit

            # Explicit
            result = getattr(s, method)(min_count=0)
            assert result == unit

            result = getattr(s, method)(min_count=1)
            assert isna(result)

            # Skipna, default
            result = getattr(s, method)(skipna=True)
            result == unit

            # Skipna, explicit
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == unit

            result = getattr(s, method)(skipna=True, min_count=1)
            assert isna(result)

            result = getattr(s, method)(skipna=False, min_count=0)
            assert result == unit

            result = getattr(s, method)(skipna=False, min_count=1)
            assert isna(result)

            # All-NA
            s = Series([np.nan], dtype=dtype)
            # NA by default
            result = getattr(s, method)()
            assert result == unit

            # Explicit
            result = getattr(s, method)(min_count=0)
            assert result == unit

            result = getattr(s, method)(min_count=1)
            assert isna(result)

            # Skipna, default
            result = getattr(s, method)(skipna=True)
            result == unit

            # skipna, explicit
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == unit

            result = getattr(s, method)(skipna=True, min_count=1)
            assert isna(result)

            # Mix of valid, empty
            s = Series([np.nan, 1], dtype=dtype)
            # Default
            result = getattr(s, method)()
            assert result == 1.0

            # Explicit
            result = getattr(s, method)(min_count=0)
            assert result == 1.0

            result = getattr(s, method)(min_count=1)
            assert result == 1.0

            # Skipna
            result = getattr(s, method)(skipna=True)
            assert result == 1.0

            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == 1.0

            # GH#844 (changed in GH#9422)
            df = DataFrame(np.empty((10, 0)), dtype=dtype)
            assert (getattr(df, method)(1) == unit).all()

            s = Series([1], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)

            result = getattr(s, method)(skipna=False, min_count=2)
            assert isna(result)

            s = Series([np.nan], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)

            s = Series([np.nan, 1], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)

    @pytest.mark.parametrize("method, unit", [("sum", 0.0), ("prod", 1.0)])
    def test_empty_multi(self, method, unit):
        s = Series(
            [1, np.nan, np.nan, np.nan],
            index=pd.MultiIndex.from_product([("a", "b"), (0, 1)]),
        )
        # 1 / 0 by default
        with tm.assert_produces_warning(FutureWarning):
            result = getattr(s, method)(level=0)
        expected = Series([1, unit], index=["a", "b"])
        tm.assert_series_equal(result, expected)

        # min_count=0
        with tm.assert_produces_warning(FutureWarning):
            result = getattr(s, method)(level=0, min_count=0)
        expected = Series([1, unit], index=["a", "b"])
        tm.assert_series_equal(result, expected)

        # min_count=1
        with tm.assert_produces_warning(FutureWarning):
            result = getattr(s, method)(level=0, min_count=1)
        expected = Series([1, np.nan], index=["a", "b"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["mean"])
    @pytest.mark.parametrize("dtype", ["Float64", "Int64", "boolean"])
    def test_ops_consistency_on_empty_nullable(self, method, dtype):

        # GH#34814
        # consistency for nullable dtypes on empty or ALL-NA mean

        # empty series
        eser = Series([], dtype=dtype)
        result = getattr(eser, method)()
        assert result is pd.NA

        # ALL-NA series
        nser = Series([np.nan], dtype=dtype)
        result = getattr(nser, method)()
        assert result is pd.NA

    @pytest.mark.parametrize("method", ["mean", "median", "std", "var"])
    def test_ops_consistency_on_empty(self, method):

        # GH#7869
        # consistency on empty

        # float
        result = getattr(Series(dtype=float), method)()
        assert isna(result)

        # timedelta64[ns]
        tdser = Series([], dtype="m8[ns]")
        if method == "var":
            msg = "|".join(
                [
                    "operation 'var' not allowed",
                    r"cannot perform var with type timedelta64\[ns\]",
                    "'TimedeltaArray' does not implement reduction 'var'",
                ]
            )
            with pytest.raises(TypeError, match=msg):
                getattr(tdser, method)()
        else:
            result = getattr(tdser, method)()
            assert result is NaT

    def test_nansum_buglet(self):
        ser = Series([1.0, np.nan], index=[0, 1])
        result = np.nansum(ser)
        tm.assert_almost_equal(result, 1)

    @pytest.mark.parametrize("use_bottleneck", [True, False])
    def test_sum_overflow(self, use_bottleneck):

        with pd.option_context("use_bottleneck", use_bottleneck):
            # GH#6915
            # overflowing on the smaller int dtypes
            for dtype in ["int32", "int64"]:
                v = np.arange(5000000, dtype=dtype)
                s = Series(v)

                result = s.sum(skipna=False)
                assert int(result) == v.sum(dtype="int64")
                result = s.min(skipna=False)
                assert int(result) == 0
                result = s.max(skipna=False)
                assert int(result) == v[-1]

            for dtype in ["float32", "float64"]:
                v = np.arange(5000000, dtype=dtype)
                s = Series(v)

                result = s.sum(skipna=False)
                assert result == v.sum(dtype=dtype)
                result = s.min(skipna=False)
                assert np.allclose(float(result), 0.0)
                result = s.max(skipna=False)
                assert np.allclose(float(result), v[-1])

    def test_empty_timeseries_reductions_return_nat(self):
        # covers GH#11245
        for dtype in ("m8[ns]", "m8[ns]", "M8[ns]", "M8[ns, UTC]"):
            assert Series([], dtype=dtype).min() is NaT
            assert Series([], dtype=dtype).max() is NaT
            assert Series([], dtype=dtype).min(skipna=False) is NaT
            assert Series([], dtype=dtype).max(skipna=False) is NaT

    def test_numpy_argmin(self):
        # See GH#16830
        data = np.arange(1, 11)

        s = Series(data, index=data)
        result = np.argmin(s)

        expected = np.argmin(data)
        assert result == expected

        result = s.argmin()

        assert result == expected

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argmin(s, out=data)

    def test_numpy_argmax(self):
        # See GH#16830
        data = np.arange(1, 11)

        s = Series(data, index=data)
        result = np.argmax(s)
        expected = np.argmax(data)
        assert result == expected

        result = s.argmax()

        assert result == expected

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argmax(s, out=data)

    def test_idxmin(self):
        # test idxmin
        # _check_stat_op approach can not be used here because of isna check.
        string_series = tm.makeStringSeries().rename("series")

        # add some NaNs
        string_series[5:15] = np.NaN

        # skipna or no
        assert string_series[string_series.idxmin()] == string_series.min()
        assert isna(string_series.idxmin(skipna=False))

        # no NaNs
        nona = string_series.dropna()
        assert nona[nona.idxmin()] == nona.min()
        assert nona.index.values.tolist().index(nona.idxmin()) == nona.values.argmin()

        # all NaNs
        allna = string_series * np.nan
        assert isna(allna.idxmin())

        # datetime64[ns]
        s = Series(date_range("20130102", periods=6))
        result = s.idxmin()
        assert result == 0

        s[0] = np.nan
        result = s.idxmin()
        assert result == 1

    def test_idxmax(self):
        # test idxmax
        # _check_stat_op approach can not be used here because of isna check.
        string_series = tm.makeStringSeries().rename("series")

        # add some NaNs
        string_series[5:15] = np.NaN

        # skipna or no
        assert string_series[string_series.idxmax()] == string_series.max()
        assert isna(string_series.idxmax(skipna=False))

        # no NaNs
        nona = string_series.dropna()
        assert nona[nona.idxmax()] == nona.max()
        assert nona.index.values.tolist().index(nona.idxmax()) == nona.values.argmax()

        # all NaNs
        allna = string_series * np.nan
        assert isna(allna.idxmax())

        from pandas import date_range

        s = Series(date_range("20130102", periods=6))
        result = s.idxmax()
        assert result == 5

        s[5] = np.nan
        result = s.idxmax()
        assert result == 4

        # Float64Index
        # GH#5914
        s = Series([1, 2, 3], [1.1, 2.1, 3.1])
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1

        s = Series(s.index, s.index)
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1

    def test_all_any(self):
        ts = tm.makeTimeSeries()
        bool_series = ts > 0
        assert not bool_series.all()
        assert bool_series.any()

        # Alternative types, with implicit 'object' dtype.
        s = Series(["abc", True])
        assert s.any()

    @pytest.mark.parametrize("klass", [Index, Series])
    def test_numpy_all_any(self, klass):
        # GH#40180
        idx = klass([0, 1, 2])
        assert not np.all(idx)
        assert np.any(idx)
        idx = Index([1, 2, 3])
        assert np.all(idx)

    def test_all_any_params(self):
        # Check skipna, with implicit 'object' dtype.
        s1 = Series([np.nan, True])
        s2 = Series([np.nan, False])
        assert s1.all(skipna=False)  # nan && True => True
        assert s1.all(skipna=True)
        assert s2.any(skipna=False)
        assert not s2.any(skipna=True)

        # Check level.
        s = Series([False, False, True, True, False, True], index=[0, 0, 1, 1, 2, 2])
        with tm.assert_produces_warning(FutureWarning):
            tm.assert_series_equal(s.all(level=0), Series([False, True, False]))
        with tm.assert_produces_warning(FutureWarning):
            tm.assert_series_equal(s.any(level=0), Series([False, True, True]))

        msg = "Option bool_only is not implemented with option level"
        with pytest.raises(NotImplementedError, match=msg):
            with tm.assert_produces_warning(FutureWarning):
                s.any(bool_only=True, level=0)
        with pytest.raises(NotImplementedError, match=msg):
            with tm.assert_produces_warning(FutureWarning):
                s.all(bool_only=True, level=0)

        # bool_only is not implemented alone.
        # TODO GH38810 change this error message to:
        # "Series.any does not implement bool_only"
        msg = "Series.any does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            s.any(bool_only=True)
        msg = "Series.all does not implement numeric_only."
        with pytest.raises(NotImplementedError, match=msg):
            s.all(bool_only=True)

    @pytest.mark.parametrize("bool_agg_func", ["any", "all"])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_any_all_object_dtype(self, bool_agg_func, skipna):
        # GH#12863
        ser = Series(["a", "b", "c", "d", "e"], dtype=object)
        result = getattr(ser, bool_agg_func)(skipna=skipna)
        expected = True

        assert result == expected

    @pytest.mark.parametrize("bool_agg_func", ["any", "all"])
    @pytest.mark.parametrize(
        "data", [[False, None], [None, False], [False, np.nan], [np.nan, False]]
    )
    def test_any_all_object_dtype_missing(self, data, bool_agg_func):
        # GH#27709
        ser = Series(data)
        result = getattr(ser, bool_agg_func)(skipna=False)

        # None is treated is False, but np.nan is treated as True
        expected = bool_agg_func == "any" and None not in data
        assert result == expected

    @pytest.mark.parametrize("bool_agg_func", ["any", "all"])
    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize(
        # expected_data indexed as [[skipna=False/any, skipna=False/all],
        #                           [skipna=True/any, skipna=True/all]]
        "data,expected_data",
        [
            ([False, False, False], [[False, False], [False, False]]),
            ([True, True, True], [[True, True], [True, True]]),
            ([pd.NA, pd.NA, pd.NA], [[pd.NA, pd.NA], [False, True]]),
            ([False, pd.NA, False], [[pd.NA, False], [False, False]]),
            ([True, pd.NA, True], [[True, pd.NA], [True, True]]),
            ([True, pd.NA, False], [[True, False], [True, False]]),
        ],
    )
    def test_any_all_boolean_kleene_logic(
        self, bool_agg_func, skipna, data, expected_data
    ):
        ser = Series(data, dtype="boolean")
        expected = expected_data[skipna][bool_agg_func == "all"]

        result = getattr(ser, bool_agg_func)(skipna=skipna)
        assert (result is pd.NA and expected is pd.NA) or result == expected

    @pytest.mark.parametrize(
        "bool_agg_func,expected",
        [("all", [False, True, False]), ("any", [False, True, True])],
    )
    def test_any_all_boolean_level(self, bool_agg_func, expected):
        # GH#33449
        ser = Series(
            [False, False, True, True, False, True],
            index=[0, 0, 1, 1, 2, 2],
            dtype="boolean",
        )
        with tm.assert_produces_warning(FutureWarning):
            result = getattr(ser, bool_agg_func)(level=0)
        expected = Series(expected, dtype="boolean")
        tm.assert_series_equal(result, expected)

    def test_any_axis1_bool_only(self):
        # GH#32432
        df = DataFrame({"A": [True, False], "B": [1, 2]})
        result = df.any(axis=1, bool_only=True)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    def test_any_all_datetimelike(self):
        # GH#38723 these may not be the desired long-term behavior (GH#34479)
        #  but in the interim should be internally consistent
        dta = date_range("1995-01-02", periods=3)._data
        ser = Series(dta)
        df = DataFrame(ser)

        assert dta.all()
        assert dta.any()

        assert ser.all()
        assert ser.any()

        assert df.any().all()
        assert df.all().all()

        dta = dta.tz_localize("UTC")
        ser = Series(dta)
        df = DataFrame(ser)

        assert dta.all()
        assert dta.any()

        assert ser.all()
        assert ser.any()

        assert df.any().all()
        assert df.all().all()

        tda = dta - dta[0]
        ser = Series(tda)
        df = DataFrame(ser)

        assert tda.any()
        assert not tda.all()

        assert ser.any()
        assert not ser.all()

        assert df.any().all()
        assert not df.all().any()

    def test_timedelta64_analytics(self):

        # index min/max
        dti = date_range("2012-1-1", periods=3, freq="D")
        td = Series(dti) - Timestamp("20120101")

        result = td.idxmin()
        assert result == 0

        result = td.idxmax()
        assert result == 2

        # GH#2982
        # with NaT
        td[0] = np.nan

        result = td.idxmin()
        assert result == 1

        result = td.idxmax()
        assert result == 2

        # abs
        s1 = Series(date_range("20120101", periods=3))
        s2 = Series(date_range("20120102", periods=3))
        expected = Series(s2 - s1)

        result = np.abs(s1 - s2)
        tm.assert_series_equal(result, expected)

        result = (s1 - s2).abs()
        tm.assert_series_equal(result, expected)

        # max/min
        result = td.max()
        expected = Timedelta("2 days")
        assert result == expected

        result = td.min()
        expected = Timedelta("1 days")
        assert result == expected

    @pytest.mark.parametrize(
        "test_input,error_type",
        [
            (Series([], dtype="float64"), ValueError),
            # For strings, or any Series with dtype 'O'
            (Series(["foo", "bar", "baz"]), TypeError),
            (Series([(1,), (2,)]), TypeError),
            # For mixed data types
            (Series(["foo", "foo", "bar", "bar", None, np.nan, "baz"]), TypeError),
        ],
    )
    def test_assert_idxminmax_raises(self, test_input, error_type):
        """
        Cases where ``Series.argmax`` and related should raise an exception
        """
        msg = (
            "reduction operation 'argmin' not allowed for this dtype|"
            "attempt to get argmin of an empty sequence"
        )
        with pytest.raises(error_type, match=msg):
            test_input.idxmin()
        with pytest.raises(error_type, match=msg):
            test_input.idxmin(skipna=False)
        msg = (
            "reduction operation 'argmax' not allowed for this dtype|"
            "attempt to get argmax of an empty sequence"
        )
        with pytest.raises(error_type, match=msg):
            test_input.idxmax()
        with pytest.raises(error_type, match=msg):
            test_input.idxmax(skipna=False)

    def test_idxminmax_with_inf(self):
        # For numeric data with NA and Inf (GH #13595)
        s = Series([0, -np.inf, np.inf, np.nan])

        assert s.idxmin() == 1
        assert np.isnan(s.idxmin(skipna=False))

        assert s.idxmax() == 2
        assert np.isnan(s.idxmax(skipna=False))

        # Using old-style behavior that treats floating point nan, -inf, and
        # +inf as missing
        with pd.option_context("mode.use_inf_as_na", True):
            assert s.idxmin() == 0
            assert np.isnan(s.idxmin(skipna=False))
            assert s.idxmax() == 0
            np.isnan(s.idxmax(skipna=False))


class TestDatetime64SeriesReductions:
    # Note: the name TestDatetime64SeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    @pytest.mark.parametrize(
        "nat_ser",
        [
            Series([NaT, NaT]),
            Series([NaT, Timedelta("nat")]),
            Series([Timedelta("nat"), Timedelta("nat")]),
        ],
    )
    def test_minmax_nat_series(self, nat_ser):
        # GH#23282
        assert nat_ser.min() is NaT
        assert nat_ser.max() is NaT
        assert nat_ser.min(skipna=False) is NaT
        assert nat_ser.max(skipna=False) is NaT

    @pytest.mark.parametrize(
        "nat_df",
        [
            DataFrame([NaT, NaT]),
            DataFrame([NaT, Timedelta("nat")]),
            DataFrame([Timedelta("nat"), Timedelta("nat")]),
        ],
    )
    def test_minmax_nat_dataframe(self, nat_df):
        # GH#23282
        assert nat_df.min()[0] is NaT
        assert nat_df.max()[0] is NaT
        assert nat_df.min(skipna=False)[0] is NaT
        assert nat_df.max(skipna=False)[0] is NaT

    def test_min_max(self):
        rng = date_range("1/1/2000", "12/31/2000")
        rng2 = rng.take(np.random.permutation(len(rng)))

        the_min = rng2.min()
        the_max = rng2.max()
        assert isinstance(the_min, Timestamp)
        assert isinstance(the_max, Timestamp)
        assert the_min == rng[0]
        assert the_max == rng[-1]

        assert rng.min() == rng[0]
        assert rng.max() == rng[-1]

    def test_min_max_series(self):
        rng = date_range("1/1/2000", periods=10, freq="4h")
        lvls = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]
        df = DataFrame({"TS": rng, "V": np.random.randn(len(rng)), "L": lvls})

        result = df.TS.max()
        exp = Timestamp(df.TS.iat[-1])
        assert isinstance(result, Timestamp)
        assert result == exp

        result = df.TS.min()
        exp = Timestamp(df.TS.iat[0])
        assert isinstance(result, Timestamp)
        assert result == exp


class TestCategoricalSeriesReductions:
    # Note: the name TestCategoricalSeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_unordered_raises(self, function):
        # unordered cats have no min/max
        cat = Series(Categorical(["a", "b", "c", "d"], ordered=False))
        msg = f"Categorical is not ordered for operation {function}"
        with pytest.raises(TypeError, match=msg):
            getattr(cat, function)()

    @pytest.mark.parametrize(
        "values, categories",
        [
            (list("abc"), list("abc")),
            (list("abc"), list("cba")),
            (list("abc") + [np.nan], list("cba")),
            ([1, 2, 3], [3, 2, 1]),
            ([1, 2, 3, np.nan], [3, 2, 1]),
        ],
    )
    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_ordered(self, values, categories, function):
        # GH 25303
        cat = Series(Categorical(values, categories=categories, ordered=True))
        result = getattr(cat, function)(skipna=True)
        expected = categories[0] if function == "min" else categories[2]
        assert result == expected

    @pytest.mark.parametrize("function", ["min", "max"])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_min_max_ordered_with_nan_only(self, function, skipna):
        # https://github.com/pandas-dev/pandas/issues/33450
        cat = Series(Categorical([np.nan], categories=[1, 2], ordered=True))
        result = getattr(cat, function)(skipna=skipna)
        assert result is np.nan

    @pytest.mark.parametrize("function", ["min", "max"])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_min_max_skipna(self, function, skipna):
        cat = Series(
            Categorical(["a", "b", np.nan, "a"], categories=["b", "a"], ordered=True)
        )
        result = getattr(cat, function)(skipna=skipna)

        if skipna is True:
            expected = "b" if function == "min" else "a"
            assert result == expected
        else:
            assert result is np.nan


class TestSeriesMode:
    # Note: the name TestSeriesMode indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    @pytest.mark.parametrize(
        "dropna, expected",
        [(True, Series([], dtype=np.float64)), (False, Series([], dtype=np.float64))],
    )
    def test_mode_empty(self, dropna, expected):
        s = Series([], dtype=np.float64)
        result = s.mode(dropna)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dropna, data, expected",
        [
            (True, [1, 1, 1, 2], [1]),
            (True, [1, 1, 1, 2, 3, 3, 3], [1, 3]),
            (False, [1, 1, 1, 2], [1]),
            (False, [1, 1, 1, 2, 3, 3, 3], [1, 3]),
        ],
    )
    @pytest.mark.parametrize(
        "dt", list(np.typecodes["AllInteger"] + np.typecodes["Float"])
    )
    def test_mode_numerical(self, dropna, data, expected, dt):
        s = Series(data, dtype=dt)
        result = s.mode(dropna)
        expected = Series(expected, dtype=dt)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dropna, expected", [(True, [1.0]), (False, [1, np.nan])])
    def test_mode_numerical_nan(self, dropna, expected):
        s = Series([1, 1, 2, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series(expected)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2, expected3",
        [(True, ["b"], ["bar"], ["nan"]), (False, ["b"], [np.nan], ["nan"])],
    )
    def test_mode_str_obj(self, dropna, expected1, expected2, expected3):
        # Test string and object types.
        data = ["a"] * 2 + ["b"] * 3

        s = Series(data, dtype="c")
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="c")
        tm.assert_series_equal(result, expected1)

        data = ["foo", "bar", "bar", np.nan, np.nan, np.nan]

        s = Series(data, dtype=object)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=object)
        tm.assert_series_equal(result, expected2)

        data = ["foo", "bar", "bar", np.nan, np.nan, np.nan]

        s = Series(data, dtype=object).astype(str)
        result = s.mode(dropna)
        expected3 = Series(expected3, dtype=str)
        tm.assert_series_equal(result, expected3)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [(True, ["foo"], ["foo"]), (False, ["foo"], [np.nan])],
    )
    def test_mode_mixeddtype(self, dropna, expected1, expected2):
        s = Series([1, "foo", "foo"])
        result = s.mode(dropna)
        expected = Series(expected1)
        tm.assert_series_equal(result, expected)

        s = Series([1, "foo", "foo", np.nan, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series(expected2, dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [
            (
                True,
                ["1900-05-03", "2011-01-03", "2013-01-02"],
                ["2011-01-03", "2013-01-02"],
            ),
            (False, [np.nan], [np.nan, "2011-01-03", "2013-01-02"]),
        ],
    )
    def test_mode_datetime(self, dropna, expected1, expected2):
        s = Series(
            ["2011-01-03", "2013-01-02", "1900-05-03", "nan", "nan"], dtype="M8[ns]"
        )
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="M8[ns]")
        tm.assert_series_equal(result, expected1)

        s = Series(
            [
                "2011-01-03",
                "2013-01-02",
                "1900-05-03",
                "2011-01-03",
                "2013-01-02",
                "nan",
                "nan",
            ],
            dtype="M8[ns]",
        )
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype="M8[ns]")
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [
            (True, ["-1 days", "0 days", "1 days"], ["2 min", "1 day"]),
            (False, [np.nan], [np.nan, "2 min", "1 day"]),
        ],
    )
    def test_mode_timedelta(self, dropna, expected1, expected2):
        # gh-5986: Test timedelta types.

        s = Series(
            ["1 days", "-1 days", "0 days", "nan", "nan"], dtype="timedelta64[ns]"
        )
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="timedelta64[ns]")
        tm.assert_series_equal(result, expected1)

        s = Series(
            [
                "1 day",
                "1 day",
                "-1 day",
                "-1 day 2 min",
                "2 min",
                "2 min",
                "nan",
                "nan",
            ],
            dtype="timedelta64[ns]",
        )
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype="timedelta64[ns]")
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2, expected3",
        [
            (
                True,
                Categorical([1, 2], categories=[1, 2]),
                Categorical(["a"], categories=[1, "a"]),
                Categorical([3, 1], categories=[3, 2, 1], ordered=True),
            ),
            (
                False,
                Categorical([np.nan], categories=[1, 2]),
                Categorical([np.nan, "a"], categories=[1, "a"]),
                Categorical([np.nan, 3, 1], categories=[3, 2, 1], ordered=True),
            ),
        ],
    )
    def test_mode_category(self, dropna, expected1, expected2, expected3):
        s = Series(Categorical([1, 2, np.nan, np.nan]))
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype="category")
        tm.assert_series_equal(result, expected1)

        s = Series(Categorical([1, "a", "a", np.nan, np.nan]))
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype="category")
        tm.assert_series_equal(result, expected2)

        s = Series(
            Categorical(
                [1, 1, 2, 3, 3, np.nan, np.nan], categories=[3, 2, 1], ordered=True
            )
        )
        result = s.mode(dropna)
        expected3 = Series(expected3, dtype="category")
        tm.assert_series_equal(result, expected3)

    @pytest.mark.parametrize(
        "dropna, expected1, expected2",
        [(True, [2 ** 63], [1, 2 ** 63]), (False, [2 ** 63], [1, 2 ** 63])],
    )
    def test_mode_intoverflow(self, dropna, expected1, expected2):
        # Test for uint64 overflow.
        s = Series([1, 2 ** 63, 2 ** 63], dtype=np.uint64)
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype=np.uint64)
        tm.assert_series_equal(result, expected1)

        s = Series([1, 2 ** 63], dtype=np.uint64)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=np.uint64)
        tm.assert_series_equal(result, expected2)

    def test_mode_sortwarning(self):
        # Check for the warning that is raised when the mode
        # results cannot be sorted

        expected = Series(["foo", np.nan])
        s = Series([1, "foo", "foo", np.nan, np.nan])

        with tm.assert_produces_warning(UserWarning):
            result = s.mode(dropna=False)
            result = result.sort_values().reset_index(drop=True)

        tm.assert_series_equal(result, expected)

    def test_mode_boolean_with_na(self):
        # GH#42107
        ser = Series([True, False, True, pd.NA], dtype="boolean")
        result = ser.mode()
        expected = Series({0: True}, dtype="boolean")
        tm.assert_series_equal(result, expected)
