from datetime import date

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm


class TestDatetimeIndex:
    def test_time_loc(self):  # GH8667
        from datetime import time

        from pandas._libs.index import _SIZE_CUTOFF

        ns = _SIZE_CUTOFF + np.array([-100, 100], dtype=np.int64)
        key = time(15, 11, 30)
        start = key.hour * 3600 + key.minute * 60 + key.second
        step = 24 * 3600

        for n in ns:
            idx = date_range("2014-11-26", periods=n, freq="S")
            ts = pd.Series(np.random.randn(n), index=idx)
            i = np.arange(start, n, step)

            tm.assert_numpy_array_equal(ts.index.get_loc(key), i, check_dtype=False)
            tm.assert_series_equal(ts[key], ts.iloc[i])

            left, right = ts.copy(), ts.copy()
            left[key] *= -10
            right.iloc[i] *= -10
            tm.assert_series_equal(left, right)

    def test_time_overflow_for_32bit_machines(self):
        # GH8943.  On some machines NumPy defaults to np.int32 (for example,
        # 32-bit Linux machines).  In the function _generate_regular_range
        # found in tseries/index.py, `periods` gets multiplied by `strides`
        # (which has value 1e9) and since the max value for np.int32 is ~2e9,
        # and since those machines won't promote np.int32 to np.int64, we get
        # overflow.
        periods = np.int_(1000)

        idx1 = date_range(start="2000", periods=periods, freq="S")
        assert len(idx1) == periods

        idx2 = date_range(end="2000", periods=periods, freq="S")
        assert len(idx2) == periods

    def test_nat(self):
        assert DatetimeIndex([np.nan])[0] is pd.NaT

    def test_week_of_month_frequency(self):
        # GH 5348: "ValueError: Could not evaluate WOM-1SUN" shouldn't raise
        d1 = date(2002, 9, 1)
        d2 = date(2013, 10, 27)
        d3 = date(2012, 9, 30)
        idx1 = DatetimeIndex([d1, d2])
        idx2 = DatetimeIndex([d3])
        result_append = idx1.append(idx2)
        expected = DatetimeIndex([d1, d2, d3])
        tm.assert_index_equal(result_append, expected)
        result_union = idx1.union(idx2)
        expected = DatetimeIndex([d1, d3, d2])
        tm.assert_index_equal(result_union, expected)

        # GH 5115
        result = date_range("2013-1-1", periods=4, freq="WOM-1SAT")
        dates = ["2013-01-05", "2013-02-02", "2013-03-02", "2013-04-06"]
        expected = DatetimeIndex(dates, freq="WOM-1SAT")
        tm.assert_index_equal(result, expected)

    def test_stringified_slice_with_tz(self):
        # GH#2658
        start = "2013-01-07"
        idx = date_range(start=start, freq="1d", periods=10, tz="US/Eastern")
        df = DataFrame(np.arange(10), index=idx)
        df["2013-01-14 23:44:34.437768-05:00":]  # no exception here

    def test_append_nondatetimeindex(self):
        rng = date_range("1/1/2000", periods=10)
        idx = Index(["a", "b", "c", "d"])

        result = rng.append(idx)
        assert isinstance(result[0], Timestamp)

    def test_iteration_preserves_tz(self):
        # see gh-8890
        index = date_range("2012-01-01", periods=3, freq="H", tz="US/Eastern")

        for i, ts in enumerate(index):
            result = ts
            expected = index[i]
            assert result == expected

        index = date_range(
            "2012-01-01", periods=3, freq="H", tz=dateutil.tz.tzoffset(None, -28800)
        )

        for i, ts in enumerate(index):
            result = ts
            expected = index[i]
            assert result._repr_base == expected._repr_base
            assert result == expected

        # 9100
        index = DatetimeIndex(
            ["2014-12-01 03:32:39.987000-08:00", "2014-12-01 04:12:34.987000-08:00"]
        )
        for i, ts in enumerate(index):
            result = ts
            expected = index[i]
            assert result._repr_base == expected._repr_base
            assert result == expected

    @pytest.mark.parametrize("periods", [0, 9999, 10000, 10001])
    def test_iteration_over_chunksize(self, periods):
        # GH21012

        index = date_range("2000-01-01 00:00:00", periods=periods, freq="min")
        num = 0
        for stamp in index:
            assert index[num] == stamp
            num += 1
        assert num == len(index)

    def test_misc_coverage(self):
        rng = date_range("1/1/2000", periods=5)
        result = rng.groupby(rng.day)
        assert isinstance(list(result.values())[0][0], Timestamp)

    def test_string_index_series_name_converted(self):
        # #1644
        df = DataFrame(np.random.randn(10, 4), index=date_range("1/1/2000", periods=10))

        result = df.loc["1/3/2000"]
        assert result.name == df.index[2]

        result = df.T["1/3/2000"]
        assert result.name == df.index[2]

    def test_groupby_function_tuple_1677(self):
        df = DataFrame(np.random.rand(100), index=date_range("1/1/2000", periods=100))
        monthly_group = df.groupby(lambda x: (x.year, x.month))

        result = monthly_group.mean()
        assert isinstance(result.index[0], tuple)

    def test_isin(self):
        index = tm.makeDateIndex(4)
        result = index.isin(index)
        assert result.all()

        result = index.isin(list(index))
        assert result.all()

        tm.assert_almost_equal(
            index.isin([index[2], 5]), np.array([False, False, True, False])
        )

    def assert_index_parameters(self, index):
        assert index.freq == "40960N"
        assert index.inferred_freq == "40960N"

    def test_ns_index(self):
        nsamples = 400
        ns = int(1e9 / 24414)
        dtstart = np.datetime64("2012-09-20T00:00:00")

        dt = dtstart + np.arange(nsamples) * np.timedelta64(ns, "ns")
        freq = ns * offsets.Nano()
        index = DatetimeIndex(dt, freq=freq, name="time")
        self.assert_index_parameters(index)

        new_index = date_range(start=index[0], end=index[-1], freq=index.freq)
        self.assert_index_parameters(new_index)

    def test_asarray_tz_naive(self):
        # This shouldn't produce a warning.
        idx = date_range("2000", periods=2)
        # M8[ns] by default
        result = np.asarray(idx)

        expected = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        tm.assert_numpy_array_equal(result, expected)

        # optionally, object
        result = np.asarray(idx, dtype=object)

        expected = np.array([Timestamp("2000-01-01"), Timestamp("2000-01-02")])
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_aware(self):
        tz = "US/Central"
        idx = date_range("2000", periods=2, tz=tz)
        expected = np.array(["2000-01-01T06", "2000-01-02T06"], dtype="M8[ns]")
        result = np.asarray(idx, dtype="datetime64[ns]")

        tm.assert_numpy_array_equal(result, expected)

        # Old behavior with no warning
        result = np.asarray(idx, dtype="M8[ns]")

        tm.assert_numpy_array_equal(result, expected)

        # Future behavior with no warning
        expected = np.array(
            [Timestamp("2000-01-01", tz=tz), Timestamp("2000-01-02", tz=tz)]
        )
        result = np.asarray(idx, dtype=object)

        tm.assert_numpy_array_equal(result, expected)
