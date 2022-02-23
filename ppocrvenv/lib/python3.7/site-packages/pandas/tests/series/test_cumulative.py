"""
Tests for Series cumulative operations.

See also
--------
tests.frame.test_cumulative
"""
from itertools import product

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def _check_accum_op(name, series, check_dtype=True):
    func = getattr(np, name)
    tm.assert_numpy_array_equal(
        func(series).values, func(np.array(series)), check_dtype=check_dtype
    )

    # with missing values
    ts = series.copy()
    ts[::2] = np.NaN

    result = func(ts)[1::2]
    expected = func(np.array(ts.dropna()))

    tm.assert_numpy_array_equal(result.values, expected, check_dtype=False)


class TestSeriesCumulativeOps:
    def test_cumsum(self, datetime_series):
        _check_accum_op("cumsum", datetime_series)

    def test_cumprod(self, datetime_series):
        _check_accum_op("cumprod", datetime_series)

    def test_cummin(self, datetime_series):
        tm.assert_numpy_array_equal(
            datetime_series.cummin().values,
            np.minimum.accumulate(np.array(datetime_series)),
        )
        ts = datetime_series.copy()
        ts[::2] = np.NaN
        result = ts.cummin()[1::2]
        expected = np.minimum.accumulate(ts.dropna())

        result.index = result.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    def test_cummax(self, datetime_series):
        tm.assert_numpy_array_equal(
            datetime_series.cummax().values,
            np.maximum.accumulate(np.array(datetime_series)),
        )
        ts = datetime_series.copy()
        ts[::2] = np.NaN
        result = ts.cummax()[1::2]
        expected = np.maximum.accumulate(ts.dropna())

        result.index = result.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "US/Pacific"])
    def test_cummin_datetime64(self, tz):
        s = pd.Series(
            pd.to_datetime(
                ["NaT", "2000-1-2", "NaT", "2000-1-1", "NaT", "2000-1-3"]
            ).tz_localize(tz)
        )

        expected = pd.Series(
            pd.to_datetime(
                ["NaT", "2000-1-2", "NaT", "2000-1-1", "NaT", "2000-1-1"]
            ).tz_localize(tz)
        )
        result = s.cummin(skipna=True)
        tm.assert_series_equal(expected, result)

        expected = pd.Series(
            pd.to_datetime(
                ["NaT", "2000-1-2", "2000-1-2", "2000-1-1", "2000-1-1", "2000-1-1"]
            ).tz_localize(tz)
        )
        result = s.cummin(skipna=False)
        tm.assert_series_equal(expected, result)

    @pytest.mark.parametrize("tz", [None, "US/Pacific"])
    def test_cummax_datetime64(self, tz):
        s = pd.Series(
            pd.to_datetime(
                ["NaT", "2000-1-2", "NaT", "2000-1-1", "NaT", "2000-1-3"]
            ).tz_localize(tz)
        )

        expected = pd.Series(
            pd.to_datetime(
                ["NaT", "2000-1-2", "NaT", "2000-1-2", "NaT", "2000-1-3"]
            ).tz_localize(tz)
        )
        result = s.cummax(skipna=True)
        tm.assert_series_equal(expected, result)

        expected = pd.Series(
            pd.to_datetime(
                ["NaT", "2000-1-2", "2000-1-2", "2000-1-2", "2000-1-2", "2000-1-3"]
            ).tz_localize(tz)
        )
        result = s.cummax(skipna=False)
        tm.assert_series_equal(expected, result)

    def test_cummin_timedelta64(self):
        s = pd.Series(pd.to_timedelta(["NaT", "2 min", "NaT", "1 min", "NaT", "3 min"]))

        expected = pd.Series(
            pd.to_timedelta(["NaT", "2 min", "NaT", "1 min", "NaT", "1 min"])
        )
        result = s.cummin(skipna=True)
        tm.assert_series_equal(expected, result)

        expected = pd.Series(
            pd.to_timedelta(["NaT", "2 min", "2 min", "1 min", "1 min", "1 min"])
        )
        result = s.cummin(skipna=False)
        tm.assert_series_equal(expected, result)

    def test_cummax_timedelta64(self):
        s = pd.Series(pd.to_timedelta(["NaT", "2 min", "NaT", "1 min", "NaT", "3 min"]))

        expected = pd.Series(
            pd.to_timedelta(["NaT", "2 min", "NaT", "2 min", "NaT", "3 min"])
        )
        result = s.cummax(skipna=True)
        tm.assert_series_equal(expected, result)

        expected = pd.Series(
            pd.to_timedelta(["NaT", "2 min", "2 min", "2 min", "2 min", "3 min"])
        )
        result = s.cummax(skipna=False)
        tm.assert_series_equal(expected, result)

    def test_cummethods_bool(self):
        # GH#6270

        a = pd.Series([False, False, False, True, True, False, False])
        b = ~a
        c = pd.Series([False] * len(b))
        d = ~c
        methods = {
            "cumsum": np.cumsum,
            "cumprod": np.cumprod,
            "cummin": np.minimum.accumulate,
            "cummax": np.maximum.accumulate,
        }
        args = product((a, b, c, d), methods)
        for s, method in args:
            expected = pd.Series(methods[method](s.values))
            result = getattr(s, method)()
            tm.assert_series_equal(result, expected)

        e = pd.Series([False, True, np.nan, False])
        cse = pd.Series([0, 1, np.nan, 1], dtype=object)
        cpe = pd.Series([False, 0, np.nan, 0])
        cmin = pd.Series([False, False, np.nan, False])
        cmax = pd.Series([False, True, np.nan, True])
        expecteds = {"cumsum": cse, "cumprod": cpe, "cummin": cmin, "cummax": cmax}

        for method in methods:
            res = getattr(e, method)()
            tm.assert_series_equal(res, expecteds[method])
