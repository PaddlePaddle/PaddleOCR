import numpy as np
import pytest

from pandas import (
    Series,
    TimedeltaIndex,
    date_range,
)
import pandas._testing as tm


class TestSeriesDiff:
    def test_diff_np(self):
        pytest.skip("skipping due to Series no longer being an ndarray")

        # no longer works as the return type of np.diff is now nd.array
        s = Series(np.arange(5))

        r = np.diff(s)
        tm.assert_series_equal(Series([np.nan, 0, 0, 0, np.nan]), r)

    def test_diff_int(self):
        # int dtype
        a = 10000000000000000
        b = a + 1
        s = Series([a, b])

        result = s.diff()
        assert result[1] == 1

    def test_diff_tz(self):
        # Combined datetime diff, normal diff and boolean diff test
        ts = tm.makeTimeSeries(name="ts")
        ts.diff()

        # neg n
        result = ts.diff(-1)
        expected = ts - ts.shift(-1)
        tm.assert_series_equal(result, expected)

        # 0
        result = ts.diff(0)
        expected = ts - ts
        tm.assert_series_equal(result, expected)

        # datetime diff (GH#3100)
        s = Series(date_range("20130102", periods=5))
        result = s.diff()
        expected = s - s.shift(1)
        tm.assert_series_equal(result, expected)

        # timedelta diff
        result = result - result.shift(1)  # previous result
        expected = expected.diff()  # previously expected
        tm.assert_series_equal(result, expected)

        # with tz
        s = Series(
            date_range("2000-01-01 09:00:00", periods=5, tz="US/Eastern"), name="foo"
        )
        result = s.diff()
        expected = Series(TimedeltaIndex(["NaT"] + ["1 days"] * 4), name="foo")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input,output,diff",
        [([False, True, True, False, False], [np.nan, True, False, True, False], 1)],
    )
    def test_diff_bool(self, input, output, diff):
        # boolean series (test for fixing #17294)
        s = Series(input)
        result = s.diff()
        expected = Series(output)
        tm.assert_series_equal(result, expected)

    def test_diff_object_dtype(self):
        # object series
        s = Series([False, True, 5.0, np.nan, True, False])
        result = s.diff()
        expected = s - s.shift(1)
        tm.assert_series_equal(result, expected)
