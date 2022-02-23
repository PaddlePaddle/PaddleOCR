"""
Note: includes tests for `last`
"""
import pytest

from pandas import (
    DataFrame,
    bdate_range,
)
import pandas._testing as tm


class TestFirst:
    def test_first_subset(self, frame_or_series):
        ts = tm.makeTimeDataFrame(freq="12h")
        if frame_or_series is not DataFrame:
            ts = ts["A"]
        result = ts.first("10d")
        assert len(result) == 20

        ts = tm.makeTimeDataFrame(freq="D")
        if frame_or_series is not DataFrame:
            ts = ts["A"]
        result = ts.first("10d")
        assert len(result) == 10

        result = ts.first("3M")
        expected = ts[:"3/31/2000"]
        tm.assert_equal(result, expected)

        result = ts.first("21D")
        expected = ts[:21]
        tm.assert_equal(result, expected)

        result = ts[:0].first("3M")
        tm.assert_equal(result, ts[:0])

    def test_first_last_raises(self, frame_or_series):
        # GH#20725
        obj = DataFrame([[1, 2, 3], [4, 5, 6]])
        if frame_or_series is not DataFrame:
            obj = obj[0]

        msg = "'first' only supports a DatetimeIndex index"
        with pytest.raises(TypeError, match=msg):  # index is not a DatetimeIndex
            obj.first("1D")

        msg = "'last' only supports a DatetimeIndex index"
        with pytest.raises(TypeError, match=msg):  # index is not a DatetimeIndex
            obj.last("1D")

    def test_last_subset(self, frame_or_series):
        ts = tm.makeTimeDataFrame(freq="12h")
        if frame_or_series is not DataFrame:
            ts = ts["A"]
        result = ts.last("10d")
        assert len(result) == 20

        ts = tm.makeTimeDataFrame(nper=30, freq="D")
        if frame_or_series is not DataFrame:
            ts = ts["A"]
        result = ts.last("10d")
        assert len(result) == 10

        result = ts.last("21D")
        expected = ts["2000-01-10":]
        tm.assert_equal(result, expected)

        result = ts.last("21D")
        expected = ts[-21:]
        tm.assert_equal(result, expected)

        result = ts[:0].last("3M")
        tm.assert_equal(result, ts[:0])

    @pytest.mark.parametrize("start, periods", [("2010-03-31", 1), ("2010-03-30", 2)])
    def test_first_with_first_day_last_of_month(self, frame_or_series, start, periods):
        # GH#29623
        x = frame_or_series([1] * 100, index=bdate_range(start, periods=100))
        result = x.first("1M")
        expected = frame_or_series(
            [1] * periods, index=bdate_range(start, periods=periods)
        )
        tm.assert_equal(result, expected)

    def test_first_with_first_day_end_of_frq_n_greater_one(self, frame_or_series):
        # GH#29623
        x = frame_or_series([1] * 100, index=bdate_range("2010-03-31", periods=100))
        result = x.first("2M")
        expected = frame_or_series(
            [1] * 23, index=bdate_range("2010-03-31", "2010-04-30")
        )
        tm.assert_equal(result, expected)
