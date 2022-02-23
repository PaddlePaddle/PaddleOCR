import numpy as np
import pytest

from pandas import (
    NaT,
    Period,
    PeriodIndex,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestPeriodRange:
    def test_required_arguments(self):
        msg = (
            "Of the three parameters: start, end, and periods, exactly two "
            "must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range("2011-1-1", "2012-1-1", "B")

    @pytest.mark.parametrize("freq", ["D", "W", "M", "Q", "A"])
    def test_construction_from_string(self, freq):
        # non-empty
        expected = date_range(
            start="2017-01-01", periods=5, freq=freq, name="foo"
        ).to_period()
        start, end = str(expected[0]), str(expected[-1])

        result = period_range(start=start, end=end, freq=freq, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=start, periods=5, freq=freq, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=5, freq=freq, name="foo")
        tm.assert_index_equal(result, expected)

        # empty
        expected = PeriodIndex([], freq=freq, name="foo")

        result = period_range(start=start, periods=0, freq=freq, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=0, freq=freq, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=end, end=start, freq=freq, name="foo")
        tm.assert_index_equal(result, expected)

    def test_construction_from_period(self):
        # upsampling
        start, end = Period("2017Q1", freq="Q"), Period("2018Q1", freq="Q")
        expected = date_range(
            start="2017-03-31", end="2018-03-31", freq="M", name="foo"
        ).to_period()
        result = period_range(start=start, end=end, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # downsampling
        start, end = Period("2017-1", freq="M"), Period("2019-12", freq="M")
        expected = date_range(
            start="2017-01-31", end="2019-12-31", freq="Q", name="foo"
        ).to_period()
        result = period_range(start=start, end=end, freq="Q", name="foo")
        tm.assert_index_equal(result, expected)

        # test for issue # 21793
        start, end = Period("2017Q1", freq="Q"), Period("2018Q1", freq="Q")
        idx = period_range(start=start, end=end, freq="Q", name="foo")
        result = idx == idx.values
        expected = np.array([True, True, True, True, True])
        tm.assert_numpy_array_equal(result, expected)

        # empty
        expected = PeriodIndex([], freq="W", name="foo")

        result = period_range(start=start, periods=0, freq="W", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=0, freq="W", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=end, end=start, freq="W", name="foo")
        tm.assert_index_equal(result, expected)

    def test_errors(self):
        # not enough params
        msg = (
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1")

        with pytest.raises(ValueError, match=msg):
            period_range(end="2017Q1")

        with pytest.raises(ValueError, match=msg):
            period_range(periods=5)

        with pytest.raises(ValueError, match=msg):
            period_range()

        # too many params
        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1", end="2018Q1", periods=8, freq="Q")

        # start/end NaT
        msg = "start and end must not be NaT"
        with pytest.raises(ValueError, match=msg):
            period_range(start=NaT, end="2018Q1")

        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1", end=NaT)

        # invalid periods param
        msg = "periods must be a number, got foo"
        with pytest.raises(TypeError, match=msg):
            period_range(start="2017Q1", periods="foo")
