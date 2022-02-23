from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    NaT,
    PeriodIndex,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestToTimestamp:
    def test_to_timestamp_freq(self):
        idx = period_range("2017", periods=12, freq="A-DEC")
        result = idx.to_timestamp()
        expected = date_range("2017", periods=12, freq="AS-JAN")
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_pi_nat(self):
        # GH#7228
        index = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="M", name="idx")

        result = index.to_timestamp("D")
        expected = DatetimeIndex(
            [NaT, datetime(2011, 1, 1), datetime(2011, 2, 1)], name="idx"
        )
        tm.assert_index_equal(result, expected)
        assert result.name == "idx"

        result2 = result.to_period(freq="M")
        tm.assert_index_equal(result2, index)
        assert result2.name == "idx"

        result3 = result.to_period(freq="3M")
        exp = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="3M", name="idx")
        tm.assert_index_equal(result3, exp)
        assert result3.freqstr == "3M"

        msg = "Frequency must be positive, because it represents span: -2A"
        with pytest.raises(ValueError, match=msg):
            result.to_period(freq="-2A")

    def test_to_timestamp_preserve_name(self):
        index = period_range(freq="A", start="1/1/2001", end="12/1/2009", name="foo")
        assert index.name == "foo"

        conv = index.to_timestamp("D")
        assert conv.name == "foo"

    def test_to_timestamp_quarterly_bug(self):
        years = np.arange(1960, 2000).repeat(4)
        quarters = np.tile(list(range(1, 5)), 40)

        pindex = PeriodIndex(year=years, quarter=quarters)

        stamps = pindex.to_timestamp("D", "end")
        expected = DatetimeIndex([x.to_timestamp("D", "end") for x in pindex])
        tm.assert_index_equal(stamps, expected)
        assert stamps.freq == expected.freq

    def test_to_timestamp_pi_mult(self):
        idx = PeriodIndex(["2011-01", "NaT", "2011-02"], freq="2M", name="idx")

        result = idx.to_timestamp()
        expected = DatetimeIndex(["2011-01-01", "NaT", "2011-02-01"], name="idx")
        tm.assert_index_equal(result, expected)

        result = idx.to_timestamp(how="E")
        expected = DatetimeIndex(["2011-02-28", "NaT", "2011-03-31"], name="idx")
        expected = expected + Timedelta(1, "D") - Timedelta(1, "ns")
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_pi_combined(self):
        idx = period_range(start="2011", periods=2, freq="1D1H", name="idx")

        result = idx.to_timestamp()
        expected = DatetimeIndex(["2011-01-01 00:00", "2011-01-02 01:00"], name="idx")
        tm.assert_index_equal(result, expected)

        result = idx.to_timestamp(how="E")
        expected = DatetimeIndex(
            ["2011-01-02 00:59:59", "2011-01-03 01:59:59"], name="idx"
        )
        expected = expected + Timedelta(1, "s") - Timedelta(1, "ns")
        tm.assert_index_equal(result, expected)

        result = idx.to_timestamp(how="E", freq="H")
        expected = DatetimeIndex(["2011-01-02 00:00", "2011-01-03 01:00"], name="idx")
        expected = expected + Timedelta(1, "h") - Timedelta(1, "ns")
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_1703(self):
        index = period_range("1/1/2012", periods=4, freq="D")

        result = index.to_timestamp()
        assert result[0] == Timestamp("1/1/2012")
