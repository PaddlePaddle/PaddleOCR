from datetime import datetime

import pytest
import pytz

from pandas.errors import NullFrequencyError

import pandas as pd
from pandas import (
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)


class TestDatetimeIndexShift:

    # -------------------------------------------------------------
    # DatetimeIndex.shift is used in integer addition

    def test_dti_shift_tzaware(self, tz_naive_fixture):
        # GH#9903
        tz = tz_naive_fixture
        idx = DatetimeIndex([], name="xxx", tz=tz)
        tm.assert_index_equal(idx.shift(0, freq="H"), idx)
        tm.assert_index_equal(idx.shift(3, freq="H"), idx)

        idx = DatetimeIndex(
            ["2011-01-01 10:00", "2011-01-01 11:00", "2011-01-01 12:00"],
            name="xxx",
            tz=tz,
            freq="H",
        )
        tm.assert_index_equal(idx.shift(0, freq="H"), idx)
        exp = DatetimeIndex(
            ["2011-01-01 13:00", "2011-01-01 14:00", "2011-01-01 15:00"],
            name="xxx",
            tz=tz,
            freq="H",
        )
        tm.assert_index_equal(idx.shift(3, freq="H"), exp)
        exp = DatetimeIndex(
            ["2011-01-01 07:00", "2011-01-01 08:00", "2011-01-01 09:00"],
            name="xxx",
            tz=tz,
            freq="H",
        )
        tm.assert_index_equal(idx.shift(-3, freq="H"), exp)

    def test_dti_shift_freqs(self):
        # test shift for DatetimeIndex and non DatetimeIndex
        # GH#8083
        drange = date_range("20130101", periods=5)
        result = drange.shift(1)
        expected = DatetimeIndex(
            ["2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05", "2013-01-06"],
            freq="D",
        )
        tm.assert_index_equal(result, expected)

        result = drange.shift(-1)
        expected = DatetimeIndex(
            ["2012-12-31", "2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04"],
            freq="D",
        )
        tm.assert_index_equal(result, expected)

        result = drange.shift(3, freq="2D")
        expected = DatetimeIndex(
            ["2013-01-07", "2013-01-08", "2013-01-09", "2013-01-10", "2013-01-11"],
            freq="D",
        )
        tm.assert_index_equal(result, expected)

    def test_dti_shift_int(self):
        rng = date_range("1/1/2000", periods=20)

        result = rng + 5 * rng.freq
        expected = rng.shift(5)
        tm.assert_index_equal(result, expected)

        result = rng - 5 * rng.freq
        expected = rng.shift(-5)
        tm.assert_index_equal(result, expected)

    def test_dti_shift_no_freq(self):
        # GH#19147
        dti = DatetimeIndex(["2011-01-01 10:00", "2011-01-01"], freq=None)
        with pytest.raises(NullFrequencyError, match="Cannot shift with no freq"):
            dti.shift(2)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_shift_localized(self, tzstr):
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        dr_tz = dr.tz_localize(tzstr)

        result = dr_tz.shift(1, "10T")
        assert result.tz == dr_tz.tz

    def test_dti_shift_across_dst(self):
        # GH 8616
        idx = date_range("2013-11-03", tz="America/Chicago", periods=7, freq="H")
        s = Series(index=idx[:-1], dtype=object)
        result = s.shift(freq="H")
        expected = Series(index=idx[1:], dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "shift, result_time",
        [
            [0, "2014-11-14 00:00:00"],
            [-1, "2014-11-13 23:00:00"],
            [1, "2014-11-14 01:00:00"],
        ],
    )
    def test_dti_shift_near_midnight(self, shift, result_time):
        # GH 8616
        dt = datetime(2014, 11, 14, 0)
        dt_est = pytz.timezone("EST").localize(dt)
        s = Series(data=[1], index=[dt_est])
        result = s.shift(shift, freq="H")
        expected = Series(1, index=DatetimeIndex([result_time], tz="EST"))
        tm.assert_series_equal(result, expected)

    def test_shift_periods(self):
        # GH#22458 : argument 'n' was deprecated in favor of 'periods'
        idx = date_range(start=START, end=END, periods=3)
        tm.assert_index_equal(idx.shift(periods=0), idx)
        tm.assert_index_equal(idx.shift(0), idx)

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_shift_bday(self, freq):
        rng = date_range(START, END, freq=freq)
        shifted = rng.shift(5)
        assert shifted[0] == rng[5]
        assert shifted.freq == rng.freq

        shifted = rng.shift(-5)
        assert shifted[5] == rng[0]
        assert shifted.freq == rng.freq

        shifted = rng.shift(0)
        assert shifted[0] == rng[0]
        assert shifted.freq == rng.freq

    def test_shift_bmonth(self):
        rng = date_range(START, END, freq=pd.offsets.BMonthEnd())
        shifted = rng.shift(1, freq=pd.offsets.BDay())
        assert shifted[0] == rng[0] + pd.offsets.BDay()

        rng = date_range(START, END, freq=pd.offsets.BMonthEnd())
        with tm.assert_produces_warning(pd.errors.PerformanceWarning):
            shifted = rng.shift(1, freq=pd.offsets.CDay())
            assert shifted[0] == rng[0] + pd.offsets.CDay()

    def test_shift_empty(self):
        # GH#14811
        dti = date_range(start="2016-10-21", end="2016-10-21", freq="BM")
        result = dti.shift(1)
        tm.assert_index_equal(result, dti)
