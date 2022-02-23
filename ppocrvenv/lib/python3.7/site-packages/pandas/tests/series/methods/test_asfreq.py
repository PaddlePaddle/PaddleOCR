from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    BDay,
    BMonthEnd,
)


class TestAsFreq:
    # TODO: de-duplicate/parametrize or move DataFrame test
    def test_asfreq_ts(self):
        index = period_range(freq="A", start="1/1/2001", end="12/31/2010")
        ts = Series(np.random.randn(len(index)), index=index)
        df = DataFrame(np.random.randn(len(index), 3), index=index)

        result = ts.asfreq("D", how="end")
        df_result = df.asfreq("D", how="end")
        exp_index = index.asfreq("D", how="end")
        assert len(result) == len(ts)
        tm.assert_index_equal(result.index, exp_index)
        tm.assert_index_equal(df_result.index, exp_index)

        result = ts.asfreq("D", how="start")
        assert len(result) == len(ts)
        tm.assert_index_equal(result.index, index.asfreq("D", how="start"))

    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_tz_aware_asfreq(self, tz):
        dr = date_range("2011-12-01", "2012-07-20", freq="D", tz=tz)

        ser = Series(np.random.randn(len(dr)), index=dr)

        # it works!
        ser.asfreq("T")

    def test_asfreq(self):
        ts = Series(
            [0.0, 1.0, 2.0],
            index=DatetimeIndex(
                [
                    datetime(2009, 10, 30),
                    datetime(2009, 11, 30),
                    datetime(2009, 12, 31),
                ],
                freq="BM",
            ),
        )

        daily_ts = ts.asfreq("B")
        monthly_ts = daily_ts.asfreq("BM")
        tm.assert_series_equal(monthly_ts, ts)

        daily_ts = ts.asfreq("B", method="pad")
        monthly_ts = daily_ts.asfreq("BM")
        tm.assert_series_equal(monthly_ts, ts)

        daily_ts = ts.asfreq(BDay())
        monthly_ts = daily_ts.asfreq(BMonthEnd())
        tm.assert_series_equal(monthly_ts, ts)

        result = ts[:0].asfreq("M")
        assert len(result) == 0
        assert result is not ts

        daily_ts = ts.asfreq("D", fill_value=-1)
        result = daily_ts.value_counts().sort_index()
        expected = Series([60, 1, 1, 1], index=[-1.0, 2.0, 1.0, 0.0]).sort_index()
        tm.assert_series_equal(result, expected)

    def test_asfreq_datetimeindex_empty_series(self):
        # GH#14320
        index = DatetimeIndex(["2016-09-29 11:00"])
        expected = Series(index=index, dtype=object).asfreq("H")
        result = Series([3], index=index.copy()).asfreq("H")
        tm.assert_index_equal(expected.index, result.index)

    def test_asfreq_keep_index_name(self):
        # GH#9854
        index_name = "bar"
        index = date_range("20130101", periods=20, name=index_name)
        df = DataFrame(list(range(20)), columns=["foo"], index=index)

        assert index_name == df.index.name
        assert index_name == df.asfreq("10D").index.name

    def test_asfreq_normalize(self):
        rng = date_range("1/1/2000 09:30", periods=20)
        norm = date_range("1/1/2000", periods=20)
        vals = np.random.randn(20)
        ts = Series(vals, index=rng)

        result = ts.asfreq("D", normalize=True)
        norm = date_range("1/1/2000", periods=20)
        expected = Series(vals, index=norm)

        tm.assert_series_equal(result, expected)

        vals = np.random.randn(20, 3)
        ts = DataFrame(vals, index=rng)

        result = ts.asfreq("D", normalize=True)
        expected = DataFrame(vals, index=norm)

        tm.assert_frame_equal(result, expected)
