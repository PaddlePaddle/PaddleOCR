from datetime import datetime

import numpy as np

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    date_range,
    to_datetime,
)
import pandas._testing as tm

from pandas.tseries import offsets


class TestAsFreq:
    def test_asfreq_resample_set_correct_freq(self):
        # GH#5613
        # we test if .asfreq() and .resample() set the correct value for .freq
        df = DataFrame(
            {"date": ["2012-01-01", "2012-01-02", "2012-01-03"], "col": [1, 2, 3]}
        )
        df = df.set_index(to_datetime(df.date))

        # testing the settings before calling .asfreq() and .resample()
        assert df.index.freq is None
        assert df.index.inferred_freq == "D"

        # does .asfreq() set .freq correctly?
        assert df.asfreq("D").index.freq == "D"

        # does .resample() set .freq correctly?
        assert df.resample("D").asfreq().index.freq == "D"

    def test_asfreq(self, datetime_frame):
        offset_monthly = datetime_frame.asfreq(offsets.BMonthEnd())
        rule_monthly = datetime_frame.asfreq("BM")

        tm.assert_almost_equal(offset_monthly["A"], rule_monthly["A"])

        filled = rule_monthly.asfreq("B", method="pad")  # noqa
        # TODO: actually check that this worked.

        # don't forget!
        filled_dep = rule_monthly.asfreq("B", method="pad")  # noqa

        # test does not blow up on length-0 DataFrame
        zero_length = datetime_frame.reindex([])
        result = zero_length.asfreq("BM")
        assert result is not zero_length

    def test_asfreq_datetimeindex(self):
        df = DataFrame(
            {"A": [1, 2, 3]},
            index=[datetime(2011, 11, 1), datetime(2011, 11, 2), datetime(2011, 11, 3)],
        )
        df = df.asfreq("B")
        assert isinstance(df.index, DatetimeIndex)

        ts = df["A"].asfreq("B")
        assert isinstance(ts.index, DatetimeIndex)

    def test_asfreq_fillvalue(self):
        # test for fill value during upsampling, related to issue 3715

        # setup
        rng = date_range("1/1/2016", periods=10, freq="2S")
        ts = Series(np.arange(len(rng)), index=rng)
        df = DataFrame({"one": ts})

        # insert pre-existing missing value
        df.loc["2016-01-01 00:00:08", "one"] = None

        actual_df = df.asfreq(freq="1S", fill_value=9.0)
        expected_df = df.asfreq(freq="1S").fillna(9.0)
        expected_df.loc["2016-01-01 00:00:08", "one"] = None
        tm.assert_frame_equal(expected_df, actual_df)

        expected_series = ts.asfreq(freq="1S").fillna(9.0)
        actual_series = ts.asfreq(freq="1S", fill_value=9.0)
        tm.assert_series_equal(expected_series, actual_series)

    def test_asfreq_with_date_object_index(self, frame_or_series):
        rng = date_range("1/1/2000", periods=20)
        ts = frame_or_series(np.random.randn(20), index=rng)

        ts2 = ts.copy()
        ts2.index = [x.date() for x in ts2.index]

        result = ts2.asfreq("4H", method="ffill")
        expected = ts.asfreq("4H", method="ffill")
        tm.assert_equal(result, expected)

    def test_asfreq_with_unsorted_index(self, frame_or_series):
        # GH#39805
        # Test that rows are not dropped when the datetime index is out of order
        index = to_datetime(["2021-01-04", "2021-01-02", "2021-01-03", "2021-01-01"])
        result = frame_or_series(range(4), index=index)

        expected = result.reindex(sorted(index))
        expected.index = expected.index._with_freq("infer")

        result = result.asfreq("D")
        tm.assert_equal(result, expected)
