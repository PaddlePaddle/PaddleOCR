import numpy as np
import pytest

from pandas import (
    Timedelta,
    timedelta_range,
    to_timedelta,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    Day,
    Second,
)


class TestTimedeltas:
    def test_timedelta_range(self):

        expected = to_timedelta(np.arange(5), unit="D")
        result = timedelta_range("0 days", periods=5, freq="D")
        tm.assert_index_equal(result, expected)

        expected = to_timedelta(np.arange(11), unit="D")
        result = timedelta_range("0 days", "10 days", freq="D")
        tm.assert_index_equal(result, expected)

        expected = to_timedelta(np.arange(5), unit="D") + Second(2) + Day()
        result = timedelta_range("1 days, 00:00:02", "5 days, 00:00:02", freq="D")
        tm.assert_index_equal(result, expected)

        expected = to_timedelta([1, 3, 5, 7, 9], unit="D") + Second(2)
        result = timedelta_range("1 days, 00:00:02", periods=5, freq="2D")
        tm.assert_index_equal(result, expected)

        expected = to_timedelta(np.arange(50), unit="T") * 30
        result = timedelta_range("0 days", freq="30T", periods=50)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, freq", [(3, "2D"), (5, "D"), (6, "19H12T"), (7, "16H"), (9, "12H")]
    )
    def test_linspace_behavior(self, periods, freq):
        # GH 20976
        result = timedelta_range(start="0 days", end="4 days", periods=periods)
        expected = timedelta_range(start="0 days", end="4 days", freq=freq)
        tm.assert_index_equal(result, expected)

    def test_errors(self):
        # not enough params
        msg = (
            "Of the four parameters: start, end, periods, and freq, "
            "exactly three must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start="0 days")

        with pytest.raises(ValueError, match=msg):
            timedelta_range(end="5 days")

        with pytest.raises(ValueError, match=msg):
            timedelta_range(periods=2)

        with pytest.raises(ValueError, match=msg):
            timedelta_range()

        # too many params
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start="0 days", end="5 days", periods=10, freq="H")

    @pytest.mark.parametrize(
        "start, end, freq, expected_periods",
        [
            ("1D", "10D", "2D", (10 - 1) // 2 + 1),
            ("2D", "30D", "3D", (30 - 2) // 3 + 1),
            ("2s", "50s", "5s", (50 - 2) // 5 + 1),
            # tests that worked before GH 33498:
            ("4D", "16D", "3D", (16 - 4) // 3 + 1),
            ("8D", "16D", "40s", (16 * 3600 * 24 - 8 * 3600 * 24) // 40 + 1),
        ],
    )
    def test_timedelta_range_freq_divide_end(self, start, end, freq, expected_periods):
        # GH 33498 only the cases where `(end % freq) == 0` used to fail
        res = timedelta_range(start=start, end=end, freq=freq)
        assert Timedelta(start) == res[0]
        assert Timedelta(end) >= res[-1]
        assert len(res) == expected_periods

    def test_timedelta_range_infer_freq(self):
        # https://github.com/pandas-dev/pandas/issues/35897
        result = timedelta_range("0s", "1s", periods=31)
        assert result.freq is None
