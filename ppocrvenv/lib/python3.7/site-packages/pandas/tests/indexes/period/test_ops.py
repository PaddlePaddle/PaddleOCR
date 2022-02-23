import pytest

import pandas as pd
import pandas._testing as tm


class TestPeriodIndexOps:
    @pytest.mark.parametrize(
        "freq,expected",
        [
            ("A", "year"),
            ("Q", "quarter"),
            ("M", "month"),
            ("D", "day"),
            ("H", "hour"),
            ("T", "minute"),
            ("S", "second"),
            ("L", "millisecond"),
            ("U", "microsecond"),
        ],
    )
    def test_resolution(self, freq, expected):
        idx = pd.period_range(start="2013-04-01", periods=30, freq=freq)
        assert idx.resolution == expected

    def test_freq_setter_deprecated(self):
        # GH 20678
        idx = pd.period_range("2018Q1", periods=4, freq="Q")

        # no warning for getter
        with tm.assert_produces_warning(None):
            idx.freq

        # warning for setter
        with pytest.raises(AttributeError, match="can't set attribute"):
            idx.freq = pd.offsets.Day()
