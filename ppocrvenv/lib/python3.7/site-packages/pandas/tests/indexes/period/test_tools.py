import numpy as np
import pytest

from pandas import (
    Period,
    PeriodIndex,
    period_range,
)
import pandas._testing as tm


class TestPeriodRepresentation:
    """
    Wish to match NumPy units
    """

    def _check_freq(self, freq, base_date):
        rng = period_range(start=base_date, periods=10, freq=freq)
        exp = np.arange(10, dtype=np.int64)

        tm.assert_numpy_array_equal(rng.asi8, exp)

    def test_annual(self):
        self._check_freq("A", 1970)

    def test_monthly(self):
        self._check_freq("M", "1970-01")

    @pytest.mark.parametrize("freq", ["W-THU", "D", "B", "H", "T", "S", "L", "U", "N"])
    def test_freq(self, freq):
        self._check_freq(freq, "1970-01-01")


class TestPeriodIndexConversion:
    def test_tolist(self):
        index = period_range(freq="A", start="1/1/2001", end="12/1/2009")
        rs = index.tolist()
        for x in rs:
            assert isinstance(x, Period)

        recon = PeriodIndex(rs)
        tm.assert_index_equal(index, recon)
