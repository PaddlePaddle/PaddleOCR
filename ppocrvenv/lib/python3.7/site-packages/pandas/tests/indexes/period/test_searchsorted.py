import numpy as np
import pytest

from pandas._libs.tslibs import IncompatibleFrequency
from pandas.compat import np_version_under1p18

from pandas import (
    NaT,
    Period,
    PeriodIndex,
    Series,
    array,
)
import pandas._testing as tm


class TestSearchsorted:
    @pytest.mark.parametrize("freq", ["D", "2D"])
    def test_searchsorted(self, freq):
        pidx = PeriodIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
            freq=freq,
        )

        p1 = Period("2014-01-01", freq=freq)
        assert pidx.searchsorted(p1) == 0

        p2 = Period("2014-01-04", freq=freq)
        assert pidx.searchsorted(p2) == 3

        if np_version_under1p18:
            # GH#36254
            # Following numpy convention, NaT goes at the beginning
            #  (unlike NaN which goes at the end)
            assert pidx.searchsorted(NaT) == 0
        else:
            assert pidx.searchsorted(NaT) == 5

        msg = "Input has different freq=H from PeriodArray"
        with pytest.raises(IncompatibleFrequency, match=msg):
            pidx.searchsorted(Period("2014-01-01", freq="H"))

        msg = "Input has different freq=5D from PeriodArray"
        with pytest.raises(IncompatibleFrequency, match=msg):
            pidx.searchsorted(Period("2014-01-01", freq="5D"))

    @pytest.mark.parametrize("klass", [list, np.array, array, Series])
    def test_searchsorted_different_argument_classes(self, klass):
        pidx = PeriodIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
            freq="D",
        )
        result = pidx.searchsorted(klass(pidx))
        expected = np.arange(len(pidx), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

        result = pidx._data.searchsorted(klass(pidx))
        tm.assert_numpy_array_equal(result, expected)

    def test_searchsorted_invalid(self):
        pidx = PeriodIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
            freq="D",
        )

        other = np.array([0, 1], dtype=np.int64)

        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Period', 'NaT', or array of those. Got",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(other)

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(other.astype("timedelta64[ns]"))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.timedelta64(4))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.timedelta64("NaT", "ms"))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.datetime64(4, "ns"))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.datetime64("NaT", "ns"))
