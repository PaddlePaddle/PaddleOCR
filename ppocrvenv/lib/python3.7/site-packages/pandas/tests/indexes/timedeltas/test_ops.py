import numpy as np
import pytest

from pandas import (
    Series,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    DateOffset,
    Day,
    Hour,
)


class TestTimedeltaIndexOps:
    def test_nonunique_contains(self):
        # GH 9512
        for idx in map(
            TimedeltaIndex,
            (
                [0, 1, 0],
                [0, 0, -1],
                [0, -1, -1],
                ["00:01:00", "00:01:00", "00:02:00"],
                ["00:01:00", "00:01:00", "00:00:01"],
            ),
        ):
            assert idx[0] in idx

    def test_unknown_attribute(self):
        # see gh-9680
        tdi = timedelta_range(start=0, periods=10, freq="1s")
        ts = Series(np.random.normal(size=10), index=tdi)
        assert "foo" not in ts.__dict__.keys()
        msg = "'Series' object has no attribute 'foo'"
        with pytest.raises(AttributeError, match=msg):
            ts.foo

    def test_infer_freq(self, freq_sample):
        # GH#11018
        idx = timedelta_range("1", freq=freq_sample, periods=10)
        result = TimedeltaIndex(idx.asi8, freq="infer")
        tm.assert_index_equal(idx, result)
        assert result.freq == freq_sample

    @pytest.mark.parametrize("values", [["0 days", "2 days", "4 days"], []])
    @pytest.mark.parametrize("freq", ["2D", Day(2), "48H", Hour(48)])
    def test_freq_setter(self, values, freq):
        # GH 20678
        idx = TimedeltaIndex(values)

        # can set to an offset, converting from string if necessary
        idx._data.freq = freq
        assert idx.freq == freq
        assert isinstance(idx.freq, DateOffset)

        # can reset to None
        idx._data.freq = None
        assert idx.freq is None

    def test_freq_setter_errors(self):
        # GH 20678
        idx = TimedeltaIndex(["0 days", "2 days", "4 days"])

        # setting with an incompatible freq
        msg = (
            "Inferred frequency 2D from passed values does not conform to "
            "passed frequency 5D"
        )
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = "5D"

        # setting with a non-fixed frequency
        msg = r"<2 \* BusinessDays> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = "2B"

        # setting with non-freq string
        with pytest.raises(ValueError, match="Invalid frequency"):
            idx._data.freq = "foo"

    def test_freq_view_safe(self):
        # Setting the freq for one TimedeltaIndex shouldn't alter the freq
        #  for another that views the same data

        tdi = TimedeltaIndex(["0 days", "2 days", "4 days"], freq="2D")
        tda = tdi._data

        tdi2 = TimedeltaIndex(tda)._with_freq(None)
        assert tdi2.freq is None

        # Original was not altered
        assert tdi.freq == "2D"
        assert tda.freq == "2D"
