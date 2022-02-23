from datetime import datetime

from dateutil.tz import tzlocal
import pytest

from pandas.compat import IS64

from pandas import (
    DateOffset,
    DatetimeIndex,
    Index,
    Series,
    bdate_range,
    date_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    BDay,
    Day,
    Hour,
)

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)


class TestDatetimeIndexOps:
    def test_ops_properties_basic(self, datetime_series):

        # sanity check that the behavior didn't change
        # GH#7206
        for op in ["year", "day", "second", "weekday"]:
            msg = f"'Series' object has no attribute '{op}'"
            with pytest.raises(AttributeError, match=msg):
                getattr(datetime_series, op)

        # attribute access should still work!
        s = Series({"year": 2000, "month": 1, "day": 10})
        assert s.year == 2000
        assert s.month == 1
        assert s.day == 10
        msg = "'Series' object has no attribute 'weekday'"
        with pytest.raises(AttributeError, match=msg):
            s.weekday

    @pytest.mark.parametrize(
        "freq,expected",
        [
            ("A", "day"),
            ("Q", "day"),
            ("M", "day"),
            ("D", "day"),
            ("H", "hour"),
            ("T", "minute"),
            ("S", "second"),
            ("L", "millisecond"),
            ("U", "microsecond"),
        ],
    )
    def test_resolution(self, request, tz_naive_fixture, freq, expected):
        tz = tz_naive_fixture
        if freq == "A" and not IS64 and isinstance(tz, tzlocal):
            request.node.add_marker(
                pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
            )

        idx = date_range(start="2013-04-01", periods=30, freq=freq, tz=tz)
        assert idx.resolution == expected

    def test_infer_freq(self, freq_sample):
        # GH 11018
        idx = date_range("2011-01-01 09:00:00", freq=freq_sample, periods=10)
        result = DatetimeIndex(idx.asi8, freq="infer")
        tm.assert_index_equal(idx, result)
        assert result.freq == freq_sample

    @pytest.mark.parametrize("values", [["20180101", "20180103", "20180105"], []])
    @pytest.mark.parametrize("freq", ["2D", Day(2), "2B", BDay(2), "48H", Hour(48)])
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_freq_setter(self, values, freq, tz):
        # GH 20678
        idx = DatetimeIndex(values, tz=tz)

        # can set to an offset, converting from string if necessary
        idx._data.freq = freq
        assert idx.freq == freq
        assert isinstance(idx.freq, DateOffset)

        # can reset to None
        idx._data.freq = None
        assert idx.freq is None

    def test_freq_setter_errors(self):
        # GH 20678
        idx = DatetimeIndex(["20180101", "20180103", "20180105"])

        # setting with an incompatible freq
        msg = (
            "Inferred frequency 2D from passed values does not conform to "
            "passed frequency 5D"
        )
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = "5D"

        # setting with non-freq string
        with pytest.raises(ValueError, match="Invalid frequency"):
            idx._data.freq = "foo"

    def test_freq_view_safe(self):
        # Setting the freq for one DatetimeIndex shouldn't alter the freq
        #  for another that views the same data

        dti = date_range("2016-01-01", periods=5)
        dta = dti._data

        dti2 = DatetimeIndex(dta)._with_freq(None)
        assert dti2.freq is None

        # Original was not altered
        assert dti.freq == "D"
        assert dta.freq == "D"


class TestBusinessDatetimeIndex:
    def setup_method(self, method):
        self.rng = bdate_range(START, END)

    def test_comparison(self):
        d = self.rng[10]

        comp = self.rng > d
        assert comp[11]
        assert not comp[9]

    def test_copy(self):
        cp = self.rng.copy()
        repr(cp)
        tm.assert_index_equal(cp, self.rng)

    def test_identical(self):
        t1 = self.rng.copy()
        t2 = self.rng.copy()
        assert t1.identical(t2)

        # name
        t1 = t1.rename("foo")
        assert t1.equals(t2)
        assert not t1.identical(t2)
        t2 = t2.rename("foo")
        assert t1.identical(t2)

        # freq
        t2v = Index(t2.values)
        assert t1.equals(t2v)
        assert not t1.identical(t2v)


class TestCustomDatetimeIndex:
    def setup_method(self, method):
        self.rng = bdate_range(START, END, freq="C")

    def test_comparison(self):
        d = self.rng[10]

        comp = self.rng > d
        assert comp[11]
        assert not comp[9]

    def test_copy(self):
        cp = self.rng.copy()
        repr(cp)
        tm.assert_index_equal(cp, self.rng)
