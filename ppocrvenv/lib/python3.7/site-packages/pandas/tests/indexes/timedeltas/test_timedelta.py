from datetime import timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    Int64Index,
    Series,
    Timedelta,
    TimedeltaIndex,
    date_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.indexes.datetimelike import DatetimeLike

randn = np.random.randn


class TestTimedeltaIndex(DatetimeLike):
    _index_cls = TimedeltaIndex

    @pytest.fixture
    def simple_index(self) -> TimedeltaIndex:
        index = pd.to_timedelta(range(5), unit="d")._with_freq("infer")
        assert index.freq == "D"
        ret = index + pd.offsets.Hour(1)
        assert ret.freq == "D"
        return ret

    @pytest.fixture
    def index(self):
        return tm.makeTimedeltaIndex(10)

    def test_numeric_compat(self):
        # Dummy method to override super's version; this test is now done
        # in test_arithmetic.py
        pass

    def test_shift(self):
        pass  # this is handled in test_arithmetic.py

    def test_pickle_after_set_freq(self):
        tdi = timedelta_range("1 day", periods=4, freq="s")
        tdi = tdi._with_freq(None)

        res = tm.round_trip_pickle(tdi)
        tm.assert_index_equal(res, tdi)

    def test_isin(self):

        index = tm.makeTimedeltaIndex(4)
        result = index.isin(index)
        assert result.all()

        result = index.isin(list(index))
        assert result.all()

        tm.assert_almost_equal(
            index.isin([index[2], 5]), np.array([False, False, True, False])
        )

    def test_misc_coverage(self):

        rng = timedelta_range("1 day", periods=5)
        result = rng.groupby(rng.days)
        assert isinstance(list(result.values())[0][0], Timedelta)

    def test_map(self):
        # test_map_dictlike generally tests

        rng = timedelta_range("1 day", periods=10)

        f = lambda x: x.days
        result = rng.map(f)
        exp = Int64Index([f(x) for x in rng])
        tm.assert_index_equal(result, exp)

    def test_pass_TimedeltaIndex_to_index(self):

        rng = timedelta_range("1 days", "10 days")
        idx = Index(rng, dtype=object)

        expected = Index(rng.to_pytimedelta(), dtype=object)

        tm.assert_numpy_array_equal(idx.values, expected.values)

    def test_fields(self):
        rng = timedelta_range("1 days, 10:11:12.100123456", periods=2, freq="s")
        tm.assert_index_equal(rng.days, Index([1, 1], dtype="int64"))
        tm.assert_index_equal(
            rng.seconds,
            Index([10 * 3600 + 11 * 60 + 12, 10 * 3600 + 11 * 60 + 13], dtype="int64"),
        )
        tm.assert_index_equal(
            rng.microseconds, Index([100 * 1000 + 123, 100 * 1000 + 123], dtype="int64")
        )
        tm.assert_index_equal(rng.nanoseconds, Index([456, 456], dtype="int64"))

        msg = "'TimedeltaIndex' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # with nat
        s = Series(rng)
        s[1] = np.nan

        tm.assert_series_equal(s.dt.days, Series([1, np.nan], index=[0, 1]))
        tm.assert_series_equal(
            s.dt.seconds, Series([10 * 3600 + 11 * 60 + 12, np.nan], index=[0, 1])
        )

        # preserve name (GH15589)
        rng.name = "name"
        assert rng.days.name == "name"

    def test_freq_conversion_always_floating(self):
        # even if we have no NaTs, we get back float64; this matches TDA and Series
        tdi = timedelta_range("1 Day", periods=30)

        res = tdi.astype("m8[s]")
        expected = Index((tdi.view("i8") / 10 ** 9).astype(np.float64))
        tm.assert_index_equal(res, expected)

        # check this matches Series and TimedeltaArray
        res = tdi._data.astype("m8[s]")
        tm.assert_numpy_array_equal(res, expected._values)

        res = tdi.to_series().astype("m8[s]")
        tm.assert_numpy_array_equal(res._values, expected._values)

    def test_freq_conversion(self):

        # doc example

        # series
        td = Series(date_range("20130101", periods=4)) - Series(
            date_range("20121201", periods=4)
        )
        td[2] += timedelta(minutes=5, seconds=3)
        td[3] = np.nan

        result = td / np.timedelta64(1, "D")
        expected = Series([31, 31, (31 * 86400 + 5 * 60 + 3) / 86400.0, np.nan])
        tm.assert_series_equal(result, expected)

        result = td.astype("timedelta64[D]")
        expected = Series([31, 31, 31, np.nan])
        tm.assert_series_equal(result, expected)

        result = td / np.timedelta64(1, "s")
        expected = Series([31 * 86400, 31 * 86400, 31 * 86400 + 5 * 60 + 3, np.nan])
        tm.assert_series_equal(result, expected)

        result = td.astype("timedelta64[s]")
        tm.assert_series_equal(result, expected)

        # tdi
        td = TimedeltaIndex(td)

        result = td / np.timedelta64(1, "D")
        expected = Index([31, 31, (31 * 86400 + 5 * 60 + 3) / 86400.0, np.nan])
        tm.assert_index_equal(result, expected)

        result = td.astype("timedelta64[D]")
        expected = Index([31, 31, 31, np.nan])
        tm.assert_index_equal(result, expected)

        result = td / np.timedelta64(1, "s")
        expected = Index([31 * 86400, 31 * 86400, 31 * 86400 + 5 * 60 + 3, np.nan])
        tm.assert_index_equal(result, expected)

        result = td.astype("timedelta64[s]")
        tm.assert_index_equal(result, expected)
