from datetime import datetime

from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pytz import utc

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    Timedelta,
    Timestamp,
    conversion,
    iNaT,
    to_offset,
)
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td

import pandas._testing as tm


class TestTimestampUnaryOps:

    # --------------------------------------------------------------
    # Timestamp.round
    @pytest.mark.parametrize(
        "timestamp, freq, expected",
        [
            ("20130101 09:10:11", "D", "20130101"),
            ("20130101 19:10:11", "D", "20130102"),
            ("20130201 12:00:00", "D", "20130202"),
            ("20130104 12:00:00", "D", "20130105"),
            ("2000-01-05 05:09:15.13", "D", "2000-01-05 00:00:00"),
            ("2000-01-05 05:09:15.13", "H", "2000-01-05 05:00:00"),
            ("2000-01-05 05:09:15.13", "S", "2000-01-05 05:09:15"),
        ],
    )
    def test_round_frequencies(self, timestamp, freq, expected):
        dt = Timestamp(timestamp)
        result = dt.round(freq)
        expected = Timestamp(expected)
        assert result == expected

    def test_round_tzaware(self):
        dt = Timestamp("20130101 09:10:11", tz="US/Eastern")
        result = dt.round("D")
        expected = Timestamp("20130101", tz="US/Eastern")
        assert result == expected

        dt = Timestamp("20130101 09:10:11", tz="US/Eastern")
        result = dt.round("s")
        assert result == dt

    def test_round_30min(self):
        # round
        dt = Timestamp("20130104 12:32:00")
        result = dt.round("30Min")
        expected = Timestamp("20130104 12:30:00")
        assert result == expected

    def test_round_subsecond(self):
        # GH#14440 & GH#15578
        result = Timestamp("2016-10-17 12:00:00.0015").round("ms")
        expected = Timestamp("2016-10-17 12:00:00.002000")
        assert result == expected

        result = Timestamp("2016-10-17 12:00:00.00149").round("ms")
        expected = Timestamp("2016-10-17 12:00:00.001000")
        assert result == expected

        ts = Timestamp("2016-10-17 12:00:00.0015")
        for freq in ["us", "ns"]:
            assert ts == ts.round(freq)

        result = Timestamp("2016-10-17 12:00:00.001501031").round("10ns")
        expected = Timestamp("2016-10-17 12:00:00.001501030")
        assert result == expected

    def test_round_nonstandard_freq(self):
        with tm.assert_produces_warning(False):
            Timestamp("2016-10-17 12:00:00.001501031").round("1010ns")

    def test_round_invalid_arg(self):
        stamp = Timestamp("2000-01-05 05:09:15.13")
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            stamp.round("foo")

    @pytest.mark.parametrize(
        "test_input, rounder, freq, expected",
        [
            ("2117-01-01 00:00:45", "floor", "15s", "2117-01-01 00:00:45"),
            ("2117-01-01 00:00:45", "ceil", "15s", "2117-01-01 00:00:45"),
            (
                "2117-01-01 00:00:45.000000012",
                "floor",
                "10ns",
                "2117-01-01 00:00:45.000000010",
            ),
            (
                "1823-01-01 00:00:01.000000012",
                "ceil",
                "10ns",
                "1823-01-01 00:00:01.000000020",
            ),
            ("1823-01-01 00:00:01", "floor", "1s", "1823-01-01 00:00:01"),
            ("1823-01-01 00:00:01", "ceil", "1s", "1823-01-01 00:00:01"),
            ("NaT", "floor", "1s", "NaT"),
            ("NaT", "ceil", "1s", "NaT"),
        ],
    )
    def test_ceil_floor_edge(self, test_input, rounder, freq, expected):
        dt = Timestamp(test_input)
        func = getattr(dt, rounder)
        result = func(freq)

        if dt is NaT:
            assert result is NaT
        else:
            expected = Timestamp(expected)
            assert result == expected

    @pytest.mark.parametrize(
        "test_input, freq, expected",
        [
            ("2018-01-01 00:02:06", "2s", "2018-01-01 00:02:06"),
            ("2018-01-01 00:02:00", "2T", "2018-01-01 00:02:00"),
            ("2018-01-01 00:04:00", "4T", "2018-01-01 00:04:00"),
            ("2018-01-01 00:15:00", "15T", "2018-01-01 00:15:00"),
            ("2018-01-01 00:20:00", "20T", "2018-01-01 00:20:00"),
            ("2018-01-01 03:00:00", "3H", "2018-01-01 03:00:00"),
        ],
    )
    @pytest.mark.parametrize("rounder", ["ceil", "floor", "round"])
    def test_round_minute_freq(self, test_input, freq, expected, rounder):
        # Ensure timestamps that shouldn't round dont!
        # GH#21262

        dt = Timestamp(test_input)
        expected = Timestamp(expected)
        func = getattr(dt, rounder)
        result = func(freq)
        assert result == expected

    def test_ceil(self):
        dt = Timestamp("20130101 09:10:11")
        result = dt.ceil("D")
        expected = Timestamp("20130102")
        assert result == expected

    def test_floor(self):
        dt = Timestamp("20130101 09:10:11")
        result = dt.floor("D")
        expected = Timestamp("20130101")
        assert result == expected

    @pytest.mark.parametrize("method", ["ceil", "round", "floor"])
    def test_round_dst_border_ambiguous(self, method):
        # GH 18946 round near "fall back" DST
        ts = Timestamp("2017-10-29 00:00:00", tz="UTC").tz_convert("Europe/Madrid")
        #
        result = getattr(ts, method)("H", ambiguous=True)
        assert result == ts

        result = getattr(ts, method)("H", ambiguous=False)
        expected = Timestamp("2017-10-29 01:00:00", tz="UTC").tz_convert(
            "Europe/Madrid"
        )
        assert result == expected

        result = getattr(ts, method)("H", ambiguous="NaT")
        assert result is NaT

        msg = "Cannot infer dst time"
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            getattr(ts, method)("H", ambiguous="raise")

    @pytest.mark.parametrize(
        "method, ts_str, freq",
        [
            ["ceil", "2018-03-11 01:59:00-0600", "5min"],
            ["round", "2018-03-11 01:59:00-0600", "5min"],
            ["floor", "2018-03-11 03:01:00-0500", "2H"],
        ],
    )
    def test_round_dst_border_nonexistent(self, method, ts_str, freq):
        # GH 23324 round near "spring forward" DST
        ts = Timestamp(ts_str, tz="America/Chicago")
        result = getattr(ts, method)(freq, nonexistent="shift_forward")
        expected = Timestamp("2018-03-11 03:00:00", tz="America/Chicago")
        assert result == expected

        result = getattr(ts, method)(freq, nonexistent="NaT")
        assert result is NaT

        msg = "2018-03-11 02:00:00"
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            getattr(ts, method)(freq, nonexistent="raise")

    @pytest.mark.parametrize(
        "timestamp",
        [
            "2018-01-01 0:0:0.124999360",
            "2018-01-01 0:0:0.125000367",
            "2018-01-01 0:0:0.125500",
            "2018-01-01 0:0:0.126500",
            "2018-01-01 12:00:00",
            "2019-01-01 12:00:00",
        ],
    )
    @pytest.mark.parametrize(
        "freq",
        [
            "2ns",
            "3ns",
            "4ns",
            "5ns",
            "6ns",
            "7ns",
            "250ns",
            "500ns",
            "750ns",
            "1us",
            "19us",
            "250us",
            "500us",
            "750us",
            "1s",
            "2s",
            "3s",
            "1D",
        ],
    )
    def test_round_int64(self, timestamp, freq):
        # check that all rounding modes are accurate to int64 precision
        # see GH#22591
        dt = Timestamp(timestamp)
        unit = to_offset(freq).nanos

        # test floor
        result = dt.floor(freq)
        assert result.value % unit == 0, f"floor not a {freq} multiple"
        assert 0 <= dt.value - result.value < unit, "floor error"

        # test ceil
        result = dt.ceil(freq)
        assert result.value % unit == 0, f"ceil not a {freq} multiple"
        assert 0 <= result.value - dt.value < unit, "ceil error"

        # test round
        result = dt.round(freq)
        assert result.value % unit == 0, f"round not a {freq} multiple"
        assert abs(result.value - dt.value) <= unit // 2, "round error"
        if unit % 2 == 0 and abs(result.value - dt.value) == unit // 2:
            # round half to even
            assert result.value // unit % 2 == 0, "round half to even error"

    def test_round_implementation_bounds(self):
        # See also: analogous test for Timedelta
        result = Timestamp.min.ceil("s")
        expected = Timestamp(1677, 9, 21, 0, 12, 44)
        assert result == expected

        result = Timestamp.max.floor("s")
        expected = Timestamp.max - Timedelta(854775807)
        assert result == expected

        with pytest.raises(OverflowError, match="value too large"):
            Timestamp.min.floor("s")

        # the second message here shows up in windows builds
        msg = "|".join(
            ["Python int too large to convert to C long", "int too big to convert"]
        )
        with pytest.raises(OverflowError, match=msg):
            Timestamp.max.ceil("s")

    @pytest.mark.parametrize("n", range(100))
    @pytest.mark.parametrize(
        "method", [Timestamp.round, Timestamp.floor, Timestamp.ceil]
    )
    def test_round_sanity(self, method, n):
        val = np.random.randint(iNaT + 1, lib.i8max, dtype=np.int64)
        ts = Timestamp(val)

        def checker(res, ts, nanos):
            if method is Timestamp.round:
                diff = np.abs((res - ts).value)
                assert diff <= nanos / 2
            elif method is Timestamp.floor:
                assert res <= ts
            elif method is Timestamp.ceil:
                assert res >= ts

        assert method(ts, "ns") == ts

        res = method(ts, "us")
        nanos = 1000
        assert np.abs((res - ts).value) < nanos
        assert res.value % nanos == 0
        checker(res, ts, nanos)

        res = method(ts, "ms")
        nanos = 1_000_000
        assert np.abs((res - ts).value) < nanos
        assert res.value % nanos == 0
        checker(res, ts, nanos)

        res = method(ts, "s")
        nanos = 1_000_000_000
        assert np.abs((res - ts).value) < nanos
        assert res.value % nanos == 0
        checker(res, ts, nanos)

        res = method(ts, "min")
        nanos = 60 * 1_000_000_000
        assert np.abs((res - ts).value) < nanos
        assert res.value % nanos == 0
        checker(res, ts, nanos)

        res = method(ts, "h")
        nanos = 60 * 60 * 1_000_000_000
        assert np.abs((res - ts).value) < nanos
        assert res.value % nanos == 0
        checker(res, ts, nanos)

        res = method(ts, "D")
        nanos = 24 * 60 * 60 * 1_000_000_000
        assert np.abs((res - ts).value) < nanos
        assert res.value % nanos == 0
        checker(res, ts, nanos)

    # --------------------------------------------------------------
    # Timestamp.replace

    def test_replace_naive(self):
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00")
        result = ts.replace(hour=0)
        expected = Timestamp("2016-01-01 00:00:00")
        assert result == expected

    def test_replace_aware(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        # replacing datetime components with and w/o presence of a timezone
        ts = Timestamp("2016-01-01 09:00:00", tz=tz)
        result = ts.replace(hour=0)
        expected = Timestamp("2016-01-01 00:00:00", tz=tz)
        assert result == expected

    def test_replace_preserves_nanos(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        result = ts.replace(hour=0)
        expected = Timestamp("2016-01-01 00:00:00.000000123", tz=tz)
        assert result == expected

    def test_replace_multiple(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        # replacing datetime components with and w/o presence of a timezone
        # test all
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        result = ts.replace(
            year=2015,
            month=2,
            day=2,
            hour=0,
            minute=5,
            second=5,
            microsecond=5,
            nanosecond=5,
        )
        expected = Timestamp("2015-02-02 00:05:05.000005005", tz=tz)
        assert result == expected

    def test_replace_invalid_kwarg(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        msg = r"replace\(\) got an unexpected keyword argument"
        with pytest.raises(TypeError, match=msg):
            ts.replace(foo=5)

    def test_replace_integer_args(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        msg = "value must be an integer, received <class 'float'> for hour"
        with pytest.raises(ValueError, match=msg):
            ts.replace(hour=0.1)

    def test_replace_tzinfo_equiv_tz_localize_none(self):
        # GH#14621, GH#7825
        # assert conversion to naive is the same as replacing tzinfo with None
        ts = Timestamp("2013-11-03 01:59:59.999999-0400", tz="US/Eastern")
        assert ts.tz_localize(None) == ts.replace(tzinfo=None)

    @td.skip_if_windows
    def test_replace_tzinfo(self):
        # GH#15683
        dt = datetime(2016, 3, 27, 1)
        tzinfo = pytz.timezone("CET").localize(dt, is_dst=False).tzinfo

        result_dt = dt.replace(tzinfo=tzinfo)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo)

        # datetime.timestamp() converts in the local timezone
        with tm.set_timezone("UTC"):
            assert result_dt.timestamp() == result_pd.timestamp()

        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()

        result_dt = dt.replace(tzinfo=tzinfo).replace(tzinfo=None)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo).replace(tzinfo=None)

        # datetime.timestamp() converts in the local timezone
        with tm.set_timezone("UTC"):
            assert result_dt.timestamp() == result_pd.timestamp()

        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()

    @pytest.mark.parametrize(
        "tz, normalize",
        [
            (pytz.timezone("US/Eastern"), lambda x: x.tzinfo.normalize(x)),
            (gettz("US/Eastern"), lambda x: x),
        ],
    )
    def test_replace_across_dst(self, tz, normalize):
        # GH#18319 check that 1) timezone is correctly normalized and
        # 2) that hour is not incorrectly changed by this normalization
        ts_naive = Timestamp("2017-12-03 16:03:30")
        ts_aware = conversion.localize_pydatetime(ts_naive, tz)

        # Preliminary sanity-check
        assert ts_aware == normalize(ts_aware)

        # Replace across DST boundary
        ts2 = ts_aware.replace(month=6)

        # Check that `replace` preserves hour literal
        assert (ts2.hour, ts2.minute) == (ts_aware.hour, ts_aware.minute)

        # Check that post-replace object is appropriately normalized
        ts2b = normalize(ts2)
        assert ts2 == ts2b

    def test_replace_dst_border(self):
        # Gh 7825
        t = Timestamp("2013-11-3", tz="America/Chicago")
        result = t.replace(hour=3)
        expected = Timestamp("2013-11-3 03:00:00", tz="America/Chicago")
        assert result == expected

    @pytest.mark.parametrize("fold", [0, 1])
    @pytest.mark.parametrize("tz", ["dateutil/Europe/London", "Europe/London"])
    def test_replace_dst_fold(self, fold, tz):
        # GH 25017
        d = datetime(2019, 10, 27, 2, 30)
        ts = Timestamp(d, tz=tz)
        result = ts.replace(hour=1, fold=fold)
        expected = Timestamp(datetime(2019, 10, 27, 1, 30)).tz_localize(
            tz, ambiguous=not fold
        )
        assert result == expected

    # --------------------------------------------------------------
    # Timestamp.normalize

    @pytest.mark.parametrize("arg", ["2013-11-30", "2013-11-30 12:00:00"])
    def test_normalize(self, tz_naive_fixture, arg):
        tz = tz_naive_fixture
        ts = Timestamp(arg, tz=tz)
        result = ts.normalize()
        expected = Timestamp("2013-11-30", tz=tz)
        assert result == expected

    def test_normalize_pre_epoch_dates(self):
        # GH: 36294
        result = Timestamp("1969-01-01 09:00:00").normalize()
        expected = Timestamp("1969-01-01 00:00:00")
        assert result == expected

    # --------------------------------------------------------------

    @td.skip_if_windows
    def test_timestamp(self):
        # GH#17329
        # tz-naive --> treat it as if it were UTC for purposes of timestamp()
        ts = Timestamp.now()
        uts = ts.replace(tzinfo=utc)
        assert ts.timestamp() == uts.timestamp()

        tsc = Timestamp("2014-10-11 11:00:01.12345678", tz="US/Central")
        utsc = tsc.tz_convert("UTC")

        # utsc is a different representation of the same time
        assert tsc.timestamp() == utsc.timestamp()

        # datetime.timestamp() converts in the local timezone
        with tm.set_timezone("UTC"):
            # should agree with datetime.timestamp method
            dt = ts.to_pydatetime()
            assert dt.timestamp() == ts.timestamp()


@pytest.mark.parametrize("fold", [0, 1])
def test_replace_preserves_fold(fold):
    # GH 37610. Check that replace preserves Timestamp fold property
    tz = gettz("Europe/Moscow")

    ts = Timestamp(year=2009, month=10, day=25, hour=2, minute=30, fold=fold, tzinfo=tz)
    ts_replaced = ts.replace(second=1)

    assert ts_replaced.fold == fold
