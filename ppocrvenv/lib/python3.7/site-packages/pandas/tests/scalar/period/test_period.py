from datetime import (
    date,
    datetime,
    timedelta,
)

import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import (
    iNaT,
    period as libperiod,
)
from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTHS,
)
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import (
    INVALID_FREQ_ERR_MSG,
    IncompatibleFrequency,
)
from pandas._libs.tslibs.timezones import (
    dateutil_gettz,
    maybe_get_tz,
)
from pandas.compat import np_datetime64_compat

import pandas as pd
from pandas import (
    NaT,
    Period,
    Timedelta,
    Timestamp,
    offsets,
)
import pandas._testing as tm


class TestPeriodConstruction:
    def test_construction(self):
        i1 = Period("1/1/2005", freq="M")
        i2 = Period("Jan 2005")

        assert i1 == i2

        i1 = Period("2005", freq="A")
        i2 = Period("2005")
        i3 = Period("2005", freq="a")

        assert i1 == i2
        assert i1 == i3

        i4 = Period("2005", freq="M")
        i5 = Period("2005", freq="m")

        assert i1 != i4
        assert i4 == i5

        i1 = Period.now("Q")
        i2 = Period(datetime.now(), freq="Q")
        i3 = Period.now("q")

        assert i1 == i2
        assert i1 == i3

        i1 = Period("1982", freq="min")
        i2 = Period("1982", freq="MIN")
        assert i1 == i2

        i1 = Period(year=2005, month=3, day=1, freq="D")
        i2 = Period("3/1/2005", freq="D")
        assert i1 == i2

        i3 = Period(year=2005, month=3, day=1, freq="d")
        assert i1 == i3

        i1 = Period("2007-01-01 09:00:00.001")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="L")
        assert i1 == expected

        expected = Period(np_datetime64_compat("2007-01-01 09:00:00.001Z"), freq="L")
        assert i1 == expected

        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="U")
        assert i1 == expected

        expected = Period(np_datetime64_compat("2007-01-01 09:00:00.00101Z"), freq="U")
        assert i1 == expected

        msg = "Must supply freq for ordinal value"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=200701)

        msg = "Invalid frequency: X"
        with pytest.raises(ValueError, match=msg):
            Period("2007-1-1", freq="X")

        # GH#34703 tuple freq disallowed
        with pytest.raises(TypeError, match="pass as a string instead"):
            Period("1982", freq=("Min", 1))

    def test_construction_bday(self):

        # Biz day construction, roll forward if non-weekday
        i1 = Period("3/10/12", freq="B")
        i2 = Period("3/10/12", freq="D")
        assert i1 == i2.asfreq("B")
        i2 = Period("3/11/12", freq="D")
        assert i1 == i2.asfreq("B")
        i2 = Period("3/12/12", freq="D")
        assert i1 == i2.asfreq("B")

        i3 = Period("3/10/12", freq="b")
        assert i1 == i3

        i1 = Period(year=2012, month=3, day=10, freq="B")
        i2 = Period("3/12/12", freq="B")
        assert i1 == i2

    def test_construction_quarter(self):

        i1 = Period(year=2005, quarter=1, freq="Q")
        i2 = Period("1/1/2005", freq="Q")
        assert i1 == i2

        i1 = Period(year=2005, quarter=3, freq="Q")
        i2 = Period("9/1/2005", freq="Q")
        assert i1 == i2

        i1 = Period("2005Q1")
        i2 = Period(year=2005, quarter=1, freq="Q")
        i3 = Period("2005q1")
        assert i1 == i2
        assert i1 == i3

        i1 = Period("05Q1")
        assert i1 == i2
        lower = Period("05q1")
        assert i1 == lower

        i1 = Period("1Q2005")
        assert i1 == i2
        lower = Period("1q2005")
        assert i1 == lower

        i1 = Period("1Q05")
        assert i1 == i2
        lower = Period("1q05")
        assert i1 == lower

        i1 = Period("4Q1984")
        assert i1.year == 1984
        lower = Period("4q1984")
        assert i1 == lower

    def test_construction_month(self):

        expected = Period("2007-01", freq="M")
        i1 = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period(200701, freq="M")
        assert i1 == expected

        i1 = Period(ordinal=200701, freq="M")
        assert i1.year == 18695

        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3 = Period(np.datetime64("2007-01-01"), freq="M")
        i4 = Period(np_datetime64_compat("2007-01-01 00:00:00Z"), freq="M")
        i5 = Period(np_datetime64_compat("2007-01-01 00:00:00.000Z"), freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

    def test_period_constructor_offsets(self):
        assert Period("1/1/2005", freq=offsets.MonthEnd()) == Period(
            "1/1/2005", freq="M"
        )
        assert Period("2005", freq=offsets.YearEnd()) == Period("2005", freq="A")
        assert Period("2005", freq=offsets.MonthEnd()) == Period("2005", freq="M")
        assert Period("3/10/12", freq=offsets.BusinessDay()) == Period(
            "3/10/12", freq="B"
        )
        assert Period("3/10/12", freq=offsets.Day()) == Period("3/10/12", freq="D")

        assert Period(
            year=2005, quarter=1, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=1, freq="Q")
        assert Period(
            year=2005, quarter=2, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=2, freq="Q")

        assert Period(year=2005, month=3, day=1, freq=offsets.Day()) == Period(
            year=2005, month=3, day=1, freq="D"
        )
        assert Period(year=2012, month=3, day=10, freq=offsets.BDay()) == Period(
            year=2012, month=3, day=10, freq="B"
        )

        expected = Period("2005-03-01", freq="3D")
        assert Period(year=2005, month=3, day=1, freq=offsets.Day(3)) == expected
        assert Period(year=2005, month=3, day=1, freq="3D") == expected

        assert Period(year=2012, month=3, day=10, freq=offsets.BDay(3)) == Period(
            year=2012, month=3, day=10, freq="3B"
        )

        assert Period(200701, freq=offsets.MonthEnd()) == Period(200701, freq="M")

        i1 = Period(ordinal=200701, freq=offsets.MonthEnd())
        i2 = Period(ordinal=200701, freq="M")
        assert i1 == i2
        assert i1.year == 18695
        assert i2.year == 18695

        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3 = Period(np.datetime64("2007-01-01"), freq="M")
        i4 = Period(np_datetime64_compat("2007-01-01 00:00:00Z"), freq="M")
        i5 = Period(np_datetime64_compat("2007-01-01 00:00:00.000Z"), freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

        i1 = Period("2007-01-01 09:00:00.001")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="L")
        assert i1 == expected

        expected = Period(np_datetime64_compat("2007-01-01 09:00:00.001Z"), freq="L")
        assert i1 == expected

        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="U")
        assert i1 == expected

        expected = Period(np_datetime64_compat("2007-01-01 09:00:00.00101Z"), freq="U")
        assert i1 == expected

    def test_invalid_arguments(self):
        msg = "Must supply freq for datetime value"
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now())
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now().date())

        msg = "Value must be Period, string, integer, or datetime"
        with pytest.raises(ValueError, match=msg):
            Period(1.6, freq="D")
        msg = "Ordinal must be an integer"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1.6, freq="D")
        msg = "Only value or ordinal but not both should be given but not both"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=2, value=1, freq="D")

        msg = "If value is None, freq cannot be None"
        with pytest.raises(ValueError, match=msg):
            Period(month=1)

        msg = "Given date string not likely a datetime"
        with pytest.raises(ValueError, match=msg):
            Period("-2000", "A")
        msg = "day is out of range for month"
        with pytest.raises(DateParseError, match=msg):
            Period("0", "A")
        msg = "Unknown datetime string format, unable to parse"
        with pytest.raises(DateParseError, match=msg):
            Period("1/1/-2000", "A")

    def test_constructor_corner(self):
        expected = Period("2007-01", freq="2M")
        assert Period(year=2007, month=1, freq="2M") == expected

        assert Period(None) is NaT

        p = Period("2007-01-01", freq="D")

        result = Period(p, freq="A")
        exp = Period("2007", freq="A")
        assert result == exp

    def test_constructor_infer_freq(self):
        p = Period("2007-01-01")
        assert p.freq == "D"

        p = Period("2007-01-01 07")
        assert p.freq == "H"

        p = Period("2007-01-01 07:10")
        assert p.freq == "T"

        p = Period("2007-01-01 07:10:15")
        assert p.freq == "S"

        p = Period("2007-01-01 07:10:15.123")
        assert p.freq == "L"

        p = Period("2007-01-01 07:10:15.123000")
        assert p.freq == "L"

        p = Period("2007-01-01 07:10:15.123400")
        assert p.freq == "U"

    def test_multiples(self):
        result1 = Period("1989", freq="2A")
        result2 = Period("1989", freq="A")
        assert result1.ordinal == result2.ordinal
        assert result1.freqstr == "2A-DEC"
        assert result2.freqstr == "A-DEC"
        assert result1.freq == offsets.YearEnd(2)
        assert result2.freq == offsets.YearEnd()

        assert (result1 + 1).ordinal == result1.ordinal + 2
        assert (1 + result1).ordinal == result1.ordinal + 2
        assert (result1 - 1).ordinal == result2.ordinal - 2
        assert (-1 + result1).ordinal == result2.ordinal - 2

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_quarterly(self, month):
        # bugs in scikits.timeseries
        freq = f"Q-{month}"
        exp = Period("1989Q3", freq=freq)
        assert "1989Q3" in str(exp)
        stamp = exp.to_timestamp("D", how="end")
        p = Period(stamp, freq=freq)
        assert p == exp

        stamp = exp.to_timestamp("3D", how="end")
        p = Period(stamp, freq=freq)
        assert p == exp

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_annual(self, month):
        # bugs in scikits.timeseries
        freq = f"A-{month}"
        exp = Period("1989", freq=freq)
        stamp = exp.to_timestamp("D", how="end") + timedelta(days=30)
        p = Period(stamp, freq=freq)

        assert p == exp + 1
        assert isinstance(p, Period)

    @pytest.mark.parametrize("day", DAYS)
    @pytest.mark.parametrize("num", range(10, 17))
    def test_period_cons_weekly(self, num, day):
        daystr = f"2011-02-{num}"
        freq = f"W-{day}"

        result = Period(daystr, freq=freq)
        expected = Period(daystr, freq="D").asfreq(freq)
        assert result == expected
        assert isinstance(result, Period)

    def test_period_from_ordinal(self):
        p = Period("2011-01", freq="M")
        res = Period._from_ordinal(p.ordinal, freq="M")
        assert p == res
        assert isinstance(res, Period)

    @pytest.mark.parametrize("freq", ["A", "M", "D", "H"])
    def test_construct_from_nat_string_and_freq(self, freq):
        per = Period("NaT", freq=freq)
        assert per is NaT

        per = Period("NaT", freq="2" + freq)
        assert per is NaT

        per = Period("NaT", freq="3" + freq)
        assert per is NaT

    def test_period_cons_nat(self):
        p = Period("nat", freq="W-SUN")
        assert p is NaT

        p = Period(iNaT, freq="D")
        assert p is NaT

        p = Period(iNaT, freq="3D")
        assert p is NaT

        p = Period(iNaT, freq="1D1H")
        assert p is NaT

        p = Period("NaT")
        assert p is NaT

        p = Period(iNaT)
        assert p is NaT

    def test_period_cons_mult(self):
        p1 = Period("2011-01", freq="3M")
        p2 = Period("2011-01", freq="M")
        assert p1.ordinal == p2.ordinal

        assert p1.freq == offsets.MonthEnd(3)
        assert p1.freqstr == "3M"

        assert p2.freq == offsets.MonthEnd()
        assert p2.freqstr == "M"

        result = p1 + 1
        assert result.ordinal == (p2 + 3).ordinal

        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        result = p1 - 1
        assert result.ordinal == (p2 - 3).ordinal
        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        msg = "Frequency must be positive, because it represents span: -3M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-3M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0M")

    def test_period_cons_combined(self):
        p = [
            (
                Period("2011-01", freq="1D1H"),
                Period("2011-01", freq="1H1D"),
                Period("2011-01", freq="H"),
            ),
            (
                Period(ordinal=1, freq="1D1H"),
                Period(ordinal=1, freq="1H1D"),
                Period(ordinal=1, freq="H"),
            ),
        ]

        for p1, p2, p3 in p:
            assert p1.ordinal == p3.ordinal
            assert p2.ordinal == p3.ordinal

            assert p1.freq == offsets.Hour(25)
            assert p1.freqstr == "25H"

            assert p2.freq == offsets.Hour(25)
            assert p2.freqstr == "25H"

            assert p3.freq == offsets.Hour()
            assert p3.freqstr == "H"

            result = p1 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25H"

            result = p2 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25H"

            result = p1 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25H"

            result = p2 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25H"

        msg = "Frequency must be positive, because it represents span: -25H"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1D1H")
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1H1D")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1D1H")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1H1D")

        msg = "Frequency must be positive, because it represents span: 0D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0D0H")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="0D0H")

        # You can only combine together day and intraday offsets
        msg = "Invalid frequency: 1W1D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1W1D")
        msg = "Invalid frequency: 1D1W"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1D1W")

    @pytest.mark.parametrize("day", ["1970/01/01 ", "2020-12-31 ", "1981/09/13 "])
    @pytest.mark.parametrize("hour", ["00:00:00", "00:00:01", "23:59:59", "12:00:59"])
    @pytest.mark.parametrize(
        "sec_float, expected",
        [
            (".000000001", 1),
            (".000000999", 999),
            (".123456789", 789),
            (".999999999", 999),
        ],
    )
    def test_period_constructor_nanosecond(self, day, hour, sec_float, expected):
        # GH 34621

        assert Period(day + hour + sec_float).start_time.nanosecond == expected

    @pytest.mark.parametrize("hour", range(24))
    def test_period_large_ordinal(self, hour):
        # Issue #36430
        # Integer overflow for Period over the maximum timestamp
        p = Period(ordinal=2562048 + hour, freq="1H")
        assert p.hour == hour


class TestPeriodMethods:
    def test_round_trip(self):
        p = Period("2000Q1")
        new_p = tm.round_trip_pickle(p)
        assert new_p == p

    def test_hash(self):
        assert hash(Period("2011-01", freq="M")) == hash(Period("2011-01", freq="M"))

        assert hash(Period("2011-01-01", freq="D")) != hash(Period("2011-01", freq="M"))

        assert hash(Period("2011-01", freq="3M")) != hash(Period("2011-01", freq="2M"))

        assert hash(Period("2011-01", freq="M")) != hash(Period("2011-02", freq="M"))

    # --------------------------------------------------------------
    # to_timestamp

    @pytest.mark.parametrize("tzstr", ["Europe/Brussels", "Asia/Tokyo", "US/Pacific"])
    def test_to_timestamp_tz_arg(self, tzstr):
        # GH#34522 tz kwarg deprecated
        with tm.assert_produces_warning(FutureWarning):
            p = Period("1/1/2005", freq="M").to_timestamp(tz=tzstr)
        exp = Timestamp("1/1/2005", tz="UTC").tz_convert(tzstr)
        exp_zone = pytz.timezone(tzstr).normalize(p)

        assert p == exp
        assert p.tz == exp_zone.tzinfo
        assert p.tz == exp.tz

        with tm.assert_produces_warning(FutureWarning):
            p = Period("1/1/2005", freq="3H").to_timestamp(tz=tzstr)
        exp = Timestamp("1/1/2005", tz="UTC").tz_convert(tzstr)
        exp_zone = pytz.timezone(tzstr).normalize(p)

        assert p == exp
        assert p.tz == exp_zone.tzinfo
        assert p.tz == exp.tz

        with tm.assert_produces_warning(FutureWarning):
            p = Period("1/1/2005", freq="A").to_timestamp(freq="A", tz=tzstr)
        exp = Timestamp("31/12/2005", tz="UTC").tz_convert(tzstr)
        exp_zone = pytz.timezone(tzstr).normalize(p)

        assert p == exp
        assert p.tz == exp_zone.tzinfo
        assert p.tz == exp.tz

        with tm.assert_produces_warning(FutureWarning):
            p = Period("1/1/2005", freq="A").to_timestamp(freq="3H", tz=tzstr)
        exp = Timestamp("1/1/2005", tz="UTC").tz_convert(tzstr)
        exp_zone = pytz.timezone(tzstr).normalize(p)

        assert p == exp
        assert p.tz == exp_zone.tzinfo
        assert p.tz == exp.tz

    @pytest.mark.parametrize(
        "tzstr",
        ["dateutil/Europe/Brussels", "dateutil/Asia/Tokyo", "dateutil/US/Pacific"],
    )
    def test_to_timestamp_tz_arg_dateutil(self, tzstr):
        tz = maybe_get_tz(tzstr)
        with tm.assert_produces_warning(FutureWarning):
            p = Period("1/1/2005", freq="M").to_timestamp(tz=tz)
        exp = Timestamp("1/1/2005", tz="UTC").tz_convert(tzstr)
        assert p == exp
        assert p.tz == dateutil_gettz(tzstr.split("/", 1)[1])
        assert p.tz == exp.tz

        with tm.assert_produces_warning(FutureWarning):
            p = Period("1/1/2005", freq="M").to_timestamp(freq="3H", tz=tz)
        exp = Timestamp("1/1/2005", tz="UTC").tz_convert(tzstr)
        assert p == exp
        assert p.tz == dateutil_gettz(tzstr.split("/", 1)[1])
        assert p.tz == exp.tz

    def test_to_timestamp_tz_arg_dateutil_from_string(self):
        with tm.assert_produces_warning(FutureWarning):
            p = Period("1/1/2005", freq="M").to_timestamp(tz="dateutil/Europe/Brussels")
        assert p.tz == dateutil_gettz("Europe/Brussels")

    def test_to_timestamp_mult(self):
        p = Period("2011-01", freq="M")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        expected = Timestamp("2011-02-01") - Timedelta(1, "ns")
        assert p.to_timestamp(how="E") == expected

        p = Period("2011-01", freq="3M")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        expected = Timestamp("2011-04-01") - Timedelta(1, "ns")
        assert p.to_timestamp(how="E") == expected

    def test_to_timestamp(self):
        p = Period("1982", freq="A")
        start_ts = p.to_timestamp(how="S")
        aliases = ["s", "StarT", "BEGIn"]
        for a in aliases:
            assert start_ts == p.to_timestamp("D", how=a)
            # freq with mult should not affect to the result
            assert start_ts == p.to_timestamp("3D", how=a)

        end_ts = p.to_timestamp(how="E")
        aliases = ["e", "end", "FINIsH"]
        for a in aliases:
            assert end_ts == p.to_timestamp("D", how=a)
            assert end_ts == p.to_timestamp("3D", how=a)

        from_lst = ["A", "Q", "M", "W", "B", "D", "H", "Min", "S"]

        def _ex(p):
            if p.freq == "B":
                return p.start_time + Timedelta(days=1, nanoseconds=-1)
            return Timestamp((p + p.freq).start_time.value - 1)

        for fcode in from_lst:
            p = Period("1982", freq=fcode)
            result = p.to_timestamp().to_period(fcode)
            assert result == p

            assert p.start_time == p.to_timestamp(how="S")

            assert p.end_time == _ex(p)

        # Frequency other than daily

        p = Period("1985", freq="A")

        result = p.to_timestamp("H", how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected
        result = p.to_timestamp("3H", how="end")
        assert result == expected

        result = p.to_timestamp("T", how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected
        result = p.to_timestamp("2T", how="end")
        assert result == expected

        result = p.to_timestamp(how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected

        expected = datetime(1985, 1, 1)
        result = p.to_timestamp("H", how="start")
        assert result == expected
        result = p.to_timestamp("T", how="start")
        assert result == expected
        result = p.to_timestamp("S", how="start")
        assert result == expected
        result = p.to_timestamp("3H", how="start")
        assert result == expected
        result = p.to_timestamp("5S", how="start")
        assert result == expected

    def test_to_timestamp_business_end(self):
        per = Period("1990-01-05", "B")  # Friday
        result = per.to_timestamp("B", how="E")

        expected = Timestamp("1990-01-06") - Timedelta(nanoseconds=1)
        assert result == expected

    @pytest.mark.parametrize(
        "ts, expected",
        [
            ("1970-01-01 00:00:00", 0),
            ("1970-01-01 00:00:00.000001", 1),
            ("1970-01-01 00:00:00.00001", 10),
            ("1970-01-01 00:00:00.499", 499000),
            ("1999-12-31 23:59:59.999", 999000),
            ("1999-12-31 23:59:59.999999", 999999),
            ("2050-12-31 23:59:59.5", 500000),
            ("2050-12-31 23:59:59.500001", 500001),
            ("2050-12-31 23:59:59.123456", 123456),
        ],
    )
    @pytest.mark.parametrize("freq", [None, "us", "ns"])
    def test_to_timestamp_microsecond(self, ts, expected, freq):
        # GH 24444
        result = Period(ts).to_timestamp(freq=freq).microsecond
        assert result == expected

    # --------------------------------------------------------------
    # Rendering: __repr__, strftime, etc

    def test_repr(self):
        p = Period("Jan-2000")
        assert "2000-01" in repr(p)

        p = Period("2000-12-15")
        assert "2000-12-15" in repr(p)

    def test_repr_nat(self):
        p = Period("nat", freq="M")
        assert repr(NaT) in repr(p)

    def test_millisecond_repr(self):
        p = Period("2000-01-01 12:15:02.123")

        assert repr(p) == "Period('2000-01-01 12:15:02.123', 'L')"

    def test_microsecond_repr(self):
        p = Period("2000-01-01 12:15:02.123567")

        assert repr(p) == "Period('2000-01-01 12:15:02.123567', 'U')"

    def test_strftime(self):
        # GH#3363
        p = Period("2000-1-1 12:34:12", freq="S")
        res = p.strftime("%Y-%m-%d %H:%M:%S")
        assert res == "2000-01-01 12:34:12"
        assert isinstance(res, str)


class TestPeriodProperties:
    """Test properties such as year, month, weekday, etc...."""

    @pytest.mark.parametrize("freq", ["A", "M", "D", "H"])
    def test_is_leap_year(self, freq):
        # GH 13727
        p = Period("2000-01-01 00:00:00", freq=freq)
        assert p.is_leap_year
        assert isinstance(p.is_leap_year, bool)

        p = Period("1999-01-01 00:00:00", freq=freq)
        assert not p.is_leap_year

        p = Period("2004-01-01 00:00:00", freq=freq)
        assert p.is_leap_year

        p = Period("2100-01-01 00:00:00", freq=freq)
        assert not p.is_leap_year

    def test_quarterly_negative_ordinals(self):
        p = Period(ordinal=-1, freq="Q-DEC")
        assert p.year == 1969
        assert p.quarter == 4
        assert isinstance(p, Period)

        p = Period(ordinal=-2, freq="Q-DEC")
        assert p.year == 1969
        assert p.quarter == 3
        assert isinstance(p, Period)

        p = Period(ordinal=-2, freq="M")
        assert p.year == 1969
        assert p.month == 11
        assert isinstance(p, Period)

    def test_freq_str(self):
        i1 = Period("1982", freq="Min")
        assert i1.freq == offsets.Minute()
        assert i1.freqstr == "T"

    def test_period_deprecated_freq(self):
        cases = {
            "M": ["MTH", "MONTH", "MONTHLY", "Mth", "month", "monthly"],
            "B": ["BUS", "BUSINESS", "BUSINESSLY", "WEEKDAY", "bus"],
            "D": ["DAY", "DLY", "DAILY", "Day", "Dly", "Daily"],
            "H": ["HR", "HOUR", "HRLY", "HOURLY", "hr", "Hour", "HRly"],
            "T": ["minute", "MINUTE", "MINUTELY", "minutely"],
            "S": ["sec", "SEC", "SECOND", "SECONDLY", "second"],
            "L": ["MILLISECOND", "MILLISECONDLY", "millisecond"],
            "U": ["MICROSECOND", "MICROSECONDLY", "microsecond"],
            "N": ["NANOSECOND", "NANOSECONDLY", "nanosecond"],
        }

        msg = INVALID_FREQ_ERR_MSG
        for exp, freqs in cases.items():
            for freq in freqs:
                with pytest.raises(ValueError, match=msg):
                    Period("2016-03-01 09:00", freq=freq)
                with pytest.raises(ValueError, match=msg):
                    Period(ordinal=1, freq=freq)

            # check supported freq-aliases still works
            p1 = Period("2016-03-01 09:00", freq=exp)
            p2 = Period(ordinal=1, freq=exp)
            assert isinstance(p1, Period)
            assert isinstance(p2, Period)

    def _period_constructor(bound, offset):
        return Period(
            year=bound.year,
            month=bound.month,
            day=bound.day,
            hour=bound.hour,
            minute=bound.minute,
            second=bound.second + offset,
            freq="us",
        )

    @pytest.mark.parametrize("bound, offset", [(Timestamp.min, -1), (Timestamp.max, 1)])
    @pytest.mark.parametrize("period_property", ["start_time", "end_time"])
    def test_outter_bounds_start_and_end_time(self, bound, offset, period_property):
        # GH #13346
        period = TestPeriodProperties._period_constructor(bound, offset)
        with pytest.raises(OutOfBoundsDatetime, match="Out of bounds nanosecond"):
            getattr(period, period_property)

    @pytest.mark.parametrize("bound, offset", [(Timestamp.min, -1), (Timestamp.max, 1)])
    @pytest.mark.parametrize("period_property", ["start_time", "end_time"])
    def test_inner_bounds_start_and_end_time(self, bound, offset, period_property):
        # GH #13346
        period = TestPeriodProperties._period_constructor(bound, -offset)
        expected = period.to_timestamp().round(freq="S")
        assert getattr(period, period_property).round(freq="S") == expected
        expected = (bound - offset * Timedelta(1, unit="S")).floor("S")
        assert getattr(period, period_property).floor("S") == expected

    def test_start_time(self):
        freq_lst = ["A", "Q", "M", "D", "H", "T", "S"]
        xp = datetime(2012, 1, 1)
        for f in freq_lst:
            p = Period("2012", freq=f)
            assert p.start_time == xp
        assert Period("2012", freq="B").start_time == datetime(2012, 1, 2)
        assert Period("2012", freq="W").start_time == datetime(2011, 12, 26)

    def test_end_time(self):
        p = Period("2012", freq="A")

        def _ex(*args):
            return Timestamp(Timestamp(datetime(*args)).value - 1)

        xp = _ex(2013, 1, 1)
        assert xp == p.end_time

        p = Period("2012", freq="Q")
        xp = _ex(2012, 4, 1)
        assert xp == p.end_time

        p = Period("2012", freq="M")
        xp = _ex(2012, 2, 1)
        assert xp == p.end_time

        p = Period("2012", freq="D")
        xp = _ex(2012, 1, 2)
        assert xp == p.end_time

        p = Period("2012", freq="H")
        xp = _ex(2012, 1, 1, 1)
        assert xp == p.end_time

        p = Period("2012", freq="B")
        xp = _ex(2012, 1, 3)
        assert xp == p.end_time

        p = Period("2012", freq="W")
        xp = _ex(2012, 1, 2)
        assert xp == p.end_time

        # Test for GH 11738
        p = Period("2012", freq="15D")
        xp = _ex(2012, 1, 16)
        assert xp == p.end_time

        p = Period("2012", freq="1D1H")
        xp = _ex(2012, 1, 2, 1)
        assert xp == p.end_time

        p = Period("2012", freq="1H1D")
        xp = _ex(2012, 1, 2, 1)
        assert xp == p.end_time

    def test_end_time_business_friday(self):
        # GH#34449
        per = Period("1990-01-05", "B")
        result = per.end_time

        expected = Timestamp("1990-01-06") - Timedelta(nanoseconds=1)
        assert result == expected

    def test_anchor_week_end_time(self):
        def _ex(*args):
            return Timestamp(Timestamp(datetime(*args)).value - 1)

        p = Period("2013-1-1", "W-SAT")
        xp = _ex(2013, 1, 6)
        assert p.end_time == xp

    def test_properties_annually(self):
        # Test properties on Periods with annually frequency.
        a_date = Period(freq="A", year=2007)
        assert a_date.year == 2007

    def test_properties_quarterly(self):
        # Test properties on Periods with daily frequency.
        qedec_date = Period(freq="Q-DEC", year=2007, quarter=1)
        qejan_date = Period(freq="Q-JAN", year=2007, quarter=1)
        qejun_date = Period(freq="Q-JUN", year=2007, quarter=1)
        #
        for x in range(3):
            for qd in (qedec_date, qejan_date, qejun_date):
                assert (qd + x).qyear == 2007
                assert (qd + x).quarter == x + 1

    def test_properties_monthly(self):
        # Test properties on Periods with daily frequency.
        m_date = Period(freq="M", year=2007, month=1)
        for x in range(11):
            m_ival_x = m_date + x
            assert m_ival_x.year == 2007
            if 1 <= x + 1 <= 3:
                assert m_ival_x.quarter == 1
            elif 4 <= x + 1 <= 6:
                assert m_ival_x.quarter == 2
            elif 7 <= x + 1 <= 9:
                assert m_ival_x.quarter == 3
            elif 10 <= x + 1 <= 12:
                assert m_ival_x.quarter == 4
            assert m_ival_x.month == x + 1

    def test_properties_weekly(self):
        # Test properties on Periods with daily frequency.
        w_date = Period(freq="W", year=2007, month=1, day=7)
        #
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31
        assert Period(freq="W", year=2012, month=2, day=1).days_in_month == 29

    def test_properties_weekly_legacy(self):
        # Test properties on Periods with daily frequency.
        w_date = Period(freq="W", year=2007, month=1, day=7)
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31

        exp = Period(freq="W", year=2012, month=2, day=1)
        assert exp.days_in_month == 29

        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK", year=2007, month=1, day=7)

    def test_properties_daily(self):
        # Test properties on Periods with daily frequency.
        b_date = Period(freq="B", year=2007, month=1, day=1)
        #
        assert b_date.year == 2007
        assert b_date.quarter == 1
        assert b_date.month == 1
        assert b_date.day == 1
        assert b_date.weekday == 0
        assert b_date.dayofyear == 1
        assert b_date.days_in_month == 31
        assert Period(freq="B", year=2012, month=2, day=1).days_in_month == 29

        d_date = Period(freq="D", year=2007, month=1, day=1)

        assert d_date.year == 2007
        assert d_date.quarter == 1
        assert d_date.month == 1
        assert d_date.day == 1
        assert d_date.weekday == 0
        assert d_date.dayofyear == 1
        assert d_date.days_in_month == 31
        assert Period(freq="D", year=2012, month=2, day=1).days_in_month == 29

    def test_properties_hourly(self):
        # Test properties on Periods with hourly frequency.
        h_date1 = Period(freq="H", year=2007, month=1, day=1, hour=0)
        h_date2 = Period(freq="2H", year=2007, month=1, day=1, hour=0)

        for h_date in [h_date1, h_date2]:
            assert h_date.year == 2007
            assert h_date.quarter == 1
            assert h_date.month == 1
            assert h_date.day == 1
            assert h_date.weekday == 0
            assert h_date.dayofyear == 1
            assert h_date.hour == 0
            assert h_date.days_in_month == 31
            assert (
                Period(freq="H", year=2012, month=2, day=1, hour=0).days_in_month == 29
            )

    def test_properties_minutely(self):
        # Test properties on Periods with minutely frequency.
        t_date = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)
        #
        assert t_date.quarter == 1
        assert t_date.month == 1
        assert t_date.day == 1
        assert t_date.weekday == 0
        assert t_date.dayofyear == 1
        assert t_date.hour == 0
        assert t_date.minute == 0
        assert t_date.days_in_month == 31
        assert (
            Period(freq="D", year=2012, month=2, day=1, hour=0, minute=0).days_in_month
            == 29
        )

    def test_properties_secondly(self):
        # Test properties on Periods with secondly frequency.
        s_date = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        #
        assert s_date.year == 2007
        assert s_date.quarter == 1
        assert s_date.month == 1
        assert s_date.day == 1
        assert s_date.weekday == 0
        assert s_date.dayofyear == 1
        assert s_date.hour == 0
        assert s_date.minute == 0
        assert s_date.second == 0
        assert s_date.days_in_month == 31
        assert (
            Period(
                freq="Min", year=2012, month=2, day=1, hour=0, minute=0, second=0
            ).days_in_month
            == 29
        )


class TestPeriodField:
    def test_get_period_field_array_raises_on_out_of_range(self):
        msg = "Buffer dtype mismatch, expected 'const int64_t' but got 'double'"
        with pytest.raises(ValueError, match=msg):
            libperiod.get_period_field_arr(-1, np.empty(1), 0)


class TestPeriodComparisons:
    def test_comparison_same_period_different_object(self):
        # Separate Period objects for the same period
        left = Period("2000-01", "M")
        right = Period("2000-01", "M")

        assert left == right
        assert left >= right
        assert left <= right
        assert not left < right
        assert not left > right

    def test_comparison_same_freq(self):
        jan = Period("2000-01", "M")
        feb = Period("2000-02", "M")

        assert not jan == feb
        assert jan != feb
        assert jan < feb
        assert jan <= feb
        assert not jan > feb
        assert not jan >= feb

    def test_comparison_mismatched_freq(self):
        jan = Period("2000-01", "M")
        day = Period("2012-01-01", "D")

        assert not jan == day
        assert jan != day
        msg = r"Input has different freq=D from Period\(freq=M\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan < day
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan <= day
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan > day
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan >= day

    def test_comparison_invalid_type(self):
        jan = Period("2000-01", "M")

        assert not jan == 1
        assert jan != 1

        int_or_per = "'(Period|int)'"
        msg = f"not supported between instances of {int_or_per} and {int_or_per}"
        for left, right in [(jan, 1), (1, jan)]:

            with pytest.raises(TypeError, match=msg):
                left > right
            with pytest.raises(TypeError, match=msg):
                left >= right
            with pytest.raises(TypeError, match=msg):
                left < right
            with pytest.raises(TypeError, match=msg):
                left <= right

    def test_sort_periods(self):
        jan = Period("2000-01", "M")
        feb = Period("2000-02", "M")
        mar = Period("2000-03", "M")
        periods = [mar, jan, feb]
        correctPeriods = [jan, feb, mar]
        assert sorted(periods) == correctPeriods

    def test_period_cmp_nat(self):
        p = Period("2011-01-01", freq="D")

        t = Timestamp("2011-01-01")
        # confirm Period('NaT') work identical with Timestamp('NaT')
        for left, right in [
            (NaT, p),
            (p, NaT),
            (NaT, t),
            (t, NaT),
        ]:
            assert not left < right
            assert not left > right
            assert not left == right
            assert left != right
            assert not left <= right
            assert not left >= right


class TestArithmetic:
    def test_sub_delta(self):
        left, right = Period("2011", freq="A"), Period("2007", freq="A")
        result = left - right
        assert result == 4 * right.freq

        msg = r"Input has different freq=M from Period\(freq=A-DEC\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            left - Period("2007-01", freq="M")

    def test_add_integer(self):
        per1 = Period(freq="D", year=2008, month=1, day=1)
        per2 = Period(freq="D", year=2008, month=1, day=2)
        assert per1 + 1 == per2
        assert 1 + per1 == per2

    def test_add_sub_nat(self):
        # GH#13071
        p = Period("2011-01", freq="M")
        assert p + NaT is NaT
        assert NaT + p is NaT
        assert p - NaT is NaT
        assert NaT - p is NaT

    def test_add_invalid(self):
        # GH#4731
        per1 = Period(freq="D", year=2008, month=1, day=1)
        per2 = Period(freq="D", year=2008, month=1, day=2)

        msg = "|".join(
            [
                r"unsupported operand type\(s\)",
                "can only concatenate str",
                "must be str, not Period",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            per1 + "str"
        with pytest.raises(TypeError, match=msg):
            "str" + per1
        with pytest.raises(TypeError, match=msg):
            per1 + per2

    boxes = [lambda x: x, lambda x: pd.Series([x]), lambda x: pd.Index([x])]
    ids = ["identity", "Series", "Index"]

    @pytest.mark.parametrize("lbox", boxes, ids=ids)
    @pytest.mark.parametrize("rbox", boxes, ids=ids)
    def test_add_timestamp_raises(self, rbox, lbox):
        # GH#17983
        ts = Timestamp("2017")
        per = Period("2017", freq="M")

        # We may get a different message depending on which class raises
        # the error.
        msg = "|".join(
            [
                "cannot add",
                "unsupported operand",
                "can only operate on a",
                "incompatible type",
                "ufunc add cannot use operands",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            lbox(ts) + rbox(per)

        with pytest.raises(TypeError, match=msg):
            lbox(per) + rbox(ts)

        with pytest.raises(TypeError, match=msg):
            lbox(per) + rbox(per)

    def test_sub(self):
        per1 = Period("2011-01-01", freq="D")
        per2 = Period("2011-01-15", freq="D")

        off = per1.freq
        assert per1 - per2 == -14 * off
        assert per2 - per1 == 14 * off

        msg = r"Input has different freq=M from Period\(freq=D\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            per1 - Period("2011-02", freq="M")

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_sub_n_gt_1_ticks(self, tick_classes, n):
        # GH 23878
        p1 = Period("19910905", freq=tick_classes(n))
        p2 = Period("19920406", freq=tick_classes(n))

        expected = Period(str(p2), freq=p2.freq.base) - Period(
            str(p1), freq=p1.freq.base
        )

        assert (p2 - p1) == expected

    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize(
        "offset, kwd_name",
        [
            (offsets.YearEnd, "month"),
            (offsets.QuarterEnd, "startingMonth"),
            (offsets.MonthEnd, None),
            (offsets.Week, "weekday"),
        ],
    )
    def test_sub_n_gt_1_offsets(self, offset, kwd_name, n, normalize):
        # GH 23878
        kwds = {kwd_name: 3} if kwd_name is not None else {}
        p1_d = "19910905"
        p2_d = "19920406"
        p1 = Period(p1_d, freq=offset(n, normalize, **kwds))
        p2 = Period(p2_d, freq=offset(n, normalize, **kwds))

        expected = Period(p2_d, freq=p2.freq.base) - Period(p1_d, freq=p1.freq.base)

        assert (p2 - p1) == expected

    def test_add_offset(self):
        # freq is DateOffset
        for freq in ["A", "2A", "3A"]:
            p = Period("2011", freq=freq)
            exp = Period("2013", freq=freq)
            assert p + offsets.YearEnd(2) == exp
            assert offsets.YearEnd(2) + p == exp

            for o in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(365, "D"),
                timedelta(365),
            ]:
                msg = "Input has different freq|Input cannot be converted to Period"
                with pytest.raises(IncompatibleFrequency, match=msg):
                    p + o

                if isinstance(o, np.timedelta64):
                    msg = "cannot use operands with types"
                    with pytest.raises(TypeError, match=msg):
                        o + p
                else:
                    msg = "|".join(
                        [
                            "Input has different freq",
                            "Input cannot be converted to Period",
                        ]
                    )
                    with pytest.raises(IncompatibleFrequency, match=msg):
                        o + p

        for freq in ["M", "2M", "3M"]:
            p = Period("2011-03", freq=freq)
            exp = Period("2011-05", freq=freq)
            assert p + offsets.MonthEnd(2) == exp
            assert offsets.MonthEnd(2) + p == exp

            exp = Period("2012-03", freq=freq)
            assert p + offsets.MonthEnd(12) == exp
            assert offsets.MonthEnd(12) + p == exp

            for o in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(365, "D"),
                timedelta(365),
            ]:
                msg = "Input has different freq|Input cannot be converted to Period"
                with pytest.raises(IncompatibleFrequency, match=msg):
                    p + o

                if isinstance(o, np.timedelta64):
                    msg = "cannot use operands with types"
                    with pytest.raises(TypeError, match=msg):
                        o + p
                else:
                    msg = "|".join(
                        [
                            "Input has different freq",
                            "Input cannot be converted to Period",
                        ]
                    )
                    with pytest.raises(IncompatibleFrequency, match=msg):
                        o + p

        # freq is Tick
        for freq in ["D", "2D", "3D"]:
            p = Period("2011-04-01", freq=freq)

            exp = Period("2011-04-06", freq=freq)
            assert p + offsets.Day(5) == exp
            assert offsets.Day(5) + p == exp

            exp = Period("2011-04-02", freq=freq)
            assert p + offsets.Hour(24) == exp
            assert offsets.Hour(24) + p == exp

            exp = Period("2011-04-03", freq=freq)
            assert p + np.timedelta64(2, "D") == exp
            msg = "cannot use operands with types"
            with pytest.raises(TypeError, match=msg):
                np.timedelta64(2, "D") + p

            exp = Period("2011-04-02", freq=freq)
            assert p + np.timedelta64(3600 * 24, "s") == exp
            with pytest.raises(TypeError, match=msg):
                np.timedelta64(3600 * 24, "s") + p

            exp = Period("2011-03-30", freq=freq)
            assert p + timedelta(-2) == exp
            assert timedelta(-2) + p == exp

            exp = Period("2011-04-03", freq=freq)
            assert p + timedelta(hours=48) == exp
            assert timedelta(hours=48) + p == exp

            for o in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(4, "h"),
                timedelta(hours=23),
            ]:
                msg = "Input has different freq|Input cannot be converted to Period"
                with pytest.raises(IncompatibleFrequency, match=msg):
                    p + o

                if isinstance(o, np.timedelta64):
                    msg = "cannot use operands with types"
                    with pytest.raises(TypeError, match=msg):
                        o + p
                else:
                    msg = "|".join(
                        [
                            "Input has different freq",
                            "Input cannot be converted to Period",
                        ]
                    )
                    with pytest.raises(IncompatibleFrequency, match=msg):
                        o + p

        for freq in ["H", "2H", "3H"]:
            p = Period("2011-04-01 09:00", freq=freq)

            exp = Period("2011-04-03 09:00", freq=freq)
            assert p + offsets.Day(2) == exp
            assert offsets.Day(2) + p == exp

            exp = Period("2011-04-01 12:00", freq=freq)
            assert p + offsets.Hour(3) == exp
            assert offsets.Hour(3) + p == exp

            msg = "cannot use operands with types"
            exp = Period("2011-04-01 12:00", freq=freq)
            assert p + np.timedelta64(3, "h") == exp
            with pytest.raises(TypeError, match=msg):
                np.timedelta64(3, "h") + p

            exp = Period("2011-04-01 10:00", freq=freq)
            assert p + np.timedelta64(3600, "s") == exp
            with pytest.raises(TypeError, match=msg):
                np.timedelta64(3600, "s") + p

            exp = Period("2011-04-01 11:00", freq=freq)
            assert p + timedelta(minutes=120) == exp
            assert timedelta(minutes=120) + p == exp

            exp = Period("2011-04-05 12:00", freq=freq)
            assert p + timedelta(days=4, minutes=180) == exp
            assert timedelta(days=4, minutes=180) + p == exp

            for o in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(3200, "s"),
                timedelta(hours=23, minutes=30),
            ]:
                msg = "Input has different freq|Input cannot be converted to Period"
                with pytest.raises(IncompatibleFrequency, match=msg):
                    p + o

                if isinstance(o, np.timedelta64):
                    msg = "cannot use operands with types"
                    with pytest.raises(TypeError, match=msg):
                        o + p
                else:
                    msg = "|".join(
                        [
                            "Input has different freq",
                            "Input cannot be converted to Period",
                        ]
                    )
                    with pytest.raises(IncompatibleFrequency, match=msg):
                        o + p

    def test_sub_offset(self):
        # freq is DateOffset
        msg = "Input has different freq|Input cannot be converted to Period"
        for freq in ["A", "2A", "3A"]:
            p = Period("2011", freq=freq)
            assert p - offsets.YearEnd(2) == Period("2009", freq=freq)

            for o in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(365, "D"),
                timedelta(365),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    p - o

        for freq in ["M", "2M", "3M"]:
            p = Period("2011-03", freq=freq)
            assert p - offsets.MonthEnd(2) == Period("2011-01", freq=freq)
            assert p - offsets.MonthEnd(12) == Period("2010-03", freq=freq)

            for o in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(365, "D"),
                timedelta(365),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    p - o

        # freq is Tick
        for freq in ["D", "2D", "3D"]:
            p = Period("2011-04-01", freq=freq)
            assert p - offsets.Day(5) == Period("2011-03-27", freq=freq)
            assert p - offsets.Hour(24) == Period("2011-03-31", freq=freq)
            assert p - np.timedelta64(2, "D") == Period("2011-03-30", freq=freq)
            assert p - np.timedelta64(3600 * 24, "s") == Period("2011-03-31", freq=freq)
            assert p - timedelta(-2) == Period("2011-04-03", freq=freq)
            assert p - timedelta(hours=48) == Period("2011-03-30", freq=freq)

            for o in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(4, "h"),
                timedelta(hours=23),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    p - o

        for freq in ["H", "2H", "3H"]:
            p = Period("2011-04-01 09:00", freq=freq)
            assert p - offsets.Day(2) == Period("2011-03-30 09:00", freq=freq)
            assert p - offsets.Hour(3) == Period("2011-04-01 06:00", freq=freq)
            assert p - np.timedelta64(3, "h") == Period("2011-04-01 06:00", freq=freq)
            assert p - np.timedelta64(3600, "s") == Period(
                "2011-04-01 08:00", freq=freq
            )
            assert p - timedelta(minutes=120) == Period("2011-04-01 07:00", freq=freq)
            assert p - timedelta(days=4, minutes=180) == Period(
                "2011-03-28 06:00", freq=freq
            )

            for o in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(3200, "s"),
                timedelta(hours=23, minutes=30),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    p - o

    @pytest.mark.parametrize("freq", ["M", "2M", "3M"])
    def test_period_addsub_nat(self, freq):
        per = Period("2011-01", freq=freq)

        # For subtraction, NaT is treated as another Period object
        assert NaT - per is NaT
        assert per - NaT is NaT

        # For addition, NaT is treated as offset-like
        assert NaT + per is NaT
        assert per + NaT is NaT

    def test_period_ops_offset(self):
        p = Period("2011-04-01", freq="D")
        result = p + offsets.Day()
        exp = Period("2011-04-02", freq="D")
        assert result == exp

        result = p - offsets.Day(2)
        exp = Period("2011-03-30", freq="D")
        assert result == exp

        msg = r"Input cannot be converted to Period\(freq=D\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            p + offsets.Hour(2)

        with pytest.raises(IncompatibleFrequency, match=msg):
            p - offsets.Hour(2)


def test_period_immutable():
    # see gh-17116
    msg = "not writable"

    per = Period("2014Q1")
    with pytest.raises(AttributeError, match=msg):
        per.ordinal = 14

    freq = per.freq
    with pytest.raises(AttributeError, match=msg):
        per.freq = 2 * freq


def test_small_year_parsing():
    per1 = Period("0001-01-07", "D")
    assert per1.year == 1
    assert per1.day == 7


def test_negone_ordinals():
    freqs = ["A", "M", "Q", "D", "H", "T", "S"]

    period = Period(ordinal=-1, freq="D")
    for freq in freqs:
        repr(period.asfreq(freq))

    for freq in freqs:
        period = Period(ordinal=-1, freq=freq)
        repr(period)
        assert period.year == 1969

    period = Period(ordinal=-1, freq="B")
    repr(period)
    period = Period(ordinal=-1, freq="W")
    repr(period)


def test_invalid_frequency_error_message():
    msg = "Invalid frequency: <WeekOfMonth: week=0, weekday=0>"
    with pytest.raises(ValueError, match=msg):
        Period("2012-01-02", freq="WOM-1MON")
