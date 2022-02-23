""" test to_datetime """

import calendar
from collections import deque
from datetime import (
    datetime,
    timedelta,
)
from decimal import Decimal
import locale

from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz

from pandas._libs import tslib
from pandas._libs.tslibs import (
    iNaT,
    parsing,
)
from pandas.errors import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_datetime64_ns_dtype

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Series,
    Timestamp,
    date_range,
    isna,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at


class TestTimeConversionFormats:
    @pytest.mark.parametrize("readonly", [True, False])
    def test_to_datetime_readonly(self, readonly):
        # GH#34857
        arr = np.array([], dtype=object)
        if readonly:
            arr.setflags(write=False)
        result = to_datetime(arr)
        expected = to_datetime([])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_format(self, cache):
        values = ["1/1/2000", "1/2/2000", "1/3/2000"]

        results1 = [Timestamp("20000101"), Timestamp("20000201"), Timestamp("20000301")]
        results2 = [Timestamp("20000101"), Timestamp("20000102"), Timestamp("20000103")]
        for vals, expecteds in [
            (values, (Index(results1), Index(results2))),
            (Series(values), (Series(results1), Series(results2))),
            (values[0], (results1[0], results2[0])),
            (values[1], (results1[1], results2[1])),
            (values[2], (results1[2], results2[2])),
        ]:

            for i, fmt in enumerate(["%d/%m/%Y", "%m/%d/%Y"]):
                result = to_datetime(vals, format=fmt, cache=cache)
                expected = expecteds[i]

                if isinstance(expected, Series):
                    tm.assert_series_equal(result, Series(expected))
                elif isinstance(expected, Timestamp):
                    assert result == expected
                else:
                    tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_format_YYYYMMDD(self, cache):
        s = Series([19801222, 19801222] + [19810105] * 5)
        expected = Series([Timestamp(x) for x in s.apply(str)])

        result = to_datetime(s, format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)

        result = to_datetime(s.apply(str), format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)

        # with NaT
        expected = Series(
            [Timestamp("19801222"), Timestamp("19801222")] + [Timestamp("19810105")] * 5
        )
        expected[2] = np.nan
        s[2] = np.nan

        result = to_datetime(s, format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)

        # string with NaT
        s = s.apply(str)
        s[2] = "nat"
        result = to_datetime(s, format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)

        # coercion
        # GH 7930
        s = Series([20121231, 20141231, 99991231])
        result = to_datetime(s, format="%Y%m%d", errors="ignore", cache=cache)
        expected = Series(
            [datetime(2012, 12, 31), datetime(2014, 12, 31), datetime(9999, 12, 31)],
            dtype=object,
        )
        tm.assert_series_equal(result, expected)

        result = to_datetime(s, format="%Y%m%d", errors="coerce", cache=cache)
        expected = Series(["20121231", "20141231", "NaT"], dtype="M8[ns]")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_s",
        [
            # Null values with Strings
            ["19801222", "20010112", None],
            ["19801222", "20010112", np.nan],
            ["19801222", "20010112", NaT],
            ["19801222", "20010112", "NaT"],
            # Null values with Integers
            [19801222, 20010112, None],
            [19801222, 20010112, np.nan],
            [19801222, 20010112, NaT],
            [19801222, 20010112, "NaT"],
        ],
    )
    def test_to_datetime_format_YYYYMMDD_with_none(self, input_s):
        # GH 30011
        # format='%Y%m%d'
        # with None
        expected = Series([Timestamp("19801222"), Timestamp("20010112"), NaT])
        result = Series(to_datetime(input_s, format="%Y%m%d"))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_s, expected",
        [
            # NaN before strings with invalid date values
            [
                Series(["19801222", np.nan, "20010012", "10019999"]),
                Series([Timestamp("19801222"), np.nan, np.nan, np.nan]),
            ],
            # NaN after strings with invalid date values
            [
                Series(["19801222", "20010012", "10019999", np.nan]),
                Series([Timestamp("19801222"), np.nan, np.nan, np.nan]),
            ],
            # NaN before integers with invalid date values
            [
                Series([20190813, np.nan, 20010012, 20019999]),
                Series([Timestamp("20190813"), np.nan, np.nan, np.nan]),
            ],
            # NaN after integers with invalid date values
            [
                Series([20190813, 20010012, np.nan, 20019999]),
                Series([Timestamp("20190813"), np.nan, np.nan, np.nan]),
            ],
        ],
    )
    def test_to_datetime_format_YYYYMMDD_overflow(self, input_s, expected):
        # GH 25512
        # format='%Y%m%d', errors='coerce'
        result = to_datetime(input_s, format="%Y%m%d", errors="coerce")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_format_integer(self, cache):
        # GH 10178
        s = Series([2000, 2001, 2002])
        expected = Series([Timestamp(x) for x in s.apply(str)])

        result = to_datetime(s, format="%Y", cache=cache)
        tm.assert_series_equal(result, expected)

        s = Series([200001, 200105, 200206])
        expected = Series([Timestamp(x[:4] + "-" + x[4:]) for x in s.apply(str)])

        result = to_datetime(s, format="%Y%m", cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "int_date, expected",
        [
            # valid date, length == 8
            [20121030, datetime(2012, 10, 30)],
            # short valid date, length == 6
            [199934, datetime(1999, 3, 4)],
            # long integer date partially parsed to datetime(2012,1,1), length > 8
            [2012010101, 2012010101],
            # invalid date partially parsed to datetime(2012,9,9), length == 8
            [20129930, 20129930],
            # short integer date partially parsed to datetime(2012,9,9), length < 8
            [2012993, 2012993],
            # short invalid date, length == 4
            [2121, 2121],
        ],
    )
    def test_int_to_datetime_format_YYYYMMDD_typeerror(self, int_date, expected):
        # GH 26583
        result = to_datetime(int_date, format="%Y%m%d", errors="ignore")
        assert result == expected

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_format_microsecond(self, cache):

        # these are locale dependent
        lang, _ = locale.getlocale()
        month_abbr = calendar.month_abbr[4]
        val = f"01-{month_abbr}-2011 00:00:01.978"

        format = "%d-%b-%Y %H:%M:%S.%f"
        result = to_datetime(val, format=format, cache=cache)
        exp = datetime.strptime(val, format)
        assert result == exp

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_format_time(self, cache):
        data = [
            ["01/10/2010 15:20", "%m/%d/%Y %H:%M", Timestamp("2010-01-10 15:20")],
            ["01/10/2010 05:43", "%m/%d/%Y %I:%M", Timestamp("2010-01-10 05:43")],
            [
                "01/10/2010 13:56:01",
                "%m/%d/%Y %H:%M:%S",
                Timestamp("2010-01-10 13:56:01"),
            ]  # ,
            # ['01/10/2010 08:14 PM', '%m/%d/%Y %I:%M %p',
            #  Timestamp('2010-01-10 20:14')],
            # ['01/10/2010 07:40 AM', '%m/%d/%Y %I:%M %p',
            #  Timestamp('2010-01-10 07:40')],
            # ['01/10/2010 09:12:56 AM', '%m/%d/%Y %I:%M:%S %p',
            #  Timestamp('2010-01-10 09:12:56')]
        ]
        for s, format, dt in data:
            assert to_datetime(s, format=format, cache=cache) == dt

    @td.skip_if_has_locale
    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_with_non_exact(self, cache):
        # GH 10834
        # 8904
        # exact kw
        s = Series(
            ["19MAY11", "foobar19MAY11", "19MAY11:00:00:00", "19MAY11 00:00:00Z"]
        )
        result = to_datetime(s, format="%d%b%y", exact=False, cache=cache)
        expected = to_datetime(
            s.str.extract(r"(\d+\w+\d+)", expand=False), format="%d%b%y", cache=cache
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_parse_nanoseconds_with_formula(self, cache):

        # GH8989
        # truncating the nanoseconds when a format was provided
        for v in [
            "2012-01-01 09:00:00.000000001",
            "2012-01-01 09:00:00.000001",
            "2012-01-01 09:00:00.001",
            "2012-01-01 09:00:00.001000",
            "2012-01-01 09:00:00.001000000",
        ]:
            expected = to_datetime(v, cache=cache)
            result = to_datetime(v, format="%Y-%m-%d %H:%M:%S.%f", cache=cache)
            assert result == expected

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_format_weeks(self, cache):
        data = [
            ["2009324", "%Y%W%w", Timestamp("2009-08-13")],
            ["2013020", "%Y%U%w", Timestamp("2013-01-13")],
        ]
        for s, format, dt in data:
            assert to_datetime(s, format=format, cache=cache) == dt

    @pytest.mark.parametrize(
        "fmt,dates,expected_dates",
        [
            [
                "%Y-%m-%d %H:%M:%S %Z",
                ["2010-01-01 12:00:00 UTC"] * 2,
                [Timestamp("2010-01-01 12:00:00", tz="UTC")] * 2,
            ],
            [
                "%Y-%m-%d %H:%M:%S %Z",
                [
                    "2010-01-01 12:00:00 UTC",
                    "2010-01-01 12:00:00 GMT",
                    "2010-01-01 12:00:00 US/Pacific",
                ],
                [
                    Timestamp("2010-01-01 12:00:00", tz="UTC"),
                    Timestamp("2010-01-01 12:00:00", tz="GMT"),
                    Timestamp("2010-01-01 12:00:00", tz="US/Pacific"),
                ],
            ],
            [
                "%Y-%m-%d %H:%M:%S%z",
                ["2010-01-01 12:00:00+0100"] * 2,
                [Timestamp("2010-01-01 12:00:00", tzinfo=pytz.FixedOffset(60))] * 2,
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100"] * 2,
                [Timestamp("2010-01-01 12:00:00", tzinfo=pytz.FixedOffset(60))] * 2,
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100", "2010-01-01 12:00:00 -0100"],
                [
                    Timestamp("2010-01-01 12:00:00", tzinfo=pytz.FixedOffset(60)),
                    Timestamp("2010-01-01 12:00:00", tzinfo=pytz.FixedOffset(-60)),
                ],
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 Z", "2010-01-01 12:00:00 Z"],
                [
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=pytz.FixedOffset(0)
                    ),  # pytz coerces to UTC
                    Timestamp("2010-01-01 12:00:00", tzinfo=pytz.FixedOffset(0)),
                ],
            ],
        ],
    )
    def test_to_datetime_parse_tzname_or_tzoffset(self, fmt, dates, expected_dates):
        # GH 13486
        result = to_datetime(dates, format=fmt)
        expected = Index(expected_dates)
        tm.assert_equal(result, expected)

    def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(self):
        # GH 32792
        dates = [
            "2010-01-01 12:00:00 +0100",
            "2010-01-01 12:00:00 -0100",
            "2010-01-01 12:00:00 +0300",
            "2010-01-01 12:00:00 +0400",
        ]
        expected_dates = [
            "2010-01-01 11:00:00+00:00",
            "2010-01-01 13:00:00+00:00",
            "2010-01-01 09:00:00+00:00",
            "2010-01-01 08:00:00+00:00",
        ]
        fmt = "%Y-%m-%d %H:%M:%S %z"

        result = to_datetime(dates, format=fmt, utc=True)
        expected = DatetimeIndex(expected_dates)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "offset", ["+0", "-1foo", "UTCbar", ":10", "+01:000:01", ""]
    )
    def test_to_datetime_parse_timezone_malformed(self, offset):
        fmt = "%Y-%m-%d %H:%M:%S %z"
        date = "2010-01-01 12:00:00 " + offset

        msg = "does not match format|unconverted data remains"
        with pytest.raises(ValueError, match=msg):
            to_datetime([date], format=fmt)

    def test_to_datetime_parse_timezone_keeps_name(self):
        # GH 21697
        fmt = "%Y-%m-%d %H:%M:%S %z"
        arg = Index(["2010-01-01 12:00:00 Z"], name="foo")
        result = to_datetime(arg, format=fmt)
        expected = DatetimeIndex(["2010-01-01 12:00:00"], tz="UTC", name="foo")
        tm.assert_index_equal(result, expected)


class TestToDatetime:
    @pytest.mark.parametrize(
        "s, _format, dt",
        [
            ["2015-1-1", "%G-%V-%u", datetime(2014, 12, 29, 0, 0)],
            ["2015-1-4", "%G-%V-%u", datetime(2015, 1, 1, 0, 0)],
            ["2015-1-7", "%G-%V-%u", datetime(2015, 1, 4, 0, 0)],
        ],
    )
    def test_to_datetime_iso_week_year_format(self, s, _format, dt):
        # See GH#16607
        assert to_datetime(s, format=_format) == dt

    @pytest.mark.parametrize(
        "msg, s, _format",
        [
            [
                "ISO week directive '%V' must be used with the ISO year directive "
                "'%G' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 50",
                "%Y %V",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 51",
                "%G %V",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 Monday",
                "%G %A",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 Mon",
                "%G %a",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 6",
                "%G %w",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 6",
                "%G %u",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "2051",
                "%G",
            ],
            [
                "Day of the year directive '%j' is not compatible with ISO year "
                "directive '%G'. Use '%Y' instead.",
                "1999 51 6 256",
                "%G %V %u %j",
            ],
            [
                "ISO week directive '%V' is incompatible with the year directive "
                "'%Y'. Use the ISO year '%G' instead.",
                "1999 51 Sunday",
                "%Y %V %A",
            ],
            [
                "ISO week directive '%V' is incompatible with the year directive "
                "'%Y'. Use the ISO year '%G' instead.",
                "1999 51 Sun",
                "%Y %V %a",
            ],
            [
                "ISO week directive '%V' is incompatible with the year directive "
                "'%Y'. Use the ISO year '%G' instead.",
                "1999 51 1",
                "%Y %V %w",
            ],
            [
                "ISO week directive '%V' is incompatible with the year directive "
                "'%Y'. Use the ISO year '%G' instead.",
                "1999 51 1",
                "%Y %V %u",
            ],
            [
                "ISO week directive '%V' must be used with the ISO year directive "
                "'%G' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "20",
                "%V",
            ],
        ],
    )
    def test_error_iso_week_year(self, msg, s, _format):
        # See GH#16607
        # This test checks for errors thrown when giving the wrong format
        # However, as discussed on PR#25541, overriding the locale
        # causes a different error to be thrown due to the format being
        # locale specific, but the test data is in english.
        # Therefore, the tests only run when locale is not overwritten,
        # as a sort of solution to this problem.
        if locale.getlocale() != ("zh_CN", "UTF-8") and locale.getlocale() != (
            "it_IT",
            "UTF-8",
        ):
            with pytest.raises(ValueError, match=msg):
                to_datetime(s, format=_format)

    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_to_datetime_dtarr(self, tz):
        # DatetimeArray
        dti = date_range("1965-04-03", periods=19, freq="2W", tz=tz)
        arr = DatetimeArray(dti)

        result = to_datetime(arr)
        assert result is arr

        result = to_datetime(arr)
        assert result is arr

    def test_to_datetime_pydatetime(self):
        actual = to_datetime(datetime(2008, 1, 15))
        assert actual == datetime(2008, 1, 15)

    def test_to_datetime_YYYYMMDD(self):
        actual = to_datetime("20080115")
        assert actual == datetime(2008, 1, 15)

    def test_to_datetime_unparseable_ignore(self):
        # unparsable
        s = "Month 1, 1999"
        assert to_datetime(s, errors="ignore") == s

    @td.skip_if_windows  # `tm.set_timezone` does not work in windows
    def test_to_datetime_now(self):
        # See GH#18666
        with tm.set_timezone("US/Eastern"):
            npnow = np.datetime64("now").astype("datetime64[ns]")
            pdnow = to_datetime("now")
            pdnow2 = to_datetime(["now"])[0]

            # These should all be equal with infinite perf; this gives
            # a generous margin of 10 seconds
            assert abs(pdnow.value - npnow.astype(np.int64)) < 1e10
            assert abs(pdnow2.value - npnow.astype(np.int64)) < 1e10

            assert pdnow.tzinfo is None
            assert pdnow2.tzinfo is None

    @td.skip_if_windows  # `tm.set_timezone` does not work in windows
    def test_to_datetime_today(self):
        # See GH#18666
        # Test with one timezone far ahead of UTC and another far behind, so
        # one of these will _almost_ always be in a different day from UTC.
        # Unfortunately this test between 12 and 1 AM Samoa time
        # this both of these timezones _and_ UTC will all be in the same day,
        # so this test will not detect the regression introduced in #18666.
        with tm.set_timezone("Pacific/Auckland"):  # 12-13 hours ahead of UTC
            nptoday = np.datetime64("today").astype("datetime64[ns]").astype(np.int64)
            pdtoday = to_datetime("today")
            pdtoday2 = to_datetime(["today"])[0]

            tstoday = Timestamp("today")
            tstoday2 = Timestamp.today()

            # These should all be equal with infinite perf; this gives
            # a generous margin of 10 seconds
            assert abs(pdtoday.normalize().value - nptoday) < 1e10
            assert abs(pdtoday2.normalize().value - nptoday) < 1e10
            assert abs(pdtoday.value - tstoday.value) < 1e10
            assert abs(pdtoday.value - tstoday2.value) < 1e10

            assert pdtoday.tzinfo is None
            assert pdtoday2.tzinfo is None

        with tm.set_timezone("US/Samoa"):  # 11 hours behind UTC
            nptoday = np.datetime64("today").astype("datetime64[ns]").astype(np.int64)
            pdtoday = to_datetime("today")
            pdtoday2 = to_datetime(["today"])[0]

            # These should all be equal with infinite perf; this gives
            # a generous margin of 10 seconds
            assert abs(pdtoday.normalize().value - nptoday) < 1e10
            assert abs(pdtoday2.normalize().value - nptoday) < 1e10

            assert pdtoday.tzinfo is None
            assert pdtoday2.tzinfo is None

    def test_to_datetime_today_now_unicode_bytes(self):
        to_datetime(["now"])
        to_datetime(["today"])

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_dt64s(self, cache):
        in_bound_dts = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]

        for dt in in_bound_dts:
            assert to_datetime(dt, cache=cache) == Timestamp(dt)

    @pytest.mark.parametrize(
        "dt", [np.datetime64("1000-01-01"), np.datetime64("5000-01-02")]
    )
    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_dt64s_out_of_bounds(self, cache, dt):
        msg = f"Out of bounds nanosecond timestamp: {dt}"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(dt, errors="raise")
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(dt)
        assert to_datetime(dt, errors="coerce", cache=cache) is NaT

    @pytest.mark.parametrize("cache", [True, False])
    @pytest.mark.parametrize("unit", ["s", "D"])
    def test_to_datetime_array_of_dt64s(self, cache, unit):
        # https://github.com/pandas-dev/pandas/issues/31491
        # Need at least 50 to ensure cache is used.
        dts = [
            np.datetime64("2000-01-01", unit),
            np.datetime64("2000-01-02", unit),
        ] * 30
        # Assuming all datetimes are in bounds, to_datetime() returns
        # an array that is equal to Timestamp() parsing
        tm.assert_index_equal(
            to_datetime(dts, cache=cache),
            DatetimeIndex([Timestamp(x).asm8 for x in dts]),
        )

        # A list of datetimes where the last one is out of bounds
        dts_with_oob = dts + [np.datetime64("9999-01-01")]

        msg = "Out of bounds nanosecond timestamp: 9999-01-01 00:00:00"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(dts_with_oob, errors="raise")

        tm.assert_index_equal(
            to_datetime(dts_with_oob, errors="coerce", cache=cache),
            DatetimeIndex(
                [Timestamp(dts_with_oob[0]).asm8, Timestamp(dts_with_oob[1]).asm8] * 30
                + [NaT],
            ),
        )

        # With errors='ignore', out of bounds datetime64s
        # are converted to their .item(), which depending on the version of
        # numpy is either a python datetime.datetime or datetime.date
        tm.assert_index_equal(
            to_datetime(dts_with_oob, errors="ignore", cache=cache),
            Index([dt.item() for dt in dts_with_oob]),
        )

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_tz(self, cache):

        # xref 8260
        # uniform returns a DatetimeIndex
        arr = [
            Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
            Timestamp("2013-01-02 14:00:00-0800", tz="US/Pacific"),
        ]
        result = to_datetime(arr, cache=cache)
        expected = DatetimeIndex(
            ["2013-01-01 13:00:00", "2013-01-02 14:00:00"], tz="US/Pacific"
        )
        tm.assert_index_equal(result, expected)

        # mixed tzs will raise
        arr = [
            Timestamp("2013-01-01 13:00:00", tz="US/Pacific"),
            Timestamp("2013-01-02 14:00:00", tz="US/Eastern"),
        ]
        msg = (
            "Tz-aware datetime.datetime cannot be "
            "converted to datetime64 unless utc=True"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_different_offsets(self, cache):
        # inspired by asv timeseries.ToDatetimeNONISO8601 benchmark
        # see GH-26097 for more
        ts_string_1 = "March 1, 2018 12:00:00+0400"
        ts_string_2 = "March 1, 2018 12:00:00+0500"
        arr = [ts_string_1] * 5 + [ts_string_2] * 5
        expected = Index([parse(x) for x in arr])
        result = to_datetime(arr, cache=cache)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_tz_pytz(self, cache):
        # see gh-8260
        us_eastern = pytz.timezone("US/Eastern")
        arr = np.array(
            [
                us_eastern.localize(
                    datetime(year=2000, month=1, day=1, hour=3, minute=0)
                ),
                us_eastern.localize(
                    datetime(year=2000, month=6, day=1, hour=3, minute=0)
                ),
            ],
            dtype=object,
        )
        result = to_datetime(arr, utc=True, cache=cache)
        expected = DatetimeIndex(
            ["2000-01-01 08:00:00+00:00", "2000-06-01 07:00:00+00:00"],
            dtype="datetime64[ns, UTC]",
            freq=None,
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    @pytest.mark.parametrize(
        "init_constructor, end_constructor, test_method",
        [
            (Index, DatetimeIndex, tm.assert_index_equal),
            (list, DatetimeIndex, tm.assert_index_equal),
            (np.array, DatetimeIndex, tm.assert_index_equal),
            (Series, Series, tm.assert_series_equal),
        ],
    )
    def test_to_datetime_utc_true(
        self, cache, init_constructor, end_constructor, test_method
    ):
        # See gh-11934 & gh-6415
        data = ["20100102 121314", "20100102 121315"]
        expected_data = [
            Timestamp("2010-01-02 12:13:14", tz="utc"),
            Timestamp("2010-01-02 12:13:15", tz="utc"),
        ]

        result = to_datetime(
            init_constructor(data), format="%Y%m%d %H%M%S", utc=True, cache=cache
        )
        expected = end_constructor(expected_data)
        test_method(result, expected)

        # Test scalar case as well
        for scalar, expected in zip(data, expected_data):
            result = to_datetime(scalar, format="%Y%m%d %H%M%S", utc=True, cache=cache)
            assert result == expected

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_utc_true_with_series_single_value(self, cache):
        # GH 15760 UTC=True with Series
        ts = 1.5e18
        result = to_datetime(Series([ts]), utc=True, cache=cache)
        expected = Series([Timestamp(ts, tz="utc")])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_utc_true_with_series_tzaware_string(self, cache):
        ts = "2013-01-01 00:00:00-01:00"
        expected_ts = "2013-01-01 01:00:00"
        data = Series([ts] * 3)
        result = to_datetime(data, utc=True, cache=cache)
        expected = Series([Timestamp(expected_ts, tz="utc")] * 3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    @pytest.mark.parametrize(
        "date, dtype",
        [
            ("2013-01-01 01:00:00", "datetime64[ns]"),
            ("2013-01-01 01:00:00", "datetime64[ns, UTC]"),
        ],
    )
    def test_to_datetime_utc_true_with_series_datetime_ns(self, cache, date, dtype):
        expected = Series([Timestamp("2013-01-01 01:00:00", tz="UTC")])
        result = to_datetime(Series([date], dtype=dtype), utc=True, cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    @td.skip_if_no("psycopg2")
    def test_to_datetime_tz_psycopg2(self, cache):

        # xref 8260
        import psycopg2

        # misc cases
        tz1 = psycopg2.tz.FixedOffsetTimezone(offset=-300, name=None)
        tz2 = psycopg2.tz.FixedOffsetTimezone(offset=-240, name=None)
        arr = np.array(
            [
                datetime(2000, 1, 1, 3, 0, tzinfo=tz1),
                datetime(2000, 6, 1, 3, 0, tzinfo=tz2),
            ],
            dtype=object,
        )

        result = to_datetime(arr, errors="coerce", utc=True, cache=cache)
        expected = DatetimeIndex(
            ["2000-01-01 08:00:00+00:00", "2000-06-01 07:00:00+00:00"],
            dtype="datetime64[ns, UTC]",
            freq=None,
        )
        tm.assert_index_equal(result, expected)

        # dtype coercion
        i = DatetimeIndex(
            ["2000-01-01 08:00:00"],
            tz=psycopg2.tz.FixedOffsetTimezone(offset=-300, name=None),
        )
        assert is_datetime64_ns_dtype(i)

        # tz coercion
        result = to_datetime(i, errors="coerce", cache=cache)
        tm.assert_index_equal(result, i)

        result = to_datetime(i, errors="coerce", utc=True, cache=cache)
        expected = DatetimeIndex(["2000-01-01 13:00:00"], dtype="datetime64[ns, UTC]")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_datetime_bool(self, cache):
        # GH13176
        msg = r"dtype bool cannot be converted to datetime64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            to_datetime(False)
        assert to_datetime(False, errors="coerce", cache=cache) is NaT
        assert to_datetime(False, errors="ignore", cache=cache) is False
        with pytest.raises(TypeError, match=msg):
            to_datetime(True)
        assert to_datetime(True, errors="coerce", cache=cache) is NaT
        assert to_datetime(True, errors="ignore", cache=cache) is True
        msg = f"{type(cache)} is not convertible to datetime"
        with pytest.raises(TypeError, match=msg):
            to_datetime([False, datetime.today()], cache=cache)
        with pytest.raises(TypeError, match=msg):
            to_datetime(["20130101", True], cache=cache)
        tm.assert_index_equal(
            to_datetime([0, False, NaT, 0.0], errors="coerce", cache=cache),
            DatetimeIndex(
                [to_datetime(0, cache=cache), NaT, NaT, to_datetime(0, cache=cache)]
            ),
        )

    def test_datetime_invalid_datatype(self):
        # GH13176
        msg = "is not convertible to datetime"
        with pytest.raises(TypeError, match=msg):
            to_datetime(bool)
        with pytest.raises(TypeError, match=msg):
            to_datetime(to_datetime)

    @pytest.mark.parametrize("value", ["a", "00:01:99"])
    @pytest.mark.parametrize("infer", [True, False])
    @pytest.mark.parametrize("format", [None, "H%:M%:S%"])
    def test_datetime_invalid_scalar(self, value, format, infer):
        # GH24763
        res = to_datetime(
            value, errors="ignore", format=format, infer_datetime_format=infer
        )
        assert res == value

        res = to_datetime(
            value, errors="coerce", format=format, infer_datetime_format=infer
        )
        assert res is NaT

        msg = (
            "is a bad directive in format|"
            "second must be in 0..59|"
            "Given date string not likely a datetime"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(
                value, errors="raise", format=format, infer_datetime_format=infer
            )

    @pytest.mark.parametrize("value", ["3000/12/11 00:00:00"])
    @pytest.mark.parametrize("infer", [True, False])
    @pytest.mark.parametrize("format", [None, "H%:M%:S%"])
    def test_datetime_outofbounds_scalar(self, value, format, infer):
        # GH24763
        res = to_datetime(
            value, errors="ignore", format=format, infer_datetime_format=infer
        )
        assert res == value

        res = to_datetime(
            value, errors="coerce", format=format, infer_datetime_format=infer
        )
        assert res is NaT

        if format is not None:
            msg = "is a bad directive in format|Out of bounds nanosecond timestamp"
            with pytest.raises(ValueError, match=msg):
                to_datetime(
                    value, errors="raise", format=format, infer_datetime_format=infer
                )
        else:
            msg = "Out of bounds nanosecond timestamp"
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime(
                    value, errors="raise", format=format, infer_datetime_format=infer
                )

    @pytest.mark.parametrize("values", [["a"], ["00:01:99"], ["a", "b", "99:00:00"]])
    @pytest.mark.parametrize("infer", [True, False])
    @pytest.mark.parametrize("format", [None, "H%:M%:S%"])
    def test_datetime_invalid_index(self, values, format, infer):
        # GH24763
        res = to_datetime(
            values, errors="ignore", format=format, infer_datetime_format=infer
        )
        tm.assert_index_equal(res, Index(values))

        res = to_datetime(
            values, errors="coerce", format=format, infer_datetime_format=infer
        )
        tm.assert_index_equal(res, DatetimeIndex([NaT] * len(values)))

        msg = (
            "is a bad directive in format|"
            "Given date string not likely a datetime|"
            "second must be in 0..59"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(
                values, errors="raise", format=format, infer_datetime_format=infer
            )

    @pytest.mark.parametrize("utc", [True, None])
    @pytest.mark.parametrize("format", ["%Y%m%d %H:%M:%S", None])
    @pytest.mark.parametrize("constructor", [list, tuple, np.array, Index, deque])
    def test_to_datetime_cache(self, utc, format, constructor):
        date = "20130101 00:00:00"
        test_dates = [date] * 10 ** 5
        data = constructor(test_dates)

        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "listlike",
        [
            (deque([Timestamp("2010-06-02 09:30:00")] * 51)),
            ([Timestamp("2010-06-02 09:30:00")] * 51),
            (tuple([Timestamp("2010-06-02 09:30:00")] * 51)),
        ],
    )
    def test_no_slicing_errors_in_should_cache(self, listlike):
        # GH 29403
        assert tools.should_cache(listlike) is True

    def test_to_datetime_from_deque(self):
        # GH 29403
        result = to_datetime(deque([Timestamp("2010-06-02 09:30:00")] * 51))
        expected = to_datetime([Timestamp("2010-06-02 09:30:00")] * 51)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("utc", [True, None])
    @pytest.mark.parametrize("format", ["%Y%m%d %H:%M:%S", None])
    def test_to_datetime_cache_series(self, utc, format):
        date = "20130101 00:00:00"
        test_dates = [date] * 10 ** 5
        data = Series(test_dates)
        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_cache_scalar(self):
        date = "20130101 00:00:00"
        result = to_datetime(date, cache=True)
        expected = Timestamp("20130101 00:00:00")
        assert result == expected

    @pytest.mark.parametrize(
        "datetimelikes,expected_values",
        (
            (
                (None, np.nan) + (NaT,) * start_caching_at,
                (NaT,) * (start_caching_at + 2),
            ),
            (
                (None, Timestamp("2012-07-26")) + (NaT,) * start_caching_at,
                (NaT, Timestamp("2012-07-26")) + (NaT,) * start_caching_at,
            ),
            (
                (None,)
                + (NaT,) * start_caching_at
                + ("2012 July 26", Timestamp("2012-07-26")),
                (NaT,) * (start_caching_at + 1)
                + (Timestamp("2012-07-26"), Timestamp("2012-07-26")),
            ),
        ),
    )
    def test_convert_object_to_datetime_with_cache(
        self, datetimelikes, expected_values
    ):
        # GH#39882
        ser = Series(
            datetimelikes,
            dtype="object",
        )
        result_series = to_datetime(ser, errors="coerce")
        expected_series = Series(
            expected_values,
            dtype="datetime64[ns]",
        )
        tm.assert_series_equal(result_series, expected_series)

    @pytest.mark.parametrize(
        "date, format",
        [
            ("2017-20", "%Y-%W"),
            ("20 Sunday", "%W %A"),
            ("20 Sun", "%W %a"),
            ("2017-21", "%Y-%U"),
            ("20 Sunday", "%U %A"),
            ("20 Sun", "%U %a"),
        ],
    )
    def test_week_without_day_and_calendar_year(self, date, format):
        # GH16774

        msg = "Cannot use '%W' or '%U' without day and year"
        with pytest.raises(ValueError, match=msg):
            to_datetime(date, format=format)

    def test_to_datetime_coerce(self):
        # GH 26122
        ts_strings = [
            "March 1, 2018 12:00:00+0400",
            "March 1, 2018 12:00:00+0500",
            "20100240",
        ]
        result = to_datetime(ts_strings, errors="coerce")
        expected = Index(
            [
                datetime(2018, 3, 1, 12, 0, tzinfo=tzoffset(None, 14400)),
                datetime(2018, 3, 1, 12, 0, tzinfo=tzoffset(None, 18000)),
                NaT,
            ]
        )
        tm.assert_index_equal(result, expected)

    def test_to_datetime_coerce_malformed(self):
        # GH 28299
        ts_strings = ["200622-12-31", "111111-24-11"]
        result = to_datetime(ts_strings, errors="coerce")
        expected = Index([NaT, NaT])
        tm.assert_index_equal(result, expected)

    def test_iso_8601_strings_with_same_offset(self):
        # GH 17697, 11736
        ts_str = "2015-11-18 15:30:00+05:30"
        result = to_datetime(ts_str)
        expected = Timestamp(ts_str)
        assert result == expected

        expected = DatetimeIndex([Timestamp(ts_str)] * 2)
        result = to_datetime([ts_str] * 2)
        tm.assert_index_equal(result, expected)

        result = DatetimeIndex([ts_str] * 2)
        tm.assert_index_equal(result, expected)

    def test_iso_8601_strings_with_different_offsets(self):
        # GH 17697, 11736
        ts_strings = ["2015-11-18 15:30:00+05:30", "2015-11-18 16:30:00+06:30", NaT]
        result = to_datetime(ts_strings)
        expected = np.array(
            [
                datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 19800)),
                datetime(2015, 11, 18, 16, 30, tzinfo=tzoffset(None, 23400)),
                NaT,
            ],
            dtype=object,
        )
        # GH 21864
        expected = Index(expected)
        tm.assert_index_equal(result, expected)

        result = to_datetime(ts_strings, utc=True)
        expected = DatetimeIndex(
            [Timestamp(2015, 11, 18, 10), Timestamp(2015, 11, 18, 10), NaT], tz="UTC"
        )
        tm.assert_index_equal(result, expected)

    def test_iso8601_strings_mixed_offsets_with_naive(self):
        # GH 24992
        result = to_datetime(
            [
                "2018-11-28T00:00:00",
                "2018-11-28T00:00:00+12:00",
                "2018-11-28T00:00:00",
                "2018-11-28T00:00:00+06:00",
                "2018-11-28T00:00:00",
            ],
            utc=True,
        )
        expected = to_datetime(
            [
                "2018-11-28T00:00:00",
                "2018-11-27T12:00:00",
                "2018-11-28T00:00:00",
                "2018-11-27T18:00:00",
                "2018-11-28T00:00:00",
            ],
            utc=True,
        )
        tm.assert_index_equal(result, expected)

        items = ["2018-11-28T00:00:00+12:00", "2018-11-28T00:00:00"]
        result = to_datetime(items, utc=True)
        expected = to_datetime(list(reversed(items)), utc=True)[::-1]
        tm.assert_index_equal(result, expected)

    def test_mixed_offsets_with_native_datetime_raises(self):
        # GH 25978
        ser = Series(
            [
                "nan",
                Timestamp("1990-01-01"),
                "2015-03-14T16:15:14.123-08:00",
                "2019-03-04T21:56:32.620-07:00",
                None,
            ]
        )
        with pytest.raises(ValueError, match="Tz-aware datetime.datetime"):
            to_datetime(ser)

    def test_non_iso_strings_with_tz_offset(self):
        result = to_datetime(["March 1, 2018 12:00:00+0400"] * 2)
        expected = DatetimeIndex(
            [datetime(2018, 3, 1, 12, tzinfo=pytz.FixedOffset(240))] * 2
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "ts, expected",
        [
            (Timestamp("2018-01-01"), Timestamp("2018-01-01", tz="UTC")),
            (
                Timestamp("2018-01-01", tz="US/Pacific"),
                Timestamp("2018-01-01 08:00", tz="UTC"),
            ),
        ],
    )
    def test_timestamp_utc_true(self, ts, expected):
        # GH 24415
        result = to_datetime(ts, utc=True)
        assert result == expected

    @pytest.mark.parametrize("dt_str", ["00010101", "13000101", "30000101", "99990101"])
    def test_to_datetime_with_format_out_of_bounds(self, dt_str):
        # GH 9107
        msg = "Out of bounds nanosecond timestamp"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(dt_str, format="%Y%m%d")

    def test_to_datetime_utc(self):
        arr = np.array([parse("2012-06-13T01:39:00Z")], dtype=object)

        result = to_datetime(arr, utc=True)
        assert result.tz is pytz.utc

    def test_to_datetime_fixed_offset(self):
        from pandas.tests.indexes.datetimes.test_timezones import fixed_off

        dates = [
            datetime(2000, 1, 1, tzinfo=fixed_off),
            datetime(2000, 1, 2, tzinfo=fixed_off),
            datetime(2000, 1, 3, tzinfo=fixed_off),
        ]
        result = to_datetime(dates)
        assert result.tz == fixed_off


class TestToDatetimeUnit:
    @pytest.mark.parametrize("cache", [True, False])
    def test_unit(self, cache):
        # GH 11758
        # test proper behavior with errors
        msg = "cannot specify both format and unit"
        with pytest.raises(ValueError, match=msg):
            to_datetime([1], unit="D", format="%Y%m%d", cache=cache)

        values = [11111111, 1, 1.0, iNaT, NaT, np.nan, "NaT", ""]
        result = to_datetime(values, unit="D", errors="ignore", cache=cache)
        expected = Index(
            [
                11111111,
                Timestamp("1970-01-02"),
                Timestamp("1970-01-02"),
                NaT,
                NaT,
                NaT,
                NaT,
                NaT,
            ],
            dtype=object,
        )
        tm.assert_index_equal(result, expected)

        result = to_datetime(values, unit="D", errors="coerce", cache=cache)
        expected = DatetimeIndex(
            ["NaT", "1970-01-02", "1970-01-02", "NaT", "NaT", "NaT", "NaT", "NaT"]
        )
        tm.assert_index_equal(result, expected)

        msg = "cannot convert input 11111111 with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, unit="D", errors="raise", cache=cache)

        values = [1420043460000, iNaT, NaT, np.nan, "NaT"]

        result = to_datetime(values, errors="ignore", unit="s", cache=cache)
        expected = Index([1420043460000, NaT, NaT, NaT, NaT], dtype=object)
        tm.assert_index_equal(result, expected)

        result = to_datetime(values, errors="coerce", unit="s", cache=cache)
        expected = DatetimeIndex(["NaT", "NaT", "NaT", "NaT", "NaT"])
        tm.assert_index_equal(result, expected)

        msg = "cannot convert input 1420043460000 with the unit 's'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, errors="raise", unit="s", cache=cache)

        # if we have a string, then we raise a ValueError
        # and NOT an OutOfBoundsDatetime
        for val in ["foo", Timestamp("20130101")]:
            try:
                to_datetime(val, errors="raise", unit="s", cache=cache)
            except OutOfBoundsDatetime as err:
                raise AssertionError("incorrect exception raised") from err
            except ValueError:
                pass

    @pytest.mark.parametrize("cache", [True, False])
    def test_unit_consistency(self, cache):

        # consistency of conversions
        expected = Timestamp("1970-05-09 14:25:11")
        result = to_datetime(11111111, unit="s", errors="raise", cache=cache)
        assert result == expected
        assert isinstance(result, Timestamp)

        result = to_datetime(11111111, unit="s", errors="coerce", cache=cache)
        assert result == expected
        assert isinstance(result, Timestamp)

        result = to_datetime(11111111, unit="s", errors="ignore", cache=cache)
        assert result == expected
        assert isinstance(result, Timestamp)

    @pytest.mark.parametrize("cache", [True, False])
    def test_unit_with_numeric(self, cache):

        # GH 13180
        # coercions from floats/ints are ok
        expected = DatetimeIndex(["2015-06-19 05:33:20", "2015-05-27 22:33:20"])
        arr1 = [1.434692e18, 1.432766e18]
        arr2 = np.array(arr1).astype("int64")
        for errors in ["ignore", "raise", "coerce"]:
            result = to_datetime(arr1, errors=errors, cache=cache)
            tm.assert_index_equal(result, expected)

            result = to_datetime(arr2, errors=errors, cache=cache)
            tm.assert_index_equal(result, expected)

        # but we want to make sure that we are coercing
        # if we have ints/strings
        expected = DatetimeIndex(["NaT", "2015-06-19 05:33:20", "2015-05-27 22:33:20"])
        arr = ["foo", 1.434692e18, 1.432766e18]
        result = to_datetime(arr, errors="coerce", cache=cache)
        tm.assert_index_equal(result, expected)

        expected = DatetimeIndex(
            ["2015-06-19 05:33:20", "2015-05-27 22:33:20", "NaT", "NaT"]
        )
        arr = [1.434692e18, 1.432766e18, "foo", "NaT"]
        result = to_datetime(arr, errors="coerce", cache=cache)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_unit_mixed(self, cache):

        # mixed integers/datetimes
        expected = DatetimeIndex(["2013-01-01", "NaT", "NaT"])
        arr = [Timestamp("20130101"), 1.434692e18, 1.432766e18]
        result = to_datetime(arr, errors="coerce", cache=cache)
        tm.assert_index_equal(result, expected)

        msg = "mixed datetimes and integers in passed array"
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, errors="raise", cache=cache)

        expected = DatetimeIndex(["NaT", "NaT", "2013-01-01"])
        arr = [1.434692e18, 1.432766e18, Timestamp("20130101")]
        result = to_datetime(arr, errors="coerce", cache=cache)
        tm.assert_index_equal(result, expected)

        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, errors="raise", cache=cache)

    @pytest.mark.parametrize("cache", [True, False])
    def test_unit_rounding(self, cache):
        # GH 14156 & GH 20445: argument will incur floating point errors
        # but no premature rounding
        result = to_datetime(1434743731.8770001, unit="s", cache=cache)
        expected = Timestamp("2015-06-19 19:55:31.877000192")
        assert result == expected

    @pytest.mark.parametrize("cache", [True, False])
    def test_unit_ignore_keeps_name(self, cache):
        # GH 21697
        expected = Index([15e9] * 2, name="name")
        result = to_datetime(expected, errors="ignore", unit="s", cache=cache)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_dataframe(self, cache):

        df = DataFrame(
            {
                "year": [2015, 2016],
                "month": [2, 3],
                "day": [4, 5],
                "hour": [6, 7],
                "minute": [58, 59],
                "second": [10, 11],
                "ms": [1, 1],
                "us": [2, 2],
                "ns": [3, 3],
            }
        )

        result = to_datetime(
            {"year": df["year"], "month": df["month"], "day": df["day"]}, cache=cache
        )
        expected = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:0:00")]
        )
        tm.assert_series_equal(result, expected)

        # dict-like
        result = to_datetime(df[["year", "month", "day"]].to_dict(), cache=cache)
        tm.assert_series_equal(result, expected)

        # dict but with constructable
        df2 = df[["year", "month", "day"]].to_dict()
        df2["month"] = 2
        result = to_datetime(df2, cache=cache)
        expected2 = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160205 00:0:00")]
        )
        tm.assert_series_equal(result, expected2)

        # unit mappings
        units = [
            {
                "year": "years",
                "month": "months",
                "day": "days",
                "hour": "hours",
                "minute": "minutes",
                "second": "seconds",
            },
            {
                "year": "year",
                "month": "month",
                "day": "day",
                "hour": "hour",
                "minute": "minute",
                "second": "second",
            },
        ]

        for d in units:
            result = to_datetime(df[list(d.keys())].rename(columns=d), cache=cache)
            expected = Series(
                [Timestamp("20150204 06:58:10"), Timestamp("20160305 07:59:11")]
            )
            tm.assert_series_equal(result, expected)

        d = {
            "year": "year",
            "month": "month",
            "day": "day",
            "hour": "hour",
            "minute": "minute",
            "second": "second",
            "ms": "ms",
            "us": "us",
            "ns": "ns",
        }

        result = to_datetime(df.rename(columns=d), cache=cache)
        expected = Series(
            [
                Timestamp("20150204 06:58:10.001002003"),
                Timestamp("20160305 07:59:11.001002003"),
            ]
        )
        tm.assert_series_equal(result, expected)

        # coerce back to int
        result = to_datetime(df.astype(str), cache=cache)
        tm.assert_series_equal(result, expected)

        # passing coerce
        df2 = DataFrame({"year": [2015, 2016], "month": [2, 20], "day": [4, 5]})

        msg = (
            "cannot assemble the datetimes: time data .+ does not "
            r"match format '%Y%m%d' \(match\)"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)
        result = to_datetime(df2, errors="coerce", cache=cache)
        expected = Series([Timestamp("20150204 00:00:00"), NaT])
        tm.assert_series_equal(result, expected)

        # extra columns
        msg = r"extra keys have been passed to the datetime assemblage: \[foo\]"
        with pytest.raises(ValueError, match=msg):
            df2 = df.copy()
            df2["foo"] = 1
            to_datetime(df2, cache=cache)

        # not enough
        msg = (
            r"to assemble mappings requires at least that \[year, month, "
            r"day\] be specified: \[.+\] is missing"
        )
        for c in [
            ["year"],
            ["year", "month"],
            ["year", "month", "second"],
            ["month", "day"],
            ["year", "day", "second"],
        ]:
            with pytest.raises(ValueError, match=msg):
                to_datetime(df[c], cache=cache)

        # duplicates
        msg = "cannot assemble with duplicate keys"
        df2 = DataFrame({"year": [2015, 2016], "month": [2, 20], "day": [4, 5]})
        df2.columns = ["year", "year", "day"]
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

        df2 = DataFrame(
            {"year": [2015, 2016], "month": [2, 20], "day": [4, 5], "hour": [4, 5]}
        )
        df2.columns = ["year", "month", "day", "day"]
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

    @pytest.mark.parametrize("cache", [True, False])
    def test_dataframe_dtypes(self, cache):
        # #13451
        df = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})

        # int16
        result = to_datetime(df.astype("int16"), cache=cache)
        expected = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")]
        )
        tm.assert_series_equal(result, expected)

        # mixed dtypes
        df["month"] = df["month"].astype("int8")
        df["day"] = df["day"].astype("int8")
        result = to_datetime(df, cache=cache)
        expected = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")]
        )
        tm.assert_series_equal(result, expected)

        # float
        df = DataFrame({"year": [2000, 2001], "month": [1.5, 1], "day": [1, 1]})
        msg = "cannot assemble the datetimes: unconverted data remains: 1"
        with pytest.raises(ValueError, match=msg):
            to_datetime(df, cache=cache)

    def test_dataframe_utc_true(self):
        # GH 23760
        df = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        result = to_datetime(df, utc=True)
        expected = Series(
            np.array(["2015-02-04", "2016-03-05"], dtype="datetime64[ns]")
        ).dt.tz_localize("UTC")
        tm.assert_series_equal(result, expected)

    def test_to_datetime_errors_ignore_utc_true(self):
        # GH 23758
        result = to_datetime([1], unit="s", utc=True, errors="ignore")
        expected = DatetimeIndex(["1970-01-01 00:00:01"], tz="UTC")
        tm.assert_index_equal(result, expected)

    # TODO: this is moved from tests.series.test_timeseries, may be redundant
    def test_to_datetime_unit(self):

        epoch = 1370745748
        s = Series([epoch + t for t in range(20)])
        result = to_datetime(s, unit="s")
        expected = Series(
            [Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)]
        )
        tm.assert_series_equal(result, expected)

        s = Series([epoch + t for t in range(20)]).astype(float)
        result = to_datetime(s, unit="s")
        expected = Series(
            [Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)]
        )
        tm.assert_series_equal(result, expected)

        s = Series([epoch + t for t in range(20)] + [iNaT])
        result = to_datetime(s, unit="s")
        expected = Series(
            [Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)]
            + [NaT]
        )
        tm.assert_series_equal(result, expected)

        s = Series([epoch + t for t in range(20)] + [iNaT]).astype(float)
        result = to_datetime(s, unit="s")
        expected = Series(
            [Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)]
            + [NaT]
        )
        tm.assert_series_equal(result, expected)

        # GH13834
        s = Series([epoch + t for t in np.arange(0, 2, 0.25)] + [iNaT]).astype(float)
        result = to_datetime(s, unit="s")
        expected = Series(
            [
                Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t)
                for t in np.arange(0, 2, 0.25)
            ]
            + [NaT]
        )
        # GH20455 argument will incur floating point errors but no premature rounding
        result = result.round("ms")
        tm.assert_series_equal(result, expected)

        s = pd.concat(
            [Series([epoch + t for t in range(20)]).astype(float), Series([np.nan])],
            ignore_index=True,
        )
        result = to_datetime(s, unit="s")
        expected = Series(
            [Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)]
            + [NaT]
        )
        tm.assert_series_equal(result, expected)

        result = to_datetime([1, 2, "NaT", NaT, np.nan], unit="D")
        expected = DatetimeIndex(
            [Timestamp("1970-01-02"), Timestamp("1970-01-03")] + ["NaT"] * 3
        )
        tm.assert_index_equal(result, expected)

        msg = "non convertible value foo with the unit 'D'"
        with pytest.raises(ValueError, match=msg):
            to_datetime([1, 2, "foo"], unit="D")
        msg = "cannot convert input 111111111 with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime([1, 2, 111111111], unit="D")

        # coerce we can process
        expected = DatetimeIndex(
            [Timestamp("1970-01-02"), Timestamp("1970-01-03")] + ["NaT"] * 1
        )
        result = to_datetime([1, 2, "foo"], unit="D", errors="coerce")
        tm.assert_index_equal(result, expected)

        result = to_datetime([1, 2, 111111111], unit="D", errors="coerce")
        tm.assert_index_equal(result, expected)


class TestToDatetimeMisc:
    def test_to_datetime_barely_out_of_bounds(self):
        # GH#19529
        # GH#19382 close enough to bounds that dropping nanos would result
        # in an in-bounds datetime
        arr = np.array(["2262-04-11 23:47:16.854775808"], dtype=object)

        msg = "Out of bounds nanosecond timestamp"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(arr)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_iso8601(self, cache):
        result = to_datetime(["2012-01-01 00:00:00"], cache=cache)
        exp = Timestamp("2012-01-01 00:00:00")
        assert result[0] == exp

        result = to_datetime(["20121001"], cache=cache)  # bad iso 8601
        exp = Timestamp("2012-10-01")
        assert result[0] == exp

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_default(self, cache):
        rs = to_datetime("2001", cache=cache)
        xp = datetime(2001, 1, 1)
        assert rs == xp

        # dayfirst is essentially broken

        # to_datetime('01-13-2012', dayfirst=True)
        # pytest.raises(ValueError, to_datetime('01-13-2012',
        #                   dayfirst=True))

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_on_datetime64_series(self, cache):
        # #2699
        s = Series(date_range("1/1/2000", periods=10))

        result = to_datetime(s, cache=cache)
        assert result[0] == s[0]

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_with_space_in_series(self, cache):
        # GH 6428
        s = Series(["10/18/2006", "10/18/2008", " "])
        msg = r"(\(')?String does not contain a date(:', ' '\))?"
        with pytest.raises(ValueError, match=msg):
            to_datetime(s, errors="raise", cache=cache)
        result_coerce = to_datetime(s, errors="coerce", cache=cache)
        expected_coerce = Series([datetime(2006, 10, 18), datetime(2008, 10, 18), NaT])
        tm.assert_series_equal(result_coerce, expected_coerce)
        result_ignore = to_datetime(s, errors="ignore", cache=cache)
        tm.assert_series_equal(result_ignore, s)

    @td.skip_if_has_locale
    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_with_apply(self, cache):
        # this is only locale tested with US/None locales
        # GH 5195
        # with a format and coerce a single item to_datetime fails
        td = Series(["May 04", "Jun 02", "Dec 11"], index=[1, 2, 3])
        expected = to_datetime(td, format="%b %y", cache=cache)
        result = td.apply(to_datetime, format="%b %y", cache=cache)
        tm.assert_series_equal(result, expected)

        td = Series(["May 04", "Jun 02", ""], index=[1, 2, 3])
        msg = r"time data '' does not match format '%b %y' \(match\)"
        with pytest.raises(ValueError, match=msg):
            to_datetime(td, format="%b %y", errors="raise", cache=cache)
        with pytest.raises(ValueError, match=msg):
            td.apply(to_datetime, format="%b %y", errors="raise", cache=cache)
        expected = to_datetime(td, format="%b %y", errors="coerce", cache=cache)

        result = td.apply(
            lambda x: to_datetime(x, format="%b %y", errors="coerce", cache=cache)
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_types(self, cache):

        # empty string
        result = to_datetime("", cache=cache)
        assert result is NaT

        result = to_datetime(["", ""], cache=cache)
        assert isna(result).all()

        # ints
        result = Timestamp(0)
        expected = to_datetime(0, cache=cache)
        assert result == expected

        # GH 3888 (strings)
        expected = to_datetime(["2012"], cache=cache)[0]
        result = to_datetime("2012", cache=cache)
        assert result == expected

        # array = ['2012','20120101','20120101 12:01:01']
        array = ["20120101", "20120101 12:01:01"]
        expected = list(to_datetime(array, cache=cache))
        result = [Timestamp(date_str) for date_str in array]
        tm.assert_almost_equal(result, expected)

        # currently fails ###
        # result = Timestamp('2012')
        # expected = to_datetime('2012')
        # assert result == expected

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_unprocessable_input(self, cache):
        # GH 4928
        # GH 21864
        result = to_datetime([1, "1"], errors="ignore", cache=cache)

        expected = Index(np.array([1, "1"], dtype="O"))
        tm.assert_equal(result, expected)
        msg = "invalid string coercion to datetime"
        with pytest.raises(TypeError, match=msg):
            to_datetime([1, "1"], errors="raise", cache=cache)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_unhashable_input(self, cache):
        series = Series([["a"]] * 100)
        result = to_datetime(series, errors="ignore", cache=cache)
        tm.assert_series_equal(series, result)

    def test_to_datetime_other_datetime64_units(self):
        # 5/25/2012
        scalar = np.int64(1337904000000000).view("M8[us]")
        as_obj = scalar.astype("O")

        index = DatetimeIndex([scalar])
        assert index[0] == scalar.astype("O")

        value = Timestamp(scalar)
        assert value == as_obj

    def test_to_datetime_list_of_integers(self):
        rng = date_range("1/1/2000", periods=20)
        rng = DatetimeIndex(rng.values)

        ints = list(rng.asi8)

        result = DatetimeIndex(ints)

        tm.assert_index_equal(rng, result)

    def test_to_datetime_overflow(self):
        # gh-17637
        # we are overflowing Timedelta range here

        msg = "|".join(
            [
                "Python int too large to convert to C long",
                "long too big to convert",
                "int too big to convert",
            ]
        )
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            date_range(start="1/1/1700", freq="B", periods=100000)

    @pytest.mark.parametrize("cache", [True, False])
    def test_string_na_nat_conversion(self, cache):
        # GH #999, #858

        strings = np.array(
            ["1/1/2000", "1/2/2000", np.nan, "1/4/2000, 12:34:56"], dtype=object
        )

        expected = np.empty(4, dtype="M8[ns]")
        for i, val in enumerate(strings):
            if isna(val):
                expected[i] = iNaT
            else:
                expected[i] = parse(val)

        result = tslib.array_to_datetime(strings)[0]
        tm.assert_almost_equal(result, expected)

        result2 = to_datetime(strings, cache=cache)
        assert isinstance(result2, DatetimeIndex)
        tm.assert_numpy_array_equal(result, result2.values)

        malformed = np.array(["1/100/2000", np.nan], dtype=object)

        # GH 10636, default is now 'raise'
        msg = r"Unknown string format:|day is out of range for month"
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)

        result = to_datetime(malformed, errors="ignore", cache=cache)
        # GH 21864
        expected = Index(malformed)
        tm.assert_index_equal(result, expected)

        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)

        idx = ["a", "b", "c", "d", "e"]
        series = Series(
            ["1/1/2000", np.nan, "1/3/2000", np.nan, "1/5/2000"], index=idx, name="foo"
        )
        dseries = Series(
            [
                to_datetime("1/1/2000", cache=cache),
                np.nan,
                to_datetime("1/3/2000", cache=cache),
                np.nan,
                to_datetime("1/5/2000", cache=cache),
            ],
            index=idx,
            name="foo",
        )

        result = to_datetime(series, cache=cache)
        dresult = to_datetime(dseries, cache=cache)

        expected = Series(np.empty(5, dtype="M8[ns]"), index=idx)
        for i in range(5):
            x = series[i]
            if isna(x):
                expected[i] = NaT
            else:
                expected[i] = to_datetime(x, cache=cache)

        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name == "foo"

        tm.assert_series_equal(dresult, expected, check_names=False)
        assert dresult.name == "foo"

    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[h]",
            "datetime64[m]",
            "datetime64[s]",
            "datetime64[ms]",
            "datetime64[us]",
            "datetime64[ns]",
        ],
    )
    @pytest.mark.parametrize("cache", [True, False])
    def test_dti_constructor_numpy_timeunits(self, cache, dtype):
        # GH 9114
        base = to_datetime(["2000-01-01T00:00", "2000-01-02T00:00", "NaT"], cache=cache)

        values = base.values.astype(dtype)

        tm.assert_index_equal(DatetimeIndex(values), base)
        tm.assert_index_equal(to_datetime(values, cache=cache), base)

    @pytest.mark.parametrize("cache", [True, False])
    def test_dayfirst(self, cache):
        # GH 5917
        arr = ["10/02/2014", "11/02/2014", "12/02/2014"]
        expected = DatetimeIndex(
            [datetime(2014, 2, 10), datetime(2014, 2, 11), datetime(2014, 2, 12)]
        )
        idx1 = DatetimeIndex(arr, dayfirst=True)
        idx2 = DatetimeIndex(np.array(arr), dayfirst=True)
        idx3 = to_datetime(arr, dayfirst=True, cache=cache)
        idx4 = to_datetime(np.array(arr), dayfirst=True, cache=cache)
        idx5 = DatetimeIndex(Index(arr), dayfirst=True)
        idx6 = DatetimeIndex(Series(arr), dayfirst=True)
        tm.assert_index_equal(expected, idx1)
        tm.assert_index_equal(expected, idx2)
        tm.assert_index_equal(expected, idx3)
        tm.assert_index_equal(expected, idx4)
        tm.assert_index_equal(expected, idx5)
        tm.assert_index_equal(expected, idx6)

    @pytest.mark.parametrize("klass", [DatetimeIndex, DatetimeArray])
    def test_to_datetime_dta_tz(self, klass):
        # GH#27733
        dti = date_range("2015-04-05", periods=3).rename("foo")
        expected = dti.tz_localize("UTC")

        obj = klass(dti)
        expected = klass(expected)

        result = to_datetime(obj, utc=True)
        tm.assert_equal(result, expected)


class TestGuessDatetimeFormat:
    @td.skip_if_not_us_locale
    def test_guess_datetime_format_for_array(self):
        expected_format = "%Y-%m-%d %H:%M:%S.%f"
        dt_string = datetime(2011, 12, 30, 0, 0, 0).strftime(expected_format)

        test_arrays = [
            np.array([dt_string, dt_string, dt_string], dtype="O"),
            np.array([np.nan, np.nan, dt_string], dtype="O"),
            np.array([dt_string, "random_string"], dtype="O"),
        ]

        for test_array in test_arrays:
            assert tools._guess_datetime_format_for_array(test_array) == expected_format

        format_for_string_of_nans = tools._guess_datetime_format_for_array(
            np.array([np.nan, np.nan, np.nan], dtype="O")
        )
        assert format_for_string_of_nans is None


class TestToDatetimeInferFormat:
    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_infer_datetime_format_consistent_format(self, cache):
        s = Series(date_range("20000101", periods=50, freq="H"))

        test_formats = ["%m-%d-%Y", "%m/%d/%Y %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"]

        for test_format in test_formats:
            s_as_dt_strings = s.apply(lambda x: x.strftime(test_format))

            with_format = to_datetime(s_as_dt_strings, format=test_format, cache=cache)
            no_infer = to_datetime(
                s_as_dt_strings, infer_datetime_format=False, cache=cache
            )
            yes_infer = to_datetime(
                s_as_dt_strings, infer_datetime_format=True, cache=cache
            )

            # Whether the format is explicitly passed, it is inferred, or
            # it is not inferred, the results should all be the same
            tm.assert_series_equal(with_format, no_infer)
            tm.assert_series_equal(no_infer, yes_infer)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_infer_datetime_format_inconsistent_format(self, cache):
        s = Series(
            np.array(
                ["01/01/2011 00:00:00", "01-02-2011 00:00:00", "2011-01-03T00:00:00"]
            )
        )

        # When the format is inconsistent, infer_datetime_format should just
        # fallback to the default parsing
        tm.assert_series_equal(
            to_datetime(s, infer_datetime_format=False, cache=cache),
            to_datetime(s, infer_datetime_format=True, cache=cache),
        )

        s = Series(np.array(["Jan/01/2011", "Feb/01/2011", "Mar/01/2011"]))

        tm.assert_series_equal(
            to_datetime(s, infer_datetime_format=False, cache=cache),
            to_datetime(s, infer_datetime_format=True, cache=cache),
        )

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_infer_datetime_format_series_with_nans(self, cache):
        s = Series(
            np.array(
                ["01/01/2011 00:00:00", np.nan, "01/03/2011 00:00:00", np.nan],
                dtype=object,
            )
        )
        tm.assert_series_equal(
            to_datetime(s, infer_datetime_format=False, cache=cache),
            to_datetime(s, infer_datetime_format=True, cache=cache),
        )

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_infer_datetime_format_series_start_with_nans(self, cache):
        s = Series(
            np.array(
                [
                    np.nan,
                    np.nan,
                    "01/01/2011 00:00:00",
                    "01/02/2011 00:00:00",
                    "01/03/2011 00:00:00",
                ],
                dtype=object,
            )
        )

        tm.assert_series_equal(
            to_datetime(s, infer_datetime_format=False, cache=cache),
            to_datetime(s, infer_datetime_format=True, cache=cache),
        )

    @pytest.mark.parametrize(
        "tz_name, offset", [("UTC", 0), ("UTC-3", 180), ("UTC+3", -180)]
    )
    def test_infer_datetime_format_tz_name(self, tz_name, offset):
        # GH 33133
        s = Series([f"2019-02-02 08:07:13 {tz_name}"])
        result = to_datetime(s, infer_datetime_format=True)
        expected = Series(
            [Timestamp("2019-02-02 08:07:13").tz_localize(pytz.FixedOffset(offset))]
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("cache", [True, False])
    def test_to_datetime_iso8601_noleading_0s(self, cache):
        # GH 11871
        s = Series(["2014-1-1", "2014-2-2", "2015-3-3"])
        expected = Series(
            [
                Timestamp("2014-01-01"),
                Timestamp("2014-02-02"),
                Timestamp("2015-03-03"),
            ]
        )
        tm.assert_series_equal(to_datetime(s, cache=cache), expected)
        tm.assert_series_equal(to_datetime(s, format="%Y-%m-%d", cache=cache), expected)


class TestDaysInMonth:
    # tests for issue #10154

    @pytest.mark.parametrize("cache", [True, False])
    def test_day_not_in_month_coerce(self, cache):
        assert isna(to_datetime("2015-02-29", errors="coerce", cache=cache))
        assert isna(
            to_datetime("2015-02-29", format="%Y-%m-%d", errors="coerce", cache=cache)
        )
        assert isna(
            to_datetime("2015-02-32", format="%Y-%m-%d", errors="coerce", cache=cache)
        )
        assert isna(
            to_datetime("2015-04-31", format="%Y-%m-%d", errors="coerce", cache=cache)
        )

    @pytest.mark.parametrize("cache", [True, False])
    def test_day_not_in_month_raise(self, cache):
        msg = "day is out of range for month"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2015-02-29", errors="raise", cache=cache)

        msg = "time data 2015-02-29 doesn't match format specified"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2015-02-29", errors="raise", format="%Y-%m-%d", cache=cache)

        msg = "time data 2015-02-32 doesn't match format specified"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2015-02-32", errors="raise", format="%Y-%m-%d", cache=cache)

        msg = "time data 2015-04-31 doesn't match format specified"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2015-04-31", errors="raise", format="%Y-%m-%d", cache=cache)

    @pytest.mark.parametrize("cache", [True, False])
    def test_day_not_in_month_ignore(self, cache):
        assert to_datetime("2015-02-29", errors="ignore", cache=cache) == "2015-02-29"
        assert (
            to_datetime("2015-02-29", errors="ignore", format="%Y-%m-%d", cache=cache)
            == "2015-02-29"
        )
        assert (
            to_datetime("2015-02-32", errors="ignore", format="%Y-%m-%d", cache=cache)
            == "2015-02-32"
        )
        assert (
            to_datetime("2015-04-31", errors="ignore", format="%Y-%m-%d", cache=cache)
            == "2015-04-31"
        )


class TestDatetimeParsingWrappers:
    @pytest.mark.parametrize(
        "date_str,expected",
        list(
            {
                "2011-01-01": datetime(2011, 1, 1),
                "2Q2005": datetime(2005, 4, 1),
                "2Q05": datetime(2005, 4, 1),
                "2005Q1": datetime(2005, 1, 1),
                "05Q1": datetime(2005, 1, 1),
                "2011Q3": datetime(2011, 7, 1),
                "11Q3": datetime(2011, 7, 1),
                "3Q2011": datetime(2011, 7, 1),
                "3Q11": datetime(2011, 7, 1),
                # quarterly without space
                "2000Q4": datetime(2000, 10, 1),
                "00Q4": datetime(2000, 10, 1),
                "4Q2000": datetime(2000, 10, 1),
                "4Q00": datetime(2000, 10, 1),
                "2000q4": datetime(2000, 10, 1),
                "2000-Q4": datetime(2000, 10, 1),
                "00-Q4": datetime(2000, 10, 1),
                "4Q-2000": datetime(2000, 10, 1),
                "4Q-00": datetime(2000, 10, 1),
                "00q4": datetime(2000, 10, 1),
                "2005": datetime(2005, 1, 1),
                "2005-11": datetime(2005, 11, 1),
                "2005 11": datetime(2005, 11, 1),
                "11-2005": datetime(2005, 11, 1),
                "11 2005": datetime(2005, 11, 1),
                "200511": datetime(2020, 5, 11),
                "20051109": datetime(2005, 11, 9),
                "20051109 10:15": datetime(2005, 11, 9, 10, 15),
                "20051109 08H": datetime(2005, 11, 9, 8, 0),
                "2005-11-09 10:15": datetime(2005, 11, 9, 10, 15),
                "2005-11-09 08H": datetime(2005, 11, 9, 8, 0),
                "2005/11/09 10:15": datetime(2005, 11, 9, 10, 15),
                "2005/11/09 08H": datetime(2005, 11, 9, 8, 0),
                "Thu Sep 25 10:36:28 2003": datetime(2003, 9, 25, 10, 36, 28),
                "Thu Sep 25 2003": datetime(2003, 9, 25),
                "Sep 25 2003": datetime(2003, 9, 25),
                "January 1 2014": datetime(2014, 1, 1),
                # GHE10537
                "2014-06": datetime(2014, 6, 1),
                "06-2014": datetime(2014, 6, 1),
                "2014-6": datetime(2014, 6, 1),
                "6-2014": datetime(2014, 6, 1),
                "20010101 12": datetime(2001, 1, 1, 12),
                "20010101 1234": datetime(2001, 1, 1, 12, 34),
                "20010101 123456": datetime(2001, 1, 1, 12, 34, 56),
            }.items()
        ),
    )
    @pytest.mark.parametrize("cache", [True, False])
    def test_parsers(self, date_str, expected, cache):

        # dateutil >= 2.5.0 defaults to yearfirst=True
        # https://github.com/dateutil/dateutil/issues/217
        yearfirst = True

        result1, _ = parsing.parse_time_string(date_str, yearfirst=yearfirst)
        result2 = to_datetime(date_str, yearfirst=yearfirst)
        result3 = to_datetime([date_str], yearfirst=yearfirst)
        # result5 is used below
        result4 = to_datetime(
            np.array([date_str], dtype=object), yearfirst=yearfirst, cache=cache
        )
        result6 = DatetimeIndex([date_str], yearfirst=yearfirst)
        # result7 is used below
        result8 = DatetimeIndex(Index([date_str]), yearfirst=yearfirst)
        result9 = DatetimeIndex(Series([date_str]), yearfirst=yearfirst)

        for res in [result1, result2]:
            assert res == expected
        for res in [result3, result4, result6, result8, result9]:
            exp = DatetimeIndex([Timestamp(expected)])
            tm.assert_index_equal(res, exp)

        # these really need to have yearfirst, but we don't support
        if not yearfirst:
            result5 = Timestamp(date_str)
            assert result5 == expected
            result7 = date_range(date_str, freq="S", periods=1, yearfirst=yearfirst)
            assert result7 == expected

    @pytest.mark.parametrize("cache", [True, False])
    def test_na_values_with_cache(
        self, cache, unique_nulls_fixture, unique_nulls_fixture2
    ):
        # GH22305
        expected = Index([NaT, NaT], dtype="datetime64[ns]")
        result = to_datetime([unique_nulls_fixture, unique_nulls_fixture2], cache=cache)
        tm.assert_index_equal(result, expected)

    def test_parsers_nat(self):
        # Test that each of several string-accepting methods return pd.NaT
        result1, _ = parsing.parse_time_string("NaT")
        result2 = to_datetime("NaT")
        result3 = Timestamp("NaT")
        result4 = DatetimeIndex(["NaT"])[0]
        assert result1 is NaT
        assert result2 is NaT
        assert result3 is NaT
        assert result4 is NaT

    @pytest.mark.parametrize("cache", [True, False])
    def test_parsers_dayfirst_yearfirst(self, cache):
        # OK
        # 2.5.1 10-11-12   [dayfirst=0, yearfirst=0] -> 2012-10-11 00:00:00
        # 2.5.2 10-11-12   [dayfirst=0, yearfirst=1] -> 2012-10-11 00:00:00
        # 2.5.3 10-11-12   [dayfirst=0, yearfirst=0] -> 2012-10-11 00:00:00

        # OK
        # 2.5.1 10-11-12   [dayfirst=0, yearfirst=1] -> 2010-11-12 00:00:00
        # 2.5.2 10-11-12   [dayfirst=0, yearfirst=1] -> 2010-11-12 00:00:00
        # 2.5.3 10-11-12   [dayfirst=0, yearfirst=1] -> 2010-11-12 00:00:00

        # bug fix in 2.5.2
        # 2.5.1 10-11-12   [dayfirst=1, yearfirst=1] -> 2010-11-12 00:00:00
        # 2.5.2 10-11-12   [dayfirst=1, yearfirst=1] -> 2010-12-11 00:00:00
        # 2.5.3 10-11-12   [dayfirst=1, yearfirst=1] -> 2010-12-11 00:00:00

        # OK
        # 2.5.1 10-11-12   [dayfirst=1, yearfirst=0] -> 2012-11-10 00:00:00
        # 2.5.2 10-11-12   [dayfirst=1, yearfirst=0] -> 2012-11-10 00:00:00
        # 2.5.3 10-11-12   [dayfirst=1, yearfirst=0] -> 2012-11-10 00:00:00

        # OK
        # 2.5.1 20/12/21   [dayfirst=0, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.2 20/12/21   [dayfirst=0, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.3 20/12/21   [dayfirst=0, yearfirst=0] -> 2021-12-20 00:00:00

        # OK
        # 2.5.1 20/12/21   [dayfirst=0, yearfirst=1] -> 2020-12-21 00:00:00
        # 2.5.2 20/12/21   [dayfirst=0, yearfirst=1] -> 2020-12-21 00:00:00
        # 2.5.3 20/12/21   [dayfirst=0, yearfirst=1] -> 2020-12-21 00:00:00

        # revert of bug in 2.5.2
        # 2.5.1 20/12/21   [dayfirst=1, yearfirst=1] -> 2020-12-21 00:00:00
        # 2.5.2 20/12/21   [dayfirst=1, yearfirst=1] -> month must be in 1..12
        # 2.5.3 20/12/21   [dayfirst=1, yearfirst=1] -> 2020-12-21 00:00:00

        # OK
        # 2.5.1 20/12/21   [dayfirst=1, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.2 20/12/21   [dayfirst=1, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.3 20/12/21   [dayfirst=1, yearfirst=0] -> 2021-12-20 00:00:00

        # str : dayfirst, yearfirst, expected
        cases = {
            "10-11-12": [
                (False, False, datetime(2012, 10, 11)),
                (True, False, datetime(2012, 11, 10)),
                (False, True, datetime(2010, 11, 12)),
                (True, True, datetime(2010, 12, 11)),
            ],
            "20/12/21": [
                (False, False, datetime(2021, 12, 20)),
                (True, False, datetime(2021, 12, 20)),
                (False, True, datetime(2020, 12, 21)),
                (True, True, datetime(2020, 12, 21)),
            ],
        }

        for date_str, values in cases.items():
            for dayfirst, yearfirst, expected in values:

                # compare with dateutil result
                dateutil_result = parse(
                    date_str, dayfirst=dayfirst, yearfirst=yearfirst
                )
                assert dateutil_result == expected

                result1, _ = parsing.parse_time_string(
                    date_str, dayfirst=dayfirst, yearfirst=yearfirst
                )

                # we don't support dayfirst/yearfirst here:
                if not dayfirst and not yearfirst:
                    result2 = Timestamp(date_str)
                    assert result2 == expected

                result3 = to_datetime(
                    date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache
                )

                result4 = DatetimeIndex(
                    [date_str], dayfirst=dayfirst, yearfirst=yearfirst
                )[0]

                assert result1 == expected
                assert result3 == expected
                assert result4 == expected

    @pytest.mark.parametrize("cache", [True, False])
    def test_parsers_timestring(self, cache):
        # must be the same as dateutil result
        cases = {
            "10:15": (parse("10:15"), datetime(1, 1, 1, 10, 15)),
            "9:05": (parse("9:05"), datetime(1, 1, 1, 9, 5)),
        }

        for date_str, (exp_now, exp_def) in cases.items():
            result1, _ = parsing.parse_time_string(date_str)
            result2 = to_datetime(date_str)
            result3 = to_datetime([date_str])
            result4 = Timestamp(date_str)
            result5 = DatetimeIndex([date_str])[0]
            # parse time string return time string based on default date
            # others are not, and can't be changed because it is used in
            # time series plot
            assert result1 == exp_def
            assert result2 == exp_now
            assert result3 == exp_now
            assert result4 == exp_now
            assert result5 == exp_now

    @pytest.mark.parametrize("cache", [True, False])
    @pytest.mark.parametrize(
        "dt_string, tz, dt_string_repr",
        [
            (
                "2013-01-01 05:45+0545",
                pytz.FixedOffset(345),
                "Timestamp('2013-01-01 05:45:00+0545', tz='pytz.FixedOffset(345)')",
            ),
            (
                "2013-01-01 05:30+0530",
                pytz.FixedOffset(330),
                "Timestamp('2013-01-01 05:30:00+0530', tz='pytz.FixedOffset(330)')",
            ),
        ],
    )
    def test_parsers_timezone_minute_offsets_roundtrip(
        self, cache, dt_string, tz, dt_string_repr
    ):
        # GH11708
        base = to_datetime("2013-01-01 00:00:00", cache=cache)
        base = base.tz_localize("UTC").tz_convert(tz)
        dt_time = to_datetime(dt_string, cache=cache)
        assert base == dt_time
        assert dt_string_repr == repr(dt_time)


@pytest.fixture(params=["D", "s", "ms", "us", "ns"])
def units(request):
    """Day and some time units.

    * D
    * s
    * ms
    * us
    * ns
    """
    return request.param


@pytest.fixture
def epoch_1960():
    """Timestamp at 1960-01-01."""
    return Timestamp("1960-01-01")


@pytest.fixture
def units_from_epochs():
    return list(range(5))


@pytest.fixture(params=["timestamp", "pydatetime", "datetime64", "str_1960"])
def epochs(epoch_1960, request):
    """Timestamp at 1960-01-01 in various forms.

    * Timestamp
    * datetime.datetime
    * numpy.datetime64
    * str
    """
    assert request.param in {"timestamp", "pydatetime", "datetime64", "str_1960"}
    if request.param == "timestamp":
        return epoch_1960
    elif request.param == "pydatetime":
        return epoch_1960.to_pydatetime()
    elif request.param == "datetime64":
        return epoch_1960.to_datetime64()
    else:
        return str(epoch_1960)


@pytest.fixture
def julian_dates():
    return date_range("2014-1-1", periods=10).to_julian_date().values


class TestOrigin:
    def test_to_basic(self, julian_dates):
        # gh-11276, gh-11745
        # for origin as julian

        result = Series(to_datetime(julian_dates, unit="D", origin="julian"))
        expected = Series(
            to_datetime(julian_dates - Timestamp(0).to_julian_date(), unit="D")
        )
        tm.assert_series_equal(result, expected)

        result = Series(to_datetime([0, 1, 2], unit="D", origin="unix"))
        expected = Series(
            [Timestamp("1970-01-01"), Timestamp("1970-01-02"), Timestamp("1970-01-03")]
        )
        tm.assert_series_equal(result, expected)

        # default
        result = Series(to_datetime([0, 1, 2], unit="D"))
        expected = Series(
            [Timestamp("1970-01-01"), Timestamp("1970-01-02"), Timestamp("1970-01-03")]
        )
        tm.assert_series_equal(result, expected)

    def test_julian_round_trip(self):
        result = to_datetime(2456658, origin="julian", unit="D")
        assert result.to_julian_date() == 2456658

        # out-of-bounds
        msg = "1 is Out of Bounds for origin='julian'"
        with pytest.raises(ValueError, match=msg):
            to_datetime(1, origin="julian", unit="D")

    def test_invalid_unit(self, units, julian_dates):

        # checking for invalid combination of origin='julian' and unit != D
        if units != "D":
            msg = "unit must be 'D' for origin='julian'"
            with pytest.raises(ValueError, match=msg):
                to_datetime(julian_dates, unit=units, origin="julian")

    def test_invalid_origin(self):

        # need to have a numeric specified
        msg = "it must be numeric with a unit specified"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2005-01-01", origin="1960-01-01")

        with pytest.raises(ValueError, match=msg):
            to_datetime("2005-01-01", origin="1960-01-01", unit="D")

    def test_epoch(self, units, epochs, epoch_1960, units_from_epochs):

        expected = Series(
            [pd.Timedelta(x, unit=units) + epoch_1960 for x in units_from_epochs]
        )

        result = Series(to_datetime(units_from_epochs, unit=units, origin=epochs))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "origin, exc",
        [
            ("random_string", ValueError),
            ("epoch", ValueError),
            ("13-24-1990", ValueError),
            (datetime(1, 1, 1), OutOfBoundsDatetime),
        ],
    )
    def test_invalid_origins(self, origin, exc, units, units_from_epochs):

        msg = f"origin {origin} (is Out of Bounds|cannot be converted to a Timestamp)"
        with pytest.raises(exc, match=msg):
            to_datetime(units_from_epochs, unit=units, origin=origin)

    def test_invalid_origins_tzinfo(self):
        # GH16842
        with pytest.raises(ValueError, match="must be tz-naive"):
            to_datetime(1, unit="D", origin=datetime(2000, 1, 1, tzinfo=pytz.utc))

    @pytest.mark.parametrize("format", [None, "%Y-%m-%d %H:%M:%S"])
    def test_to_datetime_out_of_bounds_with_format_arg(self, format):
        # see gh-23830
        msg = "Out of bounds nanosecond timestamp"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime("2417-10-27 00:00:00", format=format)

    def test_processing_order(self):
        # make sure we handle out-of-bounds *before*
        # constructing the dates

        result = to_datetime(200 * 365, unit="D")
        expected = Timestamp("2169-11-13 00:00:00")
        assert result == expected

        result = to_datetime(200 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2069-11-13 00:00:00")
        assert result == expected

        result = to_datetime(300 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2169-10-20 00:00:00")
        assert result == expected

    @pytest.mark.parametrize(
        "offset,utc,exp",
        [
            ["Z", True, "2019-01-01T00:00:00.000Z"],
            ["Z", None, "2019-01-01T00:00:00.000Z"],
            ["-01:00", True, "2019-01-01T01:00:00.000Z"],
            ["-01:00", None, "2019-01-01T00:00:00.000-01:00"],
        ],
    )
    def test_arg_tz_ns_unit(self, offset, utc, exp):
        # GH 25546
        arg = "2019-01-01T00:00:00.000" + offset
        result = to_datetime([arg], unit="ns", utc=utc)
        expected = to_datetime([exp])
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "listlike,do_caching",
    [([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], False), ([1, 1, 1, 1, 4, 5, 6, 7, 8, 9], True)],
)
def test_should_cache(listlike, do_caching):
    assert (
        tools.should_cache(listlike, check_count=len(listlike), unique_share=0.7)
        == do_caching
    )


@pytest.mark.parametrize(
    "unique_share,check_count, err_message",
    [
        (0.5, 11, r"check_count must be in next bounds: \[0; len\(arg\)\]"),
        (10, 2, r"unique_share must be in next bounds: \(0; 1\)"),
    ],
)
def test_should_cache_errors(unique_share, check_count, err_message):
    arg = [5] * 10

    with pytest.raises(AssertionError, match=err_message):
        tools.should_cache(arg, unique_share, check_count)


def test_nullable_integer_to_datetime():
    # Test for #30050
    ser = Series([1, 2, None, 2 ** 61, None])
    ser = ser.astype("Int64")
    ser_copy = ser.copy()

    res = to_datetime(ser, unit="ns")

    expected = Series(
        [
            np.datetime64("1970-01-01 00:00:00.000000001"),
            np.datetime64("1970-01-01 00:00:00.000000002"),
            np.datetime64("NaT"),
            np.datetime64("2043-01-25 23:56:49.213693952"),
            np.datetime64("NaT"),
        ]
    )
    tm.assert_series_equal(res, expected)
    # Check that ser isn't mutated
    tm.assert_series_equal(ser, ser_copy)


@pytest.mark.parametrize("klass", [np.array, list])
def test_na_to_datetime(nulls_fixture, klass):

    if isinstance(nulls_fixture, Decimal):
        with pytest.raises(TypeError, match="not convertible to datetime"):
            to_datetime(klass([nulls_fixture]))

    else:
        result = to_datetime(klass([nulls_fixture]))

        assert result[0] is NaT


def test_empty_string_datetime_coerce__format():
    # GH13044
    td = Series(["03/24/2016", "03/25/2016", ""])
    format = "%m/%d/%Y"

    # coerce empty string to pd.NaT
    result = to_datetime(td, format=format, errors="coerce")
    expected = Series(["2016-03-24", "2016-03-25", NaT], dtype="datetime64[ns]")
    tm.assert_series_equal(expected, result)

    # raise an exception in case a format is given
    with pytest.raises(ValueError, match="does not match format"):
        result = to_datetime(td, format=format, errors="raise")

    # don't raise an exception in case no format is given
    result = to_datetime(td, errors="raise")
    tm.assert_series_equal(result, expected)


def test_empty_string_datetime_coerce__unit():
    # GH13044
    # coerce empty string to pd.NaT
    result = to_datetime([1, ""], unit="s", errors="coerce")
    expected = DatetimeIndex(["1970-01-01 00:00:01", "NaT"], dtype="datetime64[ns]")
    tm.assert_index_equal(expected, result)

    # verify that no exception is raised even when errors='raise' is set
    result = to_datetime([1, ""], unit="s", errors="raise")
    tm.assert_index_equal(expected, result)
