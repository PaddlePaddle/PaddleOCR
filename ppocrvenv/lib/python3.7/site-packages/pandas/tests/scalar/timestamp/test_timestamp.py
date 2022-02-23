""" test the scalar Timestamp """

import calendar
from datetime import (
    datetime,
    timedelta,
)
import locale
import pickle
import unicodedata

from dateutil.tz import tzutc
import numpy as np
import pytest
import pytz
from pytz import (
    timezone,
    utc,
)

from pandas._libs.tslibs.timezones import (
    dateutil_gettz as gettz,
    get_timezone,
)
from pandas.compat import np_datetime64_compat
import pandas.util._test_decorators as td

from pandas import (
    NaT,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm

from pandas.tseries import offsets


class TestTimestampProperties:
    def test_freq_deprecation(self):
        # GH#41586
        msg = "The 'freq' argument in Timestamp is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # warning issued at construction
            ts = Timestamp("2021-06-01", freq="D")
            ts2 = Timestamp("2021-06-01", freq="B")

        msg = "Timestamp.freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # warning issued at attribute lookup
            ts.freq

        for per in ["month", "quarter", "year"]:
            for side in ["start", "end"]:
                attr = f"is_{per}_{side}"

                with tm.assert_produces_warning(FutureWarning, match=msg):
                    getattr(ts2, attr)

                # is_(month|quarter|year)_(start|end) does _not_ issue a warning
                #  with freq="D" bc the result will be unaffected by the deprecation
                with tm.assert_produces_warning(None):
                    getattr(ts, attr)

    @pytest.mark.filterwarnings("ignore:The 'freq' argument:FutureWarning")
    @pytest.mark.filterwarnings("ignore:Timestamp.freq is deprecated:FutureWarning")
    def test_properties_business(self):
        ts = Timestamp("2017-10-01", freq="B")
        control = Timestamp("2017-10-01")
        assert ts.dayofweek == 6
        assert ts.day_of_week == 6
        assert not ts.is_month_start  # not a weekday
        assert not ts.freq.is_month_start(ts)
        assert ts.freq.is_month_start(ts + Timedelta(days=1))
        assert not ts.is_quarter_start  # not a weekday
        assert not ts.freq.is_quarter_start(ts)
        assert ts.freq.is_quarter_start(ts + Timedelta(days=1))
        # Control case: non-business is month/qtr start
        assert control.is_month_start
        assert control.is_quarter_start

        ts = Timestamp("2017-09-30", freq="B")
        control = Timestamp("2017-09-30")
        assert ts.dayofweek == 5
        assert ts.day_of_week == 5
        assert not ts.is_month_end  # not a weekday
        assert not ts.freq.is_month_end(ts)
        assert ts.freq.is_month_end(ts - Timedelta(days=1))
        assert not ts.is_quarter_end  # not a weekday
        assert not ts.freq.is_quarter_end(ts)
        assert ts.freq.is_quarter_end(ts - Timedelta(days=1))
        # Control case: non-business is month/qtr start
        assert control.is_month_end
        assert control.is_quarter_end

    def test_fields(self):
        def check(value, equal):
            # that we are int like
            assert isinstance(value, int)
            assert value == equal

        # GH 10050
        ts = Timestamp("2015-05-10 09:06:03.000100001")
        check(ts.year, 2015)
        check(ts.month, 5)
        check(ts.day, 10)
        check(ts.hour, 9)
        check(ts.minute, 6)
        check(ts.second, 3)
        msg = "'Timestamp' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            ts.millisecond
        check(ts.microsecond, 100)
        check(ts.nanosecond, 1)
        check(ts.dayofweek, 6)
        check(ts.day_of_week, 6)
        check(ts.quarter, 2)
        check(ts.dayofyear, 130)
        check(ts.day_of_year, 130)
        check(ts.week, 19)
        check(ts.daysinmonth, 31)
        check(ts.daysinmonth, 31)

        # GH 13303
        ts = Timestamp("2014-12-31 23:59:00-05:00", tz="US/Eastern")
        check(ts.year, 2014)
        check(ts.month, 12)
        check(ts.day, 31)
        check(ts.hour, 23)
        check(ts.minute, 59)
        check(ts.second, 0)
        msg = "'Timestamp' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            ts.millisecond
        check(ts.microsecond, 0)
        check(ts.nanosecond, 0)
        check(ts.dayofweek, 2)
        check(ts.day_of_week, 2)
        check(ts.quarter, 4)
        check(ts.dayofyear, 365)
        check(ts.day_of_year, 365)
        check(ts.week, 1)
        check(ts.daysinmonth, 31)

        ts = Timestamp("2014-01-01 00:00:00+01:00")
        starts = ["is_month_start", "is_quarter_start", "is_year_start"]
        for start in starts:
            assert getattr(ts, start)
        ts = Timestamp("2014-12-31 23:59:59+01:00")
        ends = ["is_month_end", "is_year_end", "is_quarter_end"]
        for end in ends:
            assert getattr(ts, end)

    # GH 12806
    @pytest.mark.parametrize(
        "data",
        [Timestamp("2017-08-28 23:00:00"), Timestamp("2017-08-28 23:00:00", tz="EST")],
    )
    @pytest.mark.parametrize(
        "time_locale", [None] if tm.get_locales() is None else [None] + tm.get_locales()
    )
    def test_names(self, data, time_locale):
        # GH 17354
        # Test .day_name(), .month_name
        if time_locale is None:
            expected_day = "Monday"
            expected_month = "August"
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_day = calendar.day_name[0].capitalize()
                expected_month = calendar.month_name[8].capitalize()

        result_day = data.day_name(time_locale)
        result_month = data.month_name(time_locale)

        # Work around https://github.com/pandas-dev/pandas/issues/22342
        # different normalizations
        expected_day = unicodedata.normalize("NFD", expected_day)
        expected_month = unicodedata.normalize("NFD", expected_month)

        result_day = unicodedata.normalize("NFD", result_day)
        result_month = unicodedata.normalize("NFD", result_month)

        assert result_day == expected_day
        assert result_month == expected_month

        # Test NaT
        nan_ts = Timestamp(NaT)
        assert np.isnan(nan_ts.day_name(time_locale))
        assert np.isnan(nan_ts.month_name(time_locale))

    def test_is_leap_year(self, tz_naive_fixture):
        tz = tz_naive_fixture
        # GH 13727
        dt = Timestamp("2000-01-01 00:00:00", tz=tz)
        assert dt.is_leap_year
        assert isinstance(dt.is_leap_year, bool)

        dt = Timestamp("1999-01-01 00:00:00", tz=tz)
        assert not dt.is_leap_year

        dt = Timestamp("2004-01-01 00:00:00", tz=tz)
        assert dt.is_leap_year

        dt = Timestamp("2100-01-01 00:00:00", tz=tz)
        assert not dt.is_leap_year

    def test_woy_boundary(self):
        # make sure weeks at year boundaries are correct
        d = datetime(2013, 12, 31)
        result = Timestamp(d).week
        expected = 1  # ISO standard
        assert result == expected

        d = datetime(2008, 12, 28)
        result = Timestamp(d).week
        expected = 52  # ISO standard
        assert result == expected

        d = datetime(2009, 12, 31)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        d = datetime(2010, 1, 1)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        d = datetime(2010, 1, 3)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        result = np.array(
            [
                Timestamp(datetime(*args)).week
                for args in [(2000, 1, 1), (2000, 1, 2), (2005, 1, 1), (2005, 1, 2)]
            ]
        )
        assert (result == [52, 52, 53, 53]).all()

    def test_resolution(self):
        # GH#21336, GH#21365
        dt = Timestamp("2100-01-01 00:00:00")
        assert dt.resolution == Timedelta(nanoseconds=1)

        # Check that the attribute is available on the class, mirroring
        #  the stdlib datetime behavior
        assert Timestamp.resolution == Timedelta(nanoseconds=1)


class TestTimestamp:
    def test_tz(self):
        tstr = "2014-02-01 09:00"
        ts = Timestamp(tstr)
        local = ts.tz_localize("Asia/Tokyo")
        assert local.hour == 9
        assert local == Timestamp(tstr, tz="Asia/Tokyo")
        conv = local.tz_convert("US/Eastern")
        assert conv == Timestamp("2014-01-31 19:00", tz="US/Eastern")
        assert conv.hour == 19

        # preserves nanosecond
        ts = Timestamp(tstr) + offsets.Nano(5)
        local = ts.tz_localize("Asia/Tokyo")
        assert local.hour == 9
        assert local.nanosecond == 5
        conv = local.tz_convert("US/Eastern")
        assert conv.nanosecond == 5
        assert conv.hour == 19

    def test_utc_z_designator(self):
        assert get_timezone(Timestamp("2014-11-02 01:00Z").tzinfo) is utc

    def test_asm8(self):
        np.random.seed(7_960_929)
        ns = [Timestamp.min.value, Timestamp.max.value, 1000]

        for n in ns:
            assert (
                Timestamp(n).asm8.view("i8") == np.datetime64(n, "ns").view("i8") == n
            )

        assert Timestamp("nat").asm8.view("i8") == np.datetime64("nat", "ns").view("i8")

    def test_class_ops_pytz(self):
        def compare(x, y):
            assert int((Timestamp(x).value - Timestamp(y).value) / 1e9) == 0

        compare(Timestamp.now(), datetime.now())
        compare(Timestamp.now("UTC"), datetime.now(timezone("UTC")))
        compare(Timestamp.utcnow(), datetime.utcnow())
        compare(Timestamp.today(), datetime.today())
        current_time = calendar.timegm(datetime.now().utctimetuple())
        compare(
            Timestamp.utcfromtimestamp(current_time),
            datetime.utcfromtimestamp(current_time),
        )
        compare(
            Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time)
        )

        date_component = datetime.utcnow()
        time_component = (date_component + timedelta(minutes=10)).time()
        compare(
            Timestamp.combine(date_component, time_component),
            datetime.combine(date_component, time_component),
        )

    def test_class_ops_dateutil(self):
        def compare(x, y):
            assert (
                int(
                    np.round(Timestamp(x).value / 1e9)
                    - np.round(Timestamp(y).value / 1e9)
                )
                == 0
            )

        compare(Timestamp.now(), datetime.now())
        compare(Timestamp.now("UTC"), datetime.now(tzutc()))
        compare(Timestamp.utcnow(), datetime.utcnow())
        compare(Timestamp.today(), datetime.today())
        current_time = calendar.timegm(datetime.now().utctimetuple())
        compare(
            Timestamp.utcfromtimestamp(current_time),
            datetime.utcfromtimestamp(current_time),
        )
        compare(
            Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time)
        )

        date_component = datetime.utcnow()
        time_component = (date_component + timedelta(minutes=10)).time()
        compare(
            Timestamp.combine(date_component, time_component),
            datetime.combine(date_component, time_component),
        )

    def test_basics_nanos(self):
        val = np.int64(946_684_800_000_000_000).view("M8[ns]")
        stamp = Timestamp(val.view("i8") + 500)
        assert stamp.year == 2000
        assert stamp.month == 1
        assert stamp.microsecond == 0
        assert stamp.nanosecond == 500

        # GH 14415
        val = np.iinfo(np.int64).min + 80_000_000_000_000
        stamp = Timestamp(val)
        assert stamp.year == 1677
        assert stamp.month == 9
        assert stamp.day == 21
        assert stamp.microsecond == 145224
        assert stamp.nanosecond == 192

    @pytest.mark.parametrize(
        "value, check_kwargs",
        [
            [946688461000000000, {}],
            [946688461000000000 / 1000, {"unit": "us"}],
            [946688461000000000 / 1_000_000, {"unit": "ms"}],
            [946688461000000000 / 1_000_000_000, {"unit": "s"}],
            [10957, {"unit": "D", "h": 0}],
            [
                (946688461000000000 + 500000) / 1000000000,
                {"unit": "s", "us": 499, "ns": 964},
            ],
            [
                (946688461000000000 + 500000000) / 1000000000,
                {"unit": "s", "us": 500000},
            ],
            [(946688461000000000 + 500000) / 1000000, {"unit": "ms", "us": 500}],
            [(946688461000000000 + 500000) / 1000, {"unit": "us", "us": 500}],
            [(946688461000000000 + 500000000) / 1000000, {"unit": "ms", "us": 500000}],
            [946688461000000000 / 1000.0 + 5, {"unit": "us", "us": 5}],
            [946688461000000000 / 1000.0 + 5000, {"unit": "us", "us": 5000}],
            [946688461000000000 / 1000000.0 + 0.5, {"unit": "ms", "us": 500}],
            [946688461000000000 / 1000000.0 + 0.005, {"unit": "ms", "us": 5, "ns": 5}],
            [946688461000000000 / 1000000000.0 + 0.5, {"unit": "s", "us": 500000}],
            [10957 + 0.5, {"unit": "D", "h": 12}],
        ],
    )
    def test_unit(self, value, check_kwargs):
        def check(value, unit=None, h=1, s=1, us=0, ns=0):
            stamp = Timestamp(value, unit=unit)
            assert stamp.year == 2000
            assert stamp.month == 1
            assert stamp.day == 1
            assert stamp.hour == h
            if unit != "D":
                assert stamp.minute == 1
                assert stamp.second == s
                assert stamp.microsecond == us
            else:
                assert stamp.minute == 0
                assert stamp.second == 0
                assert stamp.microsecond == 0
            assert stamp.nanosecond == ns

        check(value, **check_kwargs)

    def test_roundtrip(self):

        # test value to string and back conversions
        # further test accessors
        base = Timestamp("20140101 00:00:00")

        result = Timestamp(base.value + Timedelta("5ms").value)
        assert result == Timestamp(f"{base}.005000")
        assert result.microsecond == 5000

        result = Timestamp(base.value + Timedelta("5us").value)
        assert result == Timestamp(f"{base}.000005")
        assert result.microsecond == 5

        result = Timestamp(base.value + Timedelta("5ns").value)
        assert result == Timestamp(f"{base}.000000005")
        assert result.nanosecond == 5
        assert result.microsecond == 0

        result = Timestamp(base.value + Timedelta("6ms 5us").value)
        assert result == Timestamp(f"{base}.006005")
        assert result.microsecond == 5 + 6 * 1000

        result = Timestamp(base.value + Timedelta("200ms 5us").value)
        assert result == Timestamp(f"{base}.200005")
        assert result.microsecond == 5 + 200 * 1000

    def test_hash_equivalent(self):
        d = {datetime(2011, 1, 1): 5}
        stamp = Timestamp(datetime(2011, 1, 1))
        assert d[stamp] == 5

    def test_tz_conversion_freq(self, tz_naive_fixture):
        # GH25241
        with tm.assert_produces_warning(FutureWarning, match="freq"):
            t1 = Timestamp("2019-01-01 10:00", freq="H")
            assert t1.tz_localize(tz=tz_naive_fixture).freq == t1.freq
        with tm.assert_produces_warning(FutureWarning, match="freq"):
            t2 = Timestamp("2019-01-02 12:00", tz="UTC", freq="T")
            assert t2.tz_convert(tz="UTC").freq == t2.freq

    def test_pickle_freq_no_warning(self):
        # GH#41949 we don't want a warning on unpickling
        with tm.assert_produces_warning(FutureWarning, match="freq"):
            ts = Timestamp("2019-01-01 10:00", freq="H")

        out = pickle.dumps(ts)
        with tm.assert_produces_warning(None):
            res = pickle.loads(out)

        assert res._freq == ts._freq


class TestTimestampNsOperations:
    def test_nanosecond_string_parsing(self):
        ts = Timestamp("2013-05-01 07:15:45.123456789")
        # GH 7878
        expected_repr = "2013-05-01 07:15:45.123456789"
        expected_value = 1_367_392_545_123_456_789
        assert ts.value == expected_value
        assert expected_repr in repr(ts)

        ts = Timestamp("2013-05-01 07:15:45.123456789+09:00", tz="Asia/Tokyo")
        assert ts.value == expected_value - 9 * 3600 * 1_000_000_000
        assert expected_repr in repr(ts)

        ts = Timestamp("2013-05-01 07:15:45.123456789", tz="UTC")
        assert ts.value == expected_value
        assert expected_repr in repr(ts)

        ts = Timestamp("2013-05-01 07:15:45.123456789", tz="US/Eastern")
        assert ts.value == expected_value + 4 * 3600 * 1_000_000_000
        assert expected_repr in repr(ts)

        # GH 10041
        ts = Timestamp("20130501T071545.123456789")
        assert ts.value == expected_value
        assert expected_repr in repr(ts)

    def test_nanosecond_timestamp(self):
        # GH 7610
        expected = 1_293_840_000_000_000_005
        t = Timestamp("2011-01-01") + offsets.Nano(5)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t.value == expected
        assert t.nanosecond == 5

        t = Timestamp(t)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t.value == expected
        assert t.nanosecond == 5

        t = Timestamp(np_datetime64_compat("2011-01-01 00:00:00.000000005Z"))
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t.value == expected
        assert t.nanosecond == 5

        expected = 1_293_840_000_000_000_010
        t = t + offsets.Nano(5)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t.value == expected
        assert t.nanosecond == 10

        t = Timestamp(t)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t.value == expected
        assert t.nanosecond == 10

        t = Timestamp(np_datetime64_compat("2011-01-01 00:00:00.000000010Z"))
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t.value == expected
        assert t.nanosecond == 10


class TestTimestampToJulianDate:
    def test_compare_1700(self):
        r = Timestamp("1700-06-23").to_julian_date()
        assert r == 2_342_145.5

    def test_compare_2000(self):
        r = Timestamp("2000-04-12").to_julian_date()
        assert r == 2_451_646.5

    def test_compare_2100(self):
        r = Timestamp("2100-08-12").to_julian_date()
        assert r == 2_488_292.5

    def test_compare_hour01(self):
        r = Timestamp("2000-08-12T01:00:00").to_julian_date()
        assert r == 2_451_768.5416666666666666

    def test_compare_hour13(self):
        r = Timestamp("2000-08-12T13:00:00").to_julian_date()
        assert r == 2_451_769.0416666666666666


class TestTimestampConversion:
    def test_conversion(self):
        # GH#9255
        ts = Timestamp("2000-01-01")

        result = ts.to_pydatetime()
        expected = datetime(2000, 1, 1)
        assert result == expected
        assert type(result) == type(expected)

        result = ts.to_datetime64()
        expected = np.datetime64(ts.value, "ns")
        assert result == expected
        assert type(result) == type(expected)
        assert result.dtype == expected.dtype

    def test_to_pydatetime_nonzero_nano(self):
        ts = Timestamp("2011-01-01 9:00:00.123456789")

        # Warn the user of data loss (nanoseconds).
        with tm.assert_produces_warning(UserWarning):
            expected = datetime(2011, 1, 1, 9, 0, 0, 123456)
            result = ts.to_pydatetime()
            assert result == expected

    def test_timestamp_to_datetime(self):
        stamp = Timestamp("20090415", tz="US/Eastern")
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_timestamp_to_datetime_dateutil(self):
        stamp = Timestamp("20090415", tz="dateutil/US/Eastern")
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_timestamp_to_datetime_explicit_pytz(self):
        stamp = Timestamp("20090415", tz=pytz.timezone("US/Eastern"))
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    @td.skip_if_windows_python_3
    def test_timestamp_to_datetime_explicit_dateutil(self):
        stamp = Timestamp("20090415", tz=gettz("US/Eastern"))
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_to_datetime_bijective(self):
        # Ensure that converting to datetime and back only loses precision
        # by going from nanoseconds to microseconds.
        exp_warning = None if Timestamp.max.nanosecond == 0 else UserWarning
        with tm.assert_produces_warning(exp_warning):
            pydt_max = Timestamp.max.to_pydatetime()

        assert Timestamp(pydt_max).value / 1000 == Timestamp.max.value / 1000

        exp_warning = None if Timestamp.min.nanosecond == 0 else UserWarning
        with tm.assert_produces_warning(exp_warning):
            pydt_min = Timestamp.min.to_pydatetime()

        # The next assertion can be enabled once GH#39221 is merged
        #  assert pydt_min < Timestamp.min  # this is bc nanos are dropped
        tdus = timedelta(microseconds=1)
        assert pydt_min + tdus > Timestamp.min

        assert Timestamp(pydt_min + tdus).value / 1000 == Timestamp.min.value / 1000

    def test_to_period_tz_warning(self):
        # GH#21333 make sure a warning is issued when timezone
        # info is lost
        ts = Timestamp("2009-04-15 16:17:18", tz="US/Eastern")
        with tm.assert_produces_warning(UserWarning):
            # warning that timezone info will be lost
            ts.to_period("D")

    def test_to_numpy_alias(self):
        # GH 24653: alias .to_numpy() for scalars
        ts = Timestamp(datetime.now())
        assert ts.to_datetime64() == ts.to_numpy()


class SubDatetime(datetime):
    pass


@pytest.mark.parametrize(
    "lh,rh",
    [
        (SubDatetime(2000, 1, 1), Timedelta(hours=1)),
        (Timedelta(hours=1), SubDatetime(2000, 1, 1)),
    ],
)
def test_dt_subclass_add_timedelta(lh, rh):
    # GH#25851
    # ensure that subclassed datetime works for
    # Timedelta operations
    result = lh + rh
    expected = SubDatetime(2000, 1, 1, 1)
    assert result == expected
