"""
Tests for Timestamp timezone-related methods
"""
from datetime import (
    date,
    datetime,
    timedelta,
)

import dateutil
from dateutil.tz import (
    gettz,
    tzoffset,
)
import pytest
import pytz
from pytz.exceptions import (
    AmbiguousTimeError,
    NonExistentTimeError,
)

from pandas._libs.tslibs import timezones
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td

from pandas import (
    NaT,
    Timestamp,
)


class TestTimestampTZOperations:
    # --------------------------------------------------------------
    # Timestamp.tz_localize

    def test_tz_localize_pushes_out_of_bounds(self):
        # GH#12677
        # tz_localize that pushes away from the boundary is OK
        msg = (
            f"Converting {Timestamp.min.strftime('%Y-%m-%d %H:%M:%S')} "
            f"underflows past {Timestamp.min}"
        )
        pac = Timestamp.min.tz_localize("US/Pacific")
        assert pac.value > Timestamp.min.value
        pac.tz_convert("Asia/Tokyo")  # tz_convert doesn't change value
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.tz_localize("Asia/Tokyo")

        # tz_localize that pushes away from the boundary is OK
        msg = (
            f"Converting {Timestamp.max.strftime('%Y-%m-%d %H:%M:%S')} "
            f"overflows past {Timestamp.max}"
        )
        tokyo = Timestamp.max.tz_localize("Asia/Tokyo")
        assert tokyo.value < Timestamp.max.value
        tokyo.tz_convert("US/Pacific")  # tz_convert doesn't change value
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.tz_localize("US/Pacific")

    def test_tz_localize_ambiguous_bool(self):
        # make sure that we are correctly accepting bool values as ambiguous
        # GH#14402
        ts = Timestamp("2015-11-01 01:00:03")
        expected0 = Timestamp("2015-11-01 01:00:03-0500", tz="US/Central")
        expected1 = Timestamp("2015-11-01 01:00:03-0600", tz="US/Central")

        msg = "Cannot infer dst time from 2015-11-01 01:00:03"
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            ts.tz_localize("US/Central")

        result = ts.tz_localize("US/Central", ambiguous=True)
        assert result == expected0

        result = ts.tz_localize("US/Central", ambiguous=False)
        assert result == expected1

    def test_tz_localize_ambiguous(self):
        ts = Timestamp("2014-11-02 01:00")
        ts_dst = ts.tz_localize("US/Eastern", ambiguous=True)
        ts_no_dst = ts.tz_localize("US/Eastern", ambiguous=False)

        assert (ts_no_dst.value - ts_dst.value) / 1e9 == 3600
        msg = "Cannot infer offset with only one time"
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize("US/Eastern", ambiguous="infer")

        # GH#8025
        msg = "Cannot localize tz-aware Timestamp, use tz_convert for conversions"
        with pytest.raises(TypeError, match=msg):
            Timestamp("2011-01-01", tz="US/Eastern").tz_localize("Asia/Tokyo")

        msg = "Cannot convert tz-naive Timestamp, use tz_localize to localize"
        with pytest.raises(TypeError, match=msg):
            Timestamp("2011-01-01").tz_convert("Asia/Tokyo")

    @pytest.mark.parametrize(
        "stamp, tz",
        [
            ("2015-03-08 02:00", "US/Eastern"),
            ("2015-03-08 02:30", "US/Pacific"),
            ("2015-03-29 02:00", "Europe/Paris"),
            ("2015-03-29 02:30", "Europe/Belgrade"),
        ],
    )
    def test_tz_localize_nonexistent(self, stamp, tz):
        # GH#13057
        ts = Timestamp(stamp)
        with pytest.raises(NonExistentTimeError, match=stamp):
            ts.tz_localize(tz)
        # GH 22644
        with pytest.raises(NonExistentTimeError, match=stamp):
            ts.tz_localize(tz, nonexistent="raise")
        assert ts.tz_localize(tz, nonexistent="NaT") is NaT

    def test_tz_localize_ambiguous_raise(self):
        # GH#13057
        ts = Timestamp("2015-11-1 01:00")
        msg = "Cannot infer dst time from 2015-11-01 01:00:00,"
        with pytest.raises(AmbiguousTimeError, match=msg):
            ts.tz_localize("US/Pacific", ambiguous="raise")

    def test_tz_localize_nonexistent_invalid_arg(self):
        # GH 22644
        tz = "Europe/Warsaw"
        ts = Timestamp("2015-03-29 02:00:00")
        msg = (
            "The nonexistent argument must be one of 'raise', 'NaT', "
            "'shift_forward', 'shift_backward' or a timedelta object"
        )
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent="foo")

    @pytest.mark.parametrize(
        "stamp",
        [
            "2014-02-01 09:00",
            "2014-07-08 09:00",
            "2014-11-01 17:00",
            "2014-11-05 00:00",
        ],
    )
    def test_tz_localize_roundtrip(self, stamp, tz_aware_fixture):
        tz = tz_aware_fixture
        ts = Timestamp(stamp)
        localized = ts.tz_localize(tz)
        assert localized == Timestamp(stamp, tz=tz)

        msg = "Cannot localize tz-aware Timestamp"
        with pytest.raises(TypeError, match=msg):
            localized.tz_localize(tz)

        reset = localized.tz_localize(None)
        assert reset == ts
        assert reset.tzinfo is None

    def test_tz_localize_ambiguous_compat(self):
        # validate that pytz and dateutil are compat for dst
        # when the transition happens
        naive = Timestamp("2013-10-27 01:00:00")

        pytz_zone = "Europe/London"
        dateutil_zone = "dateutil/Europe/London"
        result_pytz = naive.tz_localize(pytz_zone, ambiguous=0)
        result_dateutil = naive.tz_localize(dateutil_zone, ambiguous=0)
        assert result_pytz.value == result_dateutil.value
        assert result_pytz.value == 1382835600000000000

        # fixed ambiguous behavior
        # see gh-14621
        assert result_pytz.to_pydatetime().tzname() == "GMT"
        assert result_dateutil.to_pydatetime().tzname() == "BST"
        assert str(result_pytz) == str(result_dateutil)

        # 1 hour difference
        result_pytz = naive.tz_localize(pytz_zone, ambiguous=1)
        result_dateutil = naive.tz_localize(dateutil_zone, ambiguous=1)
        assert result_pytz.value == result_dateutil.value
        assert result_pytz.value == 1382832000000000000

        # see gh-14621
        assert str(result_pytz) == str(result_dateutil)
        assert (
            result_pytz.to_pydatetime().tzname()
            == result_dateutil.to_pydatetime().tzname()
        )

    @pytest.mark.parametrize(
        "tz",
        [
            pytz.timezone("US/Eastern"),
            gettz("US/Eastern"),
            "US/Eastern",
            "dateutil/US/Eastern",
        ],
    )
    def test_timestamp_tz_localize(self, tz):
        stamp = Timestamp("3/11/2012 04:00")

        result = stamp.tz_localize(tz)
        expected = Timestamp("3/11/2012 04:00", tz=tz)
        assert result.hour == expected.hour
        assert result == expected

    @pytest.mark.parametrize(
        "start_ts, tz, end_ts, shift",
        [
            ["2015-03-29 02:20:00", "Europe/Warsaw", "2015-03-29 03:00:00", "forward"],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:59:59.999999999",
                "backward",
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 03:20:00",
                timedelta(hours=1),
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:20:00",
                timedelta(hours=-1),
            ],
            ["2018-03-11 02:33:00", "US/Pacific", "2018-03-11 03:00:00", "forward"],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:59:59.999999999",
                "backward",
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 03:33:00",
                timedelta(hours=1),
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:33:00",
                timedelta(hours=-1),
            ],
        ],
    )
    @pytest.mark.parametrize("tz_type", ["", "dateutil/"])
    def test_timestamp_tz_localize_nonexistent_shift(
        self, start_ts, tz, end_ts, shift, tz_type
    ):
        # GH 8917, 24466
        tz = tz_type + tz
        if isinstance(shift, str):
            shift = "shift_" + shift
        ts = Timestamp(start_ts)
        result = ts.tz_localize(tz, nonexistent=shift)
        expected = Timestamp(end_ts).tz_localize(tz)
        assert result == expected

    @pytest.mark.parametrize("offset", [-1, 1])
    @pytest.mark.parametrize("tz_type", ["", "dateutil/"])
    def test_timestamp_tz_localize_nonexistent_shift_invalid(self, offset, tz_type):
        # GH 8917, 24466
        tz = tz_type + "Europe/Warsaw"
        ts = Timestamp("2015-03-29 02:20:00")
        msg = "The provided timedelta will relocalize on a nonexistent time"
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent=timedelta(seconds=offset))

    @pytest.mark.parametrize("tz", ["Europe/Warsaw", "dateutil/Europe/Warsaw"])
    def test_timestamp_tz_localize_nonexistent_NaT(self, tz):
        # GH 8917
        ts = Timestamp("2015-03-29 02:20:00")
        result = ts.tz_localize(tz, nonexistent="NaT")
        assert result is NaT

    @pytest.mark.parametrize("tz", ["Europe/Warsaw", "dateutil/Europe/Warsaw"])
    def test_timestamp_tz_localize_nonexistent_raise(self, tz):
        # GH 8917
        ts = Timestamp("2015-03-29 02:20:00")
        msg = "2015-03-29 02:20:00"
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            ts.tz_localize(tz, nonexistent="raise")
        msg = (
            "The nonexistent argument must be one of 'raise', 'NaT', "
            "'shift_forward', 'shift_backward' or a timedelta object"
        )
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent="foo")

    # ------------------------------------------------------------------
    # Timestamp.tz_convert

    @pytest.mark.parametrize(
        "stamp",
        [
            "2014-02-01 09:00",
            "2014-07-08 09:00",
            "2014-11-01 17:00",
            "2014-11-05 00:00",
        ],
    )
    def test_tz_convert_roundtrip(self, stamp, tz_aware_fixture):
        tz = tz_aware_fixture

        ts = Timestamp(stamp, tz="UTC")
        converted = ts.tz_convert(tz)

        reset = converted.tz_convert(None)
        assert reset == Timestamp(stamp)
        assert reset.tzinfo is None
        assert reset == converted.tz_convert("UTC").tz_localize(None)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_astimezone(self, tzstr):
        # astimezone is an alias for tz_convert, so keep it with
        # the tz_convert tests
        utcdate = Timestamp("3/11/2012 22:00", tz="UTC")
        expected = utcdate.tz_convert(tzstr)
        result = utcdate.astimezone(tzstr)
        assert expected == result
        assert isinstance(result, Timestamp)

    @td.skip_if_windows
    def test_tz_convert_utc_with_system_utc(self):

        # from system utc to real utc
        ts = Timestamp("2001-01-05 11:56", tz=timezones.maybe_get_tz("dateutil/UTC"))
        # check that the time hasn't changed.
        assert ts == ts.tz_convert(dateutil.tz.tzutc())

        # from system utc to real utc
        ts = Timestamp("2001-01-05 11:56", tz=timezones.maybe_get_tz("dateutil/UTC"))
        # check that the time hasn't changed.
        assert ts == ts.tz_convert(dateutil.tz.tzutc())

    # ------------------------------------------------------------------
    # Timestamp.__init__ with tz str or tzinfo

    def test_timestamp_constructor_tz_utc(self):
        utc_stamp = Timestamp("3/11/2012 05:00", tz="utc")
        assert utc_stamp.tzinfo is pytz.utc
        assert utc_stamp.hour == 5

        utc_stamp = Timestamp("3/11/2012 05:00").tz_localize("utc")
        assert utc_stamp.hour == 5

    def test_timestamp_to_datetime_tzoffset(self):
        tzinfo = tzoffset(None, 7200)
        expected = Timestamp("3/11/2012 04:00", tz=tzinfo)
        result = Timestamp(expected.to_pydatetime())
        assert expected == result

    def test_timestamp_constructor_near_dst_boundary(self):
        # GH#11481 & GH#15777
        # Naive string timestamps were being localized incorrectly
        # with tz_convert_from_utc_single instead of tz_localize_to_utc

        for tz in ["Europe/Brussels", "Europe/Prague"]:
            result = Timestamp("2015-10-25 01:00", tz=tz)
            expected = Timestamp("2015-10-25 01:00").tz_localize(tz)
            assert result == expected

            msg = "Cannot infer dst time from 2015-10-25 02:00:00"
            with pytest.raises(pytz.AmbiguousTimeError, match=msg):
                Timestamp("2015-10-25 02:00", tz=tz)

        result = Timestamp("2017-03-26 01:00", tz="Europe/Paris")
        expected = Timestamp("2017-03-26 01:00").tz_localize("Europe/Paris")
        assert result == expected

        msg = "2017-03-26 02:00"
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            Timestamp("2017-03-26 02:00", tz="Europe/Paris")

        # GH#11708
        naive = Timestamp("2015-11-18 10:00:00")
        result = naive.tz_localize("UTC").tz_convert("Asia/Kolkata")
        expected = Timestamp("2015-11-18 15:30:00+0530", tz="Asia/Kolkata")
        assert result == expected

        # GH#15823
        result = Timestamp("2017-03-26 00:00", tz="Europe/Paris")
        expected = Timestamp("2017-03-26 00:00:00+0100", tz="Europe/Paris")
        assert result == expected

        result = Timestamp("2017-03-26 01:00", tz="Europe/Paris")
        expected = Timestamp("2017-03-26 01:00:00+0100", tz="Europe/Paris")
        assert result == expected

        msg = "2017-03-26 02:00"
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            Timestamp("2017-03-26 02:00", tz="Europe/Paris")

        result = Timestamp("2017-03-26 02:00:00+0100", tz="Europe/Paris")
        naive = Timestamp(result.value)
        expected = naive.tz_localize("UTC").tz_convert("Europe/Paris")
        assert result == expected

        result = Timestamp("2017-03-26 03:00", tz="Europe/Paris")
        expected = Timestamp("2017-03-26 03:00:00+0200", tz="Europe/Paris")
        assert result == expected

    @pytest.mark.parametrize(
        "tz",
        [
            pytz.timezone("US/Eastern"),
            gettz("US/Eastern"),
            "US/Eastern",
            "dateutil/US/Eastern",
        ],
    )
    def test_timestamp_constructed_by_date_and_tz(self, tz):
        # GH#2993, Timestamp cannot be constructed by datetime.date
        # and tz correctly

        result = Timestamp(date(2012, 3, 11), tz=tz)

        expected = Timestamp("3/11/2012", tz=tz)
        assert result.hour == expected.hour
        assert result == expected

    @pytest.mark.parametrize(
        "tz",
        [
            pytz.timezone("US/Eastern"),
            gettz("US/Eastern"),
            "US/Eastern",
            "dateutil/US/Eastern",
        ],
    )
    def test_timestamp_add_timedelta_push_over_dst_boundary(self, tz):
        # GH#1389

        # 4 hours before DST transition
        stamp = Timestamp("3/10/2012 22:00", tz=tz)

        result = stamp + timedelta(hours=6)

        # spring forward, + "7" hours
        expected = Timestamp("3/11/2012 05:00", tz=tz)

        assert result == expected

    def test_timestamp_timetz_equivalent_with_datetime_tz(self, tz_naive_fixture):
        # GH21358
        tz = timezones.maybe_get_tz(tz_naive_fixture)

        stamp = Timestamp("2018-06-04 10:20:30", tz=tz)
        _datetime = datetime(2018, 6, 4, hour=10, minute=20, second=30, tzinfo=tz)

        result = stamp.timetz()
        expected = _datetime.timetz()

        assert result == expected
