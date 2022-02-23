"""
test date_range, bdate_range construction from the convenience range functions
"""

from datetime import (
    datetime,
    time,
    timedelta,
)

import numpy as np
import pytest
import pytz
from pytz import timezone

from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
    BDay,
    CDay,
    DateOffset,
    MonthEnd,
    prefix_mapping,
)
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DatetimeIndex,
    Timedelta,
    Timestamp,
    bdate_range,
    date_range,
    offsets,
)
import pandas._testing as tm
from pandas.core.arrays.datetimes import generate_range

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)


class TestTimestampEquivDateRange:
    # Older tests in TestTimeSeries constructed their `stamp` objects
    # using `date_range` instead of the `Timestamp` constructor.
    # TestTimestampEquivDateRange checks that these are equivalent in the
    # pertinent cases.

    def test_date_range_timestamp_equiv(self):
        rng = date_range("20090415", "20090519", tz="US/Eastern")
        stamp = rng[0]

        ts = Timestamp("20090415", tz="US/Eastern")
        assert ts == stamp

    def test_date_range_timestamp_equiv_dateutil(self):
        rng = date_range("20090415", "20090519", tz="dateutil/US/Eastern")
        stamp = rng[0]

        ts = Timestamp("20090415", tz="dateutil/US/Eastern")
        assert ts == stamp

    def test_date_range_timestamp_equiv_explicit_pytz(self):
        rng = date_range("20090415", "20090519", tz=pytz.timezone("US/Eastern"))
        stamp = rng[0]

        ts = Timestamp("20090415", tz=pytz.timezone("US/Eastern"))
        assert ts == stamp

    @td.skip_if_windows_python_3
    def test_date_range_timestamp_equiv_explicit_dateutil(self):
        from pandas._libs.tslibs.timezones import dateutil_gettz as gettz

        rng = date_range("20090415", "20090519", tz=gettz("US/Eastern"))
        stamp = rng[0]

        ts = Timestamp("20090415", tz=gettz("US/Eastern"))
        assert ts == stamp

    def test_date_range_timestamp_equiv_from_datetime_instance(self):
        datetime_instance = datetime(2014, 3, 4)
        # build a timestamp with a frequency, since then it supports
        # addition/subtraction of integers
        timestamp_instance = date_range(datetime_instance, periods=1, freq="D")[0]

        ts = Timestamp(datetime_instance)
        assert ts == timestamp_instance

    def test_date_range_timestamp_equiv_preserve_frequency(self):
        timestamp_instance = date_range("2014-03-05", periods=1, freq="D")[0]
        ts = Timestamp("2014-03-05")

        assert timestamp_instance == ts


class TestDateRanges:
    def test_date_range_near_implementation_bound(self):
        # GH#???
        freq = Timedelta(1)

        with pytest.raises(OutOfBoundsDatetime, match="Cannot generate range with"):
            date_range(end=Timestamp.min, periods=2, freq=freq)

    def test_date_range_nat(self):
        # GH#11587
        msg = "Neither `start` nor `end` can be NaT"
        with pytest.raises(ValueError, match=msg):
            date_range(start="2016-01-01", end=pd.NaT, freq="D")
        with pytest.raises(ValueError, match=msg):
            date_range(start=pd.NaT, end="2016-01-01", freq="D")

    def test_date_range_multiplication_overflow(self):
        # GH#24255
        # check that overflows in calculating `addend = periods * stride`
        #  are caught
        with tm.assert_produces_warning(None):
            # we should _not_ be seeing a overflow RuntimeWarning
            dti = date_range(start="1677-09-22", periods=213503, freq="D")

        assert dti[0] == Timestamp("1677-09-22")
        assert len(dti) == 213503

        msg = "Cannot generate range with"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range("1969-05-04", periods=200000000, freq="30000D")

    def test_date_range_unsigned_overflow_handling(self):
        # GH#24255
        # case where `addend = periods * stride` overflows int64 bounds
        #  but not uint64 bounds
        dti = date_range(start="1677-09-22", end="2262-04-11", freq="D")

        dti2 = date_range(start=dti[0], periods=len(dti), freq="D")
        assert dti2.equals(dti)

        dti3 = date_range(end=dti[-1], periods=len(dti), freq="D")
        assert dti3.equals(dti)

    def test_date_range_int64_overflow_non_recoverable(self):
        # GH#24255
        # case with start later than 1970-01-01, overflow int64 but not uint64
        msg = "Cannot generate range with"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(start="1970-02-01", periods=106752 * 24, freq="H")

        # case with end before 1970-01-01, overflow int64 but not uint64
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(end="1969-11-14", periods=106752 * 24, freq="H")

    @pytest.mark.slow
    def test_date_range_int64_overflow_stride_endpoint_different_signs(self):
        # cases where stride * periods overflow int64 and stride/endpoint
        #  have different signs
        start = Timestamp("2262-02-23")
        end = Timestamp("1969-11-14")

        expected = date_range(start=start, end=end, freq="-1H")
        assert expected[0] == start
        assert expected[-1] == end

        dti = date_range(end=end, periods=len(expected), freq="-1H")
        tm.assert_index_equal(dti, expected)

        start2 = Timestamp("1970-02-01")
        end2 = Timestamp("1677-10-22")

        expected2 = date_range(start=start2, end=end2, freq="-1H")
        assert expected2[0] == start2
        assert expected2[-1] == end2

        dti2 = date_range(start=start2, periods=len(expected2), freq="-1H")
        tm.assert_index_equal(dti2, expected2)

    def test_date_range_out_of_bounds(self):
        # GH#14187
        msg = "Cannot generate range"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range("2016-01-01", periods=100000, freq="D")
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(end="1763-10-12", periods=100000, freq="D")

    def test_date_range_gen_error(self):
        rng = date_range("1/1/2000 00:00", "1/1/2000 00:18", freq="5min")
        assert len(rng) == 4

    @pytest.mark.parametrize("freq", ["AS", "YS"])
    def test_begin_year_alias(self, freq):
        # see gh-9313
        rng = date_range("1/1/2013", "7/1/2017", freq=freq)
        exp = DatetimeIndex(
            ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"],
            freq=freq,
        )
        tm.assert_index_equal(rng, exp)

    @pytest.mark.parametrize("freq", ["A", "Y"])
    def test_end_year_alias(self, freq):
        # see gh-9313
        rng = date_range("1/1/2013", "7/1/2017", freq=freq)
        exp = DatetimeIndex(
            ["2013-12-31", "2014-12-31", "2015-12-31", "2016-12-31"], freq=freq
        )
        tm.assert_index_equal(rng, exp)

    @pytest.mark.parametrize("freq", ["BA", "BY"])
    def test_business_end_year_alias(self, freq):
        # see gh-9313
        rng = date_range("1/1/2013", "7/1/2017", freq=freq)
        exp = DatetimeIndex(
            ["2013-12-31", "2014-12-31", "2015-12-31", "2016-12-30"], freq=freq
        )
        tm.assert_index_equal(rng, exp)

    def test_date_range_negative_freq(self):
        # GH 11018
        rng = date_range("2011-12-31", freq="-2A", periods=3)
        exp = DatetimeIndex(["2011-12-31", "2009-12-31", "2007-12-31"], freq="-2A")
        tm.assert_index_equal(rng, exp)
        assert rng.freq == "-2A"

        rng = date_range("2011-01-31", freq="-2M", periods=3)
        exp = DatetimeIndex(["2011-01-31", "2010-11-30", "2010-09-30"], freq="-2M")
        tm.assert_index_equal(rng, exp)
        assert rng.freq == "-2M"

    def test_date_range_bms_bug(self):
        # #1645
        rng = date_range("1/1/2000", periods=10, freq="BMS")

        ex_first = Timestamp("2000-01-03")
        assert rng[0] == ex_first

    def test_date_range_normalize(self):
        snap = datetime.today()
        n = 50

        rng = date_range(snap, periods=n, normalize=False, freq="2D")

        offset = timedelta(2)
        values = DatetimeIndex([snap + i * offset for i in range(n)], freq=offset)

        tm.assert_index_equal(rng, values)

        rng = date_range("1/1/2000 08:15", periods=n, normalize=False, freq="B")
        the_time = time(8, 15)
        for val in rng:
            assert val.time() == the_time

    def test_date_range_fy5252(self):
        dr = date_range(
            start="2013-01-01",
            periods=2,
            freq=offsets.FY5253(startingMonth=1, weekday=3, variation="nearest"),
        )
        assert dr[0] == Timestamp("2013-01-31")
        assert dr[1] == Timestamp("2014-01-30")

    def test_date_range_ambiguous_arguments(self):
        # #2538
        start = datetime(2011, 1, 1, 5, 3, 40)
        end = datetime(2011, 1, 1, 8, 9, 40)

        msg = (
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            date_range(start, end, periods=10, freq="s")

    def test_date_range_convenience_periods(self):
        # GH 20808
        result = date_range("2018-04-24", "2018-04-27", periods=3)
        expected = DatetimeIndex(
            ["2018-04-24 00:00:00", "2018-04-25 12:00:00", "2018-04-27 00:00:00"],
            freq=None,
        )

        tm.assert_index_equal(result, expected)

        # Test if spacing remains linear if tz changes to dst in range
        result = date_range(
            "2018-04-01 01:00:00",
            "2018-04-01 04:00:00",
            tz="Australia/Sydney",
            periods=3,
        )
        expected = DatetimeIndex(
            [
                Timestamp("2018-04-01 01:00:00+1100", tz="Australia/Sydney"),
                Timestamp("2018-04-01 02:00:00+1000", tz="Australia/Sydney"),
                Timestamp("2018-04-01 04:00:00+1000", tz="Australia/Sydney"),
            ]
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "start,end,result_tz",
        [
            ["20180101", "20180103", "US/Eastern"],
            [datetime(2018, 1, 1), datetime(2018, 1, 3), "US/Eastern"],
            [Timestamp("20180101"), Timestamp("20180103"), "US/Eastern"],
            [
                Timestamp("20180101", tz="US/Eastern"),
                Timestamp("20180103", tz="US/Eastern"),
                "US/Eastern",
            ],
            [
                Timestamp("20180101", tz="US/Eastern"),
                Timestamp("20180103", tz="US/Eastern"),
                None,
            ],
        ],
    )
    def test_date_range_linspacing_tz(self, start, end, result_tz):
        # GH 20983
        result = date_range(start, end, periods=3, tz=result_tz)
        expected = date_range("20180101", periods=3, freq="D", tz="US/Eastern")
        tm.assert_index_equal(result, expected)

    def test_date_range_businesshour(self):
        idx = DatetimeIndex(
            [
                "2014-07-04 09:00",
                "2014-07-04 10:00",
                "2014-07-04 11:00",
                "2014-07-04 12:00",
                "2014-07-04 13:00",
                "2014-07-04 14:00",
                "2014-07-04 15:00",
                "2014-07-04 16:00",
            ],
            freq="BH",
        )
        rng = date_range("2014-07-04 09:00", "2014-07-04 16:00", freq="BH")
        tm.assert_index_equal(idx, rng)

        idx = DatetimeIndex(["2014-07-04 16:00", "2014-07-07 09:00"], freq="BH")
        rng = date_range("2014-07-04 16:00", "2014-07-07 09:00", freq="BH")
        tm.assert_index_equal(idx, rng)

        idx = DatetimeIndex(
            [
                "2014-07-04 09:00",
                "2014-07-04 10:00",
                "2014-07-04 11:00",
                "2014-07-04 12:00",
                "2014-07-04 13:00",
                "2014-07-04 14:00",
                "2014-07-04 15:00",
                "2014-07-04 16:00",
                "2014-07-07 09:00",
                "2014-07-07 10:00",
                "2014-07-07 11:00",
                "2014-07-07 12:00",
                "2014-07-07 13:00",
                "2014-07-07 14:00",
                "2014-07-07 15:00",
                "2014-07-07 16:00",
                "2014-07-08 09:00",
                "2014-07-08 10:00",
                "2014-07-08 11:00",
                "2014-07-08 12:00",
                "2014-07-08 13:00",
                "2014-07-08 14:00",
                "2014-07-08 15:00",
                "2014-07-08 16:00",
            ],
            freq="BH",
        )
        rng = date_range("2014-07-04 09:00", "2014-07-08 16:00", freq="BH")
        tm.assert_index_equal(idx, rng)

    def test_range_misspecified(self):
        # GH #1095
        msg = (
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )

        with pytest.raises(ValueError, match=msg):
            date_range(start="1/1/2000")

        with pytest.raises(ValueError, match=msg):
            date_range(end="1/1/2000")

        with pytest.raises(ValueError, match=msg):
            date_range(periods=10)

        with pytest.raises(ValueError, match=msg):
            date_range(start="1/1/2000", freq="H")

        with pytest.raises(ValueError, match=msg):
            date_range(end="1/1/2000", freq="H")

        with pytest.raises(ValueError, match=msg):
            date_range(periods=10, freq="H")

        with pytest.raises(ValueError, match=msg):
            date_range()

    def test_compat_replace(self):
        # https://github.com/statsmodels/statsmodels/issues/3349
        # replace should take ints/longs for compat
        result = date_range(Timestamp("1960-04-01 00:00:00"), periods=76, freq="QS-JAN")
        assert len(result) == 76

    def test_catch_infinite_loop(self):
        offset = offsets.DateOffset(minute=5)
        # blow up, don't loop forever
        msg = "Offset <DateOffset: minute=5> did not increment date"
        with pytest.raises(ValueError, match=msg):
            date_range(datetime(2011, 11, 11), datetime(2011, 11, 12), freq=offset)

    @pytest.mark.parametrize("periods", (1, 2))
    def test_wom_len(self, periods):
        # https://github.com/pandas-dev/pandas/issues/20517
        res = date_range(start="20110101", periods=periods, freq="WOM-1MON")
        assert len(res) == periods

    def test_construct_over_dst(self):
        # GH 20854
        pre_dst = Timestamp("2010-11-07 01:00:00").tz_localize(
            "US/Pacific", ambiguous=True
        )
        pst_dst = Timestamp("2010-11-07 01:00:00").tz_localize(
            "US/Pacific", ambiguous=False
        )
        expect_data = [
            Timestamp("2010-11-07 00:00:00", tz="US/Pacific"),
            pre_dst,
            pst_dst,
        ]
        expected = DatetimeIndex(expect_data, freq="H")
        result = date_range(start="2010-11-7", periods=3, freq="H", tz="US/Pacific")
        tm.assert_index_equal(result, expected)

    def test_construct_with_different_start_end_string_format(self):
        # GH 12064
        result = date_range(
            "2013-01-01 00:00:00+09:00", "2013/01/01 02:00:00+09:00", freq="H"
        )
        expected = DatetimeIndex(
            [
                Timestamp("2013-01-01 00:00:00+09:00"),
                Timestamp("2013-01-01 01:00:00+09:00"),
                Timestamp("2013-01-01 02:00:00+09:00"),
            ],
            freq="H",
        )
        tm.assert_index_equal(result, expected)

    def test_error_with_zero_monthends(self):
        msg = r"Offset <0 \* MonthEnds> did not increment date"
        with pytest.raises(ValueError, match=msg):
            date_range("1/1/2000", "1/1/2001", freq=MonthEnd(0))

    def test_range_bug(self):
        # GH #770
        offset = DateOffset(months=3)
        result = date_range("2011-1-1", "2012-1-31", freq=offset)

        start = datetime(2011, 1, 1)
        expected = DatetimeIndex([start + i * offset for i in range(5)], freq=offset)
        tm.assert_index_equal(result, expected)

    def test_range_tz_pytz(self):
        # see gh-2906
        tz = timezone("US/Eastern")
        start = tz.localize(datetime(2011, 1, 1))
        end = tz.localize(datetime(2011, 1, 3))

        dr = date_range(start=start, periods=3)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end

        dr = date_range(end=end, periods=3)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end

        dr = date_range(start=start, end=end)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end

    @pytest.mark.parametrize(
        "start, end",
        [
            [
                Timestamp(datetime(2014, 3, 6), tz="US/Eastern"),
                Timestamp(datetime(2014, 3, 12), tz="US/Eastern"),
            ],
            [
                Timestamp(datetime(2013, 11, 1), tz="US/Eastern"),
                Timestamp(datetime(2013, 11, 6), tz="US/Eastern"),
            ],
        ],
    )
    def test_range_tz_dst_straddle_pytz(self, start, end):
        dr = date_range(start, end, freq="D")
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)

        dr = date_range(start, end, freq="D", tz="US/Eastern")
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)

        dr = date_range(
            start.replace(tzinfo=None),
            end.replace(tzinfo=None),
            freq="D",
            tz="US/Eastern",
        )
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)

    def test_range_tz_dateutil(self):
        # see gh-2906

        # Use maybe_get_tz to fix filename in tz under dateutil.
        from pandas._libs.tslibs.timezones import maybe_get_tz

        tz = lambda x: maybe_get_tz("dateutil/" + x)

        start = datetime(2011, 1, 1, tzinfo=tz("US/Eastern"))
        end = datetime(2011, 1, 3, tzinfo=tz("US/Eastern"))

        dr = date_range(start=start, periods=3)
        assert dr.tz == tz("US/Eastern")
        assert dr[0] == start
        assert dr[2] == end

        dr = date_range(end=end, periods=3)
        assert dr.tz == tz("US/Eastern")
        assert dr[0] == start
        assert dr[2] == end

        dr = date_range(start=start, end=end)
        assert dr.tz == tz("US/Eastern")
        assert dr[0] == start
        assert dr[2] == end

    @pytest.mark.parametrize("freq", ["1D", "3D", "2M", "7W", "3H", "A"])
    def test_range_closed(self, freq):
        begin = datetime(2011, 1, 1)
        end = datetime(2014, 1, 1)

        closed = date_range(begin, end, closed=None, freq=freq)
        left = date_range(begin, end, closed="left", freq=freq)
        right = date_range(begin, end, closed="right", freq=freq)
        expected_left = left
        expected_right = right

        if end == closed[-1]:
            expected_left = closed[:-1]
        if begin == closed[0]:
            expected_right = closed[1:]

        tm.assert_index_equal(expected_left, left)
        tm.assert_index_equal(expected_right, right)

    def test_range_closed_with_tz_aware_start_end(self):
        # GH12409, GH12684
        begin = Timestamp("2011/1/1", tz="US/Eastern")
        end = Timestamp("2014/1/1", tz="US/Eastern")

        for freq in ["1D", "3D", "2M", "7W", "3H", "A"]:
            closed = date_range(begin, end, closed=None, freq=freq)
            left = date_range(begin, end, closed="left", freq=freq)
            right = date_range(begin, end, closed="right", freq=freq)
            expected_left = left
            expected_right = right

            if end == closed[-1]:
                expected_left = closed[:-1]
            if begin == closed[0]:
                expected_right = closed[1:]

            tm.assert_index_equal(expected_left, left)
            tm.assert_index_equal(expected_right, right)

        begin = Timestamp("2011/1/1")
        end = Timestamp("2014/1/1")
        begintz = Timestamp("2011/1/1", tz="US/Eastern")
        endtz = Timestamp("2014/1/1", tz="US/Eastern")

        for freq in ["1D", "3D", "2M", "7W", "3H", "A"]:
            closed = date_range(begin, end, closed=None, freq=freq, tz="US/Eastern")
            left = date_range(begin, end, closed="left", freq=freq, tz="US/Eastern")
            right = date_range(begin, end, closed="right", freq=freq, tz="US/Eastern")
            expected_left = left
            expected_right = right

            if endtz == closed[-1]:
                expected_left = closed[:-1]
            if begintz == closed[0]:
                expected_right = closed[1:]

            tm.assert_index_equal(expected_left, left)
            tm.assert_index_equal(expected_right, right)

    @pytest.mark.parametrize("closed", ["right", "left", None])
    def test_range_closed_boundary(self, closed):
        # GH#11804
        right_boundary = date_range(
            "2015-09-12", "2015-12-01", freq="QS-MAR", closed=closed
        )
        left_boundary = date_range(
            "2015-09-01", "2015-09-12", freq="QS-MAR", closed=closed
        )
        both_boundary = date_range(
            "2015-09-01", "2015-12-01", freq="QS-MAR", closed=closed
        )
        expected_right = expected_left = expected_both = both_boundary

        if closed == "right":
            expected_left = both_boundary[1:]
        if closed == "left":
            expected_right = both_boundary[:-1]
        if closed is None:
            expected_right = both_boundary[1:]
            expected_left = both_boundary[:-1]

        tm.assert_index_equal(right_boundary, expected_right)
        tm.assert_index_equal(left_boundary, expected_left)
        tm.assert_index_equal(both_boundary, expected_both)

    def test_years_only(self):
        # GH 6961
        dr = date_range("2014", "2015", freq="M")
        assert dr[0] == datetime(2014, 1, 31)
        assert dr[-1] == datetime(2014, 12, 31)

    def test_freq_divides_end_in_nanos(self):
        # GH 10885
        result_1 = date_range("2005-01-12 10:00", "2005-01-12 16:00", freq="345min")
        result_2 = date_range("2005-01-13 10:00", "2005-01-13 16:00", freq="345min")
        expected_1 = DatetimeIndex(
            ["2005-01-12 10:00:00", "2005-01-12 15:45:00"],
            dtype="datetime64[ns]",
            freq="345T",
            tz=None,
        )
        expected_2 = DatetimeIndex(
            ["2005-01-13 10:00:00", "2005-01-13 15:45:00"],
            dtype="datetime64[ns]",
            freq="345T",
            tz=None,
        )
        tm.assert_index_equal(result_1, expected_1)
        tm.assert_index_equal(result_2, expected_2)

    def test_cached_range_bug(self):
        rng = date_range("2010-09-01 05:00:00", periods=50, freq=DateOffset(hours=6))
        assert len(rng) == 50
        assert rng[0] == datetime(2010, 9, 1, 5)

    def test_timezone_comparaison_bug(self):
        # smoke test
        start = Timestamp("20130220 10:00", tz="US/Eastern")
        result = date_range(start, periods=2, tz="US/Eastern")
        assert len(result) == 2

    def test_timezone_comparaison_assert(self):
        start = Timestamp("20130220 10:00", tz="US/Eastern")
        msg = "Inferred time zone not equal to passed time zone"
        with pytest.raises(AssertionError, match=msg):
            date_range(start, periods=2, tz="Europe/Berlin")

    def test_negative_non_tick_frequency_descending_dates(self, tz_aware_fixture):
        # GH 23270
        tz = tz_aware_fixture
        result = date_range(start="2011-06-01", end="2011-01-01", freq="-1MS", tz=tz)
        expected = date_range(end="2011-06-01", start="2011-01-01", freq="1MS", tz=tz)[
            ::-1
        ]
        tm.assert_index_equal(result, expected)


class TestDateRangeTZ:
    """Tests for date_range with timezones"""

    def test_hongkong_tz_convert(self):
        # GH#1673 smoke test
        dr = date_range("2012-01-01", "2012-01-10", freq="D", tz="Hongkong")

        # it works!
        dr.hour

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_date_range_span_dst_transition(self, tzstr):
        # GH#1778

        # Standard -> Daylight Savings Time
        dr = date_range("03/06/2012 00:00", periods=200, freq="W-FRI", tz="US/Eastern")

        assert (dr.hour == 0).all()

        dr = date_range("2012-11-02", periods=10, tz=tzstr)
        result = dr.hour
        expected = pd.Index([0] * 10)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_date_range_timezone_str_argument(self, tzstr):
        tz = timezones.maybe_get_tz(tzstr)
        result = date_range("1/1/2000", periods=10, tz=tzstr)
        expected = date_range("1/1/2000", periods=10, tz=tz)

        tm.assert_index_equal(result, expected)

    def test_date_range_with_fixedoffset_noname(self):
        from pandas.tests.indexes.datetimes.test_timezones import fixed_off_no_name

        off = fixed_off_no_name
        start = datetime(2012, 3, 11, 5, 0, 0, tzinfo=off)
        end = datetime(2012, 6, 11, 5, 0, 0, tzinfo=off)
        rng = date_range(start=start, end=end)
        assert off == rng.tz

        idx = pd.Index([start, end])
        assert off == idx.tz

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_date_range_with_tz(self, tzstr):
        stamp = Timestamp("3/11/2012 05:00", tz=tzstr)
        assert stamp.hour == 5

        rng = date_range("3/11/2012 04:00", periods=10, freq="H", tz=tzstr)

        assert stamp == rng[1]


class TestGenRangeGeneration:
    def test_generate(self):
        rng1 = list(generate_range(START, END, offset=BDay()))
        rng2 = list(generate_range(START, END, offset="B"))
        assert rng1 == rng2

    def test_generate_cday(self):
        rng1 = list(generate_range(START, END, offset=CDay()))
        rng2 = list(generate_range(START, END, offset="C"))
        assert rng1 == rng2

    def test_1(self):
        rng = list(generate_range(start=datetime(2009, 3, 25), periods=2))
        expected = [datetime(2009, 3, 25), datetime(2009, 3, 26)]
        assert rng == expected

    def test_2(self):
        rng = list(generate_range(start=datetime(2008, 1, 1), end=datetime(2008, 1, 3)))
        expected = [datetime(2008, 1, 1), datetime(2008, 1, 2), datetime(2008, 1, 3)]
        assert rng == expected

    def test_3(self):
        rng = list(generate_range(start=datetime(2008, 1, 5), end=datetime(2008, 1, 6)))
        expected = []
        assert rng == expected

    def test_precision_finer_than_offset(self):
        # GH#9907
        result1 = date_range(
            start="2015-04-15 00:00:03", end="2016-04-22 00:00:00", freq="Q"
        )
        result2 = date_range(
            start="2015-04-15 00:00:03", end="2015-06-22 00:00:04", freq="W"
        )
        expected1_list = [
            "2015-06-30 00:00:03",
            "2015-09-30 00:00:03",
            "2015-12-31 00:00:03",
            "2016-03-31 00:00:03",
        ]
        expected2_list = [
            "2015-04-19 00:00:03",
            "2015-04-26 00:00:03",
            "2015-05-03 00:00:03",
            "2015-05-10 00:00:03",
            "2015-05-17 00:00:03",
            "2015-05-24 00:00:03",
            "2015-05-31 00:00:03",
            "2015-06-07 00:00:03",
            "2015-06-14 00:00:03",
            "2015-06-21 00:00:03",
        ]
        expected1 = DatetimeIndex(
            expected1_list, dtype="datetime64[ns]", freq="Q-DEC", tz=None
        )
        expected2 = DatetimeIndex(
            expected2_list, dtype="datetime64[ns]", freq="W-SUN", tz=None
        )
        tm.assert_index_equal(result1, expected1)
        tm.assert_index_equal(result2, expected2)

    dt1, dt2 = "2017-01-01", "2017-01-01"
    tz1, tz2 = "US/Eastern", "Europe/London"

    @pytest.mark.parametrize(
        "start,end",
        [
            (Timestamp(dt1, tz=tz1), Timestamp(dt2)),
            (Timestamp(dt1), Timestamp(dt2, tz=tz2)),
            (Timestamp(dt1, tz=tz1), Timestamp(dt2, tz=tz2)),
            (Timestamp(dt1, tz=tz2), Timestamp(dt2, tz=tz1)),
        ],
    )
    def test_mismatching_tz_raises_err(self, start, end):
        # issue 18488
        msg = "Start and end cannot both be tz-aware with different timezones"
        with pytest.raises(TypeError, match=msg):
            date_range(start, end)
        with pytest.raises(TypeError, match=msg):
            date_range(start, end, freq=BDay())


class TestBusinessDateRange:
    def test_constructor(self):
        bdate_range(START, END, freq=BDay())
        bdate_range(START, periods=20, freq=BDay())
        bdate_range(end=START, periods=20, freq=BDay())

        msg = "periods must be a number, got B"
        with pytest.raises(TypeError, match=msg):
            date_range("2011-1-1", "2012-1-1", "B")

        with pytest.raises(TypeError, match=msg):
            bdate_range("2011-1-1", "2012-1-1", "B")

        msg = "freq must be specified for bdate_range; use date_range instead"
        with pytest.raises(TypeError, match=msg):
            bdate_range(START, END, periods=10, freq=None)

    def test_misc(self):
        end = datetime(2009, 5, 13)
        dr = bdate_range(end=end, periods=20)
        firstDate = end - 19 * BDay()

        assert len(dr) == 20
        assert dr[0] == firstDate
        assert dr[-1] == end

    def test_date_parse_failure(self):
        badly_formed_date = "2007/100/1"

        msg = "could not convert string to Timestamp"
        with pytest.raises(ValueError, match=msg):
            Timestamp(badly_formed_date)

        with pytest.raises(ValueError, match=msg):
            bdate_range(start=badly_formed_date, periods=10)

        with pytest.raises(ValueError, match=msg):
            bdate_range(end=badly_formed_date, periods=10)

        with pytest.raises(ValueError, match=msg):
            bdate_range(badly_formed_date, badly_formed_date)

    def test_daterange_bug_456(self):
        # GH #456
        rng1 = bdate_range("12/5/2011", "12/5/2011")
        rng2 = bdate_range("12/2/2011", "12/5/2011")
        assert rng2._data.freq == BDay()

        result = rng1.union(rng2)
        assert isinstance(result, DatetimeIndex)

    @pytest.mark.parametrize("closed", ["left", "right"])
    def test_bdays_and_open_boundaries(self, closed):
        # GH 6673
        start = "2018-07-21"  # Saturday
        end = "2018-07-29"  # Sunday
        result = date_range(start, end, freq="B", closed=closed)

        bday_start = "2018-07-23"  # Monday
        bday_end = "2018-07-27"  # Friday
        expected = date_range(bday_start, bday_end, freq="D")
        tm.assert_index_equal(result, expected)
        # Note: we do _not_ expect the freqs to match here

    def test_bday_near_overflow(self):
        # GH#24252 avoid doing unnecessary addition that _would_ overflow
        start = Timestamp.max.floor("D").to_pydatetime()
        rng = date_range(start, end=None, periods=1, freq="B")
        expected = DatetimeIndex([start], freq="B")
        tm.assert_index_equal(rng, expected)

    def test_bday_overflow_error(self):
        # GH#24252 check that we get OutOfBoundsDatetime and not OverflowError
        msg = "Out of bounds nanosecond timestamp"
        start = Timestamp.max.floor("D").to_pydatetime()
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(start, periods=2, freq="B")


class TestCustomDateRange:
    def test_constructor(self):
        bdate_range(START, END, freq=CDay())
        bdate_range(START, periods=20, freq=CDay())
        bdate_range(end=START, periods=20, freq=CDay())

        msg = "periods must be a number, got C"
        with pytest.raises(TypeError, match=msg):
            date_range("2011-1-1", "2012-1-1", "C")

        with pytest.raises(TypeError, match=msg):
            bdate_range("2011-1-1", "2012-1-1", "C")

    def test_misc(self):
        end = datetime(2009, 5, 13)
        dr = bdate_range(end=end, periods=20, freq="C")
        firstDate = end - 19 * CDay()

        assert len(dr) == 20
        assert dr[0] == firstDate
        assert dr[-1] == end

    def test_daterange_bug_456(self):
        # GH #456
        rng1 = bdate_range("12/5/2011", "12/5/2011", freq="C")
        rng2 = bdate_range("12/2/2011", "12/5/2011", freq="C")
        assert rng2._data.freq == CDay()

        result = rng1.union(rng2)
        assert isinstance(result, DatetimeIndex)

    def test_cdaterange(self):
        result = bdate_range("2013-05-01", periods=3, freq="C")
        expected = DatetimeIndex(["2013-05-01", "2013-05-02", "2013-05-03"], freq="C")
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    def test_cdaterange_weekmask(self):
        result = bdate_range(
            "2013-05-01", periods=3, freq="C", weekmask="Sun Mon Tue Wed Thu"
        )
        expected = DatetimeIndex(
            ["2013-05-01", "2013-05-02", "2013-05-05"], freq=result.freq
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

        # raise with non-custom freq
        msg = (
            "a custom frequency string is required when holidays or "
            "weekmask are passed, got frequency B"
        )
        with pytest.raises(ValueError, match=msg):
            bdate_range("2013-05-01", periods=3, weekmask="Sun Mon Tue Wed Thu")

    def test_cdaterange_holidays(self):
        result = bdate_range("2013-05-01", periods=3, freq="C", holidays=["2013-05-01"])
        expected = DatetimeIndex(
            ["2013-05-02", "2013-05-03", "2013-05-06"], freq=result.freq
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

        # raise with non-custom freq
        msg = (
            "a custom frequency string is required when holidays or "
            "weekmask are passed, got frequency B"
        )
        with pytest.raises(ValueError, match=msg):
            bdate_range("2013-05-01", periods=3, holidays=["2013-05-01"])

    def test_cdaterange_weekmask_and_holidays(self):
        result = bdate_range(
            "2013-05-01",
            periods=3,
            freq="C",
            weekmask="Sun Mon Tue Wed Thu",
            holidays=["2013-05-01"],
        )
        expected = DatetimeIndex(
            ["2013-05-02", "2013-05-05", "2013-05-06"], freq=result.freq
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

        # raise with non-custom freq
        msg = (
            "a custom frequency string is required when holidays or "
            "weekmask are passed, got frequency B"
        )
        with pytest.raises(ValueError, match=msg):
            bdate_range(
                "2013-05-01",
                periods=3,
                weekmask="Sun Mon Tue Wed Thu",
                holidays=["2013-05-01"],
            )

    @pytest.mark.parametrize(
        "freq", [freq for freq in prefix_mapping if freq.startswith("C")]
    )
    def test_all_custom_freq(self, freq):
        # should not raise
        bdate_range(
            START, END, freq=freq, weekmask="Mon Wed Fri", holidays=["2009-03-14"]
        )

        bad_freq = freq + "FOO"
        msg = f"invalid custom frequency string: {bad_freq}"
        with pytest.raises(ValueError, match=msg):
            bdate_range(START, END, freq=bad_freq)

    @pytest.mark.parametrize(
        "start_end",
        [
            ("2018-01-01T00:00:01.000Z", "2018-01-03T00:00:01.000Z"),
            ("2018-01-01T00:00:00.010Z", "2018-01-03T00:00:00.010Z"),
            ("2001-01-01T00:00:00.010Z", "2001-01-03T00:00:00.010Z"),
        ],
    )
    def test_range_with_millisecond_resolution(self, start_end):
        # https://github.com/pandas-dev/pandas/issues/24110
        start, end = start_end
        result = date_range(start=start, end=end, periods=2, closed="left")
        expected = DatetimeIndex([start])
        tm.assert_index_equal(result, expected)


def test_date_range_with_custom_holidays():
    # GH 30593
    freq = offsets.CustomBusinessHour(start="15:00", holidays=["2020-11-26"])
    result = date_range(start="2020-11-25 15:00", periods=4, freq=freq)
    expected = DatetimeIndex(
        [
            "2020-11-25 15:00:00",
            "2020-11-25 16:00:00",
            "2020-11-27 15:00:00",
            "2020-11-27 16:00:00",
        ],
        freq=freq,
    )
    tm.assert_index_equal(result, expected)
