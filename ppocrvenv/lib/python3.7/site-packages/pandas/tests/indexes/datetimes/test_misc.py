import calendar
from datetime import datetime
import locale
import unicodedata

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Timedelta,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm


class TestTimeSeries:
    def test_range_edges(self):
        # GH#13672
        idx = date_range(
            start=Timestamp("1970-01-01 00:00:00.000000001"),
            end=Timestamp("1970-01-01 00:00:00.000000004"),
            freq="N",
        )
        exp = DatetimeIndex(
            [
                "1970-01-01 00:00:00.000000001",
                "1970-01-01 00:00:00.000000002",
                "1970-01-01 00:00:00.000000003",
                "1970-01-01 00:00:00.000000004",
            ],
            freq="N",
        )
        tm.assert_index_equal(idx, exp)

    def test_range_edges2(self):

        idx = date_range(
            start=Timestamp("1970-01-01 00:00:00.000000004"),
            end=Timestamp("1970-01-01 00:00:00.000000001"),
            freq="N",
        )
        exp = DatetimeIndex([], freq="N")
        tm.assert_index_equal(idx, exp)

    def test_range_edges3(self):

        idx = date_range(
            start=Timestamp("1970-01-01 00:00:00.000000001"),
            end=Timestamp("1970-01-01 00:00:00.000000001"),
            freq="N",
        )
        exp = DatetimeIndex(["1970-01-01 00:00:00.000000001"], freq="N")
        tm.assert_index_equal(idx, exp)

    def test_range_edges4(self):

        idx = date_range(
            start=Timestamp("1970-01-01 00:00:00.000001"),
            end=Timestamp("1970-01-01 00:00:00.000004"),
            freq="U",
        )
        exp = DatetimeIndex(
            [
                "1970-01-01 00:00:00.000001",
                "1970-01-01 00:00:00.000002",
                "1970-01-01 00:00:00.000003",
                "1970-01-01 00:00:00.000004",
            ],
            freq="U",
        )
        tm.assert_index_equal(idx, exp)

    def test_range_edges5(self):

        idx = date_range(
            start=Timestamp("1970-01-01 00:00:00.001"),
            end=Timestamp("1970-01-01 00:00:00.004"),
            freq="L",
        )
        exp = DatetimeIndex(
            [
                "1970-01-01 00:00:00.001",
                "1970-01-01 00:00:00.002",
                "1970-01-01 00:00:00.003",
                "1970-01-01 00:00:00.004",
            ],
            freq="L",
        )
        tm.assert_index_equal(idx, exp)

    def test_range_edges6(self):
        idx = date_range(
            start=Timestamp("1970-01-01 00:00:01"),
            end=Timestamp("1970-01-01 00:00:04"),
            freq="S",
        )
        exp = DatetimeIndex(
            [
                "1970-01-01 00:00:01",
                "1970-01-01 00:00:02",
                "1970-01-01 00:00:03",
                "1970-01-01 00:00:04",
            ],
            freq="S",
        )
        tm.assert_index_equal(idx, exp)

    def test_range_edges7(self):
        idx = date_range(
            start=Timestamp("1970-01-01 00:01"),
            end=Timestamp("1970-01-01 00:04"),
            freq="T",
        )
        exp = DatetimeIndex(
            [
                "1970-01-01 00:01",
                "1970-01-01 00:02",
                "1970-01-01 00:03",
                "1970-01-01 00:04",
            ],
            freq="T",
        )
        tm.assert_index_equal(idx, exp)

    def test_range_edges8(self):
        idx = date_range(
            start=Timestamp("1970-01-01 01:00"),
            end=Timestamp("1970-01-01 04:00"),
            freq="H",
        )
        exp = DatetimeIndex(
            [
                "1970-01-01 01:00",
                "1970-01-01 02:00",
                "1970-01-01 03:00",
                "1970-01-01 04:00",
            ],
            freq="H",
        )
        tm.assert_index_equal(idx, exp)

    def test_range_edges9(self):
        idx = date_range(
            start=Timestamp("1970-01-01"), end=Timestamp("1970-01-04"), freq="D"
        )
        exp = DatetimeIndex(
            ["1970-01-01", "1970-01-02", "1970-01-03", "1970-01-04"], freq="D"
        )
        tm.assert_index_equal(idx, exp)


class TestDatetime64:
    def test_datetimeindex_accessors(self):
        dti_naive = date_range(freq="D", start=datetime(1998, 1, 1), periods=365)
        # GH#13303
        dti_tz = date_range(
            freq="D", start=datetime(1998, 1, 1), periods=365, tz="US/Eastern"
        )
        for dti in [dti_naive, dti_tz]:

            assert dti.year[0] == 1998
            assert dti.month[0] == 1
            assert dti.day[0] == 1
            assert dti.hour[0] == 0
            assert dti.minute[0] == 0
            assert dti.second[0] == 0
            assert dti.microsecond[0] == 0
            assert dti.dayofweek[0] == 3

            assert dti.dayofyear[0] == 1
            assert dti.dayofyear[120] == 121

            assert dti.isocalendar().week[0] == 1
            assert dti.isocalendar().week[120] == 18

            assert dti.quarter[0] == 1
            assert dti.quarter[120] == 2

            assert dti.days_in_month[0] == 31
            assert dti.days_in_month[90] == 30

            assert dti.is_month_start[0]
            assert not dti.is_month_start[1]
            assert dti.is_month_start[31]
            assert dti.is_quarter_start[0]
            assert dti.is_quarter_start[90]
            assert dti.is_year_start[0]
            assert not dti.is_year_start[364]
            assert not dti.is_month_end[0]
            assert dti.is_month_end[30]
            assert not dti.is_month_end[31]
            assert dti.is_month_end[364]
            assert not dti.is_quarter_end[0]
            assert not dti.is_quarter_end[30]
            assert dti.is_quarter_end[89]
            assert dti.is_quarter_end[364]
            assert not dti.is_year_end[0]
            assert dti.is_year_end[364]

            assert len(dti.year) == 365
            assert len(dti.month) == 365
            assert len(dti.day) == 365
            assert len(dti.hour) == 365
            assert len(dti.minute) == 365
            assert len(dti.second) == 365
            assert len(dti.microsecond) == 365
            assert len(dti.dayofweek) == 365
            assert len(dti.dayofyear) == 365
            assert len(dti.isocalendar()) == 365
            assert len(dti.quarter) == 365
            assert len(dti.is_month_start) == 365
            assert len(dti.is_month_end) == 365
            assert len(dti.is_quarter_start) == 365
            assert len(dti.is_quarter_end) == 365
            assert len(dti.is_year_start) == 365
            assert len(dti.is_year_end) == 365

            dti.name = "name"

            # non boolean accessors -> return Index
            for accessor in DatetimeIndex._field_ops:
                if accessor in ["week", "weekofyear"]:
                    # GH#33595 Deprecate week and weekofyear
                    continue
                res = getattr(dti, accessor)
                assert len(res) == 365
                assert isinstance(res, Index)
                assert res.name == "name"

            # boolean accessors -> return array
            for accessor in DatetimeIndex._bool_ops:
                res = getattr(dti, accessor)
                assert len(res) == 365
                assert isinstance(res, np.ndarray)

            # test boolean indexing
            res = dti[dti.is_quarter_start]
            exp = dti[[0, 90, 181, 273]]
            tm.assert_index_equal(res, exp)
            res = dti[dti.is_leap_year]
            exp = DatetimeIndex([], freq="D", tz=dti.tz, name="name")
            tm.assert_index_equal(res, exp)

    def test_datetimeindex_accessors2(self):
        dti = date_range(freq="BQ-FEB", start=datetime(1998, 1, 1), periods=4)

        assert sum(dti.is_quarter_start) == 0
        assert sum(dti.is_quarter_end) == 4
        assert sum(dti.is_year_start) == 0
        assert sum(dti.is_year_end) == 1

    def test_datetimeindex_accessors3(self):
        # Ensure is_start/end accessors throw ValueError for CustomBusinessDay,
        bday_egypt = offsets.CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")
        dti = date_range(datetime(2013, 4, 30), periods=5, freq=bday_egypt)
        msg = "Custom business days is not supported by is_month_start"
        with pytest.raises(ValueError, match=msg):
            dti.is_month_start

    def test_datetimeindex_accessors4(self):
        dti = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"])

        assert dti.is_month_start[0] == 1

    def test_datetimeindex_accessors5(self):
        with tm.assert_produces_warning(
            FutureWarning, match="The 'freq' argument", check_stacklevel=False
        ):
            tests = [
                (Timestamp("2013-06-01", freq="M").is_month_start, 1),
                (Timestamp("2013-06-01", freq="BM").is_month_start, 0),
                (Timestamp("2013-06-03", freq="M").is_month_start, 0),
                (Timestamp("2013-06-03", freq="BM").is_month_start, 1),
                (Timestamp("2013-02-28", freq="Q-FEB").is_month_end, 1),
                (Timestamp("2013-02-28", freq="Q-FEB").is_quarter_end, 1),
                (Timestamp("2013-02-28", freq="Q-FEB").is_year_end, 1),
                (Timestamp("2013-03-01", freq="Q-FEB").is_month_start, 1),
                (Timestamp("2013-03-01", freq="Q-FEB").is_quarter_start, 1),
                (Timestamp("2013-03-01", freq="Q-FEB").is_year_start, 1),
                (Timestamp("2013-03-31", freq="QS-FEB").is_month_end, 1),
                (Timestamp("2013-03-31", freq="QS-FEB").is_quarter_end, 0),
                (Timestamp("2013-03-31", freq="QS-FEB").is_year_end, 0),
                (Timestamp("2013-02-01", freq="QS-FEB").is_month_start, 1),
                (Timestamp("2013-02-01", freq="QS-FEB").is_quarter_start, 1),
                (Timestamp("2013-02-01", freq="QS-FEB").is_year_start, 1),
                (Timestamp("2013-06-30", freq="BQ").is_month_end, 0),
                (Timestamp("2013-06-30", freq="BQ").is_quarter_end, 0),
                (Timestamp("2013-06-30", freq="BQ").is_year_end, 0),
                (Timestamp("2013-06-28", freq="BQ").is_month_end, 1),
                (Timestamp("2013-06-28", freq="BQ").is_quarter_end, 1),
                (Timestamp("2013-06-28", freq="BQ").is_year_end, 0),
                (Timestamp("2013-06-30", freq="BQS-APR").is_month_end, 0),
                (Timestamp("2013-06-30", freq="BQS-APR").is_quarter_end, 0),
                (Timestamp("2013-06-30", freq="BQS-APR").is_year_end, 0),
                (Timestamp("2013-06-28", freq="BQS-APR").is_month_end, 1),
                (Timestamp("2013-06-28", freq="BQS-APR").is_quarter_end, 1),
                (Timestamp("2013-03-29", freq="BQS-APR").is_year_end, 1),
                (Timestamp("2013-11-01", freq="AS-NOV").is_year_start, 1),
                (Timestamp("2013-10-31", freq="AS-NOV").is_year_end, 1),
                (Timestamp("2012-02-01").days_in_month, 29),
                (Timestamp("2013-02-01").days_in_month, 28),
            ]

        for ts, value in tests:
            assert ts == value

    def test_datetimeindex_accessors6(self):
        # GH 6538: Check that DatetimeIndex and its TimeStamp elements
        # return the same weekofyear accessor close to new year w/ tz
        dates = ["2013/12/29", "2013/12/30", "2013/12/31"]
        dates = DatetimeIndex(dates, tz="Europe/Brussels")
        expected = [52, 1, 1]
        assert dates.isocalendar().week.tolist() == expected
        assert [d.weekofyear for d in dates] == expected

    # GH 12806
    @pytest.mark.parametrize(
        "time_locale", [None] if tm.get_locales() is None else [None] + tm.get_locales()
    )
    def test_datetime_name_accessors(self, time_locale):
        # Test Monday -> Sunday and January -> December, in that sequence
        if time_locale is None:
            # If the time_locale is None, day-name and month_name should
            # return the english attributes
            expected_days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            expected_months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_days = calendar.day_name[:]
                expected_months = calendar.month_name[1:]

        # GH#11128
        dti = date_range(freq="D", start=datetime(1998, 1, 1), periods=365)
        english_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for day, name, eng_name in zip(range(4, 11), expected_days, english_days):
            name = name.capitalize()
            assert dti.day_name(locale=time_locale)[day] == name
            assert dti.day_name(locale=None)[day] == eng_name
            ts = Timestamp(datetime(2016, 4, day))
            assert ts.day_name(locale=time_locale) == name
        dti = dti.append(DatetimeIndex([pd.NaT]))
        assert np.isnan(dti.day_name(locale=time_locale)[-1])
        ts = Timestamp(pd.NaT)
        assert np.isnan(ts.day_name(locale=time_locale))

        # GH#12805
        dti = date_range(freq="M", start="2012", end="2013")
        result = dti.month_name(locale=time_locale)
        expected = Index([month.capitalize() for month in expected_months])

        # work around different normalization schemes
        # https://github.com/pandas-dev/pandas/issues/22342
        result = result.str.normalize("NFD")
        expected = expected.str.normalize("NFD")

        tm.assert_index_equal(result, expected)

        for date, expected in zip(dti, expected_months):
            result = date.month_name(locale=time_locale)
            expected = expected.capitalize()

            result = unicodedata.normalize("NFD", result)
            expected = unicodedata.normalize("NFD", result)

            assert result == expected
        dti = dti.append(DatetimeIndex([pd.NaT]))
        assert np.isnan(dti.month_name(locale=time_locale)[-1])

    def test_nanosecond_field(self):
        dti = DatetimeIndex(np.arange(10))

        tm.assert_index_equal(dti.nanosecond, Index(np.arange(10, dtype=np.int64)))


def test_iter_readonly():
    # GH#28055 ints_to_pydatetime with readonly array
    arr = np.array([np.datetime64("2012-02-15T12:00:00.000000000")])
    arr.setflags(write=False)
    dti = pd.to_datetime(arr)
    list(dti)


def test_week_and_weekofyear_are_deprecated():
    # GH#33595 Deprecate week and weekofyear
    idx = date_range(start="2019-12-29", freq="D", periods=4)
    with tm.assert_produces_warning(FutureWarning):
        idx.week
    with tm.assert_produces_warning(FutureWarning):
        idx.weekofyear


def test_isocalendar_returns_correct_values_close_to_new_year_with_tz():
    # GH 6538: Check that DatetimeIndex and its TimeStamp elements
    # return the same weekofyear accessor close to new year w/ tz
    dates = ["2013/12/29", "2013/12/30", "2013/12/31"]
    dates = DatetimeIndex(dates, tz="Europe/Brussels")
    result = dates.isocalendar()
    expected_data_frame = pd.DataFrame(
        [[2013, 52, 7], [2014, 1, 1], [2014, 1, 2]],
        columns=["year", "week", "day"],
        index=dates,
        dtype="UInt32",
    )
    tm.assert_frame_equal(result, expected_data_frame)


def test_add_timedelta_preserves_freq():
    # GH#37295 should hold for any DTI with freq=None or Tick freq
    tz = "Canada/Eastern"
    dti = date_range(
        start=Timestamp("2019-03-26 00:00:00-0400", tz=tz),
        end=Timestamp("2020-10-17 00:00:00-0400", tz=tz),
        freq="D",
    )
    result = dti + Timedelta(days=1)
    assert result.freq == dti.freq
