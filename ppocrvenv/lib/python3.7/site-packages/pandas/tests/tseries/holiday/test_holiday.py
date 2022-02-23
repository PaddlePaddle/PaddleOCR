from datetime import datetime

import pytest
from pytz import utc

import pandas._testing as tm

from pandas.tseries.holiday import (
    MO,
    SA,
    AbstractHolidayCalendar,
    DateOffset,
    EasterMonday,
    GoodFriday,
    Holiday,
    HolidayCalendarFactory,
    Timestamp,
    USColumbusDay,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    get_calendar,
    next_monday,
)


def _check_holiday_results(holiday, start, end, expected):
    """
    Check that the dates for a given holiday match in date and timezone.

    Parameters
    ----------
    holiday : Holiday
        The holiday to check.
    start : datetime-like
        The start date of range in which to collect dates for a given holiday.
    end : datetime-like
        The end date of range in which to collect dates for a given holiday.
    expected : list
        The list of dates we expect to get.
    """
    assert list(holiday.dates(start, end)) == expected

    # Verify that timezone info is preserved.
    assert list(
        holiday.dates(utc.localize(Timestamp(start)), utc.localize(Timestamp(end)))
    ) == [utc.localize(dt) for dt in expected]


@pytest.mark.parametrize(
    "holiday,start_date,end_date,expected",
    [
        (
            USMemorialDay,
            datetime(2011, 1, 1),
            datetime(2020, 12, 31),
            [
                datetime(2011, 5, 30),
                datetime(2012, 5, 28),
                datetime(2013, 5, 27),
                datetime(2014, 5, 26),
                datetime(2015, 5, 25),
                datetime(2016, 5, 30),
                datetime(2017, 5, 29),
                datetime(2018, 5, 28),
                datetime(2019, 5, 27),
                datetime(2020, 5, 25),
            ],
        ),
        (
            Holiday("July 4th Eve", month=7, day=3),
            "2001-01-01",
            "2003-03-03",
            [Timestamp("2001-07-03 00:00:00"), Timestamp("2002-07-03 00:00:00")],
        ),
        (
            Holiday("July 4th Eve", month=7, day=3, days_of_week=(0, 1, 2, 3)),
            "2001-01-01",
            "2008-03-03",
            [
                Timestamp("2001-07-03 00:00:00"),
                Timestamp("2002-07-03 00:00:00"),
                Timestamp("2003-07-03 00:00:00"),
                Timestamp("2006-07-03 00:00:00"),
                Timestamp("2007-07-03 00:00:00"),
            ],
        ),
        (
            EasterMonday,
            datetime(2011, 1, 1),
            datetime(2020, 12, 31),
            [
                Timestamp("2011-04-25 00:00:00"),
                Timestamp("2012-04-09 00:00:00"),
                Timestamp("2013-04-01 00:00:00"),
                Timestamp("2014-04-21 00:00:00"),
                Timestamp("2015-04-06 00:00:00"),
                Timestamp("2016-03-28 00:00:00"),
                Timestamp("2017-04-17 00:00:00"),
                Timestamp("2018-04-02 00:00:00"),
                Timestamp("2019-04-22 00:00:00"),
                Timestamp("2020-04-13 00:00:00"),
            ],
        ),
        (
            GoodFriday,
            datetime(2011, 1, 1),
            datetime(2020, 12, 31),
            [
                Timestamp("2011-04-22 00:00:00"),
                Timestamp("2012-04-06 00:00:00"),
                Timestamp("2013-03-29 00:00:00"),
                Timestamp("2014-04-18 00:00:00"),
                Timestamp("2015-04-03 00:00:00"),
                Timestamp("2016-03-25 00:00:00"),
                Timestamp("2017-04-14 00:00:00"),
                Timestamp("2018-03-30 00:00:00"),
                Timestamp("2019-04-19 00:00:00"),
                Timestamp("2020-04-10 00:00:00"),
            ],
        ),
        (
            USThanksgivingDay,
            datetime(2011, 1, 1),
            datetime(2020, 12, 31),
            [
                datetime(2011, 11, 24),
                datetime(2012, 11, 22),
                datetime(2013, 11, 28),
                datetime(2014, 11, 27),
                datetime(2015, 11, 26),
                datetime(2016, 11, 24),
                datetime(2017, 11, 23),
                datetime(2018, 11, 22),
                datetime(2019, 11, 28),
                datetime(2020, 11, 26),
            ],
        ),
    ],
)
def test_holiday_dates(holiday, start_date, end_date, expected):
    _check_holiday_results(holiday, start_date, end_date, expected)


@pytest.mark.parametrize(
    "holiday,start,expected",
    [
        (USMemorialDay, datetime(2015, 7, 1), []),
        (USMemorialDay, "2015-05-25", "2015-05-25"),
        (USLaborDay, datetime(2015, 7, 1), []),
        (USLaborDay, "2015-09-07", "2015-09-07"),
        (USColumbusDay, datetime(2015, 7, 1), []),
        (USColumbusDay, "2015-10-12", "2015-10-12"),
        (USThanksgivingDay, datetime(2015, 7, 1), []),
        (USThanksgivingDay, "2015-11-26", "2015-11-26"),
        (USMartinLutherKingJr, datetime(2015, 7, 1), []),
        (USMartinLutherKingJr, "2015-01-19", "2015-01-19"),
        (USPresidentsDay, datetime(2015, 7, 1), []),
        (USPresidentsDay, "2015-02-16", "2015-02-16"),
        (GoodFriday, datetime(2015, 7, 1), []),
        (GoodFriday, "2015-04-03", "2015-04-03"),
        (EasterMonday, "2015-04-06", "2015-04-06"),
        (EasterMonday, datetime(2015, 7, 1), []),
        (EasterMonday, "2015-04-05", []),
        ("New Years Day", "2015-01-01", "2015-01-01"),
        ("New Years Day", "2010-12-31", "2010-12-31"),
        ("New Years Day", datetime(2015, 7, 1), []),
        ("New Years Day", "2011-01-01", []),
        ("July 4th", "2015-07-03", "2015-07-03"),
        ("July 4th", datetime(2015, 7, 1), []),
        ("July 4th", "2015-07-04", []),
        ("Veterans Day", "2012-11-12", "2012-11-12"),
        ("Veterans Day", datetime(2015, 7, 1), []),
        ("Veterans Day", "2012-11-11", []),
        ("Christmas", "2011-12-26", "2011-12-26"),
        ("Christmas", datetime(2015, 7, 1), []),
        ("Christmas", "2011-12-25", []),
    ],
)
def test_holidays_within_dates(holiday, start, expected):
    # see gh-11477
    #
    # Fix holiday behavior where holiday.dates returned dates outside
    # start/end date, or observed rules could not be applied because the
    # holiday was not in the original date range (e.g., 7/4/2015 -> 7/3/2015).
    if isinstance(holiday, str):
        calendar = get_calendar("USFederalHolidayCalendar")
        holiday = calendar.rule_from_name(holiday)

    if isinstance(expected, str):
        expected = [Timestamp(expected)]

    _check_holiday_results(holiday, start, start, expected)


@pytest.mark.parametrize(
    "transform", [lambda x: x.strftime("%Y-%m-%d"), lambda x: Timestamp(x)]
)
def test_argument_types(transform):
    start_date = datetime(2011, 1, 1)
    end_date = datetime(2020, 12, 31)

    holidays = USThanksgivingDay.dates(start_date, end_date)
    holidays2 = USThanksgivingDay.dates(transform(start_date), transform(end_date))
    tm.assert_index_equal(holidays, holidays2)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("One-Time", {"year": 2012, "month": 5, "day": 28}),
        (
            "Range",
            {
                "month": 5,
                "day": 28,
                "start_date": datetime(2012, 1, 1),
                "end_date": datetime(2012, 12, 31),
                "offset": DateOffset(weekday=MO(1)),
            },
        ),
    ],
)
def test_special_holidays(name, kwargs):
    base_date = [datetime(2012, 5, 28)]
    holiday = Holiday(name, **kwargs)

    start_date = datetime(2011, 1, 1)
    end_date = datetime(2020, 12, 31)

    assert base_date == holiday.dates(start_date, end_date)


def test_get_calendar():
    class TestCalendar(AbstractHolidayCalendar):
        rules = []

    calendar = get_calendar("TestCalendar")
    assert TestCalendar == type(calendar)


def test_factory():
    class_1 = HolidayCalendarFactory(
        "MemorialDay", AbstractHolidayCalendar, USMemorialDay
    )
    class_2 = HolidayCalendarFactory(
        "Thanksgiving", AbstractHolidayCalendar, USThanksgivingDay
    )
    class_3 = HolidayCalendarFactory("Combined", class_1, class_2)

    assert len(class_1.rules) == 1
    assert len(class_2.rules) == 1
    assert len(class_3.rules) == 2


def test_both_offset_observance_raises():
    # see gh-10217
    msg = "Cannot use both offset and observance"
    with pytest.raises(NotImplementedError, match=msg):
        Holiday(
            "Cyber Monday",
            month=11,
            day=1,
            offset=[DateOffset(weekday=SA(4))],
            observance=next_monday,
        )
