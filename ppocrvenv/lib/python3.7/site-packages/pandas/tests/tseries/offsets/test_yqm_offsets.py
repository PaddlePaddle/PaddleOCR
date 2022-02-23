"""
Tests for Year, Quarter, and Month-based DateOffset subclasses
"""
from datetime import datetime

import pytest

import pandas as pd
from pandas import Timestamp
from pandas.tests.tseries.offsets.common import (
    Base,
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    YearBegin,
    YearEnd,
)

# --------------------------------------------------------------------
# Misc


def test_quarterly_dont_normalize():
    date = datetime(2012, 3, 31, 5, 30)

    offsets = (QuarterBegin, QuarterEnd, BQuarterEnd, BQuarterBegin)

    for klass in offsets:
        result = date + klass()
        assert result.time() == date.time()


@pytest.mark.parametrize("n", [-2, 1])
@pytest.mark.parametrize(
    "cls",
    [
        MonthBegin,
        MonthEnd,
        BMonthBegin,
        BMonthEnd,
        QuarterBegin,
        QuarterEnd,
        BQuarterBegin,
        BQuarterEnd,
        YearBegin,
        YearEnd,
        BYearBegin,
        BYearEnd,
    ],
)
def test_apply_index(cls, n):
    offset = cls(n=n)
    rng = pd.date_range(start="1/1/2000", periods=100000, freq="T")
    ser = pd.Series(rng)

    res = rng + offset
    assert res.freq is None  # not retained
    assert res[0] == rng[0] + offset
    assert res[-1] == rng[-1] + offset
    res2 = ser + offset
    # apply_index is only for indexes, not series, so no res2_v2
    assert res2.iloc[0] == ser.iloc[0] + offset
    assert res2.iloc[-1] == ser.iloc[-1] + offset


@pytest.mark.parametrize(
    "offset", [QuarterBegin(), QuarterEnd(), BQuarterBegin(), BQuarterEnd()]
)
def test_on_offset(offset):
    dates = [
        datetime(2016, m, d)
        for m in [10, 11, 12]
        for d in [1, 2, 3, 28, 29, 30, 31]
        if not (m == 11 and d == 31)
    ]
    for date in dates:
        res = offset.is_on_offset(date)
        slow_version = date == (date + offset) - offset
        assert res == slow_version


# --------------------------------------------------------------------
# Months


class TestMonthBegin(Base):
    _offset = MonthBegin

    offset_cases = []
    # NOTE: I'm not entirely happy with the logic here for Begin -ss
    # see thread 'offset conventions' on the ML
    offset_cases.append(
        (
            MonthBegin(),
            {
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 2, 1): datetime(2008, 3, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
                datetime(2007, 1, 31): datetime(2007, 2, 1),
            },
        )
    )

    offset_cases.append(
        (
            MonthBegin(0),
            {
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2006, 12, 3): datetime(2007, 1, 1),
                datetime(2007, 1, 31): datetime(2007, 2, 1),
            },
        )
    )

    offset_cases.append(
        (
            MonthBegin(2),
            {
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 1, 31): datetime(2008, 3, 1),
                datetime(2006, 12, 31): datetime(2007, 2, 1),
                datetime(2007, 12, 28): datetime(2008, 2, 1),
                datetime(2007, 1, 1): datetime(2007, 3, 1),
                datetime(2006, 11, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            MonthBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),
                datetime(2008, 5, 31): datetime(2008, 5, 1),
                datetime(2008, 12, 31): datetime(2008, 12, 1),
                datetime(2006, 12, 29): datetime(2006, 12, 1),
                datetime(2006, 1, 2): datetime(2006, 1, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestMonthEnd(Base):
    _offset = MonthEnd

    def test_day_of_month(self):
        dt = datetime(2007, 1, 1)
        offset = MonthEnd()

        result = dt + offset
        assert result == Timestamp(2007, 1, 31)

        result = result + offset
        assert result == Timestamp(2007, 2, 28)

    def test_normalize(self):
        dt = datetime(2007, 1, 1, 3)

        result = dt + MonthEnd(normalize=True)
        expected = dt.replace(hour=0) + MonthEnd()
        assert result == expected

    offset_cases = []
    offset_cases.append(
        (
            MonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2006, 12, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            MonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            MonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 3, 31),
                datetime(2006, 12, 29): datetime(2007, 1, 31),
                datetime(2006, 12, 31): datetime(2007, 2, 28),
                datetime(2007, 1, 1): datetime(2007, 2, 28),
                datetime(2006, 11, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            MonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 5, 31),
                datetime(2008, 12, 31): datetime(2008, 11, 30),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 30): datetime(2006, 11, 30),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (MonthEnd(), datetime(2007, 12, 31), True),
        (MonthEnd(), datetime(2008, 1, 1), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBMonthBegin(Base):
    _offset = BMonthBegin

    def test_offsets_compare_equal(self):
        # root cause of #456
        offset1 = BMonthBegin()
        offset2 = BMonthBegin()
        assert not offset1 != offset2

    offset_cases = []
    offset_cases.append(
        (
            BMonthBegin(),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 9, 1): datetime(2006, 10, 2),
                datetime(2007, 1, 1): datetime(2007, 2, 1),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2006, 10, 2): datetime(2006, 10, 2),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 9, 15): datetime(2006, 10, 2),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(2),
            {
                datetime(2008, 1, 1): datetime(2008, 3, 3),
                datetime(2008, 1, 15): datetime(2008, 3, 3),
                datetime(2006, 12, 29): datetime(2007, 2, 1),
                datetime(2006, 12, 31): datetime(2007, 2, 1),
                datetime(2007, 1, 1): datetime(2007, 3, 1),
                datetime(2006, 11, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),
                datetime(2008, 6, 30): datetime(2008, 6, 2),
                datetime(2008, 6, 1): datetime(2008, 5, 1),
                datetime(2008, 3, 10): datetime(2008, 3, 3),
                datetime(2008, 12, 31): datetime(2008, 12, 1),
                datetime(2006, 12, 29): datetime(2006, 12, 1),
                datetime(2006, 12, 30): datetime(2006, 12, 1),
                datetime(2007, 1, 1): datetime(2006, 12, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BMonthBegin(), datetime(2007, 12, 31), False),
        (BMonthBegin(), datetime(2008, 1, 1), True),
        (BMonthBegin(), datetime(2001, 4, 2), True),
        (BMonthBegin(), datetime(2008, 3, 3), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBMonthEnd(Base):
    _offset = BMonthEnd

    def test_normalize(self):
        dt = datetime(2007, 1, 1, 3)

        result = dt + BMonthEnd(normalize=True)
        expected = dt.replace(hour=0) + BMonthEnd()
        assert result == expected

    def test_offsets_compare_equal(self):
        # root cause of #456
        offset1 = BMonthEnd()
        offset2 = BMonthEnd()
        assert not offset1 != offset2

    offset_cases = []
    offset_cases.append(
        (
            BMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2007, 1, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2006, 12, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 29),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 3, 31),
                datetime(2006, 12, 29): datetime(2007, 2, 28),
                datetime(2006, 12, 31): datetime(2007, 2, 28),
                datetime(2007, 1, 1): datetime(2007, 2, 28),
                datetime(2006, 11, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 29),
                datetime(2008, 6, 30): datetime(2008, 5, 30),
                datetime(2008, 12, 31): datetime(2008, 11, 28),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 30): datetime(2006, 12, 29),
                datetime(2007, 1, 1): datetime(2006, 12, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BMonthEnd(), datetime(2007, 12, 31), True),
        (BMonthEnd(), datetime(2008, 1, 1), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


# --------------------------------------------------------------------
# Quarters


class TestQuarterBegin(Base):
    def test_repr(self):
        expected = "<QuarterBegin: startingMonth=3>"
        assert repr(QuarterBegin()) == expected
        expected = "<QuarterBegin: startingMonth=3>"
        assert repr(QuarterBegin(startingMonth=3)) == expected
        expected = "<QuarterBegin: startingMonth=1>"
        assert repr(QuarterBegin(startingMonth=1)) == expected

    def test_is_anchored(self):
        assert QuarterBegin(startingMonth=1).is_anchored()
        assert QuarterBegin().is_anchored()
        assert not QuarterBegin(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        # corner
        offset = QuarterBegin(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 1)

    offset_cases = []
    offset_cases.append(
        (
            QuarterBegin(startingMonth=1),
            {
                datetime(2007, 12, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 1): datetime(2008, 4, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2008, 3, 31): datetime(2008, 4, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2008, 4, 1): datetime(2008, 7, 1),
            },
        )
    )

    offset_cases.append(
        (
            QuarterBegin(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 15): datetime(2008, 2, 1),
                datetime(2008, 2, 29): datetime(2008, 5, 1),
                datetime(2008, 3, 15): datetime(2008, 5, 1),
                datetime(2008, 3, 31): datetime(2008, 5, 1),
                datetime(2008, 4, 15): datetime(2008, 5, 1),
                datetime(2008, 4, 30): datetime(2008, 5, 1),
            },
        )
    )

    offset_cases.append(
        (
            QuarterBegin(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 12, 1): datetime(2009, 1, 1),
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2008, 3, 31): datetime(2008, 4, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2008, 4, 30): datetime(2008, 7, 1),
            },
        )
    )

    offset_cases.append(
        (
            QuarterBegin(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 10, 1),
                datetime(2008, 1, 31): datetime(2008, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 1, 1),
                datetime(2008, 2, 29): datetime(2008, 1, 1),
                datetime(2008, 3, 15): datetime(2008, 1, 1),
                datetime(2008, 3, 31): datetime(2008, 1, 1),
                datetime(2008, 4, 15): datetime(2008, 4, 1),
                datetime(2008, 4, 30): datetime(2008, 4, 1),
                datetime(2008, 7, 1): datetime(2008, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            QuarterBegin(startingMonth=1, n=2),
            {
                datetime(2008, 1, 1): datetime(2008, 7, 1),
                datetime(2008, 2, 15): datetime(2008, 7, 1),
                datetime(2008, 2, 29): datetime(2008, 7, 1),
                datetime(2008, 3, 15): datetime(2008, 7, 1),
                datetime(2008, 3, 31): datetime(2008, 7, 1),
                datetime(2008, 4, 15): datetime(2008, 10, 1),
                datetime(2008, 4, 1): datetime(2008, 10, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestQuarterEnd(Base):
    _offset = QuarterEnd

    def test_repr(self):
        expected = "<QuarterEnd: startingMonth=3>"
        assert repr(QuarterEnd()) == expected
        expected = "<QuarterEnd: startingMonth=3>"
        assert repr(QuarterEnd(startingMonth=3)) == expected
        expected = "<QuarterEnd: startingMonth=1>"
        assert repr(QuarterEnd(startingMonth=1)) == expected

    def test_is_anchored(self):
        assert QuarterEnd(startingMonth=1).is_anchored()
        assert QuarterEnd().is_anchored()
        assert not QuarterEnd(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        # corner
        offset = QuarterEnd(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 31)

    offset_cases = []
    offset_cases.append(
        (
            QuarterEnd(startingMonth=1),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 4, 30),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 7, 31),
            },
        )
    )

    offset_cases.append(
        (
            QuarterEnd(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2008, 2, 15): datetime(2008, 2, 29),
                datetime(2008, 2, 29): datetime(2008, 5, 31),
                datetime(2008, 3, 15): datetime(2008, 5, 31),
                datetime(2008, 3, 31): datetime(2008, 5, 31),
                datetime(2008, 4, 15): datetime(2008, 5, 31),
                datetime(2008, 4, 30): datetime(2008, 5, 31),
            },
        )
    )

    offset_cases.append(
        (
            QuarterEnd(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 4, 30),
            },
        )
    )

    offset_cases.append(
        (
            QuarterEnd(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 10, 31),
                datetime(2008, 1, 31): datetime(2007, 10, 31),
                datetime(2008, 2, 15): datetime(2008, 1, 31),
                datetime(2008, 2, 29): datetime(2008, 1, 31),
                datetime(2008, 3, 15): datetime(2008, 1, 31),
                datetime(2008, 3, 31): datetime(2008, 1, 31),
                datetime(2008, 4, 15): datetime(2008, 1, 31),
                datetime(2008, 4, 30): datetime(2008, 1, 31),
                datetime(2008, 7, 1): datetime(2008, 4, 30),
            },
        )
    )

    offset_cases.append(
        (
            QuarterEnd(startingMonth=1, n=2),
            {
                datetime(2008, 1, 31): datetime(2008, 7, 31),
                datetime(2008, 2, 15): datetime(2008, 7, 31),
                datetime(2008, 2, 29): datetime(2008, 7, 31),
                datetime(2008, 3, 15): datetime(2008, 7, 31),
                datetime(2008, 3, 31): datetime(2008, 7, 31),
                datetime(2008, 4, 15): datetime(2008, 7, 31),
                datetime(2008, 4, 30): datetime(2008, 10, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (QuarterEnd(1, startingMonth=1), datetime(2008, 1, 31), True),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 12, 31), False),
        (QuarterEnd(1, startingMonth=1), datetime(2008, 2, 29), False),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 3, 30), False),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 3, 31), False),
        (QuarterEnd(1, startingMonth=1), datetime(2008, 4, 30), True),
        (QuarterEnd(1, startingMonth=1), datetime(2008, 5, 30), False),
        (QuarterEnd(1, startingMonth=1), datetime(2008, 5, 31), False),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 6, 29), False),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 6, 30), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 1, 31), False),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 12, 31), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 2, 29), True),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 3, 30), False),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 3, 31), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 4, 30), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 5, 30), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 5, 31), True),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 6, 29), False),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 6, 30), False),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 1, 31), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 12, 31), True),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 2, 29), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 3, 30), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 3, 31), True),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 4, 30), False),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 5, 30), False),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 5, 31), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 6, 29), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 6, 30), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBQuarterBegin(Base):
    _offset = BQuarterBegin

    def test_repr(self):
        expected = "<BusinessQuarterBegin: startingMonth=3>"
        assert repr(BQuarterBegin()) == expected
        expected = "<BusinessQuarterBegin: startingMonth=3>"
        assert repr(BQuarterBegin(startingMonth=3)) == expected
        expected = "<BusinessQuarterBegin: startingMonth=1>"
        assert repr(BQuarterBegin(startingMonth=1)) == expected

    def test_is_anchored(self):
        assert BQuarterBegin(startingMonth=1).is_anchored()
        assert BQuarterBegin().is_anchored()
        assert not BQuarterBegin(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        # corner
        offset = BQuarterBegin(n=-1, startingMonth=1)
        assert datetime(2007, 4, 3) + offset == datetime(2007, 4, 2)

    offset_cases = []
    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1),
            {
                datetime(2008, 1, 1): datetime(2008, 4, 1),
                datetime(2008, 1, 31): datetime(2008, 4, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2008, 3, 31): datetime(2008, 4, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2007, 3, 15): datetime(2007, 4, 2),
                datetime(2007, 2, 28): datetime(2007, 4, 2),
                datetime(2007, 1, 1): datetime(2007, 4, 2),
                datetime(2007, 4, 15): datetime(2007, 7, 2),
                datetime(2007, 7, 1): datetime(2007, 7, 2),
                datetime(2007, 4, 1): datetime(2007, 4, 2),
                datetime(2007, 4, 2): datetime(2007, 7, 2),
                datetime(2008, 4, 30): datetime(2008, 7, 1),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterBegin(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 15): datetime(2008, 2, 1),
                datetime(2008, 2, 29): datetime(2008, 5, 1),
                datetime(2008, 3, 15): datetime(2008, 5, 1),
                datetime(2008, 3, 31): datetime(2008, 5, 1),
                datetime(2008, 4, 15): datetime(2008, 5, 1),
                datetime(2008, 8, 15): datetime(2008, 11, 3),
                datetime(2008, 9, 15): datetime(2008, 11, 3),
                datetime(2008, 11, 1): datetime(2008, 11, 3),
                datetime(2008, 4, 30): datetime(2008, 5, 1),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2007, 12, 31): datetime(2008, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 1, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 27): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2007, 4, 1): datetime(2007, 4, 2),
                datetime(2007, 4, 2): datetime(2007, 4, 2),
                datetime(2007, 7, 1): datetime(2007, 7, 2),
                datetime(2007, 4, 15): datetime(2007, 7, 2),
                datetime(2007, 7, 2): datetime(2007, 7, 2),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 10, 1),
                datetime(2008, 1, 31): datetime(2008, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 1, 1),
                datetime(2008, 2, 29): datetime(2008, 1, 1),
                datetime(2008, 3, 15): datetime(2008, 1, 1),
                datetime(2008, 3, 31): datetime(2008, 1, 1),
                datetime(2008, 4, 15): datetime(2008, 4, 1),
                datetime(2007, 7, 3): datetime(2007, 7, 2),
                datetime(2007, 4, 3): datetime(2007, 4, 2),
                datetime(2007, 7, 2): datetime(2007, 4, 2),
                datetime(2008, 4, 1): datetime(2008, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterBegin(startingMonth=1, n=2),
            {
                datetime(2008, 1, 1): datetime(2008, 7, 1),
                datetime(2008, 1, 15): datetime(2008, 7, 1),
                datetime(2008, 2, 29): datetime(2008, 7, 1),
                datetime(2008, 3, 15): datetime(2008, 7, 1),
                datetime(2007, 3, 31): datetime(2007, 7, 2),
                datetime(2007, 4, 15): datetime(2007, 10, 1),
                datetime(2008, 4, 30): datetime(2008, 10, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestBQuarterEnd(Base):
    _offset = BQuarterEnd

    def test_repr(self):
        expected = "<BusinessQuarterEnd: startingMonth=3>"
        assert repr(BQuarterEnd()) == expected
        expected = "<BusinessQuarterEnd: startingMonth=3>"
        assert repr(BQuarterEnd(startingMonth=3)) == expected
        expected = "<BusinessQuarterEnd: startingMonth=1>"
        assert repr(BQuarterEnd(startingMonth=1)) == expected

    def test_is_anchored(self):
        assert BQuarterEnd(startingMonth=1).is_anchored()
        assert BQuarterEnd().is_anchored()
        assert not BQuarterEnd(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        # corner
        offset = BQuarterEnd(n=-1, startingMonth=1)
        assert datetime(2010, 1, 31) + offset == datetime(2010, 1, 29)

    offset_cases = []
    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 4, 30),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 7, 31),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterEnd(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2008, 2, 15): datetime(2008, 2, 29),
                datetime(2008, 2, 29): datetime(2008, 5, 30),
                datetime(2008, 3, 15): datetime(2008, 5, 30),
                datetime(2008, 3, 31): datetime(2008, 5, 30),
                datetime(2008, 4, 15): datetime(2008, 5, 30),
                datetime(2008, 4, 30): datetime(2008, 5, 30),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 4, 30),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 10, 31),
                datetime(2008, 1, 31): datetime(2007, 10, 31),
                datetime(2008, 2, 15): datetime(2008, 1, 31),
                datetime(2008, 2, 29): datetime(2008, 1, 31),
                datetime(2008, 3, 15): datetime(2008, 1, 31),
                datetime(2008, 3, 31): datetime(2008, 1, 31),
                datetime(2008, 4, 15): datetime(2008, 1, 31),
                datetime(2008, 4, 30): datetime(2008, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            BQuarterEnd(startingMonth=1, n=2),
            {
                datetime(2008, 1, 31): datetime(2008, 7, 31),
                datetime(2008, 2, 15): datetime(2008, 7, 31),
                datetime(2008, 2, 29): datetime(2008, 7, 31),
                datetime(2008, 3, 15): datetime(2008, 7, 31),
                datetime(2008, 3, 31): datetime(2008, 7, 31),
                datetime(2008, 4, 15): datetime(2008, 7, 31),
                datetime(2008, 4, 30): datetime(2008, 10, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 1, 31), True),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 12, 31), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 2, 29), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 3, 30), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 3, 31), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 4, 30), True),
        (BQuarterEnd(1, startingMonth=1), datetime(2008, 5, 30), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 6, 29), False),
        (BQuarterEnd(1, startingMonth=1), datetime(2007, 6, 30), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 1, 31), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 12, 31), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 2, 29), True),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 3, 30), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 3, 31), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 4, 30), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2008, 5, 30), True),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 6, 29), False),
        (BQuarterEnd(1, startingMonth=2), datetime(2007, 6, 30), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 1, 31), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 12, 31), True),
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 2, 29), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 3, 30), True),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 3, 31), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 4, 30), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2008, 5, 30), False),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 6, 29), True),
        (BQuarterEnd(1, startingMonth=3), datetime(2007, 6, 30), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


# --------------------------------------------------------------------
# Years


class TestYearBegin(Base):
    _offset = YearBegin

    def test_misspecified(self):
        with pytest.raises(ValueError, match="Month must go from 1 to 12"):
            YearBegin(month=13)

    offset_cases = []
    offset_cases.append(
        (
            YearBegin(),
            {
                datetime(2008, 1, 1): datetime(2009, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 1),
                datetime(2005, 12, 31): datetime(2006, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 1),
                datetime(2005, 12, 31): datetime(2006, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(3),
            {
                datetime(2008, 1, 1): datetime(2011, 1, 1),
                datetime(2008, 6, 30): datetime(2011, 1, 1),
                datetime(2008, 12, 31): datetime(2011, 1, 1),
                datetime(2005, 12, 30): datetime(2008, 1, 1),
                datetime(2005, 12, 31): datetime(2008, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 1, 1),
                datetime(2007, 1, 15): datetime(2007, 1, 1),
                datetime(2008, 6, 30): datetime(2008, 1, 1),
                datetime(2008, 12, 31): datetime(2008, 1, 1),
                datetime(2006, 12, 29): datetime(2006, 1, 1),
                datetime(2006, 12, 30): datetime(2006, 1, 1),
                datetime(2007, 1, 1): datetime(2006, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 1, 1),
                datetime(2008, 6, 30): datetime(2007, 1, 1),
                datetime(2008, 12, 31): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(month=4),
            {
                datetime(2007, 4, 1): datetime(2008, 4, 1),
                datetime(2007, 4, 15): datetime(2008, 4, 1),
                datetime(2007, 3, 1): datetime(2007, 4, 1),
                datetime(2007, 12, 15): datetime(2008, 4, 1),
                datetime(2012, 1, 31): datetime(2012, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(0, month=4),
            {
                datetime(2007, 4, 1): datetime(2007, 4, 1),
                datetime(2007, 3, 1): datetime(2007, 4, 1),
                datetime(2007, 12, 15): datetime(2008, 4, 1),
                datetime(2012, 1, 31): datetime(2012, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(4, month=4),
            {
                datetime(2007, 4, 1): datetime(2011, 4, 1),
                datetime(2007, 4, 15): datetime(2011, 4, 1),
                datetime(2007, 3, 1): datetime(2010, 4, 1),
                datetime(2007, 12, 15): datetime(2011, 4, 1),
                datetime(2012, 1, 31): datetime(2015, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-1, month=4),
            {
                datetime(2007, 4, 1): datetime(2006, 4, 1),
                datetime(2007, 3, 1): datetime(2006, 4, 1),
                datetime(2007, 12, 15): datetime(2007, 4, 1),
                datetime(2012, 1, 31): datetime(2011, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            YearBegin(-3, month=4),
            {
                datetime(2007, 4, 1): datetime(2004, 4, 1),
                datetime(2007, 3, 1): datetime(2004, 4, 1),
                datetime(2007, 12, 15): datetime(2005, 4, 1),
                datetime(2012, 1, 31): datetime(2009, 4, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (YearBegin(), datetime(2007, 1, 3), False),
        (YearBegin(), datetime(2008, 1, 1), True),
        (YearBegin(), datetime(2006, 12, 31), False),
        (YearBegin(), datetime(2006, 1, 2), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestYearEnd(Base):
    _offset = YearEnd

    def test_misspecified(self):
        with pytest.raises(ValueError, match="Month must go from 1 to 12"):
            YearEnd(month=13)

    offset_cases = []
    offset_cases.append(
        (
            YearEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2009, 12, 31),
                datetime(2005, 12, 30): datetime(2005, 12, 31),
                datetime(2005, 12, 31): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2008, 12, 31),
                datetime(2005, 12, 30): datetime(2005, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2007, 12, 31),
                datetime(2008, 12, 31): datetime(2007, 12, 31),
                datetime(2006, 12, 29): datetime(2005, 12, 31),
                datetime(2006, 12, 30): datetime(2005, 12, 31),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 12, 31),
                datetime(2008, 6, 30): datetime(2006, 12, 31),
                datetime(2008, 12, 31): datetime(2006, 12, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (YearEnd(), datetime(2007, 12, 31), True),
        (YearEnd(), datetime(2008, 1, 1), False),
        (YearEnd(), datetime(2006, 12, 31), True),
        (YearEnd(), datetime(2006, 12, 29), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestYearEndDiffMonth(Base):
    offset_cases = []
    offset_cases.append(
        (
            YearEnd(month=3),
            {
                datetime(2008, 1, 1): datetime(2008, 3, 31),
                datetime(2008, 2, 15): datetime(2008, 3, 31),
                datetime(2008, 3, 31): datetime(2009, 3, 31),
                datetime(2008, 3, 30): datetime(2008, 3, 31),
                datetime(2005, 3, 31): datetime(2006, 3, 31),
                datetime(2006, 7, 30): datetime(2007, 3, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(0, month=3),
            {
                datetime(2008, 1, 1): datetime(2008, 3, 31),
                datetime(2008, 2, 28): datetime(2008, 3, 31),
                datetime(2008, 3, 31): datetime(2008, 3, 31),
                datetime(2005, 3, 30): datetime(2005, 3, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-1, month=3),
            {
                datetime(2007, 1, 1): datetime(2006, 3, 31),
                datetime(2008, 2, 28): datetime(2007, 3, 31),
                datetime(2008, 3, 31): datetime(2007, 3, 31),
                datetime(2006, 3, 29): datetime(2005, 3, 31),
                datetime(2006, 3, 30): datetime(2005, 3, 31),
                datetime(2007, 3, 1): datetime(2006, 3, 31),
            },
        )
    )

    offset_cases.append(
        (
            YearEnd(-2, month=3),
            {
                datetime(2007, 1, 1): datetime(2005, 3, 31),
                datetime(2008, 6, 30): datetime(2007, 3, 31),
                datetime(2008, 3, 31): datetime(2006, 3, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (YearEnd(month=3), datetime(2007, 3, 31), True),
        (YearEnd(month=3), datetime(2008, 1, 1), False),
        (YearEnd(month=3), datetime(2006, 3, 31), True),
        (YearEnd(month=3), datetime(2006, 3, 29), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBYearBegin(Base):
    _offset = BYearBegin

    def test_misspecified(self):
        msg = "Month must go from 1 to 12"
        with pytest.raises(ValueError, match=msg):
            BYearBegin(month=13)
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)

    offset_cases = []
    offset_cases.append(
        (
            BYearBegin(),
            {
                datetime(2008, 1, 1): datetime(2009, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2011, 1, 1): datetime(2011, 1, 3),
                datetime(2011, 1, 3): datetime(2012, 1, 2),
                datetime(2005, 12, 30): datetime(2006, 1, 2),
                datetime(2005, 12, 31): datetime(2006, 1, 2),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 2),
                datetime(2005, 12, 31): datetime(2006, 1, 2),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 1, 2),
                datetime(2009, 1, 4): datetime(2009, 1, 1),
                datetime(2009, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2008, 1, 1),
                datetime(2008, 12, 31): datetime(2008, 1, 1),
                datetime(2006, 12, 29): datetime(2006, 1, 2),
                datetime(2006, 12, 30): datetime(2006, 1, 2),
                datetime(2006, 1, 1): datetime(2005, 1, 3),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 1, 3),
                datetime(2007, 6, 30): datetime(2006, 1, 2),
                datetime(2008, 12, 31): datetime(2007, 1, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestBYearEnd(Base):
    _offset = BYearEnd

    offset_cases = []
    offset_cases.append(
        (
            BYearEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2009, 12, 31),
                datetime(2005, 12, 30): datetime(2006, 12, 29),
                datetime(2005, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2008, 12, 31),
                datetime(2005, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 29),
                datetime(2008, 6, 30): datetime(2007, 12, 31),
                datetime(2008, 12, 31): datetime(2007, 12, 31),
                datetime(2006, 12, 29): datetime(2005, 12, 30),
                datetime(2006, 12, 30): datetime(2006, 12, 29),
                datetime(2007, 1, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 12, 30),
                datetime(2008, 6, 30): datetime(2006, 12, 29),
                datetime(2008, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BYearEnd(), datetime(2007, 12, 31), True),
        (BYearEnd(), datetime(2008, 1, 1), False),
        (BYearEnd(), datetime(2006, 12, 31), False),
        (BYearEnd(), datetime(2006, 12, 29), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBYearEndLagged(Base):
    _offset = BYearEnd

    def test_bad_month_fail(self):
        msg = "Month must go from 1 to 12"
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=0)

    offset_cases = []
    offset_cases.append(
        (
            BYearEnd(month=6),
            {
                datetime(2008, 1, 1): datetime(2008, 6, 30),
                datetime(2007, 6, 30): datetime(2008, 6, 30),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(n=-1, month=6),
            {
                datetime(2008, 1, 1): datetime(2007, 6, 29),
                datetime(2007, 6, 30): datetime(2007, 6, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_roll(self):
        offset = BYearEnd(month=6)
        date = datetime(2009, 11, 30)

        assert offset.rollforward(date) == datetime(2010, 6, 30)
        assert offset.rollback(date) == datetime(2009, 6, 30)

    on_offset_cases = [
        (BYearEnd(month=2), datetime(2007, 2, 28), True),
        (BYearEnd(month=6), datetime(2007, 6, 30), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
