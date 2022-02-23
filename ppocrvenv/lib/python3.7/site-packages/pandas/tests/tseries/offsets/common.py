"""
Assertion helpers and base class for offsets tests
"""
from __future__ import annotations

from datetime import datetime

from dateutil.tz.tz import tzlocal
import pytest

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timestamp,
)
from pandas._libs.tslibs.offsets import (
    FY5253,
    BusinessHour,
    CustomBusinessHour,
    DateOffset,
    FY5253Quarter,
    LastWeekOfMonth,
    Week,
    WeekOfMonth,
)
from pandas.compat import IS64


def assert_offset_equal(offset, base, expected):
    actual = offset + base
    actual_swapped = base + offset
    actual_apply = offset.apply(base)
    try:
        assert actual == expected
        assert actual_swapped == expected
        assert actual_apply == expected
    except AssertionError as err:
        raise AssertionError(
            f"\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})"
            f"\nAt Date: {base}"
        ) from err


def assert_is_on_offset(offset, date, expected):
    actual = offset.is_on_offset(date)
    assert actual == expected, (
        f"\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})"
        f"\nAt Date: {date}"
    )


class WeekDay:
    MON = 0
    TUE = 1
    WED = 2
    THU = 3
    FRI = 4
    SAT = 5
    SUN = 6


class Base:
    _offset: type[DateOffset] | None = None
    d = Timestamp(datetime(2008, 1, 2))

    timezones = [
        None,
        "UTC",
        "Asia/Tokyo",
        "US/Eastern",
        "dateutil/Asia/Tokyo",
        "dateutil/US/Pacific",
    ]

    def _get_offset(self, klass, value=1, normalize=False):
        # create instance from offset class
        if klass is FY5253:
            klass = klass(
                n=value,
                startingMonth=1,
                weekday=1,
                variation="last",
                normalize=normalize,
            )
        elif klass is FY5253Quarter:
            klass = klass(
                n=value,
                startingMonth=1,
                weekday=1,
                qtr_with_extra_week=1,
                variation="last",
                normalize=normalize,
            )
        elif klass is LastWeekOfMonth:
            klass = klass(n=value, weekday=5, normalize=normalize)
        elif klass is WeekOfMonth:
            klass = klass(n=value, week=1, weekday=5, normalize=normalize)
        elif klass is Week:
            klass = klass(n=value, weekday=5, normalize=normalize)
        elif klass is DateOffset:
            klass = klass(days=value, normalize=normalize)
        else:
            klass = klass(value, normalize=normalize)
        return klass

    def test_apply_out_of_range(self, request, tz_naive_fixture):
        tz = tz_naive_fixture
        if self._offset is None:
            return

        # try to create an out-of-bounds result timestamp; if we can't create
        # the offset skip
        try:
            if self._offset in (BusinessHour, CustomBusinessHour):
                # Using 10000 in BusinessHour fails in tz check because of DST
                # difference
                offset = self._get_offset(self._offset, value=100000)
            else:
                offset = self._get_offset(self._offset, value=10000)

            result = Timestamp("20080101") + offset
            assert isinstance(result, datetime)
            assert result.tzinfo is None

            # Check tz is preserved
            t = Timestamp("20080101", tz=tz)
            result = t + offset
            assert isinstance(result, datetime)

            if isinstance(tz, tzlocal) and not IS64:
                # If we hit OutOfBoundsDatetime on non-64 bit machines
                # we'll drop out of the try clause before the next test
                request.node.add_marker(
                    pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
                )
            assert t.tzinfo == result.tzinfo

        except OutOfBoundsDatetime:
            pass
        except (ValueError, KeyError):
            # we are creating an invalid offset
            # so ignore
            pass

    def test_offsets_compare_equal(self):
        # root cause of GH#456: __ne__ was not implemented
        if self._offset is None:
            return
        offset1 = self._offset()
        offset2 = self._offset()
        assert not offset1 != offset2
        assert offset1 == offset2

    def test_rsub(self):
        if self._offset is None or not hasattr(self, "offset2"):
            # i.e. skip for TestCommon and YQM subclasses that do not have
            # offset2 attr
            return
        assert self.d - self.offset2 == (-self.offset2).apply(self.d)

    def test_radd(self):
        if self._offset is None or not hasattr(self, "offset2"):
            # i.e. skip for TestCommon and YQM subclasses that do not have
            # offset2 attr
            return
        assert self.d + self.offset2 == self.offset2 + self.d

    def test_sub(self):
        if self._offset is None or not hasattr(self, "offset2"):
            # i.e. skip for TestCommon and YQM subclasses that do not have
            # offset2 attr
            return
        off = self.offset2
        msg = "Cannot subtract datetime from offset"
        with pytest.raises(TypeError, match=msg):
            off - self.d

        assert 2 * off - off == off
        assert self.d - self.offset2 == self.d + self._offset(-2)
        assert self.d - self.offset2 == self.d - (2 * off - off)

    def testMult1(self):
        if self._offset is None or not hasattr(self, "offset1"):
            # i.e. skip for TestCommon and YQM subclasses that do not have
            # offset1 attr
            return
        assert self.d + 10 * self.offset1 == self.d + self._offset(10)
        assert self.d + 5 * self.offset1 == self.d + self._offset(5)

    def testMult2(self):
        if self._offset is None:
            return
        assert self.d + (-5 * self._offset(-10)) == self.d + self._offset(50)
        assert self.d + (-3 * self._offset(-2)) == self.d + self._offset(6)

    def test_compare_str(self):
        # GH#23524
        # comparing to strings that cannot be cast to DateOffsets should
        #  not raise for __eq__ or __ne__
        if self._offset is None:
            return
        off = self._get_offset(self._offset)

        assert not off == "infer"
        assert off != "foo"
        # Note: inequalities are only implemented for Tick subclasses;
        #  tests for this are in test_ticks
