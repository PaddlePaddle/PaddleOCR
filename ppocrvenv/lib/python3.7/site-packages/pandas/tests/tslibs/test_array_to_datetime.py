from datetime import (
    date,
    datetime,
)

from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz

from pandas._libs import (
    iNaT,
    tslib,
)
from pandas.compat import np_array_datetime64_compat

from pandas import Timestamp
import pandas._testing as tm


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            ["01-01-2013", "01-02-2013"],
            [
                "2013-01-01T00:00:00.000000000-0000",
                "2013-01-02T00:00:00.000000000-0000",
            ],
        ),
        (
            ["Mon Sep 16 2013", "Tue Sep 17 2013"],
            [
                "2013-09-16T00:00:00.000000000-0000",
                "2013-09-17T00:00:00.000000000-0000",
            ],
        ),
    ],
)
def test_parsing_valid_dates(data, expected):
    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr)

    expected = np_array_datetime64_compat(expected, dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "dt_string, expected_tz",
    [
        ["01-01-2013 08:00:00+08:00", 480],
        ["2013-01-01T08:00:00.000000000+0800", 480],
        ["2012-12-31T16:00:00.000000000-0800", -480],
        ["12-31-2012 23:00:00-01:00", -60],
    ],
)
def test_parsing_timezone_offsets(dt_string, expected_tz):
    # All of these datetime strings with offsets are equivalent
    # to the same datetime after the timezone offset is added.
    arr = np.array(["01-01-2013 00:00:00"], dtype=object)
    expected, _ = tslib.array_to_datetime(arr)

    arr = np.array([dt_string], dtype=object)
    result, result_tz = tslib.array_to_datetime(arr)

    tm.assert_numpy_array_equal(result, expected)
    assert result_tz is pytz.FixedOffset(expected_tz)


def test_parsing_non_iso_timezone_offset():
    dt_string = "01-01-2013T00:00:00.000000000+0000"
    arr = np.array([dt_string], dtype=object)

    result, result_tz = tslib.array_to_datetime(arr)
    expected = np.array([np.datetime64("2013-01-01 00:00:00.000000000")])

    tm.assert_numpy_array_equal(result, expected)
    assert result_tz is pytz.FixedOffset(0)


def test_parsing_different_timezone_offsets():
    # see gh-17697
    data = ["2015-11-18 15:30:00+05:30", "2015-11-18 15:30:00+06:30"]
    data = np.array(data, dtype=object)

    result, result_tz = tslib.array_to_datetime(data)
    expected = np.array(
        [
            datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 19800)),
            datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 23400)),
        ],
        dtype=object,
    )

    tm.assert_numpy_array_equal(result, expected)
    assert result_tz is None


@pytest.mark.parametrize(
    "data", [["-352.737091", "183.575577"], ["1", "2", "3", "4", "5"]]
)
def test_number_looking_strings_not_into_datetime(data):
    # see gh-4601
    #
    # These strings don't look like datetimes, so
    # they shouldn't be attempted to be converted.
    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr, errors="ignore")

    tm.assert_numpy_array_equal(result, arr)


@pytest.mark.parametrize(
    "invalid_date",
    [
        date(1000, 1, 1),
        datetime(1000, 1, 1),
        "1000-01-01",
        "Jan 1, 1000",
        np.datetime64("1000-01-01"),
    ],
)
@pytest.mark.parametrize("errors", ["coerce", "raise"])
def test_coerce_outside_ns_bounds(invalid_date, errors):
    arr = np.array([invalid_date], dtype="object")
    kwargs = {"values": arr, "errors": errors}

    if errors == "raise":
        msg = "Out of bounds nanosecond timestamp"

        with pytest.raises(ValueError, match=msg):
            tslib.array_to_datetime(**kwargs)
    else:  # coerce.
        result, _ = tslib.array_to_datetime(**kwargs)
        expected = np.array([iNaT], dtype="M8[ns]")

        tm.assert_numpy_array_equal(result, expected)


def test_coerce_outside_ns_bounds_one_valid():
    arr = np.array(["1/1/1000", "1/1/2000"], dtype=object)
    result, _ = tslib.array_to_datetime(arr, errors="coerce")

    expected = [iNaT, "2000-01-01T00:00:00.000000000-0000"]
    expected = np_array_datetime64_compat(expected, dtype="M8[ns]")

    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("errors", ["ignore", "coerce"])
def test_coerce_of_invalid_datetimes(errors):
    arr = np.array(["01-01-2013", "not_a_date", "1"], dtype=object)
    kwargs = {"values": arr, "errors": errors}

    if errors == "ignore":
        # Without coercing, the presence of any invalid
        # dates prevents any values from being converted.
        result, _ = tslib.array_to_datetime(**kwargs)
        tm.assert_numpy_array_equal(result, arr)
    else:  # coerce.
        # With coercing, the invalid dates becomes iNaT
        result, _ = tslib.array_to_datetime(arr, errors="coerce")
        expected = ["2013-01-01T00:00:00.000000000-0000", iNaT, iNaT]

        tm.assert_numpy_array_equal(
            result, np_array_datetime64_compat(expected, dtype="M8[ns]")
        )


def test_to_datetime_barely_out_of_bounds():
    # see gh-19382, gh-19529
    #
    # Close enough to bounds that dropping nanos
    # would result in an in-bounds datetime.
    arr = np.array(["2262-04-11 23:47:16.854775808"], dtype=object)
    msg = "Out of bounds nanosecond timestamp: 2262-04-11 23:47:16"

    with pytest.raises(tslib.OutOfBoundsDatetime, match=msg):
        tslib.array_to_datetime(arr)


class SubDatetime(datetime):
    pass


@pytest.mark.parametrize(
    "data,expected",
    [
        ([SubDatetime(2000, 1, 1)], ["2000-01-01T00:00:00.000000000-0000"]),
        ([datetime(2000, 1, 1)], ["2000-01-01T00:00:00.000000000-0000"]),
        ([Timestamp(2000, 1, 1)], ["2000-01-01T00:00:00.000000000-0000"]),
    ],
)
def test_datetime_subclass(data, expected):
    # GH 25851
    # ensure that subclassed datetime works with
    # array_to_datetime

    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr)

    expected = np_array_datetime64_compat(expected, dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)
