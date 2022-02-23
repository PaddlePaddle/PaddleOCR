from datetime import timedelta

import numpy as np
import pytest

from pandas import (
    Interval,
    Timedelta,
    Timestamp,
)


@pytest.mark.parametrize("method", ["__add__", "__sub__"])
@pytest.mark.parametrize(
    "interval",
    [
        Interval(Timestamp("2017-01-01 00:00:00"), Timestamp("2018-01-01 00:00:00")),
        Interval(Timedelta(days=7), Timedelta(days=14)),
    ],
)
@pytest.mark.parametrize(
    "delta", [Timedelta(days=7), timedelta(7), np.timedelta64(7, "D")]
)
def test_time_interval_add_subtract_timedelta(interval, delta, method):
    # https://github.com/pandas-dev/pandas/issues/32023
    result = getattr(interval, method)(delta)
    left = getattr(interval.left, method)(delta)
    right = getattr(interval.right, method)(delta)
    expected = Interval(left, right)

    assert result == expected


@pytest.mark.parametrize("interval", [Interval(1, 2), Interval(1.0, 2.0)])
@pytest.mark.parametrize(
    "delta", [Timedelta(days=7), timedelta(7), np.timedelta64(7, "D")]
)
def test_numeric_interval_add_timedelta_raises(interval, delta):
    # https://github.com/pandas-dev/pandas/issues/32023
    msg = "|".join(
        [
            "unsupported operand",
            "cannot use operands",
            "Only numeric, Timestamp and Timedelta endpoints are allowed",
        ]
    )
    with pytest.raises((TypeError, ValueError), match=msg):
        interval + delta

    with pytest.raises((TypeError, ValueError), match=msg):
        delta + interval


@pytest.mark.parametrize("klass", [timedelta, np.timedelta64, Timedelta])
def test_timdelta_add_timestamp_interval(klass):
    delta = klass(0)
    expected = Interval(Timestamp("2020-01-01"), Timestamp("2020-02-01"))

    result = delta + expected
    assert result == expected

    result = expected + delta
    assert result == expected
