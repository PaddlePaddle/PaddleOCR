import pytest

from pandas import Timedelta


@pytest.mark.parametrize(
    "td, expected_repr",
    [
        (Timedelta(10, unit="d"), "Timedelta('10 days 00:00:00')"),
        (Timedelta(10, unit="s"), "Timedelta('0 days 00:00:10')"),
        (Timedelta(10, unit="ms"), "Timedelta('0 days 00:00:00.010000')"),
        (Timedelta(-10, unit="ms"), "Timedelta('-1 days +23:59:59.990000')"),
    ],
)
def test_repr(td, expected_repr):
    assert repr(td) == expected_repr


@pytest.mark.parametrize(
    "td, expected_iso",
    [
        (
            Timedelta(
                days=6,
                minutes=50,
                seconds=3,
                milliseconds=10,
                microseconds=10,
                nanoseconds=12,
            ),
            "P6DT0H50M3.010010012S",
        ),
        (Timedelta(days=4, hours=12, minutes=30, seconds=5), "P4DT12H30M5S"),
        (Timedelta(nanoseconds=123), "P0DT0H0M0.000000123S"),
        # trim nano
        (Timedelta(microseconds=10), "P0DT0H0M0.00001S"),
        # trim micro
        (Timedelta(milliseconds=1), "P0DT0H0M0.001S"),
        # don't strip every 0
        (Timedelta(minutes=1), "P0DT0H1M0S"),
    ],
)
def test_isoformat(td, expected_iso):
    assert td.isoformat() == expected_iso
