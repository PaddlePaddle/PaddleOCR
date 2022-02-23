import pytest

from pandas._libs.tslibs import to_offset
from pandas._libs.tslibs.period import (
    period_asfreq,
    period_ordinal,
)


def get_freq_code(freqstr: str) -> int:
    off = to_offset(freqstr)
    code = off._period_dtype_code
    return code


@pytest.mark.parametrize(
    "freq1,freq2,expected",
    [
        ("D", "H", 24),
        ("D", "T", 1440),
        ("D", "S", 86400),
        ("D", "L", 86400000),
        ("D", "U", 86400000000),
        ("D", "N", 86400000000000),
        ("H", "T", 60),
        ("H", "S", 3600),
        ("H", "L", 3600000),
        ("H", "U", 3600000000),
        ("H", "N", 3600000000000),
        ("T", "S", 60),
        ("T", "L", 60000),
        ("T", "U", 60000000),
        ("T", "N", 60000000000),
        ("S", "L", 1000),
        ("S", "U", 1000000),
        ("S", "N", 1000000000),
        ("L", "U", 1000),
        ("L", "N", 1000000),
        ("U", "N", 1000),
    ],
)
def test_intra_day_conversion_factors(freq1, freq2, expected):
    assert (
        period_asfreq(1, get_freq_code(freq1), get_freq_code(freq2), False) == expected
    )


@pytest.mark.parametrize(
    "freq,expected", [("A", 0), ("M", 0), ("W", 1), ("D", 0), ("B", 0)]
)
def test_period_ordinal_start_values(freq, expected):
    # information for Jan. 1, 1970.
    assert period_ordinal(1970, 1, 1, 0, 0, 0, 0, 0, get_freq_code(freq)) == expected


@pytest.mark.parametrize(
    "dt,expected",
    [
        ((1970, 1, 4, 0, 0, 0, 0, 0), 1),
        ((1970, 1, 5, 0, 0, 0, 0, 0), 2),
        ((2013, 10, 6, 0, 0, 0, 0, 0), 2284),
        ((2013, 10, 7, 0, 0, 0, 0, 0), 2285),
    ],
)
def test_period_ordinal_week(dt, expected):
    args = dt + (get_freq_code("W"),)
    assert period_ordinal(*args) == expected


@pytest.mark.parametrize(
    "day,expected",
    [
        # Thursday (Oct. 3, 2013).
        (3, 11415),
        # Friday (Oct. 4, 2013).
        (4, 11416),
        # Saturday (Oct. 5, 2013).
        (5, 11417),
        # Sunday (Oct. 6, 2013).
        (6, 11417),
        # Monday (Oct. 7, 2013).
        (7, 11417),
        # Tuesday (Oct. 8, 2013).
        (8, 11418),
    ],
)
def test_period_ordinal_business_day(day, expected):
    # 5000 is PeriodDtypeCode for BusinessDay
    args = (2013, 10, day, 0, 0, 0, 0, 0, 5000)
    assert period_ordinal(*args) == expected
