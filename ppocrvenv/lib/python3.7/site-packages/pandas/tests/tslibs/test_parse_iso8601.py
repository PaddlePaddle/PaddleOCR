from datetime import datetime

import pytest

from pandas._libs import tslib


@pytest.mark.parametrize(
    "date_str, exp",
    [
        ("2011-01-02", datetime(2011, 1, 2)),
        ("2011-1-2", datetime(2011, 1, 2)),
        ("2011-01", datetime(2011, 1, 1)),
        ("2011-1", datetime(2011, 1, 1)),
        ("2011 01 02", datetime(2011, 1, 2)),
        ("2011.01.02", datetime(2011, 1, 2)),
        ("2011/01/02", datetime(2011, 1, 2)),
        ("2011\\01\\02", datetime(2011, 1, 2)),
        ("2013-01-01 05:30:00", datetime(2013, 1, 1, 5, 30)),
        ("2013-1-1 5:30:00", datetime(2013, 1, 1, 5, 30)),
    ],
)
def test_parsers_iso8601(date_str, exp):
    # see gh-12060
    #
    # Test only the ISO parser - flexibility to
    # different separators and leading zero's.
    actual = tslib._test_parse_iso8601(date_str)
    assert actual == exp


@pytest.mark.parametrize(
    "date_str",
    [
        "2011-01/02",
        "2011=11=11",
        "201401",
        "201111",
        "200101",
        # Mixed separated and unseparated.
        "2005-0101",
        "200501-01",
        "20010101 12:3456",
        "20010101 1234:56",
        # HHMMSS must have two digits in
        # each component if unseparated.
        "20010101 1",
        "20010101 123",
        "20010101 12345",
        "20010101 12345Z",
    ],
)
def test_parsers_iso8601_invalid(date_str):
    msg = f'Error parsing datetime string "{date_str}"'

    with pytest.raises(ValueError, match=msg):
        tslib._test_parse_iso8601(date_str)


def test_parsers_iso8601_invalid_offset_invalid():
    date_str = "2001-01-01 12-34-56"
    msg = f'Timezone hours offset out of range in datetime string "{date_str}"'

    with pytest.raises(ValueError, match=msg):
        tslib._test_parse_iso8601(date_str)


def test_parsers_iso8601_leading_space():
    # GH#25895 make sure isoparser doesn't overflow with long input
    date_str, expected = ("2013-1-1 5:30:00", datetime(2013, 1, 1, 5, 30))
    actual = tslib._test_parse_iso8601(" " * 200 + date_str)
    assert actual == expected
