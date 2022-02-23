"""
Tests for Timestamp parsing, aimed at pandas/_libs/tslibs/parsing.pyx
"""
from datetime import datetime
import re

from dateutil.parser import parse
import numpy as np
import pytest

from pandas._libs.tslibs import parsing
from pandas._libs.tslibs.parsing import parse_time_string
import pandas.util._test_decorators as td

import pandas._testing as tm


def test_parse_time_string():
    (parsed, reso) = parse_time_string("4Q1984")
    (parsed_lower, reso_lower) = parse_time_string("4q1984")

    assert reso == reso_lower
    assert parsed == parsed_lower


def test_parse_time_string_invalid_type():
    # Raise on invalid input, don't just return it
    msg = "Argument 'arg' has incorrect type (expected str, got tuple)"
    with pytest.raises(TypeError, match=re.escape(msg)):
        parse_time_string((4, 5))


@pytest.mark.parametrize(
    "dashed,normal", [("1988-Q2", "1988Q2"), ("2Q-1988", "2Q1988")]
)
def test_parse_time_quarter_with_dash(dashed, normal):
    # see gh-9688
    (parsed_dash, reso_dash) = parse_time_string(dashed)
    (parsed, reso) = parse_time_string(normal)

    assert parsed_dash == parsed
    assert reso_dash == reso


@pytest.mark.parametrize("dashed", ["-2Q1992", "2-Q1992", "4-4Q1992"])
def test_parse_time_quarter_with_dash_error(dashed):
    msg = f"Unknown datetime string format, unable to parse: {dashed}"

    with pytest.raises(parsing.DateParseError, match=msg):
        parse_time_string(dashed)


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("123.1234", False),
        ("-50000", False),
        ("999", False),
        ("m", False),
        ("T", False),
        ("Mon Sep 16, 2013", True),
        ("2012-01-01", True),
        ("01/01/2012", True),
        ("01012012", True),
        ("0101", True),
        ("1-1", True),
    ],
)
def test_does_not_convert_mixed_integer(date_string, expected):
    assert parsing._does_string_look_like_datetime(date_string) is expected


@pytest.mark.parametrize(
    "date_str,kwargs,msg",
    [
        (
            "2013Q5",
            {},
            (
                "Incorrect quarterly string is given, "
                "quarter must be between 1 and 4: 2013Q5"
            ),
        ),
        # see gh-5418
        (
            "2013Q1",
            {"freq": "INVLD-L-DEC-SAT"},
            (
                "Unable to retrieve month information "
                "from given freq: INVLD-L-DEC-SAT"
            ),
        ),
    ],
)
def test_parsers_quarterly_with_freq_error(date_str, kwargs, msg):
    with pytest.raises(parsing.DateParseError, match=msg):
        parsing.parse_time_string(date_str, **kwargs)


@pytest.mark.parametrize(
    "date_str,freq,expected",
    [
        ("2013Q2", None, datetime(2013, 4, 1)),
        ("2013Q2", "A-APR", datetime(2012, 8, 1)),
        ("2013-Q2", "A-DEC", datetime(2013, 4, 1)),
    ],
)
def test_parsers_quarterly_with_freq(date_str, freq, expected):
    result, _ = parsing.parse_time_string(date_str, freq=freq)
    assert result == expected


@pytest.mark.parametrize(
    "date_str", ["2Q 2005", "2Q-200A", "2Q-200", "22Q2005", "2Q200.", "6Q-20"]
)
def test_parsers_quarter_invalid(date_str):
    if date_str == "6Q-20":
        msg = (
            "Incorrect quarterly string is given, quarter "
            f"must be between 1 and 4: {date_str}"
        )
    else:
        msg = f"Unknown datetime string format, unable to parse: {date_str}"

    with pytest.raises(ValueError, match=msg):
        parsing.parse_time_string(date_str)


@pytest.mark.parametrize(
    "date_str,expected",
    [("201101", datetime(2011, 1, 1, 0, 0)), ("200005", datetime(2000, 5, 1, 0, 0))],
)
def test_parsers_month_freq(date_str, expected):
    result, _ = parsing.parse_time_string(date_str, freq="M")
    assert result == expected


@td.skip_if_not_us_locale
@pytest.mark.parametrize(
    "string,fmt",
    [
        ("20111230", "%Y%m%d"),
        ("2011-12-30", "%Y-%m-%d"),
        ("30-12-2011", "%d-%m-%Y"),
        ("2011-12-30 00:00:00", "%Y-%m-%d %H:%M:%S"),
        ("2011-12-30T00:00:00", "%Y-%m-%dT%H:%M:%S"),
        ("2011-12-30 00:00:00.000000", "%Y-%m-%d %H:%M:%S.%f"),
    ],
)
def test_guess_datetime_format_with_parseable_formats(string, fmt):
    result = parsing.guess_datetime_format(string)
    assert result == fmt


@pytest.mark.parametrize("dayfirst,expected", [(True, "%d/%m/%Y"), (False, "%m/%d/%Y")])
def test_guess_datetime_format_with_dayfirst(dayfirst, expected):
    ambiguous_string = "01/01/2011"
    result = parsing.guess_datetime_format(ambiguous_string, dayfirst=dayfirst)
    assert result == expected


@td.skip_if_has_locale
@pytest.mark.parametrize(
    "string,fmt",
    [
        ("30/Dec/2011", "%d/%b/%Y"),
        ("30/December/2011", "%d/%B/%Y"),
        ("30/Dec/2011 00:00:00", "%d/%b/%Y %H:%M:%S"),
    ],
)
def test_guess_datetime_format_with_locale_specific_formats(string, fmt):
    result = parsing.guess_datetime_format(string)
    assert result == fmt


@pytest.mark.parametrize(
    "invalid_dt",
    [
        "2013",
        "01/2013",
        "12:00:00",
        "1/1/1/1",
        "this_is_not_a_datetime",
        "51a",
        9,
        datetime(2011, 1, 1),
    ],
)
def test_guess_datetime_format_invalid_inputs(invalid_dt):
    # A datetime string must include a year, month and a day for it to be
    # guessable, in addition to being a string that looks like a datetime.
    assert parsing.guess_datetime_format(invalid_dt) is None


@pytest.mark.parametrize(
    "string,fmt",
    [
        ("2011-1-1", "%Y-%m-%d"),
        ("1/1/2011", "%m/%d/%Y"),
        ("30-1-2011", "%d-%m-%Y"),
        ("2011-1-1 0:0:0", "%Y-%m-%d %H:%M:%S"),
        ("2011-1-3T00:00:0", "%Y-%m-%dT%H:%M:%S"),
        ("2011-1-1 00:00:00", "%Y-%m-%d %H:%M:%S"),
    ],
)
def test_guess_datetime_format_no_padding(string, fmt):
    # see gh-11142
    result = parsing.guess_datetime_format(string)
    assert result == fmt


def test_try_parse_dates():
    arr = np.array(["5/1/2000", "6/1/2000", "7/1/2000"], dtype=object)
    result = parsing.try_parse_dates(arr, dayfirst=True)

    expected = np.array([parse(d, dayfirst=True) for d in arr])
    tm.assert_numpy_array_equal(result, expected)


def test_parse_time_string_check_instance_type_raise_exception():
    # issue 20684
    msg = "Argument 'arg' has incorrect type (expected str, got tuple)"
    with pytest.raises(TypeError, match=re.escape(msg)):
        parse_time_string((1, 2, 3))

    result = parse_time_string("2019")
    expected = (datetime(2019, 1, 1), "year")
    assert result == expected
