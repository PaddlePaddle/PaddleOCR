"""
Parsing functions for datetime and datetime-like strings.
"""
import re
import time

from libc.string cimport strchr

import cython
from cython import Py_ssize_t

from cpython.datetime cimport (
    datetime,
    datetime_new,
    import_datetime,
    tzinfo,
)
from cpython.object cimport PyObject_Str
from cpython.version cimport PY_VERSION_HEX

import_datetime()

import numpy as np

cimport numpy as cnp
from numpy cimport (
    PyArray_GETITEM,
    PyArray_ITER_DATA,
    PyArray_ITER_NEXT,
    PyArray_IterNew,
    flatiter,
    float64_t,
)

cnp.import_array()

# dateutil compat

from dateutil.parser import (
    DEFAULTPARSER,
    parse as du_parse,
)
from dateutil.relativedelta import relativedelta
from dateutil.tz import (
    tzlocal as _dateutil_tzlocal,
    tzoffset,
    tzstr as _dateutil_tzstr,
    tzutc as _dateutil_tzutc,
)

from pandas._config import get_option

from pandas._libs.tslibs.ccalendar cimport c_MONTH_NUMBERS
from pandas._libs.tslibs.nattype cimport (
    c_NaT as NaT,
    c_nat_strings as nat_strings,
)
from pandas._libs.tslibs.offsets cimport is_offset_object
from pandas._libs.tslibs.util cimport (
    get_c_string_buf_and_size,
    is_array,
)


cdef extern from "../src/headers/portable.h":
    int getdigit_ascii(char c, int default) nogil

cdef extern from "../src/parser/tokenizer.h":
    double xstrtod(const char *p, char **q, char decimal, char sci, char tsep,
                   int skip_trailing, int *error, int *maybe_int)


# ----------------------------------------------------------------------
# Constants


class DateParseError(ValueError):
    pass


_DEFAULT_DATETIME = datetime(1, 1, 1).replace(hour=0, minute=0,
                                              second=0, microsecond=0)

cdef:
    set _not_datelike_strings = {'a', 'A', 'm', 'M', 'p', 'P', 't', 'T'}

# ----------------------------------------------------------------------
cdef:
    const char* delimiters = " /-."
    int MAX_DAYS_IN_MONTH = 31, MAX_MONTH = 12


cdef inline bint _is_not_delimiter(const char ch):
    return strchr(delimiters, ch) == NULL


cdef inline int _parse_2digit(const char* s):
    cdef int result = 0
    result += getdigit_ascii(s[0], -10) * 10
    result += getdigit_ascii(s[1], -100) * 1
    return result


cdef inline int _parse_4digit(const char* s):
    cdef int result = 0
    result += getdigit_ascii(s[0], -10) * 1000
    result += getdigit_ascii(s[1], -100) * 100
    result += getdigit_ascii(s[2], -1000) * 10
    result += getdigit_ascii(s[3], -10000) * 1
    return result


cdef inline object _parse_delimited_date(str date_string, bint dayfirst):
    """
    Parse special cases of dates: MM/DD/YYYY, DD/MM/YYYY, MM/YYYY.

    At the beginning function tries to parse date in MM/DD/YYYY format, but
    if month > 12 - in DD/MM/YYYY (`dayfirst == False`).
    With `dayfirst == True` function makes an attempt to parse date in
    DD/MM/YYYY, if an attempt is wrong - in DD/MM/YYYY

    For MM/DD/YYYY, DD/MM/YYYY: delimiter can be a space or one of /-.
    For MM/YYYY: delimiter can be a space or one of /-
    If `date_string` can't be converted to date, then function returns
    None, None

    Parameters
    ----------
    date_string : str
    dayfirst : bool

    Returns:
    --------
    datetime or None
    str or None
        Describing resolution of the parsed string.
    """
    cdef:
        const char* buf
        Py_ssize_t length
        int day = 1, month = 1, year
        bint can_swap = 0

    buf = get_c_string_buf_and_size(date_string, &length)
    if length == 10:
        # parsing MM?DD?YYYY and DD?MM?YYYY dates
        if _is_not_delimiter(buf[2]) or _is_not_delimiter(buf[5]):
            return None, None
        month = _parse_2digit(buf)
        day = _parse_2digit(buf + 3)
        year = _parse_4digit(buf + 6)
        reso = 'day'
        can_swap = 1
    elif length == 7:
        # parsing MM?YYYY dates
        if buf[2] == b'.' or _is_not_delimiter(buf[2]):
            # we cannot reliably tell whether e.g. 10.2010 is a float
            # or a date, thus we refuse to parse it here
            return None, None
        month = _parse_2digit(buf)
        year = _parse_4digit(buf + 3)
        reso = 'month'
    else:
        return None, None

    if month < 0 or day < 0 or year < 1000:
        # some part is not an integer, so
        # date_string can't be converted to date, above format
        return None, None

    if 1 <= month <= MAX_DAYS_IN_MONTH and 1 <= day <= MAX_DAYS_IN_MONTH \
            and (month <= MAX_MONTH or day <= MAX_MONTH):
        if (month > MAX_MONTH or (day <= MAX_MONTH and dayfirst)) and can_swap:
            day, month = month, day
        if PY_VERSION_HEX >= 0x03060100:
            # In Python <= 3.6.0 there is no range checking for invalid dates
            # in C api, thus we call faster C version for 3.6.1 or newer
            return datetime_new(year, month, day, 0, 0, 0, 0, None), reso
        return datetime(year, month, day, 0, 0, 0, 0, None), reso

    raise DateParseError(f"Invalid date specified ({month}/{day})")


cdef inline bint does_string_look_like_time(str parse_string):
    """
    Checks whether given string is a time: it has to start either from
    H:MM or from HH:MM, and hour and minute values must be valid.

    Parameters
    ----------
    parse_string : str

    Returns:
    --------
    bool
        Whether given string is potentially a time.
    """
    cdef:
        const char* buf
        Py_ssize_t length
        int hour = -1, minute = -1

    buf = get_c_string_buf_and_size(parse_string, &length)
    if length >= 4:
        if buf[1] == b':':
            # h:MM format
            hour = getdigit_ascii(buf[0], -1)
            minute = _parse_2digit(buf + 2)
        elif buf[2] == b':':
            # HH:MM format
            hour = _parse_2digit(buf)
            minute = _parse_2digit(buf + 3)

    return 0 <= hour <= 23 and 0 <= minute <= 59


def parse_datetime_string(
    str date_string,
    bint dayfirst=False,
    bint yearfirst=False,
    **kwargs,
) -> datetime:
    """
    Parse datetime string, only returns datetime.
    Also cares special handling matching time patterns.

    Returns
    -------
    datetime
    """

    cdef:
        object dt

    if not _does_string_look_like_datetime(date_string):
        raise ValueError('Given date string not likely a datetime.')

    if does_string_look_like_time(date_string):
        # use current datetime as default, not pass _DEFAULT_DATETIME
        dt = du_parse(date_string, dayfirst=dayfirst,
                      yearfirst=yearfirst, **kwargs)
        return dt

    dt, _ = _parse_delimited_date(date_string, dayfirst)
    if dt is not None:
        return dt

    try:
        dt, _ = _parse_dateabbr_string(date_string, _DEFAULT_DATETIME, freq=None)
        return dt
    except DateParseError:
        raise
    except ValueError:
        pass

    try:
        dt = du_parse(date_string, default=_DEFAULT_DATETIME,
                      dayfirst=dayfirst, yearfirst=yearfirst, **kwargs)
    except TypeError:
        # following may be raised from dateutil
        # TypeError: 'NoneType' object is not iterable
        raise ValueError('Given date string not likely a datetime.')

    return dt


def parse_time_string(arg: str, freq=None, dayfirst=None, yearfirst=None):
    """
    Try hard to parse datetime string, leveraging dateutil plus some extra
    goodies like quarter recognition.

    Parameters
    ----------
    arg : str
    freq : str or DateOffset, default None
        Helps with interpreting time string if supplied
    dayfirst : bool, default None
        If None uses default from print_config
    yearfirst : bool, default None
        If None uses default from print_config

    Returns
    -------
    datetime
    str
        Describing resolution of parsed string.
    """
    if is_offset_object(freq):
        freq = freq.rule_code

    if dayfirst is None or yearfirst is None:
        if dayfirst is None:
            dayfirst = get_option("display.date_dayfirst")
        if yearfirst is None:
            yearfirst = get_option("display.date_yearfirst")

    res = parse_datetime_string_with_reso(arg, freq=freq,
                                          dayfirst=dayfirst,
                                          yearfirst=yearfirst)
    return res


cdef parse_datetime_string_with_reso(
    str date_string, str freq=None, bint dayfirst=False, bint yearfirst=False,
):
    """
    Parse datetime string and try to identify its resolution.

    Returns
    -------
    datetime
    str
        Inferred resolution of the parsed string.

    Raises
    ------
    ValueError : preliminary check suggests string is not datetime
    DateParseError : error within dateutil
    """
    cdef:
        object parsed, reso

    if not _does_string_look_like_datetime(date_string):
        raise ValueError('Given date string not likely a datetime.')

    parsed, reso = _parse_delimited_date(date_string, dayfirst)
    if parsed is not None:
        return parsed, reso

    try:
        return _parse_dateabbr_string(date_string, _DEFAULT_DATETIME, freq)
    except DateParseError:
        raise
    except ValueError:
        pass

    try:
        parsed, reso = dateutil_parse(date_string, _DEFAULT_DATETIME,
                                      dayfirst=dayfirst, yearfirst=yearfirst,
                                      ignoretz=False, tzinfos=None)
    except (ValueError, OverflowError) as err:
        # TODO: allow raise of errors within instead
        raise DateParseError(err)
    if parsed is None:
        raise DateParseError(f"Could not parse {date_string}")
    return parsed, reso


cpdef bint _does_string_look_like_datetime(str py_string):
    """
    Checks whether given string is a datetime: it has to start with '0' or
    be greater than 1000.

    Parameters
    ----------
    py_string: str

    Returns
    -------
    bool
        Whether given string is potentially a datetime.
    """
    cdef:
        const char *buf
        char *endptr = NULL
        Py_ssize_t length = -1
        double converted_date
        char first
        int error = 0

    buf = get_c_string_buf_and_size(py_string, &length)
    if length >= 1:
        first = buf[0]
        if first == b'0':
            # Strings starting with 0 are more consistent with a
            # date-like string than a number
            return True
        elif py_string in _not_datelike_strings:
            return False
        else:
            # xstrtod with such parameters copies behavior of python `float`
            # cast; for example, " 35.e-1 " is valid string for this cast so,
            # for correctly xstrtod call necessary to pass these params:
            # b'.' - a dot is used as separator, b'e' - an exponential form of
            # a float number can be used, b'\0' - not to use a thousand
            # separator, 1 - skip extra spaces before and after,
            converted_date = xstrtod(buf, &endptr,
                                     b'.', b'e', b'\0', 1, &error, NULL)
            # if there were no errors and the whole line was parsed, then ...
            if error == 0 and endptr == buf + length:
                return converted_date >= 1000

    return True


cdef inline object _parse_dateabbr_string(object date_string, datetime default,
                                          str freq=None):
    cdef:
        object ret
        # year initialized to prevent compiler warnings
        int year = -1, quarter = -1, month, mnum, date_len

    # special handling for possibilities eg, 2Q2005, 2Q05, 2005Q1, 05Q1
    assert isinstance(date_string, str)

    if date_string in nat_strings:
        return NaT, ''

    date_string = date_string.upper()
    date_len = len(date_string)

    if date_len == 4:
        # parse year only like 2000
        try:
            ret = default.replace(year=int(date_string))
            return ret, 'year'
        except ValueError:
            pass

    try:
        if 4 <= date_len <= 7:
            i = date_string.index('Q', 1, 6)
            if i == 1:
                quarter = int(date_string[0])
                if date_len == 4 or (date_len == 5
                                     and date_string[i + 1] == '-'):
                    # r'(\d)Q-?(\d\d)')
                    year = 2000 + int(date_string[-2:])
                elif date_len == 6 or (date_len == 7
                                       and date_string[i + 1] == '-'):
                    # r'(\d)Q-?(\d\d\d\d)')
                    year = int(date_string[-4:])
                else:
                    raise ValueError
            elif i == 2 or i == 3:
                # r'(\d\d)-?Q(\d)'
                if date_len == 4 or (date_len == 5
                                     and date_string[i - 1] == '-'):
                    quarter = int(date_string[-1])
                    year = 2000 + int(date_string[:2])
                else:
                    raise ValueError
            elif i == 4 or i == 5:
                if date_len == 6 or (date_len == 7
                                     and date_string[i - 1] == '-'):
                    # r'(\d\d\d\d)-?Q(\d)'
                    quarter = int(date_string[-1])
                    year = int(date_string[:4])
                else:
                    raise ValueError

            if not (1 <= quarter <= 4):
                raise DateParseError(f'Incorrect quarterly string is given, '
                                     f'quarter must be '
                                     f'between 1 and 4: {date_string}')

            try:
                # GH#1228
                year, month = quarter_to_myear(year, quarter, freq)
            except KeyError:
                raise DateParseError("Unable to retrieve month "
                                     "information from given "
                                     f"freq: {freq}")

            ret = default.replace(year=year, month=month)
            return ret, 'quarter'

    except DateParseError:
        raise
    except ValueError:
        pass

    if date_len == 6 and (freq == 'M' or
                          getattr(freq, 'rule_code', None) == 'M'):
        year = int(date_string[:4])
        month = int(date_string[4:6])
        try:
            ret = default.replace(year=year, month=month)
            return ret, 'month'
        except ValueError:
            pass

    for pat in ['%Y-%m', '%b %Y', '%b-%Y']:
        try:
            ret = datetime.strptime(date_string, pat)
            return ret, 'month'
        except ValueError:
            pass

    raise ValueError(f'Unable to parse {date_string}')


cpdef quarter_to_myear(int year, int quarter, str freq):
    """
    A quarterly frequency defines a "year" which may not coincide with
    the calendar-year.  Find the calendar-year and calendar-month associated
    with the given year and quarter under the `freq`-derived calendar.

    Parameters
    ----------
    year : int
    quarter : int
    freq : str or None

    Returns
    -------
    year : int
    month : int

    See Also
    --------
    Period.qyear
    """
    if quarter <= 0 or quarter > 4:
        raise ValueError("Quarter must be 1 <= q <= 4")

    if freq is not None:
        mnum = c_MONTH_NUMBERS[get_rule_month(freq)] + 1
        month = (mnum + (quarter - 1) * 3) % 12 + 1
        if month > mnum:
            year -= 1
    else:
        month = (quarter - 1) * 3 + 1

    return year, month


cdef dateutil_parse(
    str timestr,
    object default,
    bint ignoretz=False,
    object tzinfos=None,
    bint dayfirst=False,
    bint yearfirst=False,
):
    """ lifted from dateutil to get resolution"""

    cdef:
        object res, attr, ret, tzdata
        object reso = None
        dict repl = {}

    res, _ = DEFAULTPARSER._parse(timestr, dayfirst=dayfirst, yearfirst=yearfirst)

    if res is None:
        raise ValueError(f"Unknown datetime string format, unable to parse: {timestr}")

    for attr in ["year", "month", "day", "hour",
                 "minute", "second", "microsecond"]:
        value = getattr(res, attr)
        if value is not None:
            repl[attr] = value
            reso = attr

    if reso is None:
        raise ValueError(f"Unable to parse datetime string: {timestr}")

    if reso == 'microsecond':
        if repl['microsecond'] == 0:
            reso = 'second'
        elif repl['microsecond'] % 1000 == 0:
            reso = 'millisecond'

    ret = default.replace(**repl)
    if res.weekday is not None and not res.day:
        ret = ret + relativedelta.relativedelta(weekday=res.weekday)
    if not ignoretz:
        if callable(tzinfos) or tzinfos and res.tzname in tzinfos:
            # Note: as of 1.0 this is not reached because
            #  we never pass tzinfos, see GH#22234
            if callable(tzinfos):
                tzdata = tzinfos(res.tzname, res.tzoffset)
            else:
                tzdata = tzinfos.get(res.tzname)
            if isinstance(tzdata, tzinfo):
                new_tzinfo = tzdata
            elif isinstance(tzdata, str):
                new_tzinfo = _dateutil_tzstr(tzdata)
            elif isinstance(tzdata, int):
                new_tzinfo = tzoffset(res.tzname, tzdata)
            else:
                raise ValueError("offset must be tzinfo subclass, "
                                 "tz string, or int offset")
            ret = ret.replace(tzinfo=new_tzinfo)
        elif res.tzname and res.tzname in time.tzname:
            ret = ret.replace(tzinfo=_dateutil_tzlocal())
        elif res.tzoffset == 0:
            ret = ret.replace(tzinfo=_dateutil_tzutc())
        elif res.tzoffset:
            ret = ret.replace(tzinfo=tzoffset(res.tzname, res.tzoffset))
    return ret, reso


# ----------------------------------------------------------------------
# Parsing for type-inference


def try_parse_dates(
    object[:] values, parser=None, bint dayfirst=False, default=None,
) -> np.ndarray:
    cdef:
        Py_ssize_t i, n
        object[:] result

    n = len(values)
    result = np.empty(n, dtype='O')

    if parser is None:
        if default is None:  # GH2618
            date = datetime.now()
            default = datetime(date.year, date.month, 1)

        parse_date = lambda x: du_parse(x, dayfirst=dayfirst, default=default)

        # EAFP here
        try:
            for i in range(n):
                if values[i] == '':
                    result[i] = np.nan
                else:
                    result[i] = parse_date(values[i])
        except Exception:
            # Since parser is user-defined, we can't guess what it might raise
            return values
    else:
        parse_date = parser

        for i in range(n):
            if values[i] == '':
                result[i] = np.nan
            else:
                result[i] = parse_date(values[i])

    return result.base  # .base to access underlying ndarray


def try_parse_date_and_time(
    object[:] dates,
    object[:] times,
    date_parser=None,
    time_parser=None,
    bint dayfirst=False,
    default=None,
) -> np.ndarray:
    cdef:
        Py_ssize_t i, n
        object[:] result

    n = len(dates)
    # TODO(cython 3.0): Use len instead of `shape[0]`
    if times.shape[0] != n:
        raise ValueError('Length of dates and times must be equal')
    result = np.empty(n, dtype='O')

    if date_parser is None:
        if default is None:  # GH2618
            date = datetime.now()
            default = datetime(date.year, date.month, 1)

        parse_date = lambda x: du_parse(x, dayfirst=dayfirst, default=default)

    else:
        parse_date = date_parser

    if time_parser is None:
        parse_time = lambda x: du_parse(x)

    else:
        parse_time = time_parser

    for i in range(n):
        d = parse_date(str(dates[i]))
        t = parse_time(str(times[i]))
        result[i] = datetime(d.year, d.month, d.day,
                             t.hour, t.minute, t.second)

    return result.base  # .base to access underlying ndarray


def try_parse_year_month_day(
    object[:] years, object[:] months, object[:] days
) -> np.ndarray:
    cdef:
        Py_ssize_t i, n
        object[:] result

    n = len(years)
    # TODO(cython 3.0): Use len instead of `shape[0]`
    if months.shape[0] != n or days.shape[0] != n:
        raise ValueError('Length of years/months/days must all be equal')
    result = np.empty(n, dtype='O')

    for i in range(n):
        result[i] = datetime(int(years[i]), int(months[i]), int(days[i]))

    return result.base  # .base to access underlying ndarray


def try_parse_datetime_components(object[:] years,
                                  object[:] months,
                                  object[:] days,
                                  object[:] hours,
                                  object[:] minutes,
                                  object[:] seconds) -> np.ndarray:

    cdef:
        Py_ssize_t i, n
        object[:] result
        int secs
        double float_secs
        double micros

    n = len(years)
    # TODO(cython 3.0): Use len instead of `shape[0]`
    if (
        months.shape[0] != n
        or days.shape[0] != n
        or hours.shape[0] != n
        or minutes.shape[0] != n
        or seconds.shape[0] != n
    ):
        raise ValueError('Length of all datetime components must be equal')
    result = np.empty(n, dtype='O')

    for i in range(n):
        float_secs = float(seconds[i])
        secs = int(float_secs)

        micros = float_secs - secs
        if micros > 0:
            micros = micros * 1000000

        result[i] = datetime(int(years[i]), int(months[i]), int(days[i]),
                             int(hours[i]), int(minutes[i]), secs,
                             int(micros))

    return result.base  # .base to access underlying ndarray


# ----------------------------------------------------------------------
# Miscellaneous


# Class copied verbatim from https://github.com/dateutil/dateutil/pull/732
#
# We use this class to parse and tokenize date strings. However, as it is
# a private class in the dateutil library, relying on backwards compatibility
# is not practical. In fact, using this class issues warnings (xref gh-21322).
# Thus, we port the class over so that both issues are resolved.
#
# Copyright (c) 2017 - dateutil contributors
class _timelex:
    def __init__(self, instream):
        if getattr(instream, 'decode', None) is not None:
            instream = instream.decode()

        if isinstance(instream, str):
            self.stream = instream
        elif getattr(instream, 'read', None) is None:
            raise TypeError(
                'Parser must be a string or character stream, not '
                f'{type(instream).__name__}')
        else:
            self.stream = instream.read()

    def get_tokens(self):
        """
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.
        The main complication arises from the fact that dots ('.') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it is necessary to read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        """
        cdef:
            Py_ssize_t n

        stream = self.stream.replace('\x00', '')

        # TODO: Change \s --> \s+ (this doesn't match existing behavior)
        # TODO: change the punctuation block to punc+ (does not match existing)
        # TODO: can we merge the two digit patterns?
        tokens = re.findall('\s|'
                            '(?<![\.\d])\d+\.\d+(?![\.\d])'
                            '|\d+'
                            '|[a-zA-Z]+'
                            '|[\./:]+'
                            '|[^\da-zA-Z\./:\s]+', stream)

        # Re-combine token tuples of the form ["59", ",", "456"] because
        # in this context the "," is treated as a decimal
        # (e.g. in python's default logging format)
        for n, token in enumerate(tokens[:-2]):
            # Kludge to match ,-decimal behavior; it'd be better to do this
            # later in the process and have a simpler tokenization
            if (token is not None and token.isdigit() and
                    tokens[n + 1] == ',' and tokens[n + 2].isdigit()):
                # Have to check None b/c it might be replaced during the loop
                # TODO: I _really_ don't faking the value here
                tokens[n] = token + '.' + tokens[n + 2]
                tokens[n + 1] = None
                tokens[n + 2] = None

        tokens = [x for x in tokens if x is not None]
        return tokens

    @classmethod
    def split(cls, s):
        return cls(s).get_tokens()


_DATEUTIL_LEXER_SPLIT = _timelex.split


def format_is_iso(f: str) -> bint:
    """
    Does format match the iso8601 set that can be handled by the C parser?
    Generally of form YYYY-MM-DDTHH:MM:SS - date separator can be different
    but must be consistent.  Leading 0s in dates and times are optional.
    """
    iso_template = '%Y{date_sep}%m{date_sep}%d{time_sep}%H:%M:%S.%f'.format
    excluded_formats = ['%Y%m%d', '%Y%m', '%Y']

    for date_sep in [' ', '/', '\\', '-', '.', '']:
        for time_sep in [' ', 'T']:
            if (iso_template(date_sep=date_sep,
                             time_sep=time_sep
                             ).startswith(f) and f not in excluded_formats):
                return True
    return False


def guess_datetime_format(
    dt_str,
    bint dayfirst=False,
    dt_str_parse=du_parse,
    dt_str_split=_DATEUTIL_LEXER_SPLIT,
):
    """
    Guess the datetime format of a given datetime string.

    Parameters
    ----------
    dt_str : str
        Datetime string to guess the format of.
    dayfirst : bool, default False
        If True parses dates with the day first, eg 20/01/2005
        Warning: dayfirst=True is not strict, but will prefer to parse
        with day first (this is a known bug).
    dt_str_parse : function, defaults to `dateutil.parser.parse`
        This function should take in a datetime string and return
        a `datetime.datetime` guess that the datetime string represents
    dt_str_split : function, defaults to `_DATEUTIL_LEXER_SPLIT` (dateutil)
        This function should take in a datetime string and return
        a list of strings, the guess of the various specific parts
        e.g. '2011/12/30' -> ['2011', '/', '12', '/', '30']

    Returns
    -------
    ret : datetime format string (for `strftime` or `strptime`)
    """
    if dt_str_parse is None or dt_str_split is None:
        return None

    if not isinstance(dt_str, str):
        return None

    day_attribute_and_format = (('day',), '%d', 2)

    # attr name, format, padding (if any)
    datetime_attrs_to_format = [
        (('year', 'month', 'day'), '%Y%m%d', 0),
        (('year',), '%Y', 0),
        (('month',), '%B', 0),
        (('month',), '%b', 0),
        (('month',), '%m', 2),
        day_attribute_and_format,
        (('hour',), '%H', 2),
        (('minute',), '%M', 2),
        (('second',), '%S', 2),
        (('microsecond',), '%f', 6),
        (('second', 'microsecond'), '%S.%f', 0),
        (('tzinfo',), '%Z', 0),
    ]

    if dayfirst:
        datetime_attrs_to_format.remove(day_attribute_and_format)
        datetime_attrs_to_format.insert(0, day_attribute_and_format)

    try:
        parsed_datetime = dt_str_parse(dt_str, dayfirst=dayfirst)
    except (ValueError, OverflowError):
        # In case the datetime can't be parsed, its format cannot be guessed
        return None

    if parsed_datetime is None:
        return None

    # the default dt_str_split from dateutil will never raise here; we assume
    #  that any user-provided function will not either.
    tokens = dt_str_split(dt_str)

    format_guess = [None] * len(tokens)
    found_attrs = set()

    for attrs, attr_format, padding in datetime_attrs_to_format:
        # If a given attribute has been placed in the format string, skip
        # over other formats for that same underlying attribute (IE, month
        # can be represented in multiple different ways)
        if set(attrs) & found_attrs:
            continue

        if all(getattr(parsed_datetime, attr) is not None for attr in attrs):
            for i, token_format in enumerate(format_guess):
                token_filled = tokens[i].zfill(padding)
                if (token_format is None and
                        token_filled == parsed_datetime.strftime(attr_format)):
                    format_guess[i] = attr_format
                    tokens[i] = token_filled
                    found_attrs.update(attrs)
                    break

    # Only consider it a valid guess if we have a year, month and day
    if len({'year', 'month', 'day'} & found_attrs) != 3:
        return None

    output_format = []
    for i, guess in enumerate(format_guess):
        if guess is not None:
            # Either fill in the format placeholder (like %Y)
            output_format.append(guess)
        else:
            # Or just the token separate (IE, the dashes in "01-01-2013")
            try:
                # If the token is numeric, then we likely didn't parse it
                # properly, so our guess is wrong
                float(tokens[i])
                return None
            except ValueError:
                pass

            output_format.append(tokens[i])

    guessed_format = ''.join(output_format)

    # rebuild string, capturing any inferred padding
    dt_str = ''.join(tokens)
    if parsed_datetime.strftime(guessed_format) == dt_str:
        return guessed_format
    else:
        return None


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline object convert_to_unicode(object item, bint keep_trivial_numbers):
    """
    Convert `item` to str.

    Parameters
    ----------
    item : object
    keep_trivial_numbers : bool
        if True, then conversion (to string from integer/float zero)
        is not performed

    Returns
    -------
    str or int or float
    """
    cdef:
        float64_t float_item

    if keep_trivial_numbers:
        if isinstance(item, int):
            if <int>item == 0:
                return item
        elif isinstance(item, float):
            float_item = item
            if float_item == 0.0 or float_item != float_item:
                return item

    if not isinstance(item, str):
        item = PyObject_Str(item)

    return item


@cython.wraparound(False)
@cython.boundscheck(False)
def concat_date_cols(tuple date_cols, bint keep_trivial_numbers=True) -> np.ndarray:
    """
    Concatenates elements from numpy arrays in `date_cols` into strings.

    Parameters
    ----------
    date_cols : tuple[ndarray]
    keep_trivial_numbers : bool, default True
        if True and len(date_cols) == 1, then
        conversion (to string from integer/float zero) is not performed

    Returns
    -------
    arr_of_rows : ndarray[object]

    Examples
    --------
    >>> dates=np.array(['3/31/2019', '4/31/2019'], dtype=object)
    >>> times=np.array(['11:20', '10:45'], dtype=object)
    >>> result = concat_date_cols((dates, times))
    >>> result
    array(['3/31/2019 11:20', '4/31/2019 10:45'], dtype=object)
    """
    cdef:
        Py_ssize_t rows_count = 0, col_count = len(date_cols)
        Py_ssize_t col_idx, row_idx
        list list_to_join
        cnp.ndarray[object] iters
        object[::1] iters_view
        flatiter it
        cnp.ndarray[object] result
        object[:] result_view

    if col_count == 0:
        return np.zeros(0, dtype=object)

    if not all(is_array(array) for array in date_cols):
        raise ValueError("not all elements from date_cols are numpy arrays")

    rows_count = min(len(array) for array in date_cols)
    result = np.zeros(rows_count, dtype=object)
    result_view = result

    if col_count == 1:
        array = date_cols[0]
        it = <flatiter>PyArray_IterNew(array)
        for row_idx in range(rows_count):
            item = PyArray_GETITEM(array, PyArray_ITER_DATA(it))
            result_view[row_idx] = convert_to_unicode(item,
                                                      keep_trivial_numbers)
            PyArray_ITER_NEXT(it)
    else:
        # create fixed size list - more efficient memory allocation
        list_to_join = [None] * col_count
        iters = np.zeros(col_count, dtype=object)

        # create memoryview of iters ndarray, that will contain some
        # flatiter's for each array in `date_cols` - more efficient indexing
        iters_view = iters
        for col_idx, array in enumerate(date_cols):
            iters_view[col_idx] = PyArray_IterNew(array)

        # array elements that are on the same line are converted to one string
        for row_idx in range(rows_count):
            for col_idx, array in enumerate(date_cols):
                # this cast is needed, because we did not find a way
                # to efficiently store `flatiter` type objects in ndarray
                it = <flatiter>iters_view[col_idx]
                item = PyArray_GETITEM(array, PyArray_ITER_DATA(it))
                list_to_join[col_idx] = convert_to_unicode(item, False)
                PyArray_ITER_NEXT(it)
            result_view[row_idx] = " ".join(list_to_join)

    return result


cpdef str get_rule_month(str source):
    """
    Return starting month of given freq, default is December.

    Parameters
    ----------
    source : str
        Derived from `freq.rule_code` or `freq.freqstr`.

    Returns
    -------
    rule_month: str

    Examples
    --------
    >>> get_rule_month('D')
    'DEC'

    >>> get_rule_month('A-JAN')
    'JAN'
    """
    source = source.upper()
    if "-" not in source:
        return "DEC"
    else:
        return source.split("-")[1]
