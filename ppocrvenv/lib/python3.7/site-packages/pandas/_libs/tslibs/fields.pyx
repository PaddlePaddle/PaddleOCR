"""
Functions for accessing attributes of Timestamp/datetime64/datetime-like
objects and arrays
"""
from locale import LC_TIME

import cython
from cython import Py_ssize_t
import numpy as np

cimport numpy as cnp
from numpy cimport (
    int8_t,
    int32_t,
    int64_t,
    ndarray,
    uint32_t,
)

cnp.import_array()

from pandas._config.localization import set_locale

from pandas._libs.tslibs.ccalendar import (
    DAYS_FULL,
    MONTHS_FULL,
)

from pandas._libs.tslibs.ccalendar cimport (
    dayofweek,
    get_day_of_year,
    get_days_in_month,
    get_firstbday,
    get_iso_calendar,
    get_lastbday,
    get_week_of_year,
    is_leapyear,
    iso_calendar_t,
    month_offset,
)
from pandas._libs.tslibs.nattype cimport NPY_NAT
from pandas._libs.tslibs.np_datetime cimport (
    dt64_to_dtstruct,
    npy_datetimestruct,
    pandas_timedeltastruct,
    td64_to_tdstruct,
)

from pandas._libs.tslibs.strptime import LocaleTime


@cython.wraparound(False)
@cython.boundscheck(False)
def build_field_sarray(const int64_t[:] dtindex):
    """
    Datetime as int64 representation to a structured array of fields
    """
    cdef:
        Py_ssize_t i, count = len(dtindex)
        npy_datetimestruct dts
        ndarray[int32_t] years, months, days, hours, minutes, seconds, mus

    sa_dtype = [
        ("Y", "i4"),  # year
        ("M", "i4"),  # month
        ("D", "i4"),  # day
        ("h", "i4"),  # hour
        ("m", "i4"),  # min
        ("s", "i4"),  # second
        ("u", "i4"),  # microsecond
    ]

    out = np.empty(count, dtype=sa_dtype)

    years = out['Y']
    months = out['M']
    days = out['D']
    hours = out['h']
    minutes = out['m']
    seconds = out['s']
    mus = out['u']

    for i in range(count):
        dt64_to_dtstruct(dtindex[i], &dts)
        years[i] = dts.year
        months[i] = dts.month
        days[i] = dts.day
        hours[i] = dts.hour
        minutes[i] = dts.min
        seconds[i] = dts.sec
        mus[i] = dts.us

    return out


def month_position_check(fields, weekdays) -> str | None:
    cdef:
        int32_t daysinmonth, y, m, d
        bint calendar_end = True
        bint business_end = True
        bint calendar_start = True
        bint business_start = True
        bint cal
        int32_t[:] years = fields["Y"]
        int32_t[:] months = fields["M"]
        int32_t[:] days = fields["D"]

    for y, m, d, wd in zip(years, months, days, weekdays):
        if calendar_start:
            calendar_start &= d == 1
        if business_start:
            business_start &= d == 1 or (d <= 3 and wd == 0)

        if calendar_end or business_end:
            daysinmonth = get_days_in_month(y, m)
            cal = d == daysinmonth
            if calendar_end:
                calendar_end &= cal
            if business_end:
                business_end &= cal or (daysinmonth - d < 3 and wd == 4)
        elif not calendar_start and not business_start:
            break

    if calendar_end:
        return "ce"
    elif business_end:
        return "be"
    elif calendar_start:
        return "cs"
    elif business_start:
        return "bs"
    else:
        return None


@cython.wraparound(False)
@cython.boundscheck(False)
def get_date_name_field(const int64_t[:] dtindex, str field, object locale=None):
    """
    Given a int64-based datetime index, return array of strings of date
    name based on requested field (e.g. day_name)
    """
    cdef:
        Py_ssize_t i, count = len(dtindex)
        ndarray[object] out, names
        npy_datetimestruct dts
        int dow

    out = np.empty(count, dtype=object)

    if field == 'day_name':
        if locale is None:
            names = np.array(DAYS_FULL, dtype=np.object_)
        else:
            names = np.array(get_locale_names('f_weekday', locale),
                             dtype=np.object_)
        for i in range(count):
            if dtindex[i] == NPY_NAT:
                out[i] = np.nan
                continue

            dt64_to_dtstruct(dtindex[i], &dts)
            dow = dayofweek(dts.year, dts.month, dts.day)
            out[i] = names[dow].capitalize()

    elif field == 'month_name':
        if locale is None:
            names = np.array(MONTHS_FULL, dtype=np.object_)
        else:
            names = np.array(get_locale_names('f_month', locale),
                             dtype=np.object_)
        for i in range(count):
            if dtindex[i] == NPY_NAT:
                out[i] = np.nan
                continue

            dt64_to_dtstruct(dtindex[i], &dts)
            out[i] = names[dts.month].capitalize()

    else:
        raise ValueError(f"Field {field} not supported")

    return out


cdef inline bint _is_on_month(int month, int compare_month, int modby) nogil:
    """
    Analogous to DateOffset.is_on_offset checking for the month part of a date.
    """
    if modby == 1:
        return True
    elif modby == 3:
        return (month - compare_month) % 3 == 0
    else:
        return month == compare_month


@cython.wraparound(False)
@cython.boundscheck(False)
def get_start_end_field(const int64_t[:] dtindex, str field,
                        object freqstr=None, int month_kw=12):
    """
    Given an int64-based datetime index return array of indicators
    of whether timestamps are at the start/end of the month/quarter/year
    (defined by frequency).
    """
    cdef:
        Py_ssize_t i
        int count = len(dtindex)
        bint is_business = 0
        int end_month = 12
        int start_month = 1
        ndarray[int8_t] out
        npy_datetimestruct dts
        int compare_month, modby

    out = np.zeros(count, dtype='int8')

    if freqstr:
        if freqstr == 'C':
            raise ValueError(f"Custom business days is not supported by {field}")
        is_business = freqstr[0] == 'B'

        # YearBegin(), BYearBegin() use month = starting month of year.
        # QuarterBegin(), BQuarterBegin() use startingMonth = starting
        # month of year. Other offsets use month, startingMonth as ending
        # month of year.

        if (freqstr[0:2] in ['MS', 'QS', 'AS']) or (
                freqstr[1:3] in ['MS', 'QS', 'AS']):
            end_month = 12 if month_kw == 1 else month_kw - 1
            start_month = month_kw
        else:
            end_month = month_kw
            start_month = (end_month % 12) + 1
    else:
        end_month = 12
        start_month = 1

    compare_month = start_month if "start" in field else end_month
    if "month" in field:
        modby = 1
    elif "quarter" in field:
        modby = 3
    else:
        modby = 12

    if field in ["is_month_start", "is_quarter_start", "is_year_start"]:
        if is_business:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)

                if _is_on_month(dts.month, compare_month, modby) and (
                        dts.day == get_firstbday(dts.year, dts.month)):
                    out[i] = 1

        else:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)

                if _is_on_month(dts.month, compare_month, modby) and dts.day == 1:
                    out[i] = 1

    elif field in ["is_month_end", "is_quarter_end", "is_year_end"]:
        if is_business:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)

                if _is_on_month(dts.month, compare_month, modby) and (
                        dts.day == get_lastbday(dts.year, dts.month)):
                    out[i] = 1

        else:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = 0
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)

                if _is_on_month(dts.month, compare_month, modby) and (
                        dts.day == get_days_in_month(dts.year, dts.month)):
                    out[i] = 1

    else:
        raise ValueError(f"Field {field} not supported")

    return out.view(bool)


@cython.wraparound(False)
@cython.boundscheck(False)
def get_date_field(const int64_t[:] dtindex, str field):
    """
    Given a int64-based datetime index, extract the year, month, etc.,
    field and return an array of these values.
    """
    cdef:
        Py_ssize_t i, count = len(dtindex)
        ndarray[int32_t] out
        npy_datetimestruct dts

    out = np.empty(count, dtype='i4')

    if field == 'Y':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dts.year
        return out

    elif field == 'M':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dts.month
        return out

    elif field == 'D':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dts.day
        return out

    elif field == 'h':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dts.hour
        return out

    elif field == 'm':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dts.min
        return out

    elif field == 's':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dts.sec
        return out

    elif field == 'us':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dts.us
        return out

    elif field == 'ns':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dts.ps // 1000
        return out
    elif field == 'doy':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = get_day_of_year(dts.year, dts.month, dts.day)
        return out

    elif field == 'dow':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dayofweek(dts.year, dts.month, dts.day)
        return out

    elif field == 'woy':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = get_week_of_year(dts.year, dts.month, dts.day)
        return out

    elif field == 'q':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = dts.month
                out[i] = ((out[i] - 1) // 3) + 1
        return out

    elif field == 'dim':
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                out[i] = get_days_in_month(dts.year, dts.month)
        return out
    elif field == 'is_leap_year':
        return isleapyear_arr(get_date_field(dtindex, 'Y'))

    raise ValueError(f"Field {field} not supported")


@cython.wraparound(False)
@cython.boundscheck(False)
def get_timedelta_field(const int64_t[:] tdindex, str field):
    """
    Given a int64-based timedelta index, extract the days, hrs, sec.,
    field and return an array of these values.
    """
    cdef:
        Py_ssize_t i, count = len(tdindex)
        ndarray[int32_t] out
        pandas_timedeltastruct tds

    out = np.empty(count, dtype='i4')

    if field == 'days':
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                td64_to_tdstruct(tdindex[i], &tds)
                out[i] = tds.days
        return out

    elif field == 'h':
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                td64_to_tdstruct(tdindex[i], &tds)
                out[i] = tds.hrs
        return out

    elif field == 's':
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                td64_to_tdstruct(tdindex[i], &tds)
                out[i] = tds.sec
        return out

    elif field == 'seconds':
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                td64_to_tdstruct(tdindex[i], &tds)
                out[i] = tds.seconds
        return out

    elif field == 'ms':
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                td64_to_tdstruct(tdindex[i], &tds)
                out[i] = tds.ms
        return out

    elif field == 'microseconds':
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                td64_to_tdstruct(tdindex[i], &tds)
                out[i] = tds.microseconds
        return out

    elif field == 'us':
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                td64_to_tdstruct(tdindex[i], &tds)
                out[i] = tds.us
        return out

    elif field == 'ns':
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                td64_to_tdstruct(tdindex[i], &tds)
                out[i] = tds.ns
        return out

    elif field == 'nanoseconds':
        with nogil:
            for i in range(count):
                if tdindex[i] == NPY_NAT:
                    out[i] = -1
                    continue

                td64_to_tdstruct(tdindex[i], &tds)
                out[i] = tds.nanoseconds
        return out

    raise ValueError(f"Field {field} not supported")


cpdef isleapyear_arr(ndarray years):
    """vectorized version of isleapyear; NaT evaluates as False"""
    cdef:
        ndarray[int8_t] out

    out = np.zeros(len(years), dtype='int8')
    out[np.logical_or(years % 400 == 0,
                      np.logical_and(years % 4 == 0,
                                     years % 100 > 0))] = 1
    return out.view(bool)


@cython.wraparound(False)
@cython.boundscheck(False)
def build_isocalendar_sarray(const int64_t[:] dtindex):
    """
    Given a int64-based datetime array, return the ISO 8601 year, week, and day
    as a structured array.
    """
    cdef:
        Py_ssize_t i, count = len(dtindex)
        npy_datetimestruct dts
        ndarray[uint32_t] iso_years, iso_weeks, days
        iso_calendar_t ret_val

    sa_dtype = [
        ("year", "u4"),
        ("week", "u4"),
        ("day", "u4"),
    ]

    out = np.empty(count, dtype=sa_dtype)

    iso_years = out["year"]
    iso_weeks = out["week"]
    days = out["day"]

    with nogil:
        for i in range(count):
            if dtindex[i] == NPY_NAT:
                ret_val = 0, 0, 0
            else:
                dt64_to_dtstruct(dtindex[i], &dts)
                ret_val = get_iso_calendar(dts.year, dts.month, dts.day)

            iso_years[i] = ret_val[0]
            iso_weeks[i] = ret_val[1]
            days[i] = ret_val[2]
    return out


def get_locale_names(name_type: str, locale: object = None):
    """
    Returns an array of localized day or month names.

    Parameters
    ----------
    name_type : str
        Attribute of LocaleTime() in which to return localized names.
    locale : str

    Returns
    -------
    list of locale names
    """
    with set_locale(locale, LC_TIME):
        return getattr(LocaleTime(), name_type)


# ---------------------------------------------------------------------
# Rounding


class RoundTo:
    """
    enumeration defining the available rounding modes

    Attributes
    ----------
    MINUS_INFTY
        round towards -∞, or floor [2]_
    PLUS_INFTY
        round towards +∞, or ceil [3]_
    NEAREST_HALF_EVEN
        round to nearest, tie-break half to even [6]_
    NEAREST_HALF_MINUS_INFTY
        round to nearest, tie-break half to -∞ [5]_
    NEAREST_HALF_PLUS_INFTY
        round to nearest, tie-break half to +∞ [4]_


    References
    ----------
    .. [1] "Rounding - Wikipedia"
           https://en.wikipedia.org/wiki/Rounding
    .. [2] "Rounding down"
           https://en.wikipedia.org/wiki/Rounding#Rounding_down
    .. [3] "Rounding up"
           https://en.wikipedia.org/wiki/Rounding#Rounding_up
    .. [4] "Round half up"
           https://en.wikipedia.org/wiki/Rounding#Round_half_up
    .. [5] "Round half down"
           https://en.wikipedia.org/wiki/Rounding#Round_half_down
    .. [6] "Round half to even"
           https://en.wikipedia.org/wiki/Rounding#Round_half_to_even
    """
    @property
    def MINUS_INFTY(self) -> int:
        return 0

    @property
    def PLUS_INFTY(self) -> int:
        return 1

    @property
    def NEAREST_HALF_EVEN(self) -> int:
        return 2

    @property
    def NEAREST_HALF_PLUS_INFTY(self) -> int:
        return 3

    @property
    def NEAREST_HALF_MINUS_INFTY(self) -> int:
        return 4


cdef inline ndarray[int64_t] _floor_int64(int64_t[:] values, int64_t unit):
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        int64_t res, value

    with cython.overflowcheck(True):
        for i in range(n):
            value = values[i]
            if value == NPY_NAT:
                res = NPY_NAT
            else:
                res = value - value % unit
            result[i] = res

    return result


cdef inline ndarray[int64_t] _ceil_int64(int64_t[:] values, int64_t unit):
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int64_t] result = np.empty(n, dtype="i8")
        int64_t res, value

    with cython.overflowcheck(True):
        for i in range(n):
            value = values[i]

            if value == NPY_NAT:
                res = NPY_NAT
            else:
                remainder = value % unit
                if remainder == 0:
                    res = value
                else:
                    res = value + (unit - remainder)

            result[i] = res

    return result


cdef inline ndarray[int64_t] _rounddown_int64(values, int64_t unit):
    return _ceil_int64(values - unit // 2, unit)


cdef inline ndarray[int64_t] _roundup_int64(values, int64_t unit):
    return _floor_int64(values + unit // 2, unit)


def round_nsint64(values: np.ndarray, mode: RoundTo, nanos: int) -> np.ndarray:
    """
    Applies rounding mode at given frequency

    Parameters
    ----------
    values : np.ndarray[int64_t]`
    mode : instance of `RoundTo` enumeration
    nanos : np.int64
        Freq to round to, expressed in nanoseconds

    Returns
    -------
    np.ndarray[int64_t]
    """
    cdef:
        int64_t unit = nanos

    if mode == RoundTo.MINUS_INFTY:
        return _floor_int64(values, unit)
    elif mode == RoundTo.PLUS_INFTY:
        return _ceil_int64(values, unit)
    elif mode == RoundTo.NEAREST_HALF_MINUS_INFTY:
        return _rounddown_int64(values, unit)
    elif mode == RoundTo.NEAREST_HALF_PLUS_INFTY:
        return _roundup_int64(values, unit)
    elif mode == RoundTo.NEAREST_HALF_EVEN:
        # for odd unit there is no need of a tie break
        if unit % 2:
            return _rounddown_int64(values, unit)
        quotient, remainder = np.divmod(values, unit)
        mask = np.logical_or(
            remainder > (unit // 2),
            np.logical_and(remainder == (unit // 2), quotient % 2)
        )
        quotient[mask] += 1
        return quotient * unit

    # if/elif above should catch all rounding modes defined in enum 'RoundTo':
    # if flow of control arrives here, it is a bug
    raise ValueError("round_nsint64 called with an unrecognized rounding mode")
