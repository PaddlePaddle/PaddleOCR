# cython: boundscheck=False
"""
Cython implementations of functions resembling the stdlib calendar module
"""

import cython

from numpy cimport (
    int32_t,
    int64_t,
)

# ----------------------------------------------------------------------
# Constants

# Slightly more performant cython lookups than a 2D table
# The first 12 entries correspond to month lengths for non-leap years.
# The remaining 12 entries give month lengths for leap years
cdef int32_t* days_per_month_array = [
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

cdef int* sakamoto_arr = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]

# The first 13 entries give the month days elapsed as of the first of month N
# (or the total number of days in the year for N=13) in non-leap years.
# The remaining 13 entries give the days elapsed in leap years.
cdef int32_t* month_offset = [
    0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365,
    0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]

# Canonical location for other modules to find name constants
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
          'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
# The first blank line is consistent with calendar.month_name in the calendar
# standard library
MONTHS_FULL = ['', 'January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November',
               'December']
MONTH_NUMBERS = {name: num for num, name in enumerate(MONTHS)}
cdef dict c_MONTH_NUMBERS = MONTH_NUMBERS
MONTH_ALIASES = {(num + 1): name for num, name in enumerate(MONTHS)}
MONTH_TO_CAL_NUM = {name: num + 1 for num, name in enumerate(MONTHS)}

DAYS = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
DAYS_FULL = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
             'Saturday', 'Sunday']
int_to_weekday = {num: name for num, name in enumerate(DAYS)}
weekday_to_int = {int_to_weekday[key]: key for key in int_to_weekday}

DAY_SECONDS = 86400
HOUR_SECONDS = 3600

cdef int64_t DAY_NANOS = DAY_SECONDS * 1_000_000_000
cdef int64_t HOUR_NANOS = HOUR_SECONDS * 1_000_000_000

# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int32_t get_days_in_month(int year, Py_ssize_t month) nogil:
    """
    Return the number of days in the given month of the given year.

    Parameters
    ----------
    year : int
    month : int

    Returns
    -------
    days_in_month : int

    Notes
    -----
    Assumes that the arguments are valid.  Passing a month not between 1 and 12
    risks a segfault.
    """
    return days_per_month_array[12 * is_leapyear(year) + month - 1]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision
cdef int dayofweek(int y, int m, int d) nogil:
    """
    Find the day of week for the date described by the Y/M/D triple y, m, d
    using Sakamoto's method, from wikipedia.

    0 represents Monday.  See [1]_.

    Parameters
    ----------
    y : int
    m : int
    d : int

    Returns
    -------
    weekday : int

    Notes
    -----
    Assumes that y, m, d, represents a valid date.

    See Also
    --------
    [1] https://docs.python.org/3/library/calendar.html#calendar.weekday

    [2] https://en.wikipedia.org/wiki/\
    Determination_of_the_day_of_the_week#Sakamoto.27s_methods
    """
    cdef:
        int day

    y -= m < 3
    day = (y + y / 4 - y / 100 + y / 400 + sakamoto_arr[m - 1] + d) % 7
    # convert to python day
    return (day + 6) % 7


cdef bint is_leapyear(int64_t year) nogil:
    """
    Returns 1 if the given year is a leap year, 0 otherwise.

    Parameters
    ----------
    year : int

    Returns
    -------
    is_leap : bool
    """
    return ((year & 0x3) == 0 and  # year % 4 == 0
            ((year % 100) != 0 or (year % 400) == 0))


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int32_t get_week_of_year(int year, int month, int day) nogil:
    """
    Return the ordinal week-of-year for the given day.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    week_of_year : int32_t

    Notes
    -----
    Assumes the inputs describe a valid date.
    """
    return get_iso_calendar(year, month, day)[1]


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef iso_calendar_t get_iso_calendar(int year, int month, int day) nogil:
    """
    Return the year, week, and day of year corresponding to ISO 8601

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    year : int32_t
    week : int32_t
    day : int32_t

    Notes
    -----
    Assumes the inputs describe a valid date.
    """
    cdef:
        int32_t doy, dow
        int32_t iso_year, iso_week

    doy = get_day_of_year(year, month, day)
    dow = dayofweek(year, month, day)

    # estimate
    iso_week = (doy - 1) - dow + 3
    if iso_week >= 0:
        iso_week = iso_week // 7 + 1

    # verify
    if iso_week < 0:
        if (iso_week > -2) or (iso_week == -2 and is_leapyear(year - 1)):
            iso_week = 53
        else:
            iso_week = 52
    elif iso_week == 53:
        if 31 - day + dow < 3:
            iso_week = 1

    iso_year = year
    if iso_week == 1 and month == 12:
        iso_year += 1

    elif iso_week >= 52 and month == 1:
        iso_year -= 1

    return iso_year, iso_week, dow + 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int32_t get_day_of_year(int year, int month, int day) nogil:
    """
    Return the ordinal day-of-year for the given day.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    day_of_year : int32_t

    Notes
    -----
    Assumes the inputs describe a valid date.
    """
    cdef:
        bint isleap
        int32_t mo_off
        int day_of_year

    isleap = is_leapyear(year)

    mo_off = month_offset[isleap * 13 + month - 1]

    day_of_year = mo_off + day
    return day_of_year


# ---------------------------------------------------------------------
# Business Helpers

cpdef int get_lastbday(int year, int month) nogil:
    """
    Find the last day of the month that is a business day.

    Parameters
    ----------
    year : int
    month : int

    Returns
    -------
    last_bday : int
    """
    cdef:
        int wkday, days_in_month

    wkday = dayofweek(year, month, 1)
    days_in_month = get_days_in_month(year, month)
    return days_in_month - max(((wkday + days_in_month - 1) % 7) - 4, 0)


cpdef int get_firstbday(int year, int month) nogil:
    """
    Find the first day of the month that is a business day.

    Parameters
    ----------
    year : int
    month : int

    Returns
    -------
    first_bday : int
    """
    cdef:
        int first, wkday

    wkday = dayofweek(year, month, 1)
    first = 1
    if wkday == 5:  # on Saturday
        first = 3
    elif wkday == 6:  # on Sunday
        first = 2
    return first
