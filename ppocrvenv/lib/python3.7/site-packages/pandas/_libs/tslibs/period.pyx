import warnings

cimport numpy as cnp
from cpython.object cimport (
    Py_EQ,
    Py_NE,
    PyObject_RichCompareBool,
)
from numpy cimport (
    int64_t,
    ndarray,
)

import numpy as np

cnp.import_array()

from libc.stdlib cimport (
    free,
    malloc,
)
from libc.string cimport (
    memset,
    strlen,
)
from libc.time cimport (
    strftime,
    tm,
)

import cython

from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDateTime_IMPORT,
    PyDelta_Check,
    datetime,
)

# import datetime C API
PyDateTime_IMPORT

from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_D,
    NPY_FR_us,
    check_dts_bounds,
    dt64_to_dtstruct,
    dtstruct_to_dt64,
    npy_datetimestruct,
    pandas_datetime_to_datetimestruct,
)


cdef extern from "src/datetime/np_datetime.h":
    int64_t npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT fr,
                                           npy_datetimestruct *d) nogil

cimport pandas._libs.tslibs.util as util

from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp

from pandas._libs.tslibs.ccalendar cimport (
    c_MONTH_NUMBERS,
    dayofweek,
    get_day_of_year,
    get_days_in_month,
    get_week_of_year,
    is_leapyear,
)
from pandas._libs.tslibs.timedeltas cimport (
    delta_to_nanoseconds,
    is_any_td_scalar,
)

from pandas._libs.tslibs.conversion import ensure_datetime64ns

from pandas._libs.tslibs.dtypes cimport (
    FR_ANN,
    FR_BUS,
    FR_DAY,
    FR_HR,
    FR_MIN,
    FR_MS,
    FR_MTH,
    FR_NS,
    FR_QTR,
    FR_SEC,
    FR_UND,
    FR_US,
    FR_WK,
    PeriodDtypeBase,
    attrname_to_abbrevs,
)
from pandas._libs.tslibs.parsing cimport quarter_to_myear

from pandas._libs.tslibs.parsing import parse_time_string

from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    _nat_scalar_rules,
    c_NaT as NaT,
    c_nat_strings as nat_strings,
    is_null_datetimelike,
)
from pandas._libs.tslibs.offsets cimport (
    BaseOffset,
    is_offset_object,
    is_tick_object,
    to_offset,
)

from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG

cdef:
    enum:
        INT32_MIN = -2_147_483_648LL


ctypedef struct asfreq_info:
    int64_t intraday_conversion_factor
    int is_end
    int to_end
    int from_end

ctypedef int64_t (*freq_conv_func)(int64_t, asfreq_info*) nogil


cdef extern from *:
    """
    // must use npy typedef b/c int64_t is aliased in cython-generated c
    // unclear why we need LL for that row.
    // see https://github.com/pandas-dev/pandas/pull/34416/
    static npy_int64 daytime_conversion_factor_matrix[7][7] = {
        {1, 24, 1440, 86400, 86400000, 86400000000, 86400000000000},
        {0LL,  1LL,   60LL,  3600LL,  3600000LL,  3600000000LL,  3600000000000LL},
        {0,  0,   1,     60,    60000,    60000000,    60000000000},
        {0,  0,   0,      1,     1000,     1000000,     1000000000},
        {0,  0,   0,      0,        1,        1000,        1000000},
        {0,  0,   0,      0,        0,           1,           1000},
        {0,  0,   0,      0,        0,           0,              1}};
    """
    int64_t daytime_conversion_factor_matrix[7][7]


cdef int max_value(int left, int right) nogil:
    if left > right:
        return left
    return right


cdef int min_value(int left, int right) nogil:
    if left < right:
        return left
    return right


cdef int64_t get_daytime_conversion_factor(int from_index, int to_index) nogil:
    cdef:
        int row = min_value(from_index, to_index)
        int col = max_value(from_index, to_index)
    # row or col < 6 means frequency strictly lower than Daily, which
    # do not use daytime_conversion_factors
    if row < 6:
        return 0
    elif col < 6:
        return 0
    return daytime_conversion_factor_matrix[row - 6][col - 6]


cdef int64_t nofunc(int64_t ordinal, asfreq_info *af_info) nogil:
    return INT32_MIN


cdef int64_t no_op(int64_t ordinal, asfreq_info *af_info) nogil:
    return ordinal


cdef freq_conv_func get_asfreq_func(int from_freq, int to_freq) nogil:
    cdef:
        int from_group = get_freq_group(from_freq)
        int to_group = get_freq_group(to_freq)

    if from_group == FR_UND:
        from_group = FR_DAY

    if from_group == FR_BUS:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_BtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_BtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_BtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_BtoW
        elif to_group == FR_BUS:
            return <freq_conv_func>no_op
        elif to_group  in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_BtoDT
        else:
            return <freq_conv_func>nofunc

    elif to_group == FR_BUS:
        if from_group == FR_ANN:
            return <freq_conv_func>asfreq_AtoB
        elif from_group == FR_QTR:
            return <freq_conv_func>asfreq_QtoB
        elif from_group == FR_MTH:
            return <freq_conv_func>asfreq_MtoB
        elif from_group == FR_WK:
            return <freq_conv_func>asfreq_WtoB
        elif from_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_DTtoB
        else:
            return <freq_conv_func>nofunc

    elif from_group == FR_ANN:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_AtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_AtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_AtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_AtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_AtoDT
        else:
            return <freq_conv_func>nofunc

    elif from_group == FR_QTR:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_QtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_QtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_QtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_QtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_QtoDT
        else:
            return <freq_conv_func>nofunc

    elif from_group == FR_MTH:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_MtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_MtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>no_op
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_MtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_MtoDT
        else:
            return <freq_conv_func>nofunc

    elif from_group == FR_WK:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_WtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_WtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_WtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_WtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            return <freq_conv_func>asfreq_WtoDT
        else:
            return <freq_conv_func>nofunc

    elif from_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
        if to_group == FR_ANN:
            return <freq_conv_func>asfreq_DTtoA
        elif to_group == FR_QTR:
            return <freq_conv_func>asfreq_DTtoQ
        elif to_group == FR_MTH:
            return <freq_conv_func>asfreq_DTtoM
        elif to_group == FR_WK:
            return <freq_conv_func>asfreq_DTtoW
        elif to_group in [FR_DAY, FR_HR, FR_MIN, FR_SEC, FR_MS, FR_US, FR_NS]:
            if from_group > to_group:
                return <freq_conv_func>downsample_daytime
            else:
                return <freq_conv_func>upsample_daytime

        else:
            return <freq_conv_func>nofunc

    else:
        return <freq_conv_func>nofunc


# --------------------------------------------------------------------
# Frequency Conversion Helpers

cdef int64_t DtoB_weekday(int64_t unix_date) nogil:
    return ((unix_date + 4) // 7) * 5 + ((unix_date + 4) % 7) - 4


cdef int64_t DtoB(npy_datetimestruct *dts, int roll_back,
                  int64_t unix_date) nogil:
    # calculate the current week (counting from 1970-01-01) treating
    # sunday as last day of a week
    cdef:
        int day_of_week = dayofweek(dts.year, dts.month, dts.day)

    if roll_back == 1:
        if day_of_week > 4:
            # change to friday before weekend
            unix_date -= (day_of_week - 4)
    else:
        if day_of_week > 4:
            # change to Monday after weekend
            unix_date += (7 - day_of_week)

    return DtoB_weekday(unix_date)


cdef inline int64_t upsample_daytime(int64_t ordinal, asfreq_info *af_info) nogil:
    if af_info.is_end:
        return (ordinal + 1) * af_info.intraday_conversion_factor - 1
    else:
        return ordinal * af_info.intraday_conversion_factor


cdef inline int64_t downsample_daytime(int64_t ordinal, asfreq_info *af_info) nogil:
    return ordinal // af_info.intraday_conversion_factor


cdef inline int64_t transform_via_day(int64_t ordinal,
                                      asfreq_info *af_info,
                                      freq_conv_func first_func,
                                      freq_conv_func second_func) nogil:
    cdef:
        int64_t result

    result = first_func(ordinal, af_info)
    result = second_func(result, af_info)
    return result


# --------------------------------------------------------------------
# Conversion _to_ Daily Freq

cdef int64_t asfreq_AtoDT(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        int64_t unix_date
        npy_datetimestruct dts

    ordinal += af_info.is_end

    dts.year = ordinal + 1970
    dts.month = 1
    adjust_dts_for_month(&dts, af_info.from_end)

    unix_date = unix_date_from_ymd(dts.year, dts.month, 1)
    unix_date -= af_info.is_end
    return upsample_daytime(unix_date, af_info)


cdef int64_t asfreq_QtoDT(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        int64_t unix_date
        npy_datetimestruct dts

    ordinal += af_info.is_end

    dts.year = ordinal // 4 + 1970
    dts.month = (ordinal % 4) * 3 + 1
    adjust_dts_for_month(&dts, af_info.from_end)

    unix_date = unix_date_from_ymd(dts.year, dts.month, 1)
    unix_date -= af_info.is_end
    return upsample_daytime(unix_date, af_info)


cdef int64_t asfreq_MtoDT(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        int64_t unix_date
        int year, month

    ordinal += af_info.is_end

    year = ordinal // 12 + 1970
    month = ordinal % 12 + 1

    unix_date = unix_date_from_ymd(year, month, 1)
    unix_date -= af_info.is_end
    return upsample_daytime(unix_date, af_info)


cdef int64_t asfreq_WtoDT(int64_t ordinal, asfreq_info *af_info) nogil:
    ordinal = (ordinal * 7 + af_info.from_end - 4 +
               (7 - 1) * (af_info.is_end - 1))
    return upsample_daytime(ordinal, af_info)


# --------------------------------------------------------------------
# Conversion _to_ BusinessDay Freq

cdef int64_t asfreq_AtoB(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = asfreq_AtoDT(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    roll_back = af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


cdef int64_t asfreq_QtoB(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = asfreq_QtoDT(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    roll_back = af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


cdef int64_t asfreq_MtoB(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = asfreq_MtoDT(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    roll_back = af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


cdef int64_t asfreq_WtoB(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = asfreq_WtoDT(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    roll_back = af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


cdef int64_t asfreq_DTtoB(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        int roll_back
        npy_datetimestruct dts
        int64_t unix_date = downsample_daytime(ordinal, af_info)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, &dts)
    # This usage defines roll_back the opposite way from the others
    roll_back = 1 - af_info.is_end
    return DtoB(&dts, roll_back, unix_date)


# ----------------------------------------------------------------------
# Conversion _from_ Daily Freq

cdef int64_t asfreq_DTtoA(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        npy_datetimestruct dts

    ordinal = downsample_daytime(ordinal, af_info)
    pandas_datetime_to_datetimestruct(ordinal, NPY_FR_D, &dts)
    return dts_to_year_ordinal(&dts, af_info.to_end)


cdef int DtoQ_yq(int64_t ordinal, asfreq_info *af_info, npy_datetimestruct* dts) nogil:
    cdef:
        int quarter

    pandas_datetime_to_datetimestruct(ordinal, NPY_FR_D, dts)
    adjust_dts_for_qtr(dts, af_info.to_end)

    quarter = month_to_quarter(dts.month)
    return quarter


cdef int64_t asfreq_DTtoQ(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        int quarter
        npy_datetimestruct dts

    ordinal = downsample_daytime(ordinal, af_info)

    quarter = DtoQ_yq(ordinal, af_info, &dts)
    return <int64_t>((dts.year - 1970) * 4 + quarter - 1)


cdef int64_t asfreq_DTtoM(int64_t ordinal, asfreq_info *af_info) nogil:
    cdef:
        npy_datetimestruct dts

    ordinal = downsample_daytime(ordinal, af_info)
    pandas_datetime_to_datetimestruct(ordinal, NPY_FR_D, &dts)
    return dts_to_month_ordinal(&dts)


cdef int64_t asfreq_DTtoW(int64_t ordinal, asfreq_info *af_info) nogil:
    ordinal = downsample_daytime(ordinal, af_info)
    return unix_date_to_week(ordinal, af_info.to_end)


cdef int64_t unix_date_to_week(int64_t unix_date, int to_end) nogil:
    return (unix_date + 3 - to_end) // 7 + 1


# --------------------------------------------------------------------
# Conversion _from_ BusinessDay Freq

cdef int64_t asfreq_BtoDT(int64_t ordinal, asfreq_info *af_info) nogil:
    ordinal = ((ordinal + 3) // 5) * 7 + (ordinal + 3) % 5 - 3
    return upsample_daytime(ordinal, af_info)


cdef int64_t asfreq_BtoA(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_BtoQ(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_BtoM(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoM)


cdef int64_t asfreq_BtoW(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_BtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Annual Freq

cdef int64_t asfreq_AtoA(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_AtoQ(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_AtoM(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoM)


cdef int64_t asfreq_AtoW(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_AtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Quarterly Freq

cdef int64_t asfreq_QtoQ(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_QtoA(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_QtoM(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoM)


cdef int64_t asfreq_QtoW(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_QtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Monthly Freq

cdef int64_t asfreq_MtoA(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_MtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_MtoQ(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_MtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_MtoW(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_MtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------
# Conversion _from_ Weekly Freq

cdef int64_t asfreq_WtoA(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoA)


cdef int64_t asfreq_WtoQ(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoQ)


cdef int64_t asfreq_WtoM(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoM)


cdef int64_t asfreq_WtoW(int64_t ordinal, asfreq_info *af_info) nogil:
    return transform_via_day(ordinal, af_info,
                             <freq_conv_func>asfreq_WtoDT,
                             <freq_conv_func>asfreq_DTtoW)


# ----------------------------------------------------------------------

@cython.cdivision
cdef char* c_strftime(npy_datetimestruct *dts, char *fmt):
    """
    Generate a nice string representation of the period
    object, originally from DateObject_strftime

    Parameters
    ----------
    dts : npy_datetimestruct*
    fmt : char*

    Returns
    -------
    result : char*
    """
    cdef:
        tm c_date
        char *result
        int result_len = strlen(fmt) + 50

    c_date.tm_sec = dts.sec
    c_date.tm_min = dts.min
    c_date.tm_hour = dts.hour
    c_date.tm_mday = dts.day
    c_date.tm_mon = dts.month - 1
    c_date.tm_year = dts.year - 1900
    c_date.tm_wday = (dayofweek(dts.year, dts.month, dts.day) + 1) % 7
    c_date.tm_yday = get_day_of_year(dts.year, dts.month, dts.day) - 1
    c_date.tm_isdst = -1

    result = <char*>malloc(result_len * sizeof(char))

    strftime(result, result_len, fmt, &c_date)

    return result


# ----------------------------------------------------------------------
# Conversion between date_info and npy_datetimestruct

cdef inline int get_freq_group(int freq) nogil:
    # Note: this is equivalent to libfrequencies.get_freq_group, specialized
    #  to integer argument.
    return (freq // 1000) * 1000


cdef inline int get_freq_group_index(int freq) nogil:
    return freq // 1000


cdef void adjust_dts_for_month(npy_datetimestruct* dts, int from_end) nogil:
    if from_end != 12:
        dts.month += from_end
        if dts.month > 12:
            dts.month -= 12
        else:
            dts.year -= 1


cdef void adjust_dts_for_qtr(npy_datetimestruct* dts, int to_end) nogil:
    if to_end != 12:
        dts.month -= to_end
        if dts.month <= 0:
            dts.month += 12
        else:
            dts.year += 1


# Find the unix_date (days elapsed since datetime(1970, 1, 1)
# for the given year/month/day.
# Assumes GREGORIAN_CALENDAR */
cdef int64_t unix_date_from_ymd(int year, int month, int day) nogil:
    # Calculate the absolute date
    cdef:
        npy_datetimestruct dts
        int64_t unix_date

    memset(&dts, 0, sizeof(npy_datetimestruct))
    dts.year = year
    dts.month = month
    dts.day = day
    unix_date = npy_datetimestruct_to_datetime(NPY_FR_D, &dts)
    return unix_date


cdef inline int64_t dts_to_month_ordinal(npy_datetimestruct* dts) nogil:
    # AKA: use npy_datetimestruct_to_datetime(NPY_FR_M, &dts)
    return <int64_t>((dts.year - 1970) * 12 + dts.month - 1)


cdef inline int64_t dts_to_year_ordinal(npy_datetimestruct *dts, int to_end) nogil:
    cdef:
        int64_t result

    result = npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT.NPY_FR_Y, dts)
    if dts.month > to_end:
        return result + 1
    else:
        return result


cdef inline int64_t dts_to_qtr_ordinal(npy_datetimestruct* dts, int to_end) nogil:
    cdef:
        int quarter

    adjust_dts_for_qtr(dts, to_end)
    quarter = month_to_quarter(dts.month)
    return <int64_t>((dts.year - 1970) * 4 + quarter - 1)


cdef inline int get_anchor_month(int freq, int freq_group) nogil:
    cdef:
        int fmonth
    fmonth = freq - freq_group
    if fmonth == 0:
        fmonth = 12
    return fmonth


# specifically _dont_ use cdvision or else ordinals near -1 are assigned to
# incorrect dates GH#19643
@cython.cdivision(False)
cdef int64_t get_period_ordinal(npy_datetimestruct *dts, int freq) nogil:
    """
    Generate an ordinal in period space

    Parameters
    ----------
    dts: npy_datetimestruct*
    freq : int

    Returns
    -------
    period_ordinal : int64_t
    """
    cdef:
        int64_t unix_date
        int freq_group, fmonth, mdiff
        NPY_DATETIMEUNIT unit

    freq_group = get_freq_group(freq)

    if freq_group == FR_ANN:
        fmonth = get_anchor_month(freq, freq_group)
        return dts_to_year_ordinal(dts, fmonth)

    elif freq_group == FR_QTR:
        fmonth = get_anchor_month(freq, freq_group)
        return dts_to_qtr_ordinal(dts, fmonth)

    elif freq_group == FR_WK:
        unix_date = npy_datetimestruct_to_datetime(NPY_FR_D, dts)
        return unix_date_to_week(unix_date, freq - FR_WK)

    elif freq == FR_BUS:
        unix_date = npy_datetimestruct_to_datetime(NPY_FR_D, dts)
        return DtoB(dts, 0, unix_date)

    unit = get_unit(freq)
    return npy_datetimestruct_to_datetime(unit, dts)


cdef NPY_DATETIMEUNIT get_unit(int freq) nogil:
    """
    Convert the freq to the corresponding NPY_DATETIMEUNIT to pass
    to npy_datetimestruct_to_datetime.
    """
    if freq == FR_MTH:
        return NPY_DATETIMEUNIT.NPY_FR_M
    elif freq == FR_DAY:
        return NPY_DATETIMEUNIT.NPY_FR_D
    elif freq == FR_HR:
        return NPY_DATETIMEUNIT.NPY_FR_h
    elif freq == FR_MIN:
        return NPY_DATETIMEUNIT.NPY_FR_m
    elif freq == FR_SEC:
        return NPY_DATETIMEUNIT.NPY_FR_s
    elif freq == FR_MS:
        return NPY_DATETIMEUNIT.NPY_FR_ms
    elif freq == FR_US:
        return NPY_DATETIMEUNIT.NPY_FR_us
    elif freq == FR_NS:
        return NPY_DATETIMEUNIT.NPY_FR_ns
    elif freq == FR_UND:
        # Default to Day
        return NPY_DATETIMEUNIT.NPY_FR_D


cdef void get_date_info(int64_t ordinal, int freq, npy_datetimestruct *dts) nogil:
    cdef:
        int64_t unix_date, nanos
        npy_datetimestruct dts2

    unix_date = get_unix_date(ordinal, freq)
    nanos = get_time_nanos(freq, unix_date, ordinal)

    pandas_datetime_to_datetimestruct(unix_date, NPY_FR_D, dts)

    dt64_to_dtstruct(nanos, &dts2)
    dts.hour = dts2.hour
    dts.min = dts2.min
    dts.sec = dts2.sec
    dts.us = dts2.us
    dts.ps = dts2.ps


cdef int64_t get_unix_date(int64_t period_ordinal, int freq) nogil:
    """
    Returns the proleptic Gregorian ordinal of the date, as an integer.
    This corresponds to the number of days since Jan., 1st, 1970 AD.
    When the instance has a frequency less than daily, the proleptic date
    is calculated for the last day of the period.

    Parameters
    ----------
    period_ordinal : int64_t
    freq : int

    Returns
    -------
    unix_date : int64_t number of days since datetime(1970, 1, 1)
    """
    cdef:
        asfreq_info af_info
        freq_conv_func toDaily = NULL

    if freq == FR_DAY:
        return period_ordinal

    toDaily = get_asfreq_func(freq, FR_DAY)
    get_asfreq_info(freq, FR_DAY, True, &af_info)
    return toDaily(period_ordinal, &af_info)


@cython.cdivision
cdef int64_t get_time_nanos(int freq, int64_t unix_date, int64_t ordinal) nogil:
    """
    Find the number of nanoseconds after midnight on the given unix_date
    that the ordinal represents in the given frequency.

    Parameters
    ----------
    freq : int
    unix_date : int64_t
    ordinal : int64_t

    Returns
    -------
    int64_t
    """
    cdef:
        int64_t sub, factor
        int64_t nanos_in_day = 24 * 3600 * 10**9

    freq = get_freq_group(freq)

    if freq <= FR_DAY:
        return 0

    elif freq == FR_NS:
        factor = 1

    elif freq == FR_US:
        factor = 10**3

    elif freq == FR_MS:
        factor = 10**6

    elif freq == FR_SEC:
        factor = 10 **9

    elif freq == FR_MIN:
        factor = 10**9 * 60

    else:
        # We must have freq == FR_HR
        factor = 10**9 * 3600

    sub = ordinal - unix_date * (nanos_in_day / factor)
    return sub * factor


cdef int get_yq(int64_t ordinal, int freq, npy_datetimestruct* dts):
    """
    Find the year and quarter of a Period with the given ordinal and frequency

    Parameters
    ----------
    ordinal : int64_t
    freq : int
    dts : *npy_datetimestruct

    Returns
    -------
    quarter : int
        describes the implied quarterly frequency associated with `freq`

    Notes
    -----
    Sets dts.year in-place.
    """
    cdef:
        asfreq_info af_info
        int qtr_freq
        int64_t unix_date
        int quarter

    unix_date = get_unix_date(ordinal, freq)

    if get_freq_group(freq) == FR_QTR:
        qtr_freq = freq
    else:
        qtr_freq = FR_QTR

    get_asfreq_info(FR_DAY, qtr_freq, True, &af_info)

    quarter = DtoQ_yq(unix_date, &af_info, dts)
    return quarter


cdef inline int month_to_quarter(int month) nogil:
    return (month - 1) // 3 + 1


# ----------------------------------------------------------------------
# Period logic

@cython.wraparound(False)
@cython.boundscheck(False)
def periodarr_to_dt64arr(const int64_t[:] periodarr, int freq):
    """
    Convert array to datetime64 values from a set of ordinals corresponding to
    periods per period convention.
    """
    cdef:
        int64_t[:] out
        Py_ssize_t i, l

    if freq < 6000:  # i.e. FR_DAY, hard-code to avoid need to cast
        l = len(periodarr)
        out = np.empty(l, dtype="i8")

        # We get here with freqs that do not correspond to a datetime64 unit
        for i in range(l):
            out[i] = period_ordinal_to_dt64(periodarr[i], freq)

        return out.base  # .base to access underlying np.ndarray

    else:
        # Short-circuit for performance
        if freq == FR_NS:
            return periodarr.base

        if freq == FR_US:
            dta = periodarr.base.view("M8[us]")
        elif freq == FR_MS:
            dta = periodarr.base.view("M8[ms]")
        elif freq == FR_SEC:
            dta = periodarr.base.view("M8[s]")
        elif freq == FR_MIN:
            dta = periodarr.base.view("M8[m]")
        elif freq == FR_HR:
            dta = periodarr.base.view("M8[h]")
        elif freq == FR_DAY:
            dta = periodarr.base.view("M8[D]")
        return ensure_datetime64ns(dta)


cdef void get_asfreq_info(int from_freq, int to_freq,
                          bint is_end, asfreq_info *af_info) nogil:
    """
    Construct the `asfreq_info` object used to convert an ordinal from
    `from_freq` to `to_freq`.

    Parameters
    ----------
    from_freq : int
    to_freq int
    is_end : bool
    af_info : *asfreq_info
    """
    cdef:
        int from_group = get_freq_group(from_freq)
        int to_group = get_freq_group(to_freq)

    af_info.is_end = is_end

    af_info.intraday_conversion_factor = get_daytime_conversion_factor(
        get_freq_group_index(max_value(from_group, FR_DAY)),
        get_freq_group_index(max_value(to_group, FR_DAY)))

    if from_group == FR_WK:
        af_info.from_end = calc_week_end(from_freq, from_group)
    elif from_group == FR_ANN:
        af_info.from_end = calc_a_year_end(from_freq, from_group)
    elif from_group == FR_QTR:
        af_info.from_end = calc_a_year_end(from_freq, from_group)

    if to_group == FR_WK:
        af_info.to_end = calc_week_end(to_freq, to_group)
    elif to_group == FR_ANN:
        af_info.to_end = calc_a_year_end(to_freq, to_group)
    elif to_group == FR_QTR:
        af_info.to_end = calc_a_year_end(to_freq, to_group)


@cython.cdivision
cdef int calc_a_year_end(int freq, int group) nogil:
    cdef:
        int result = (freq - group) % 12
    if result == 0:
        return 12
    else:
        return result


cdef inline int calc_week_end(int freq, int group) nogil:
    return freq - group


cpdef int64_t period_asfreq(int64_t ordinal, int freq1, int freq2, bint end):
    """
    Convert period ordinal from one frequency to another, and if upsampling,
    choose to use start ('S') or end ('E') of period.
    """
    cdef:
        int64_t retval

    _period_asfreq(&ordinal, &retval, 1, freq1, freq2, end)
    return retval


@cython.wraparound(False)
@cython.boundscheck(False)
def period_asfreq_arr(ndarray[int64_t] arr, int freq1, int freq2, bint end):
    """
    Convert int64-array of period ordinals from one frequency to another, and
    if upsampling, choose to use start ('S') or end ('E') of period.
    """
    cdef:
        Py_ssize_t n = len(arr)
        ndarray[int64_t] result = np.empty(n, dtype=np.int64)

    _period_asfreq(
        <int64_t*>cnp.PyArray_DATA(arr),
        <int64_t*>cnp.PyArray_DATA(result),
        n,
        freq1,
        freq2,
        end,
    )
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _period_asfreq(
    int64_t* ordinals,
    int64_t* out,
    Py_ssize_t length,
    int freq1,
    int freq2,
    bint end,
):
    """See period_asfreq.__doc__"""
    cdef:
        Py_ssize_t i
        freq_conv_func func
        asfreq_info af_info
        int64_t val

    if length == 1 and ordinals[0] == NPY_NAT:
        # fastpath avoid calling get_asfreq_func
        out[0] = NPY_NAT
        return

    func = get_asfreq_func(freq1, freq2)
    get_asfreq_info(freq1, freq2, end, &af_info)

    for i in range(length):
        val = ordinals[i]
        if val != NPY_NAT:
            val = func(val, &af_info)
        out[i] = val


cpdef int64_t period_ordinal(int y, int m, int d, int h, int min,
                             int s, int us, int ps, int freq):
    """
    Find the ordinal representation of the given datetime components at the
    frequency `freq`.

    Parameters
    ----------
    y : int
    m : int
    d : int
    h : int
    min : int
    s : int
    us : int
    ps : int

    Returns
    -------
    ordinal : int64_t
    """
    cdef:
        npy_datetimestruct dts
    dts.year = y
    dts.month = m
    dts.day = d
    dts.hour = h
    dts.min = min
    dts.sec = s
    dts.us = us
    dts.ps = ps
    return get_period_ordinal(&dts, freq)


cdef int64_t period_ordinal_to_dt64(int64_t ordinal, int freq) except? -1:
    cdef:
        npy_datetimestruct dts

    if ordinal == NPY_NAT:
        return NPY_NAT

    get_date_info(ordinal, freq, &dts)

    check_dts_bounds(&dts)
    return dtstruct_to_dt64(&dts)


cdef str period_format(int64_t value, int freq, object fmt=None):
    cdef:
        int freq_group

    if value == NPY_NAT:
        return "NaT"

    if isinstance(fmt, str):
        fmt = fmt.encode("utf-8")

    if fmt is None:
        freq_group = get_freq_group(freq)
        if freq_group == FR_ANN:
            fmt = b'%Y'
        elif freq_group == FR_QTR:
            fmt = b'%FQ%q'
        elif freq_group == FR_MTH:
            fmt = b'%Y-%m'
        elif freq_group == FR_WK:
            left = period_asfreq(value, freq, FR_DAY, 0)
            right = period_asfreq(value, freq, FR_DAY, 1)
            return f"{period_format(left, FR_DAY)}/{period_format(right, FR_DAY)}"
        elif freq_group == FR_BUS or freq_group == FR_DAY:
            fmt = b'%Y-%m-%d'
        elif freq_group == FR_HR:
            fmt = b'%Y-%m-%d %H:00'
        elif freq_group == FR_MIN:
            fmt = b'%Y-%m-%d %H:%M'
        elif freq_group == FR_SEC:
            fmt = b'%Y-%m-%d %H:%M:%S'
        elif freq_group == FR_MS:
            fmt = b'%Y-%m-%d %H:%M:%S.%l'
        elif freq_group == FR_US:
            fmt = b'%Y-%m-%d %H:%M:%S.%u'
        elif freq_group == FR_NS:
            fmt = b'%Y-%m-%d %H:%M:%S.%n'
        else:
            raise ValueError(f"Unknown freq: {freq}")

    return _period_strftime(value, freq, fmt)


cdef list extra_fmts = [(b"%q", b"^`AB`^"),
                        (b"%f", b"^`CD`^"),
                        (b"%F", b"^`EF`^"),
                        (b"%l", b"^`GH`^"),
                        (b"%u", b"^`IJ`^"),
                        (b"%n", b"^`KL`^")]

cdef list str_extra_fmts = ["^`AB`^", "^`CD`^", "^`EF`^",
                            "^`GH`^", "^`IJ`^", "^`KL`^"]

cdef str _period_strftime(int64_t value, int freq, bytes fmt):
    cdef:
        Py_ssize_t i
        npy_datetimestruct dts
        char *formatted
        bytes pat, brepl
        list found_pat = [False] * len(extra_fmts)
        int year, quarter
        str result, repl

    get_date_info(value, freq, &dts)
    for i in range(len(extra_fmts)):
        pat = extra_fmts[i][0]
        brepl = extra_fmts[i][1]
        if pat in fmt:
            fmt = fmt.replace(pat, brepl)
            found_pat[i] = True

    formatted = c_strftime(&dts, <char*>fmt)

    result = util.char_to_string(formatted)
    free(formatted)

    for i in range(len(extra_fmts)):
        if found_pat[i]:

            quarter = get_yq(value, freq, &dts)

            if i == 0:
                repl = str(quarter)
            elif i == 1:  # %f, 2-digit year
                repl = f"{(dts.year % 100):02d}"
            elif i == 2:
                repl = str(dts.year)
            elif i == 3:
                repl = f"{(value % 1_000):03d}"
            elif i == 4:
                repl = f"{(value % 1_000_000):06d}"
            elif i == 5:
                repl = f"{(value % 1_000_000_000):09d}"

            result = result.replace(str_extra_fmts[i], repl)

    return result


# ----------------------------------------------------------------------
# period accessors

ctypedef int (*accessor)(int64_t ordinal, int freq) except INT32_MIN


cdef int pyear(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.year


cdef int pqyear(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts

    get_yq(ordinal, freq, &dts)
    return dts.year


cdef int pquarter(int64_t ordinal, int freq):
    cdef:
        int quarter
        npy_datetimestruct dts
    quarter = get_yq(ordinal, freq, &dts)
    return quarter


cdef int pmonth(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.month


cdef int pday(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.day


cdef int pweekday(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dayofweek(dts.year, dts.month, dts.day)


cdef int pday_of_year(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return get_day_of_year(dts.year, dts.month, dts.day)


cdef int pweek(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return get_week_of_year(dts.year, dts.month, dts.day)


cdef int phour(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.hour


cdef int pminute(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return dts.min


cdef int psecond(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return <int>dts.sec


cdef int pdays_in_month(int64_t ordinal, int freq):
    cdef:
        npy_datetimestruct dts
    get_date_info(ordinal, freq, &dts)
    return get_days_in_month(dts.year, dts.month)


@cython.wraparound(False)
@cython.boundscheck(False)
def get_period_field_arr(str field, const int64_t[:] arr, int freq):
    cdef:
        Py_ssize_t i, sz
        int64_t[:] out
        accessor f

    func = _get_accessor_func(field)
    if func is NULL:
        raise ValueError(f"Unrecognized field name: {field}")

    sz = len(arr)
    out = np.empty(sz, dtype=np.int64)

    for i in range(sz):
        if arr[i] == NPY_NAT:
            out[i] = -1
            continue
        out[i] = func(arr[i], freq)

    return out.base  # .base to access underlying np.ndarray


cdef accessor _get_accessor_func(str field):
    if field == "year":
        return <accessor>pyear
    elif field == "qyear":
        return <accessor>pqyear
    elif field == "quarter":
        return <accessor>pquarter
    elif field == "month":
        return <accessor>pmonth
    elif field == "day":
        return <accessor>pday
    elif field == "hour":
        return <accessor>phour
    elif field == "minute":
        return <accessor>pminute
    elif field == "second":
        return <accessor>psecond
    elif field == "week":
        return <accessor>pweek
    elif field == "day_of_year":
        return <accessor>pday_of_year
    elif field == "weekday" or field == "day_of_week":
        return <accessor>pweekday
    elif field == "days_in_month":
        return <accessor>pdays_in_month
    return NULL


@cython.wraparound(False)
@cython.boundscheck(False)
def from_ordinals(const int64_t[:] values, freq):
    cdef:
        Py_ssize_t i, n = len(values)
        int64_t[:] result = np.empty(len(values), dtype="i8")
        int64_t val

    freq = to_offset(freq)
    if not isinstance(freq, BaseOffset):
        raise ValueError("freq not specified and cannot be inferred")

    for i in range(n):
        val = values[i]
        if val == NPY_NAT:
            result[i] = NPY_NAT
        else:
            result[i] = Period(val, freq=freq).ordinal

    return result.base


@cython.wraparound(False)
@cython.boundscheck(False)
def extract_ordinals(ndarray[object] values, freq) -> np.ndarray:
    # TODO: Change type to const object[:] when Cython supports that.

    cdef:
        Py_ssize_t i, n = len(values)
        int64_t[:] ordinals = np.empty(n, dtype=np.int64)
        object p

    freqstr = Period._maybe_convert_freq(freq).freqstr

    for i in range(n):
        p = values[i]

        if is_null_datetimelike(p):
            ordinals[i] = NPY_NAT
        elif util.is_integer_object(p):
            raise TypeError(p)
        else:
            try:
                ordinals[i] = p.ordinal

                if p.freqstr != freqstr:
                    msg = DIFFERENT_FREQ.format(cls="PeriodIndex",
                                                own_freq=freqstr,
                                                other_freq=p.freqstr)
                    raise IncompatibleFrequency(msg)

            except AttributeError:
                p = Period(p, freq=freq)
                if p is NaT:
                    # input may contain NaT-like string
                    ordinals[i] = NPY_NAT
                else:
                    ordinals[i] = p.ordinal

    return ordinals.base  # .base to access underlying np.ndarray


def extract_freq(ndarray[object] values) -> BaseOffset:
    # TODO: Change type to const object[:] when Cython supports that.

    cdef:
        Py_ssize_t i, n = len(values)
        object value

    for i in range(n):
        value = values[i]

        if is_period_object(value):
            return value.freq

    raise ValueError('freq not specified and cannot be inferred')

# -----------------------------------------------------------------------
# period helpers


DIFFERENT_FREQ = ("Input has different freq={other_freq} "
                  "from {cls}(freq={own_freq})")


class IncompatibleFrequency(ValueError):
    pass


cdef class PeriodMixin:
    # Methods shared between Period and PeriodArray

    cpdef int _get_to_timestamp_base(self):
        """
        Return frequency code group used for base of to_timestamp against
        frequency code.

        Return day freq code against longer freq than day.
        Return second freq code against hour between second.

        Returns
        -------
        int
        """
        base = self._dtype._dtype_code
        if base < FR_BUS:
            return FR_DAY
        elif FR_HR <= base <= FR_SEC:
            return FR_SEC
        return base

    @property
    def start_time(self) -> Timestamp:
        """
        Get the Timestamp for the start of the period.

        Returns
        -------
        Timestamp

        See Also
        --------
        Period.end_time : Return the end Timestamp.
        Period.dayofyear : Return the day of year.
        Period.daysinmonth : Return the days in that month.
        Period.dayofweek : Return the day of the week.

        Examples
        --------
        >>> period = pd.Period('2012-1-1', freq='D')
        >>> period
        Period('2012-01-01', 'D')

        >>> period.start_time
        Timestamp('2012-01-01 00:00:00')

        >>> period.end_time
        Timestamp('2012-01-01 23:59:59.999999999')
        """
        return self.to_timestamp(how="start")

    @property
    def end_time(self) -> Timestamp:
        return self.to_timestamp(how="end")

    def _require_matching_freq(self, other, base=False):
        # See also arrays.period.raise_on_incompatible
        if is_offset_object(other):
            other_freq = other
        else:
            other_freq = other.freq

        if base:
            condition = self.freq.base != other_freq.base
        else:
            condition = self.freq != other_freq

        if condition:
            msg = DIFFERENT_FREQ.format(
                cls=type(self).__name__,
                own_freq=self.freqstr,
                other_freq=other_freq.freqstr,
            )
            raise IncompatibleFrequency(msg)


cdef class _Period(PeriodMixin):

    cdef readonly:
        int64_t ordinal
        PeriodDtypeBase _dtype
        BaseOffset freq

    dayofweek = _Period.day_of_week
    dayofyear = _Period.day_of_year

    def __cinit__(self, int64_t ordinal, BaseOffset freq):
        self.ordinal = ordinal
        self.freq = freq
        # Note: this is more performant than PeriodDtype.from_date_offset(freq)
        #  because from_date_offset cannot be made a cdef method (until cython
        #  supported cdef classmethods)
        self._dtype = PeriodDtypeBase(freq._period_dtype_code)

    @classmethod
    def _maybe_convert_freq(cls, object freq) -> BaseOffset:
        """
        Internally we allow integer and tuple representations (for now) that
        are not recognized by to_offset, so we convert them here.  Also, a
        Period's freq attribute must have `freq.n > 0`, which we check for here.

        Returns
        -------
        DateOffset
        """
        if isinstance(freq, int):
            # We already have a dtype code
            dtype = PeriodDtypeBase(freq)
            freq = dtype.date_offset

        freq = to_offset(freq)

        if freq.n <= 0:
            raise ValueError("Frequency must be positive, because it "
                             f"represents span: {freq.freqstr}")

        return freq

    @classmethod
    def _from_ordinal(cls, ordinal: int, freq) -> "Period":
        """
        Fast creation from an ordinal and freq that are already validated!
        """
        if ordinal == NPY_NAT:
            return NaT
        else:
            freq = cls._maybe_convert_freq(freq)
            self = _Period.__new__(cls, ordinal, freq)
            return self

    def __richcmp__(self, other, op):
        if is_period_object(other):
            if other.freq != self.freq:
                if op == Py_EQ:
                    return False
                elif op == Py_NE:
                    return True
                self._require_matching_freq(other)
            return PyObject_RichCompareBool(self.ordinal, other.ordinal, op)
        elif other is NaT:
            return _nat_scalar_rules[op]
        return NotImplemented  # TODO: ndarray[object]?

    def __hash__(self):
        return hash((self.ordinal, self.freqstr))

    def _add_timedeltalike_scalar(self, other) -> "Period":
        cdef:
            int64_t nanos, base_nanos

        if is_tick_object(self.freq):
            nanos = delta_to_nanoseconds(other)
            base_nanos = self.freq.base.nanos
            if nanos % base_nanos == 0:
                ordinal = self.ordinal + (nanos // base_nanos)
                return Period(ordinal=ordinal, freq=self.freq)
        raise IncompatibleFrequency("Input cannot be converted to "
                                    f"Period(freq={self.freqstr})")

    def _add_offset(self, other) -> "Period":
        # Non-Tick DateOffset other
        cdef:
            int64_t ordinal

        self._require_matching_freq(other, base=True)

        ordinal = self.ordinal + other.n
        return Period(ordinal=ordinal, freq=self.freq)

    def __add__(self, other):
        if not is_period_object(self):
            # cython semantics; this is analogous to a call to __radd__
            if self is NaT:
                return NaT
            return other.__add__(self)

        if is_any_td_scalar(other):
            return self._add_timedeltalike_scalar(other)
        elif is_offset_object(other):
            return self._add_offset(other)
        elif other is NaT:
            return NaT
        elif util.is_integer_object(other):
            ordinal = self.ordinal + other * self.freq.n
            return Period(ordinal=ordinal, freq=self.freq)
        elif (PyDateTime_Check(other) or
              is_period_object(other) or util.is_datetime64_object(other)):
            # can't add datetime-like
            # GH#17983
            sname = type(self).__name__
            oname = type(other).__name__
            raise TypeError(f"unsupported operand type(s) for +: '{sname}' "
                            f"and '{oname}'")

        return NotImplemented

    def __sub__(self, other):
        if not is_period_object(self):
            # cython semantics; this is like a call to __rsub__
            if self is NaT:
                return NaT
            return NotImplemented

        elif is_any_td_scalar(other):
            neg_other = -other
            return self + neg_other
        elif is_offset_object(other):
            # Non-Tick DateOffset
            neg_other = -other
            return self + neg_other
        elif util.is_integer_object(other):
            ordinal = self.ordinal - other * self.freq.n
            return Period(ordinal=ordinal, freq=self.freq)
        elif is_period_object(other):
            self._require_matching_freq(other)
            # GH 23915 - mul by base freq since __add__ is agnostic of n
            return (self.ordinal - other.ordinal) * self.freq.base
        elif other is NaT:
            return NaT

        return NotImplemented

    def asfreq(self, freq, how='E') -> "Period":
        """
        Convert Period to desired frequency, at the start or end of the interval.

        Parameters
        ----------
        freq : str
            The desired frequency.
        how : {'E', 'S', 'end', 'start'}, default 'end'
            Start or end of the timespan.

        Returns
        -------
        resampled : Period
        """
        freq = self._maybe_convert_freq(freq)
        how = validate_end_alias(how)
        base1 = self._dtype._dtype_code
        base2 = freq_to_dtype_code(freq)

        # self.n can't be negative or 0
        end = how == 'E'
        if end:
            ordinal = self.ordinal + self.freq.n - 1
        else:
            ordinal = self.ordinal
        ordinal = period_asfreq(ordinal, base1, base2, end)

        return Period(ordinal=ordinal, freq=freq)

    def to_timestamp(self, freq=None, how='start', tz=None) -> Timestamp:
        """
        Return the Timestamp representation of the Period.

        Uses the target frequency specified at the part of the period specified
        by `how`, which is either `Start` or `Finish`.

        Parameters
        ----------
        freq : str or DateOffset
            Target frequency. Default is 'D' if self.freq is week or
            longer and 'S' otherwise.
        how : str, default 'S' (start)
            One of 'S', 'E'. Can be aliased as case insensitive
            'Start', 'Finish', 'Begin', 'End'.

        Returns
        -------
        Timestamp
        """
        if tz is not None:
            # GH#34522
            warnings.warn(
                "Period.to_timestamp `tz` argument is deprecated and will "
                "be removed in a future version.  Use "
                "`per.to_timestamp(...).tz_localize(tz)` instead.",
                FutureWarning,
                stacklevel=1,
            )

        how = validate_end_alias(how)

        end = how == 'E'
        if end:
            if freq == "B" or self.freq == "B":
                # roll forward to ensure we land on B date
                adjust = Timedelta(1, "D") - Timedelta(1, "ns")
                return self.to_timestamp(how="start") + adjust
            endpoint = (self + self.freq).to_timestamp(how='start')
            return endpoint - Timedelta(1, 'ns')

        if freq is None:
            freq = self._get_to_timestamp_base()
            base = freq
        else:
            freq = self._maybe_convert_freq(freq)
            base = freq._period_dtype_code

        val = self.asfreq(freq, how)

        dt64 = period_ordinal_to_dt64(val.ordinal, base)
        return Timestamp(dt64, tz=tz)

    @property
    def year(self) -> int:
        base = self._dtype._dtype_code
        return pyear(self.ordinal, base)

    @property
    def month(self) -> int:
        base = self._dtype._dtype_code
        return pmonth(self.ordinal, base)

    @property
    def day(self) -> int:
        """
        Get day of the month that a Period falls on.

        Returns
        -------
        int

        See Also
        --------
        Period.dayofweek : Get the day of the week.
        Period.dayofyear : Get the day of the year.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", freq='H')
        >>> p.day
        11
        """
        base = self._dtype._dtype_code
        return pday(self.ordinal, base)

    @property
    def hour(self) -> int:
        """
        Get the hour of the day component of the Period.

        Returns
        -------
        int
            The hour as an integer, between 0 and 23.

        See Also
        --------
        Period.second : Get the second component of the Period.
        Period.minute : Get the minute component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11 13:03:12.050000")
        >>> p.hour
        13

        Period longer than a day

        >>> p = pd.Period("2018-03-11", freq="M")
        >>> p.hour
        0
        """
        base = self._dtype._dtype_code
        return phour(self.ordinal, base)

    @property
    def minute(self) -> int:
        """
        Get minute of the hour component of the Period.

        Returns
        -------
        int
            The minute as an integer, between 0 and 59.

        See Also
        --------
        Period.hour : Get the hour component of the Period.
        Period.second : Get the second component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11 13:03:12.050000")
        >>> p.minute
        3
        """
        base = self._dtype._dtype_code
        return pminute(self.ordinal, base)

    @property
    def second(self) -> int:
        """
        Get the second component of the Period.

        Returns
        -------
        int
            The second of the Period (ranges from 0 to 59).

        See Also
        --------
        Period.hour : Get the hour component of the Period.
        Period.minute : Get the minute component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11 13:03:12.050000")
        >>> p.second
        12
        """
        base = self._dtype._dtype_code
        return psecond(self.ordinal, base)

    @property
    def weekofyear(self) -> int:
        base = self._dtype._dtype_code
        return pweek(self.ordinal, base)

    @property
    def week(self) -> int:
        """
        Get the week of the year on the given Period.

        Returns
        -------
        int

        See Also
        --------
        Period.dayofweek : Get the day component of the Period.
        Period.weekday : Get the day component of the Period.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", "H")
        >>> p.week
        10

        >>> p = pd.Period("2018-02-01", "D")
        >>> p.week
        5

        >>> p = pd.Period("2018-01-06", "D")
        >>> p.week
        1
        """
        return self.weekofyear

    @property
    def day_of_week(self) -> int:
        """
        Day of the week the period lies in, with Monday=0 and Sunday=6.

        If the period frequency is lower than daily (e.g. hourly), and the
        period spans over multiple days, the day at the start of the period is
        used.

        If the frequency is higher than daily (e.g. monthly), the last day
        of the period is used.

        Returns
        -------
        int
            Day of the week.

        See Also
        --------
        Period.day_of_week : Day of the week the period lies in.
        Period.weekday : Alias of Period.day_of_week.
        Period.day : Day of the month.
        Period.dayofyear : Day of the year.

        Examples
        --------
        >>> per = pd.Period('2017-12-31 22:00', 'H')
        >>> per.day_of_week
        6

        For periods that span over multiple days, the day at the beginning of
        the period is returned.

        >>> per = pd.Period('2017-12-31 22:00', '4H')
        >>> per.day_of_week
        6
        >>> per.start_time.day_of_week
        6

        For periods with a frequency higher than days, the last day of the
        period is returned.

        >>> per = pd.Period('2018-01', 'M')
        >>> per.day_of_week
        2
        >>> per.end_time.day_of_week
        2
        """
        base = self._dtype._dtype_code
        return pweekday(self.ordinal, base)

    @property
    def weekday(self) -> int:
        """
        Day of the week the period lies in, with Monday=0 and Sunday=6.

        If the period frequency is lower than daily (e.g. hourly), and the
        period spans over multiple days, the day at the start of the period is
        used.

        If the frequency is higher than daily (e.g. monthly), the last day
        of the period is used.

        Returns
        -------
        int
            Day of the week.

        See Also
        --------
        Period.dayofweek : Day of the week the period lies in.
        Period.weekday : Alias of Period.dayofweek.
        Period.day : Day of the month.
        Period.dayofyear : Day of the year.

        Examples
        --------
        >>> per = pd.Period('2017-12-31 22:00', 'H')
        >>> per.dayofweek
        6

        For periods that span over multiple days, the day at the beginning of
        the period is returned.

        >>> per = pd.Period('2017-12-31 22:00', '4H')
        >>> per.dayofweek
        6
        >>> per.start_time.dayofweek
        6

        For periods with a frequency higher than days, the last day of the
        period is returned.

        >>> per = pd.Period('2018-01', 'M')
        >>> per.dayofweek
        2
        >>> per.end_time.dayofweek
        2
        """
        # Docstring is a duplicate from dayofweek. Reusing docstrings with
        # Appender doesn't work for properties in Cython files, and setting
        # the __doc__ attribute is also not possible.
        return self.dayofweek

    @property
    def day_of_year(self) -> int:
        """
        Return the day of the year.

        This attribute returns the day of the year on which the particular
        date occurs. The return value ranges between 1 to 365 for regular
        years and 1 to 366 for leap years.

        Returns
        -------
        int
            The day of year.

        See Also
        --------
        Period.day : Return the day of the month.
        Period.day_of_week : Return the day of week.
        PeriodIndex.day_of_year : Return the day of year of all indexes.

        Examples
        --------
        >>> period = pd.Period("2015-10-23", freq='H')
        >>> period.day_of_year
        296
        >>> period = pd.Period("2012-12-31", freq='D')
        >>> period.day_of_year
        366
        >>> period = pd.Period("2013-01-01", freq='D')
        >>> period.day_of_year
        1
        """
        base = self._dtype._dtype_code
        return pday_of_year(self.ordinal, base)

    @property
    def quarter(self) -> int:
        base = self._dtype._dtype_code
        return pquarter(self.ordinal, base)

    @property
    def qyear(self) -> int:
        """
        Fiscal year the Period lies in according to its starting-quarter.

        The `year` and the `qyear` of the period will be the same if the fiscal
        and calendar years are the same. When they are not, the fiscal year
        can be different from the calendar year of the period.

        Returns
        -------
        int
            The fiscal year of the period.

        See Also
        --------
        Period.year : Return the calendar year of the period.

        Examples
        --------
        If the natural and fiscal year are the same, `qyear` and `year` will
        be the same.

        >>> per = pd.Period('2018Q1', freq='Q')
        >>> per.qyear
        2018
        >>> per.year
        2018

        If the fiscal year starts in April (`Q-MAR`), the first quarter of
        2018 will start in April 2017. `year` will then be 2018, but `qyear`
        will be the fiscal year, 2018.

        >>> per = pd.Period('2018Q1', freq='Q-MAR')
        >>> per.start_time
        Timestamp('2017-04-01 00:00:00')
        >>> per.qyear
        2018
        >>> per.year
        2017
        """
        base = self._dtype._dtype_code
        return pqyear(self.ordinal, base)

    @property
    def days_in_month(self) -> int:
        """
        Get the total number of days in the month that this period falls on.

        Returns
        -------
        int

        See Also
        --------
        Period.daysinmonth : Gets the number of days in the month.
        DatetimeIndex.daysinmonth : Gets the number of days in the month.
        calendar.monthrange : Returns a tuple containing weekday
            (0-6 ~ Mon-Sun) and number of days (28-31).

        Examples
        --------
        >>> p = pd.Period('2018-2-17')
        >>> p.days_in_month
        28

        >>> pd.Period('2018-03-01').days_in_month
        31

        Handles the leap year case as well:

        >>> p = pd.Period('2016-2-17')
        >>> p.days_in_month
        29
        """
        base = self._dtype._dtype_code
        return pdays_in_month(self.ordinal, base)

    @property
    def daysinmonth(self) -> int:
        """
        Get the total number of days of the month that the Period falls in.

        Returns
        -------
        int

        See Also
        --------
        Period.days_in_month : Return the days of the month.
        Period.dayofyear : Return the day of the year.

        Examples
        --------
        >>> p = pd.Period("2018-03-11", freq='H')
        >>> p.daysinmonth
        31
        """
        return self.days_in_month

    @property
    def is_leap_year(self) -> bool:
        return bool(is_leapyear(self.year))

    @classmethod
    def now(cls, freq=None):
        return Period(datetime.now(), freq=freq)

    @property
    def freqstr(self) -> str:
        return self.freq.freqstr

    def __repr__(self) -> str:
        base = self._dtype._dtype_code
        formatted = period_format(self.ordinal, base)
        return f"Period('{formatted}', '{self.freqstr}')"

    def __str__(self) -> str:
        """
        Return a string representation for a particular DataFrame
        """
        base = self._dtype._dtype_code
        formatted = period_format(self.ordinal, base)
        value = str(formatted)
        return value

    def __setstate__(self, state):
        self.freq = state[1]
        self.ordinal = state[2]

    def __reduce__(self):
        object_state = None, self.freq, self.ordinal
        return (Period, object_state)

    def strftime(self, fmt: str) -> str:
        """
        Returns the string representation of the :class:`Period`, depending
        on the selected ``fmt``. ``fmt`` must be a string
        containing one or several directives.  The method recognizes the same
        directives as the :func:`time.strftime` function of the standard Python
        distribution, as well as the specific additional directives ``%f``,
        ``%F``, ``%q``. (formatting & docs originally from scikits.timeries).

        +-----------+--------------------------------+-------+
        | Directive | Meaning                        | Notes |
        +===========+================================+=======+
        | ``%a``    | Locale's abbreviated weekday   |       |
        |           | name.                          |       |
        +-----------+--------------------------------+-------+
        | ``%A``    | Locale's full weekday name.    |       |
        +-----------+--------------------------------+-------+
        | ``%b``    | Locale's abbreviated month     |       |
        |           | name.                          |       |
        +-----------+--------------------------------+-------+
        | ``%B``    | Locale's full month name.      |       |
        +-----------+--------------------------------+-------+
        | ``%c``    | Locale's appropriate date and  |       |
        |           | time representation.           |       |
        +-----------+--------------------------------+-------+
        | ``%d``    | Day of the month as a decimal  |       |
        |           | number [01,31].                |       |
        +-----------+--------------------------------+-------+
        | ``%f``    | 'Fiscal' year without a        | \(1)  |
        |           | century  as a decimal number   |       |
        |           | [00,99]                        |       |
        +-----------+--------------------------------+-------+
        | ``%F``    | 'Fiscal' year with a century   | \(2)  |
        |           | as a decimal number            |       |
        +-----------+--------------------------------+-------+
        | ``%H``    | Hour (24-hour clock) as a      |       |
        |           | decimal number [00,23].        |       |
        +-----------+--------------------------------+-------+
        | ``%I``    | Hour (12-hour clock) as a      |       |
        |           | decimal number [01,12].        |       |
        +-----------+--------------------------------+-------+
        | ``%j``    | Day of the year as a decimal   |       |
        |           | number [001,366].              |       |
        +-----------+--------------------------------+-------+
        | ``%m``    | Month as a decimal number      |       |
        |           | [01,12].                       |       |
        +-----------+--------------------------------+-------+
        | ``%M``    | Minute as a decimal number     |       |
        |           | [00,59].                       |       |
        +-----------+--------------------------------+-------+
        | ``%p``    | Locale's equivalent of either  | \(3)  |
        |           | AM or PM.                      |       |
        +-----------+--------------------------------+-------+
        | ``%q``    | Quarter as a decimal number    |       |
        |           | [01,04]                        |       |
        +-----------+--------------------------------+-------+
        | ``%S``    | Second as a decimal number     | \(4)  |
        |           | [00,61].                       |       |
        +-----------+--------------------------------+-------+
        | ``%U``    | Week number of the year        | \(5)  |
        |           | (Sunday as the first day of    |       |
        |           | the week) as a decimal number  |       |
        |           | [00,53].  All days in a new    |       |
        |           | year preceding the first       |       |
        |           | Sunday are considered to be in |       |
        |           | week 0.                        |       |
        +-----------+--------------------------------+-------+
        | ``%w``    | Weekday as a decimal number    |       |
        |           | [0(Sunday),6].                 |       |
        +-----------+--------------------------------+-------+
        | ``%W``    | Week number of the year        | \(5)  |
        |           | (Monday as the first day of    |       |
        |           | the week) as a decimal number  |       |
        |           | [00,53].  All days in a new    |       |
        |           | year preceding the first       |       |
        |           | Monday are considered to be in |       |
        |           | week 0.                        |       |
        +-----------+--------------------------------+-------+
        | ``%x``    | Locale's appropriate date      |       |
        |           | representation.                |       |
        +-----------+--------------------------------+-------+
        | ``%X``    | Locale's appropriate time      |       |
        |           | representation.                |       |
        +-----------+--------------------------------+-------+
        | ``%y``    | Year without century as a      |       |
        |           | decimal number [00,99].        |       |
        +-----------+--------------------------------+-------+
        | ``%Y``    | Year with century as a decimal |       |
        |           | number.                        |       |
        +-----------+--------------------------------+-------+
        | ``%Z``    | Time zone name (no characters  |       |
        |           | if no time zone exists).       |       |
        +-----------+--------------------------------+-------+
        | ``%%``    | A literal ``'%'`` character.   |       |
        +-----------+--------------------------------+-------+

        Notes
        -----

        (1)
            The ``%f`` directive is the same as ``%y`` if the frequency is
            not quarterly.
            Otherwise, it corresponds to the 'fiscal' year, as defined by
            the :attr:`qyear` attribute.

        (2)
            The ``%F`` directive is the same as ``%Y`` if the frequency is
            not quarterly.
            Otherwise, it corresponds to the 'fiscal' year, as defined by
            the :attr:`qyear` attribute.

        (3)
            The ``%p`` directive only affects the output hour field
            if the ``%I`` directive is used to parse the hour.

        (4)
            The range really is ``0`` to ``61``; this accounts for leap
            seconds and the (very rare) double leap seconds.

        (5)
            The ``%U`` and ``%W`` directives are only used in calculations
            when the day of the week and the year are specified.

        Examples
        --------

        >>> a = Period(freq='Q-JUL', year=2006, quarter=1)
        >>> a.strftime('%F-Q%q')
        '2006-Q1'
        >>> # Output the last month in the quarter of this date
        >>> a.strftime('%b-%Y')
        'Oct-2005'
        >>>
        >>> a = Period(freq='D', year=2001, month=1, day=1)
        >>> a.strftime('%d-%b-%Y')
        '01-Jan-2006'
        >>> a.strftime('%b. %d, %Y was a %A')
        'Jan. 01, 2001 was a Monday'
        """
        base = self._dtype._dtype_code
        return period_format(self.ordinal, base, fmt)


class Period(_Period):
    """
    Represents a period of time.

    Parameters
    ----------
    value : Period or str, default None
        The time period represented (e.g., '4Q2005').
    freq : str, default None
        One of pandas period strings or corresponding objects.
    ordinal : int, default None
        The period offset from the proleptic Gregorian epoch.
    year : int, default None
        Year value of the period.
    month : int, default 1
        Month value of the period.
    quarter : int, default None
        Quarter value of the period.
    day : int, default 1
        Day value of the period.
    hour : int, default 0
        Hour value of the period.
    minute : int, default 0
        Minute value of the period.
    second : int, default 0
        Second value of the period.
    """

    def __new__(cls, value=None, freq=None, ordinal=None,
                year=None, month=None, quarter=None, day=None,
                hour=None, minute=None, second=None):
        # freq points to a tuple (base, mult);  base is one of the defined
        # periods such as A, Q, etc. Every five minutes would be, e.g.,
        # ('T', 5) but may be passed in as a string like '5T'

        # ordinal is the period offset from the gregorian proleptic epoch
        cdef _Period self

        if freq is not None:
            freq = cls._maybe_convert_freq(freq)
        nanosecond = 0

        if ordinal is not None and value is not None:
            raise ValueError("Only value or ordinal but not both should be "
                             "given but not both")
        elif ordinal is not None:
            if not util.is_integer_object(ordinal):
                raise ValueError("Ordinal must be an integer")
            if freq is None:
                raise ValueError('Must supply freq for ordinal value')

        elif value is None:
            if (year is None and month is None and
                    quarter is None and day is None and
                    hour is None and minute is None and second is None):
                ordinal = NPY_NAT
            else:
                if freq is None:
                    raise ValueError("If value is None, freq cannot be None")

                # set defaults
                month = 1 if month is None else month
                day = 1 if day is None else day
                hour = 0 if hour is None else hour
                minute = 0 if minute is None else minute
                second = 0 if second is None else second

                ordinal = _ordinal_from_fields(year, month, quarter, day,
                                               hour, minute, second, freq)

        elif is_period_object(value):
            other = value
            if freq is None or freq._period_dtype_code == other.freq._period_dtype_code:
                ordinal = other.ordinal
                freq = other.freq
            else:
                converted = other.asfreq(freq)
                ordinal = converted.ordinal

        elif is_null_datetimelike(value) or (isinstance(value, str) and
                                             value in nat_strings):
            # explicit str check is necessary to avoid raising incorrectly
            #  if we have a non-hashable value.
            ordinal = NPY_NAT

        elif isinstance(value, str) or util.is_integer_object(value):
            if util.is_integer_object(value):
                value = str(value)
            value = value.upper()
            dt, reso = parse_time_string(value, freq)
            try:
                ts = Timestamp(value)
            except ValueError:
                nanosecond = 0
            else:
                nanosecond = ts.nanosecond
                if nanosecond != 0:
                    reso = 'nanosecond'
            if dt is NaT:
                ordinal = NPY_NAT

            if freq is None:
                try:
                    freq = attrname_to_abbrevs[reso]
                except KeyError:
                    raise ValueError(f"Invalid frequency or could not "
                                     f"infer: {reso}")
                freq = to_offset(freq)

        elif PyDateTime_Check(value):
            dt = value
            if freq is None:
                raise ValueError('Must supply freq for datetime value')
        elif util.is_datetime64_object(value):
            dt = Timestamp(value)
            if freq is None:
                raise ValueError('Must supply freq for datetime value')
        elif PyDate_Check(value):
            dt = datetime(year=value.year, month=value.month, day=value.day)
            if freq is None:
                raise ValueError('Must supply freq for datetime value')
        else:
            msg = "Value must be Period, string, integer, or datetime"
            raise ValueError(msg)

        if ordinal is None:
            base = freq_to_dtype_code(freq)
            ordinal = period_ordinal(dt.year, dt.month, dt.day,
                                     dt.hour, dt.minute, dt.second,
                                     dt.microsecond, 1000*nanosecond, base)

        return cls._from_ordinal(ordinal, freq)


cdef bint is_period_object(object obj):
    return isinstance(obj, _Period)


cpdef int freq_to_dtype_code(BaseOffset freq) except? -1:
    try:
        return freq._period_dtype_code
    except AttributeError as err:
        raise ValueError(INVALID_FREQ_ERR_MSG.format(freq)) from err


cdef int64_t _ordinal_from_fields(int year, int month, quarter, int day,
                                  int hour, int minute, int second,
                                  BaseOffset freq):
    base = freq_to_dtype_code(freq)
    if quarter is not None:
        year, month = quarter_to_myear(year, quarter, freq.freqstr)

    return period_ordinal(year, month, day, hour,
                          minute, second, 0, 0, base)


def validate_end_alias(how: str) -> str:  # Literal["E", "S"]
    how_dict = {'S': 'S', 'E': 'E',
                'START': 'S', 'FINISH': 'E',
                'BEGIN': 'S', 'END': 'E'}
    how = how_dict.get(str(how).upper())
    if how not in {'S', 'E'}:
        raise ValueError('How must be one of S or E')
    return how
