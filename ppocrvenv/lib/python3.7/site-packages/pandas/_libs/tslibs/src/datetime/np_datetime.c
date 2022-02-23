/*

Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Copyright (c) 2005-2011, NumPy Developers
All rights reserved.

This file is derived from NumPy 1.7. See NUMPY_LICENSE.txt

*/

#define NO_IMPORT

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif  // NPY_NO_DEPRECATED_API

#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ndarraytypes.h>
#include "np_datetime.h"

#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#endif  // PyInt_AsLong

const npy_datetimestruct _NS_MIN_DTS = {
    1677, 9, 21, 0, 12, 43, 145224, 193000, 0};
const npy_datetimestruct _NS_MAX_DTS = {
    2262, 4, 11, 23, 47, 16, 854775, 807000, 0};


const int days_per_month_table[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};

/*
 * Returns 1 if the given year is a leap year, 0 otherwise.
 */
int is_leapyear(npy_int64 year) {
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 || (year % 400) == 0);
}

/*
 * Adjusts a datetimestruct based on a minutes offset. Assumes
 * the current values are valid.g
 */
void add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes) {
    int isleap;

    /* MINUTES */
    dts->min += minutes;
    while (dts->min < 0) {
        dts->min += 60;
        dts->hour--;
    }
    while (dts->min >= 60) {
        dts->min -= 60;
        dts->hour++;
    }

    /* HOURS */
    while (dts->hour < 0) {
        dts->hour += 24;
        dts->day--;
    }
    while (dts->hour >= 24) {
        dts->hour -= 24;
        dts->day++;
    }

    /* DAYS */
    if (dts->day < 1) {
        dts->month--;
        if (dts->month < 1) {
            dts->year--;
            dts->month = 12;
        }
        isleap = is_leapyear(dts->year);
        dts->day += days_per_month_table[isleap][dts->month - 1];
    } else if (dts->day > 28) {
        isleap = is_leapyear(dts->year);
        if (dts->day > days_per_month_table[isleap][dts->month - 1]) {
            dts->day -= days_per_month_table[isleap][dts->month - 1];
            dts->month++;
            if (dts->month > 12) {
                dts->year++;
                dts->month = 1;
            }
        }
    }
}

/*
 * Calculates the days offset from the 1970 epoch.
 */
npy_int64 get_datetimestruct_days(const npy_datetimestruct *dts) {
    int i, month;
    npy_int64 year, days = 0;
    const int *month_lengths;

    year = dts->year - 1970;
    days = year * 365;

    /* Adjust for leap years */
    if (days >= 0) {
        /*
         * 1968 is the closest leap year before 1970.
         * Exclude the current year, so add 1.
         */
        year += 1;
        /* Add one day for each 4 years */
        days += year / 4;
        /* 1900 is the closest previous year divisible by 100 */
        year += 68;
        /* Subtract one day for each 100 years */
        days -= year / 100;
        /* 1600 is the closest previous year divisible by 400 */
        year += 300;
        /* Add one day for each 400 years */
        days += year / 400;
    } else {
        /*
         * 1972 is the closest later year after 1970.
         * Include the current year, so subtract 2.
         */
        year -= 2;
        /* Subtract one day for each 4 years */
        days += year / 4;
        /* 2000 is the closest later year divisible by 100 */
        year -= 28;
        /* Add one day for each 100 years */
        days -= year / 100;
        /* 2000 is also the closest later year divisible by 400 */
        /* Subtract one day for each 400 years */
        days += year / 400;
    }

    month_lengths = days_per_month_table[is_leapyear(dts->year)];
    month = dts->month - 1;

    /* Add the months */
    for (i = 0; i < month; ++i) {
        days += month_lengths[i];
    }

    /* Add the days */
    days += dts->day - 1;

    return days;
}

/*
 * Modifies '*days_' to be the day offset within the year,
 * and returns the year.
 */
static npy_int64 days_to_yearsdays(npy_int64 *days_) {
    const npy_int64 days_per_400years = (400 * 365 + 100 - 4 + 1);
    /* Adjust so it's relative to the year 2000 (divisible by 400) */
    npy_int64 days = (*days_) - (365 * 30 + 7);
    npy_int64 year;

    /* Break down the 400 year cycle to get the year and day within the year */
    if (days >= 0) {
        year = 400 * (days / days_per_400years);
        days = days % days_per_400years;
    } else {
        year = 400 * ((days - (days_per_400years - 1)) / days_per_400years);
        days = days % days_per_400years;
        if (days < 0) {
            days += days_per_400years;
        }
    }

    /* Work out the year/day within the 400 year cycle */
    if (days >= 366) {
        year += 100 * ((days - 1) / (100 * 365 + 25 - 1));
        days = (days - 1) % (100 * 365 + 25 - 1);
        if (days >= 365) {
            year += 4 * ((days + 1) / (4 * 365 + 1));
            days = (days + 1) % (4 * 365 + 1);
            if (days >= 366) {
                year += (days - 1) / 365;
                days = (days - 1) % 365;
            }
        }
    }

    *days_ = days;
    return year + 2000;
}

/*
 * Adjusts a datetimestruct based on a seconds offset. Assumes
 * the current values are valid.
 */
NPY_NO_EXPORT void add_seconds_to_datetimestruct(npy_datetimestruct *dts,
                                                 int seconds) {
    int minutes;

    dts->sec += seconds;
    if (dts->sec < 0) {
        minutes = dts->sec / 60;
        dts->sec = dts->sec % 60;
        if (dts->sec < 0) {
            --minutes;
            dts->sec += 60;
        }
        add_minutes_to_datetimestruct(dts, minutes);
    } else if (dts->sec >= 60) {
        minutes = dts->sec / 60;
        dts->sec = dts->sec % 60;
        add_minutes_to_datetimestruct(dts, minutes);
    }
}

/*
 * Fills in the year, month, day in 'dts' based on the days
 * offset from 1970.
 */
static void set_datetimestruct_days(npy_int64 days, npy_datetimestruct *dts) {
    const int *month_lengths;
    int i;

    dts->year = days_to_yearsdays(&days);
    month_lengths = days_per_month_table[is_leapyear(dts->year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            dts->month = i + 1;
            dts->day = days + 1;
            return;
        } else {
            days -= month_lengths[i];
        }
    }
}

/*
 * Compares two npy_datetimestruct objects chronologically
 */
int cmp_npy_datetimestruct(const npy_datetimestruct *a,
                           const npy_datetimestruct *b) {
    if (a->year > b->year) {
        return 1;
    } else if (a->year < b->year) {
        return -1;
    }

    if (a->month > b->month) {
        return 1;
    } else if (a->month < b->month) {
        return -1;
    }

    if (a->day > b->day) {
        return 1;
    } else if (a->day < b->day) {
        return -1;
    }

    if (a->hour > b->hour) {
        return 1;
    } else if (a->hour < b->hour) {
        return -1;
    }

    if (a->min > b->min) {
        return 1;
    } else if (a->min < b->min) {
        return -1;
    }

    if (a->sec > b->sec) {
        return 1;
    } else if (a->sec < b->sec) {
        return -1;
    }

    if (a->us > b->us) {
        return 1;
    } else if (a->us < b->us) {
        return -1;
    }

    if (a->ps > b->ps) {
        return 1;
    } else if (a->ps < b->ps) {
        return -1;
    }

    if (a->as > b->as) {
        return 1;
    } else if (a->as < b->as) {
        return -1;
    }

    return 0;
}

/*
 *
 * Converts a Python datetime.datetime or datetime.date
 * object into a NumPy npy_datetimestruct.  Uses tzinfo (if present)
 * to convert to UTC time.
 *
 * The following implementation just asks for attributes, and thus
 * supports datetime duck typing. The tzinfo time zone conversion
 * requires this style of access as well.
 *
 * Returns -1 on error, 0 on success, and 1 (with no error set)
 * if obj doesn't have the needed date or datetime attributes.
 */
int convert_pydatetime_to_datetimestruct(PyObject *dtobj,
                                         npy_datetimestruct *out) {
    // Assumes that obj is a valid datetime object
    PyObject *tmp;
    PyObject *obj = (PyObject*)dtobj;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->month = 1;
    out->day = 1;

    out->year = PyInt_AsLong(PyObject_GetAttrString(obj, "year"));
    out->month = PyInt_AsLong(PyObject_GetAttrString(obj, "month"));
    out->day = PyInt_AsLong(PyObject_GetAttrString(obj, "day"));

    // TODO(anyone): If we can get PyDateTime_IMPORT to work, we could use
    // PyDateTime_Check here, and less verbose attribute lookups.

    /* Check for time attributes (if not there, return success as a date) */
    if (!PyObject_HasAttrString(obj, "hour") ||
        !PyObject_HasAttrString(obj, "minute") ||
        !PyObject_HasAttrString(obj, "second") ||
        !PyObject_HasAttrString(obj, "microsecond")) {
        return 0;
    }

    out->hour = PyInt_AsLong(PyObject_GetAttrString(obj, "hour"));
    out->min = PyInt_AsLong(PyObject_GetAttrString(obj, "minute"));
    out->sec = PyInt_AsLong(PyObject_GetAttrString(obj, "second"));
    out->us = PyInt_AsLong(PyObject_GetAttrString(obj, "microsecond"));

    /* Apply the time zone offset if datetime obj is tz-aware */
    if (PyObject_HasAttrString((PyObject*)obj, "tzinfo")) {
        tmp = PyObject_GetAttrString(obj, "tzinfo");
        if (tmp == NULL) {
            return -1;
        }
        if (tmp == Py_None) {
            Py_DECREF(tmp);
        } else {
            PyObject *offset;
            int seconds_offset, minutes_offset;

            /* The utcoffset function should return a timedelta */
            offset = PyObject_CallMethod(tmp, "utcoffset", "O", obj);
            if (offset == NULL) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /*
             * The timedelta should have a function "total_seconds"
             * which contains the value we want.
             */
            tmp = PyObject_CallMethod(offset, "total_seconds", "");
            if (tmp == NULL) {
                return -1;
            }
            seconds_offset = PyInt_AsLong(tmp);
            if (seconds_offset == -1 && PyErr_Occurred()) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /* Convert to a minutes offset and apply it */
            minutes_offset = seconds_offset / 60;

            add_minutes_to_datetimestruct(out, -minutes_offset);
        }
    }

    return 0;
}


/*
 * Converts a datetime from a datetimestruct to a datetime based
 * on a metadata unit. The date is assumed to be valid.
 */
npy_datetime npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT base,
                                            const npy_datetimestruct *dts) {
    npy_datetime ret;

    if (base == NPY_FR_Y) {
        /* Truncate to the year */
        ret = dts->year - 1970;
    } else if (base == NPY_FR_M) {
        /* Truncate to the month */
        ret = 12 * (dts->year - 1970) + (dts->month - 1);
    } else {
        /* Otherwise calculate the number of days to start */
        npy_int64 days = get_datetimestruct_days(dts);

        switch (base) {
            case NPY_FR_W:
                /* Truncate to weeks */
                if (days >= 0) {
                    ret = days / 7;
                } else {
                    ret = (days - 6) / 7;
                }
                break;
            case NPY_FR_D:
                ret = days;
                break;
            case NPY_FR_h:
                ret = days * 24 + dts->hour;
                break;
            case NPY_FR_m:
                ret = (days * 24 + dts->hour) * 60 + dts->min;
                break;
            case NPY_FR_s:
                ret = ((days * 24 + dts->hour) * 60 + dts->min) * 60 + dts->sec;
                break;
            case NPY_FR_ms:
                ret = (((days * 24 + dts->hour) * 60 + dts->min) * 60 +
                       dts->sec) *
                          1000 +
                      dts->us / 1000;
                break;
            case NPY_FR_us:
                ret = (((days * 24 + dts->hour) * 60 + dts->min) * 60 +
                       dts->sec) *
                          1000000 +
                      dts->us;
                break;
            case NPY_FR_ns:
                ret = ((((days * 24 + dts->hour) * 60 + dts->min) * 60 +
                        dts->sec) *
                           1000000 +
                       dts->us) *
                          1000 +
                      dts->ps / 1000;
                break;
            case NPY_FR_ps:
                ret = ((((days * 24 + dts->hour) * 60 + dts->min) * 60 +
                        dts->sec) *
                           1000000 +
                       dts->us) *
                          1000000 +
                      dts->ps;
                break;
            case NPY_FR_fs:
                /* only 2.6 hours */
                ret = (((((days * 24 + dts->hour) * 60 + dts->min) * 60 +
                         dts->sec) *
                            1000000 +
                        dts->us) *
                           1000000 +
                       dts->ps) *
                          1000 +
                      dts->as / 1000;
                break;
            case NPY_FR_as:
                /* only 9.2 secs */
                ret = (((((days * 24 + dts->hour) * 60 + dts->min) * 60 +
                         dts->sec) *
                            1000000 +
                        dts->us) *
                           1000000 +
                       dts->ps) *
                          1000000 +
                      dts->as;
                break;
            default:
                /* Something got corrupted */
                PyErr_SetString(
                    PyExc_ValueError,
                    "NumPy datetime metadata with corrupt unit value");
                return -1;
        }
    }
    return ret;
}

/*
 * Port numpy#13188 https://github.com/numpy/numpy/pull/13188/
 *
 * Computes the python `ret, d = divmod(d, unit)`.
 *
 * Note that GCC is smart enough at -O2 to eliminate the `if(*d < 0)` branch
 * for subsequent calls to this command - it is able to deduce that `*d >= 0`.
 */
npy_int64 extract_unit(npy_datetime *d, npy_datetime unit) {
    assert(unit > 0);
    npy_int64 div = *d / unit;
    npy_int64 mod = *d % unit;
    if (mod < 0) {
        mod += unit;
        div -= 1;
    }
    assert(mod >= 0);
    *d = mod;
    return div;
}

/*
 * Converts a datetime based on the given metadata into a datetimestruct
 */
void pandas_datetime_to_datetimestruct(npy_datetime dt,
                                       NPY_DATETIMEUNIT base,
                                       npy_datetimestruct *out) {
    npy_int64 perday;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->year = 1970;
    out->month = 1;
    out->day = 1;

    /*
     * Note that care must be taken with the / and % operators
     * for negative values.
     */
    switch (base) {
        case NPY_FR_Y:
            out->year = 1970 + dt;
            break;

        case NPY_FR_M:
            out->year  = 1970 + extract_unit(&dt, 12);
            out->month = dt + 1;
            break;

        case NPY_FR_W:
            /* A week is 7 days */
            set_datetimestruct_days(dt * 7, out);
            break;

        case NPY_FR_D:
            set_datetimestruct_days(dt, out);
            break;

        case NPY_FR_h:
            perday = 24LL;

            set_datetimestruct_days(extract_unit(&dt, perday), out);
            out->hour = dt;
            break;

        case NPY_FR_m:
            perday = 24LL * 60;

            set_datetimestruct_days(extract_unit(&dt, perday), out);
            out->hour = (int)extract_unit(&dt, 60);
            out->min = (int)dt;
            break;

        case NPY_FR_s:
            perday = 24LL * 60 * 60;

            set_datetimestruct_days(extract_unit(&dt, perday), out);
            out->hour = (int)extract_unit(&dt, 60 * 60);
            out->min  = (int)extract_unit(&dt, 60);
            out->sec  = (int)dt;
            break;

        case NPY_FR_ms:
            perday = 24LL * 60 * 60 * 1000;

            set_datetimestruct_days(extract_unit(&dt, perday), out);
            out->hour = (int)extract_unit(&dt, 1000LL * 60 * 60);
            out->min  = (int)extract_unit(&dt, 1000LL * 60);
            out->sec  = (int)extract_unit(&dt, 1000LL);
            out->us   = (int)(dt * 1000);
            break;

        case NPY_FR_us:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL;

            set_datetimestruct_days(extract_unit(&dt, perday), out);
            out->hour = (int)extract_unit(&dt, 1000LL * 1000 * 60 * 60);
            out->min  = (int)extract_unit(&dt, 1000LL * 1000 * 60);
            out->sec  = (int)extract_unit(&dt, 1000LL * 1000);
            out->us   = (int)dt;
            break;

        case NPY_FR_ns:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;

            set_datetimestruct_days(extract_unit(&dt, perday), out);
            out->hour = (int)extract_unit(&dt, 1000LL * 1000 * 1000 * 60 * 60);
            out->min  = (int)extract_unit(&dt, 1000LL * 1000 * 1000 * 60);
            out->sec  = (int)extract_unit(&dt, 1000LL * 1000 * 1000);
            out->us   = (int)extract_unit(&dt, 1000LL);
            out->ps   = (int)(dt * 1000);
            break;

        case NPY_FR_ps:
            perday = 24LL * 60 * 60 * 1000 * 1000 * 1000 * 1000;

            set_datetimestruct_days(extract_unit(&dt, perday), out);
            out->hour = (int)extract_unit(&dt, 1000LL * 1000 * 1000 * 60 * 60);
            out->min  = (int)extract_unit(&dt, 1000LL * 1000 * 1000 * 60);
            out->sec  = (int)extract_unit(&dt, 1000LL * 1000 * 1000);
            out->us   = (int)extract_unit(&dt, 1000LL);
            out->ps   = (int)(dt * 1000);
            break;

        case NPY_FR_fs:
            /* entire range is only +- 2.6 hours */
            out->hour = (int)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000 *
                                        1000 * 60 * 60);
            if (out->hour < 0) {
                out->year  = 1969;
                out->month = 12;
                out->day   = 31;
                out->hour  += 24;
                assert(out->hour >= 0);
            }
            out->min  = (int)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000 *
                                        1000 * 60);
            out->sec  = (int)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000 *
                                        1000);
            out->us   = (int)extract_unit(&dt, 1000LL * 1000 * 1000);
            out->ps   = (int)extract_unit(&dt, 1000LL);
            out->as   = (int)(dt * 1000);
            break;

        case NPY_FR_as:
            /* entire range is only +- 9.2 seconds */
            out->sec = (int)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000 *
                                        1000 * 1000);
            if (out->sec < 0) {
                out->year  = 1969;
                out->month = 12;
                out->day   = 31;
                out->hour  = 23;
                out->min   = 59;
                out->sec   += 60;
                assert(out->sec >= 0);
            }
            out->us   = (int)extract_unit(&dt, 1000LL * 1000 * 1000 * 1000);
            out->ps   = (int)extract_unit(&dt, 1000LL * 1000);
            out->as   = (int)dt;
            break;

        default:
            PyErr_SetString(PyExc_RuntimeError,
                            "NumPy datetime metadata is corrupted with invalid "
                            "base unit");
    }
}

/*
 * Converts a timedelta from a timedeltastruct to a timedelta based
 * on a metadata unit. The timedelta is assumed to be valid.
 *
 * Returns 0 on success, -1 on failure.
 */
void pandas_timedelta_to_timedeltastruct(npy_timedelta td,
                                         NPY_DATETIMEUNIT base,
                                         pandas_timedeltastruct *out) {
    npy_int64 frac;
    npy_int64 sfrac;
    npy_int64 ifrac;
    int sign;
    npy_int64 DAY_NS = 86400000000000LL;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(pandas_timedeltastruct));

    switch (base) {
        case NPY_FR_ns:

        // put frac in seconds
        if (td < 0 && td % (1000LL * 1000LL * 1000LL) != 0)
            frac = td / (1000LL * 1000LL * 1000LL) - 1;
        else
            frac = td / (1000LL * 1000LL * 1000LL);

        if (frac < 0) {
            sign = -1;

            // even fraction
            if ((-frac % 86400LL) != 0) {
              out->days = -frac / 86400LL + 1;
              frac += 86400LL * out->days;
            } else {
              frac = -frac;
            }
        } else {
            sign = 1;
            out->days = 0;
        }

        if (frac >= 86400) {
            out->days += frac / 86400LL;
            frac -= out->days * 86400LL;
        }

        if (frac >= 3600) {
            out->hrs = frac / 3600LL;
            frac -= out->hrs * 3600LL;
        } else {
            out->hrs = 0;
        }

        if (frac >= 60) {
            out->min = frac / 60LL;
            frac -= out->min * 60LL;
        } else {
            out->min = 0;
        }

        if (frac >= 0) {
            out->sec = frac;
            frac -= out->sec;
        } else {
            out->sec = 0;
        }

        sfrac = (out->hrs * 3600LL + out->min * 60LL
                 + out->sec) * (1000LL * 1000LL * 1000LL);

        if (sign < 0)
            out->days = -out->days;

        ifrac = td - (out->days * DAY_NS + sfrac);

        if (ifrac != 0) {
            out->ms = ifrac / (1000LL * 1000LL);
            ifrac -= out->ms * 1000LL * 1000LL;
            out->us = ifrac / 1000LL;
            ifrac -= out->us * 1000LL;
            out->ns = ifrac;
        } else {
            out->ms = 0;
            out->us = 0;
            out->ns = 0;
        }

        out->seconds = out->hrs * 3600 + out->min * 60 + out->sec;
        out->microseconds = out->ms * 1000 + out->us;
        out->nanoseconds = out->ns;
        break;

        default:
            PyErr_SetString(PyExc_RuntimeError,
                            "NumPy timedelta metadata is corrupted with "
                            "invalid base unit");
    }
}
