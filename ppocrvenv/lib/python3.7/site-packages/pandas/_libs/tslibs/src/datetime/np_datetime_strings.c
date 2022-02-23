/*

Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Written by Mark Wiebe (mwwiebe@gmail.com)
Copyright (c) 2011 by Enthought, Inc.

Copyright (c) 2005-2011, NumPy Developers
All rights reserved.

See NUMPY_LICENSE.txt for the license.

This file implements string parsing and creation for NumPy datetime.

*/

#define PY_SSIZE_T_CLEAN
#define NO_IMPORT

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif  // NPY_NO_DEPRECATED_API

#include <Python.h>

#include <time.h>

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ndarraytypes.h>

#include "np_datetime.h"
#include "np_datetime_strings.h"


/*
 * Parses (almost) standard ISO 8601 date strings. The differences are:
 *
 * + Only seconds may have a decimal point, with up to 18 digits after it
 *   (maximum attoseconds precision).
 * + Either a 'T' as in ISO 8601 or a ' ' may be used to separate
 *   the date and the time. Both are treated equivalently.
 * + Doesn't (yet) handle the "YYYY-DDD" or "YYYY-Www" formats.
 * + Doesn't handle leap seconds (seconds value has 60 in these cases).
 * + Doesn't handle 24:00:00 as synonym for midnight (00:00:00) tomorrow
 * + Accepts special values "NaT" (not a time), "Today", (current
 *   day according to local time) and "Now" (current time in UTC).
 * + ':' separator between hours, minutes, and seconds is optional. When
 *   omitted, each component must be 2 digits if it appears. (GH-10041)
 *
 * 'str' must be a NULL-terminated string, and 'len' must be its length.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_local' gets set to 1 if the parsed time contains timezone,
 *      to 0 otherwise.
 * 'out_tzoffset' gets set to timezone offset by minutes
 *      if the parsed time was in local time,
 *      to 0 otherwise. The values 'now' and 'today' don't get counted
 *      as local, and neither do UTC +/-#### timezone offsets, because
 *      they aren't using the computer's local timezone offset.
 *
 * Returns 0 on success, -1 on failure.
 */
int parse_iso_8601_datetime(const char *str, int len, int want_exc,
                            npy_datetimestruct *out,
                            int *out_local, int *out_tzoffset) {
    int year_leap = 0;
    int i, numdigits;
    const char *substr;
    int sublen;

    /* If year-month-day are separated by a valid separator,
     * months/days without leading zeroes will be parsed
     * (though not iso8601). If the components aren't separated,
     * 4 (YYYY) or 8 (YYYYMMDD) digits are expected. 6 digits are
     * forbidden here (but parsed as YYMMDD elsewhere).
    */
    int has_ymd_sep = 0;
    char ymd_sep = '\0';
    char valid_ymd_sep[] = {'-', '.', '/', '\\', ' '};
    int valid_ymd_sep_len = sizeof(valid_ymd_sep);

    /* hour-minute-second may or may not separated by ':'. If not, then
     * each component must be 2 digits. */
    int has_hms_sep = 0;
    int hour_was_2_digits = 0;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->month = 1;
    out->day = 1;

    substr = str;
    sublen = len;

    /* Skip leading whitespace */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    /* Leading '-' sign for negative year */
    if (*substr == '-') {
        ++substr;
        --sublen;
    }

    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE YEAR (4 digits) */
    out->year = 0;
    if (sublen >= 4 && isdigit(substr[0]) && isdigit(substr[1]) &&
        isdigit(substr[2]) && isdigit(substr[3])) {
        out->year = 1000 * (substr[0] - '0') + 100 * (substr[1] - '0') +
                    10 * (substr[2] - '0') + (substr[3] - '0');

        substr += 4;
        sublen -= 4;
    }

    /* Negate the year if necessary */
    if (str[0] == '-') {
        out->year = -out->year;
    }
    /* Check whether it's a leap-year */
    year_leap = is_leapyear(out->year);

    /* Next character must be a separator, start of month, or end of string */
    if (sublen == 0) {
        if (out_local != NULL) {
            *out_local = 0;
        }
        goto finish;
    }

    if (!isdigit(*substr)) {
        for (i = 0; i < valid_ymd_sep_len; ++i) {
            if (*substr == valid_ymd_sep[i]) {
                break;
            }
        }
        if (i == valid_ymd_sep_len) {
            goto parse_error;
        }
        has_ymd_sep = 1;
        ymd_sep = valid_ymd_sep[i];
        ++substr;
        --sublen;
        /* Cannot have trailing separator */
        if (sublen == 0 || !isdigit(*substr)) {
            goto parse_error;
        }
    }

    /* PARSE THE MONTH */
    /* First digit required */
    out->month = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional if there was a separator */
    if (isdigit(*substr)) {
        out->month = 10 * out->month + (*substr - '0');
        ++substr;
        --sublen;
    } else if (!has_ymd_sep) {
        goto parse_error;
    }
    if (out->month < 1 || out->month > 12) {
        if (want_exc) {
            PyErr_Format(PyExc_ValueError,
                        "Month out of range in datetime string \"%s\"", str);
        }
        goto error;
    }

    /* Next character must be the separator, start of day, or end of string */
    if (sublen == 0) {
        /* Forbid YYYYMM. Parsed instead as YYMMDD by someone else. */
        if (!has_ymd_sep) {
            goto parse_error;
        }
        if (out_local != NULL) {
            *out_local = 0;
        }
        goto finish;
    }

    if (has_ymd_sep) {
        /* Must have separator, but cannot be trailing */
        if (*substr != ymd_sep || sublen == 1) {
            goto parse_error;
        }
        ++substr;
        --sublen;
    }

    /* PARSE THE DAY */
    /* First digit required */
    if (!isdigit(*substr)) {
        goto parse_error;
    }
    out->day = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional if there was a separator */
    if (isdigit(*substr)) {
        out->day = 10 * out->day + (*substr - '0');
        ++substr;
        --sublen;
    } else if (!has_ymd_sep) {
        goto parse_error;
    }
    if (out->day < 1 ||
        out->day > days_per_month_table[year_leap][out->month - 1]) {
        if (want_exc) {
            PyErr_Format(PyExc_ValueError,
                        "Day out of range in datetime string \"%s\"", str);
        }
        goto error;
    }

    /* Next character must be a 'T', ' ', or end of string */
    if (sublen == 0) {
        if (out_local != NULL) {
            *out_local = 0;
        }
        goto finish;
    }

    if ((*substr != 'T' && *substr != ' ') || sublen == 1) {
        goto parse_error;
    }
    ++substr;
    --sublen;

    /* PARSE THE HOURS */
    /* First digit required */
    if (!isdigit(*substr)) {
        goto parse_error;
    }
    out->hour = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional */
    if (isdigit(*substr)) {
        hour_was_2_digits = 1;
        out->hour = 10 * out->hour + (*substr - '0');
        ++substr;
        --sublen;
        if (out->hour >= 24) {
            if (want_exc) {
                PyErr_Format(PyExc_ValueError,
                             "Hours out of range in datetime string \"%s\"",
                             str);
            }
            goto error;
        }
    }

    /* Next character must be a ':' or the end of the string */
    if (sublen == 0) {
        if (!hour_was_2_digits) {
            goto parse_error;
        }
        goto finish;
    }

    if (*substr == ':') {
        has_hms_sep = 1;
        ++substr;
        --sublen;
        /* Cannot have a trailing separator */
        if (sublen == 0 || !isdigit(*substr)) {
            goto parse_error;
        }
    } else if (!isdigit(*substr)) {
        if (!hour_was_2_digits) {
            goto parse_error;
        }
        goto parse_timezone;
    }

    /* PARSE THE MINUTES */
    /* First digit required */
    out->min = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional if there was a separator */
    if (isdigit(*substr)) {
        out->min = 10 * out->min + (*substr - '0');
        ++substr;
        --sublen;
        if (out->min >= 60) {
            if (want_exc) {
                PyErr_Format(PyExc_ValueError,
                             "Minutes out of range in datetime string \"%s\"",
                             str);
            }
            goto error;
        }
    } else if (!has_hms_sep) {
        goto parse_error;
    }

    if (sublen == 0) {
        goto finish;
    }

    /* If we make it through this condition block, then the next
     * character is a digit. */
    if (has_hms_sep && *substr == ':') {
        ++substr;
        --sublen;
        /* Cannot have a trailing ':' */
        if (sublen == 0 || !isdigit(*substr)) {
            goto parse_error;
        }
    } else if (!has_hms_sep && isdigit(*substr)) {
    } else {
        goto parse_timezone;
    }

    /* PARSE THE SECONDS */
    /* First digit required */
    out->sec = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional if there was a separator */
    if (isdigit(*substr)) {
        out->sec = 10 * out->sec + (*substr - '0');
        ++substr;
        --sublen;
        if (out->sec >= 60) {
            if (want_exc) {
                PyErr_Format(PyExc_ValueError,
                             "Seconds out of range in datetime string \"%s\"",
                             str);
            }
            goto error;
        }
    } else if (!has_hms_sep) {
        goto parse_error;
    }

    /* Next character may be a '.' indicating fractional seconds */
    if (sublen > 0 && *substr == '.') {
        ++substr;
        --sublen;
    } else {
        goto parse_timezone;
    }

    /* PARSE THE MICROSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->us *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->us += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (sublen == 0 || !isdigit(*substr)) {
        goto parse_timezone;
    }

    /* PARSE THE PICOSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->ps *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->ps += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (sublen == 0 || !isdigit(*substr)) {
        goto parse_timezone;
    }

    /* PARSE THE ATTOSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->as *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->as += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

parse_timezone:
    /* trim any whitepsace between time/timeezone */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    if (sublen == 0) {
        // Unlike NumPy, treating no time zone as naive
        goto finish;
    }

    /* UTC specifier */
    if (*substr == 'Z') {
        /* "Z" should be equivalent to tz offset "+00:00" */
        if (out_local != NULL) {
            *out_local = 1;
        }

        if (out_tzoffset != NULL) {
            *out_tzoffset = 0;
        }

        if (sublen == 1) {
            goto finish;
        } else {
            ++substr;
            --sublen;
        }
    } else if (*substr == '-' || *substr == '+') {
        /* Time zone offset */
        int offset_neg = 0, offset_hour = 0, offset_minute = 0;

        /*
         * Since "local" means local with respect to the current
         * machine, we say this is non-local.
         */

        if (*substr == '-') {
            offset_neg = 1;
        }
        ++substr;
        --sublen;

        /* The hours offset */
        if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
            offset_hour = 10 * (substr[0] - '0') + (substr[1] - '0');
            substr += 2;
            sublen -= 2;
            if (offset_hour >= 24) {
                if (want_exc) {
                    PyErr_Format(PyExc_ValueError,
                                "Timezone hours offset out of range "
                                "in datetime string \"%s\"",
                                str);
                }
                goto error;
            }
        } else if (sublen >= 1 && isdigit(substr[0])) {
            offset_hour = substr[0] - '0';
            ++substr;
            --sublen;
        } else {
            goto parse_error;
        }

        /* The minutes offset is optional */
        if (sublen > 0) {
            /* Optional ':' */
            if (*substr == ':') {
                ++substr;
                --sublen;
            }

            /* The minutes offset (at the end of the string) */
            if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
                offset_minute = 10 * (substr[0] - '0') + (substr[1] - '0');
                substr += 2;
                sublen -= 2;
                if (offset_minute >= 60) {
                    if (want_exc) {
                        PyErr_Format(PyExc_ValueError,
                                    "Timezone minutes offset out of range "
                                    "in datetime string \"%s\"",
                                    str);
                    }
                    goto error;
                }
            } else if (sublen >= 1 && isdigit(substr[0])) {
                offset_minute = substr[0] - '0';
                ++substr;
                --sublen;
            } else {
                goto parse_error;
            }
        }

        /* Apply the time zone offset */
        if (offset_neg) {
            offset_hour = -offset_hour;
            offset_minute = -offset_minute;
        }
        if (out_local != NULL) {
            *out_local = 1;
            // Unlike NumPy, do not change internal value to local time
            *out_tzoffset = 60 * offset_hour + offset_minute;
        }
    }

    /* Skip trailing whitespace */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    if (sublen != 0) {
        goto parse_error;
    }

finish:
    return 0;

parse_error:
    if (want_exc) {
        PyErr_Format(PyExc_ValueError,
                    "Error parsing datetime string \"%s\" at position %d", str,
                    (int)(substr - str));
    }
    return -1;

error:
    return -1;
}

/*
 * Provides a string length to use for converting datetime
 * objects with the given local and unit settings.
 */
int get_datetime_iso_8601_strlen(int local, NPY_DATETIMEUNIT base) {
    int len = 0;

    switch (base) {
        /* Generic units can only be used to represent NaT */
        /*    return 4;*/
        case NPY_FR_as:
            len += 3; /* "###" */
        case NPY_FR_fs:
            len += 3; /* "###" */
        case NPY_FR_ps:
            len += 3; /* "###" */
        case NPY_FR_ns:
            len += 3; /* "###" */
        case NPY_FR_us:
            len += 3; /* "###" */
        case NPY_FR_ms:
            len += 4; /* ".###" */
        case NPY_FR_s:
            len += 3; /* ":##" */
        case NPY_FR_m:
            len += 3; /* ":##" */
        case NPY_FR_h:
            len += 3; /* "T##" */
        case NPY_FR_D:
        case NPY_FR_W:
            len += 3; /* "-##" */
        case NPY_FR_M:
            len += 3; /* "-##" */
        case NPY_FR_Y:
            len += 21; /* 64-bit year */
            break;
        default:
            len += 3; /* handle the now defunct NPY_FR_B */
            break;
    }

    if (base >= NPY_FR_h) {
        if (local) {
            len += 5; /* "+####" or "-####" */
        } else {
            len += 1; /* "Z" */
        }
    }

    len += 1; /* NULL terminator */

    return len;
}


/*
 * Converts an npy_datetimestruct to an (almost) ISO 8601
 * NULL-terminated string using timezone Z (UTC). If the string fits in
 * the space exactly, it leaves out the NULL terminator and returns success.
 *
 * The differences from ISO 8601 are the 'NaT' string, and
 * the number of year digits is >= 4 instead of strictly 4.
 *
 * 'base' restricts the output to that unit. Set 'base' to
 * -1 to auto-detect a base after which all the values are zero.
 *
 *  Returns 0 on success, -1 on failure (for example if the output
 *  string was too short).
 */
int make_iso_8601_datetime(npy_datetimestruct *dts, char *outstr, int outlen,
                           NPY_DATETIMEUNIT base) {
    char *substr = outstr;
    int sublen = outlen;
    int tmplen;

    /*
     * Print weeks with the same precision as days.
     *
     * TODO: Could print weeks with YYYY-Www format if the week
     *       epoch is a Monday.
     */
    if (base == NPY_FR_W) {
        base = NPY_FR_D;
    }

/* YEAR */
/*
 * Can't use PyOS_snprintf, because it always produces a '\0'
 * character at the end, and NumPy string types are permitted
 * to have data all the way to the end of the buffer.
 */
#ifdef _WIN32
    tmplen = _snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#else
    tmplen = snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#endif  // _WIN32
    /* If it ran out of space or there isn't space for the NULL terminator */
    if (tmplen < 0 || tmplen > sublen) {
        goto string_too_short;
    }
    substr += tmplen;
    sublen -= tmplen;

    /* Stop if the unit is years */
    if (base == NPY_FR_Y) {
        if (sublen > 0) {
            *substr = '\0';
        }
        return 0;
    }

    /* MONTH */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = '-';
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->month / 10) + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->month % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is months */
    if (base == NPY_FR_M) {
        if (sublen > 0) {
            *substr = '\0';
        }
        return 0;
    }

    /* DAY */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = '-';
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->day / 10) + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->day % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is days */
    if (base == NPY_FR_D) {
        if (sublen > 0) {
            *substr = '\0';
        }
        return 0;
    }

    /* HOUR */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = 'T';
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->hour / 10) + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->hour % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is hours */
    if (base == NPY_FR_h) {
        goto add_time_zone;
    }

    /* MINUTE */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = ':';
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->min / 10) + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->min % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is minutes */
    if (base == NPY_FR_m) {
        goto add_time_zone;
    }

    /* SECOND */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = ':';
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->sec / 10) + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->sec % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is seconds */
    if (base == NPY_FR_s) {
        goto add_time_zone;
    }

    /* MILLISECOND */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = '.';
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->us / 100000) % 10 + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->us / 10000) % 10 + '0');
    if (sublen < 4) {
        goto string_too_short;
    }
    substr[3] = (char)((dts->us / 1000) % 10 + '0');
    substr += 4;
    sublen -= 4;

    /* Stop if the unit is milliseconds */
    if (base == NPY_FR_ms) {
        goto add_time_zone;
    }

    /* MICROSECOND */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->us / 100) % 10 + '0');
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->us / 10) % 10 + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->us % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is microseconds */
    if (base == NPY_FR_us) {
        goto add_time_zone;
    }

    /* NANOSECOND */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->ps / 100000) % 10 + '0');
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->ps / 10000) % 10 + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->ps / 1000) % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is nanoseconds */
    if (base == NPY_FR_ns) {
        goto add_time_zone;
    }

    /* PICOSECOND */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->ps / 100) % 10 + '0');
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->ps / 10) % 10 + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->ps % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is picoseconds */
    if (base == NPY_FR_ps) {
        goto add_time_zone;
    }

    /* FEMTOSECOND */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->as / 100000) % 10 + '0');
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->as / 10000) % 10 + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->as / 1000) % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is femtoseconds */
    if (base == NPY_FR_fs) {
        goto add_time_zone;
    }

    /* ATTOSECOND */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->as / 100) % 10 + '0');
    if (sublen < 2) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->as / 10) % 10 + '0');
    if (sublen < 3) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->as % 10 + '0');
    substr += 3;
    sublen -= 3;

add_time_zone:
    /* UTC "Zulu" time */
    if (sublen < 1) {
        goto string_too_short;
    }
    substr[0] = 'Z';
    substr += 1;
    sublen -= 1;

    /* Add a NULL terminator, and return */
    if (sublen > 0) {
        substr[0] = '\0';
    }

    return 0;

string_too_short:
    PyErr_Format(PyExc_RuntimeError,
                 "The string provided for NumPy ISO datetime formatting "
                 "was too short, with length %d",
                 outlen);
    return -1;
}


int make_iso_8601_timedelta(pandas_timedeltastruct *tds,
                            char *outstr, size_t *outlen) {
  *outlen = 0;
  *outlen += snprintf(outstr, 60,  // NOLINT
                     "P%" NPY_INT64_FMT
                     "DT%" NPY_INT32_FMT
                     "H%" NPY_INT32_FMT
                     "M%" NPY_INT32_FMT,
                     tds->days, tds->hrs, tds->min, tds->sec);
  outstr += *outlen;

  if (tds->ns != 0) {
    *outlen += snprintf(outstr, 12,  // NOLINT
                       ".%03" NPY_INT32_FMT
                       "%03" NPY_INT32_FMT
                       "%03" NPY_INT32_FMT
                       "S", tds->ms, tds->us, tds->ns);
  } else if (tds->us != 0) {
    *outlen += snprintf(outstr, 9,  // NOLINT
                       ".%03" NPY_INT32_FMT
                       "%03" NPY_INT32_FMT
                       "S", tds->ms, tds->us);
  } else if (tds->ms != 0) {
    *outlen += snprintf(outstr, 6,  // NOLINT
                        ".%03" NPY_INT32_FMT "S", tds->ms);
  } else {
    *outlen += snprintf(outstr, 2,  // NOLINT
                        "%s", "S");
  }

  return 0;
}
