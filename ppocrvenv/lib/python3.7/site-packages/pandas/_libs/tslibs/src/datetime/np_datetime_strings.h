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

#ifndef PANDAS__LIBS_TSLIBS_SRC_DATETIME_NP_DATETIME_STRINGS_H_
#define PANDAS__LIBS_TSLIBS_SRC_DATETIME_NP_DATETIME_STRINGS_H_

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif  // NPY_NO_DEPRECATED_API

/*
 * Parses (almost) standard ISO 8601 date strings. The differences are:
 *
 * + The date "20100312" is parsed as the year 20100312, not as
 *   equivalent to "2010-03-12". The '-' in the dates are not optional.
 * + Only seconds may have a decimal point, with up to 18 digits after it
 *   (maximum attoseconds precision).
 * + Either a 'T' as in ISO 8601 or a ' ' may be used to separate
 *   the date and the time. Both are treated equivalently.
 * + Doesn't (yet) handle the "YYYY-DDD" or "YYYY-Www" formats.
 * + Doesn't handle leap seconds (seconds value has 60 in these cases).
 * + Doesn't handle 24:00:00 as synonym for midnight (00:00:00) tomorrow
 * + Accepts special values "NaT" (not a time), "Today", (current
 *   day according to local time) and "Now" (current time in UTC).
 *
 * 'str' must be a NULL-terminated string, and 'len' must be its length.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_local' gets whether returned value contains timezone. 0 for UTC, 1 for local time.
 * 'out_tzoffset' gets set to timezone offset by minutes
 *      if the parsed time was in local time,
 *      to 0 otherwise. The values 'now' and 'today' don't get counted
 *      as local, and neither do UTC +/-#### timezone offsets, because
 *      they aren't using the computer's local timezone offset.
 *
 * Returns 0 on success, -1 on failure.
 */
int
parse_iso_8601_datetime(const char *str, int len, int want_exc,
                        npy_datetimestruct *out,
                        int *out_local,
                        int *out_tzoffset);

/*
 * Provides a string length to use for converting datetime
 * objects with the given local and unit settings.
 */
int
get_datetime_iso_8601_strlen(int local, NPY_DATETIMEUNIT base);

/*
 * Converts an npy_datetimestruct to an (almost) ISO 8601
 * NULL-terminated string using timezone Z (UTC).
 *
 * 'base' restricts the output to that unit. Set 'base' to
 * -1 to auto-detect a base after which all the values are zero.
 *
 *  Returns 0 on success, -1 on failure (for example if the output
 *  string was too short).
 */
int
make_iso_8601_datetime(npy_datetimestruct *dts, char *outstr, int outlen,
                       NPY_DATETIMEUNIT base);

/*
 * Converts an pandas_timedeltastruct to an ISO 8601 string.
 *
 * Mutates outlen to provide size of (non-NULL terminated) string.
 *
 * Currently has no error handling
 */
int make_iso_8601_timedelta(pandas_timedeltastruct *tds, char *outstr,
                            size_t *outlen);
#endif  // PANDAS__LIBS_TSLIBS_SRC_DATETIME_NP_DATETIME_STRINGS_H_
