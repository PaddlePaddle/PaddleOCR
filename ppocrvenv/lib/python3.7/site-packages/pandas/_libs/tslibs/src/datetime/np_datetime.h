/*

Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Copyright (c) 2005-2011, NumPy Developers
All rights reserved.

This file is derived from NumPy 1.7. See NUMPY_LICENSE.txt

*/

#ifndef PANDAS__LIBS_TSLIBS_SRC_DATETIME_NP_DATETIME_H_
#define PANDAS__LIBS_TSLIBS_SRC_DATETIME_NP_DATETIME_H_

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif  // NPY_NO_DEPRECATED_API

#include <numpy/ndarraytypes.h>

typedef struct {
        npy_int64 days;
        npy_int32 hrs, min, sec, ms, us, ns, seconds, microseconds, nanoseconds;
} pandas_timedeltastruct;

extern const npy_datetimestruct _NS_MIN_DTS;
extern const npy_datetimestruct _NS_MAX_DTS;

// stuff pandas needs
// ----------------------------------------------------------------------------

int convert_pydatetime_to_datetimestruct(PyObject *dtobj,
                                         npy_datetimestruct *out);

npy_datetime npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT base,
                                            const npy_datetimestruct *dts);

void pandas_datetime_to_datetimestruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                       npy_datetimestruct *result);

void pandas_timedelta_to_timedeltastruct(npy_timedelta val,
                                         NPY_DATETIMEUNIT fr,
                                         pandas_timedeltastruct *result);

extern const int days_per_month_table[2][12];

// stuff numpy-derived code needs in header
// ----------------------------------------------------------------------------

int is_leapyear(npy_int64 year);

/*
 * Calculates the days offset from the 1970 epoch.
 */
npy_int64
get_datetimestruct_days(const npy_datetimestruct *dts);


/*
 * Compares two npy_datetimestruct objects chronologically
 */
int cmp_npy_datetimestruct(const npy_datetimestruct *a,
                           const npy_datetimestruct *b);


/*
 * Adjusts a datetimestruct based on a minutes offset. Assumes
 * the current values are valid.
 */
void
add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes);


#endif  // PANDAS__LIBS_TSLIBS_SRC_DATETIME_NP_DATETIME_H_
