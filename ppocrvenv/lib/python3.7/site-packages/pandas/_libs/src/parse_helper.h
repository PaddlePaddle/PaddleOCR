/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.
*/

#ifndef PANDAS__LIBS_SRC_PARSE_HELPER_H_
#define PANDAS__LIBS_SRC_PARSE_HELPER_H_

#include <float.h>
#include "parser/tokenizer.h"

int to_double(char *item, double *p_value, char sci, char decimal,
              int *maybe_int) {
    char *p_end = NULL;
    int error = 0;

    /* Switch to precise xstrtod GH 31364 */
    *p_value = precise_xstrtod(item, &p_end, decimal, sci, '\0', 1,
                               &error, maybe_int);

    return (error == 0) && (!*p_end);
}

int floatify(PyObject *str, double *result, int *maybe_int) {
    int status;
    char *data;
    PyObject *tmp = NULL;
    const char sci = 'E';
    const char dec = '.';

    if (PyBytes_Check(str)) {
        data = PyBytes_AS_STRING(str);
    } else if (PyUnicode_Check(str)) {
        tmp = PyUnicode_AsUTF8String(str);
        if (tmp == NULL) {
            return -1;
        }
        data = PyBytes_AS_STRING(tmp);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid object type");
        return -1;
    }

    status = to_double(data, result, sci, dec, maybe_int);

    if (!status) {
        /* handle inf/-inf infinity/-infinity */
        if (strlen(data) == 3) {
            if (0 == strcasecmp(data, "inf")) {
                *result = HUGE_VAL;
                *maybe_int = 0;
            } else {
                goto parsingerror;
            }
        } else if (strlen(data) == 4) {
            if (0 == strcasecmp(data, "-inf")) {
                *result = -HUGE_VAL;
                *maybe_int = 0;
            } else if (0 == strcasecmp(data, "+inf")) {
                *result = HUGE_VAL;
                *maybe_int = 0;
            } else {
                goto parsingerror;
            }
        } else if (strlen(data) == 8) {
            if (0 == strcasecmp(data, "infinity")) {
                *result = HUGE_VAL;
                *maybe_int = 0;
            } else {
                goto parsingerror;
            }
        } else if (strlen(data) == 9) {
            if (0 == strcasecmp(data, "-infinity")) {
                *result = -HUGE_VAL;
                *maybe_int = 0;
            } else if (0 == strcasecmp(data, "+infinity")) {
                *result = HUGE_VAL;
                *maybe_int = 0;
            } else {
                goto parsingerror;
            }
        } else {
            goto parsingerror;
        }
    }

    Py_XDECREF(tmp);
    return 0;

parsingerror:
    PyErr_Format(PyExc_ValueError, "Unable to parse string \"%s\"", data);
    Py_XDECREF(tmp);
    return -1;
}

#endif  // PANDAS__LIBS_SRC_PARSE_HELPER_H_
