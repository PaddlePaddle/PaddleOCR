/*
Copyright (c) 2011-2013, ESN Social Software AB and Jonas Tarnstrom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the ESN Social Software AB nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ESN SOCIAL SOFTWARE AB OR JONAS TARNSTROM BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Portions of code from MODP_ASCII - Ascii transformations (upper/lower, etc)
https://github.com/client9/stringencoders
Copyright (c) 2007  Nick Galbreath -- nickg [at] modp [dot] com. All rights
reserved.

Numeric decoder derived from from TCL library
https://www.opensource.apple.com/source/tcl/tcl-14/tcl/license.terms
* Copyright (c) 1988-1993 The Regents of the University of California.
* Copyright (c) 1994 Sun Microsystems, Inc.
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL UJSON_NUMPY
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>
#include <ultrajson.h>
#include "date_conversions.h"
#include "datetime.h"

static PyTypeObject *type_decimal;
static PyTypeObject *cls_dataframe;
static PyTypeObject *cls_series;
static PyTypeObject *cls_index;
static PyTypeObject *cls_nat;
static PyTypeObject *cls_na;
PyObject *cls_timedelta;

npy_int64 get_nat(void) { return NPY_MIN_INT64; }

typedef char *(*PFN_PyTypeToUTF8)(JSOBJ obj, JSONTypeContext *ti,
                                  size_t *_outLen);

typedef struct __NpyArrContext {
    PyObject *array;
    char *dataptr;
    int curdim;     // current dimension in array's order
    int stridedim;  // dimension we are striding over
    int inc;        // stride dimension increment (+/- 1)
    npy_intp dim;
    npy_intp stride;
    npy_intp ndim;
    npy_intp index[NPY_MAXDIMS];
    int type_num;
    PyArray_GetItemFunc *getitem;

    char **rowLabels;
    char **columnLabels;
} NpyArrContext;

typedef struct __PdBlockContext {
    int colIdx;
    int ncols;
    int transpose;

    NpyArrContext **npyCtxts;  // NpyArrContext for each column
} PdBlockContext;

typedef struct __TypeContext {
    JSPFN_ITERBEGIN iterBegin;
    JSPFN_ITEREND iterEnd;
    JSPFN_ITERNEXT iterNext;
    JSPFN_ITERGETNAME iterGetName;
    JSPFN_ITERGETVALUE iterGetValue;
    PFN_PyTypeToUTF8 PyTypeToUTF8;
    PyObject *newObj;
    PyObject *dictObj;
    Py_ssize_t index;
    Py_ssize_t size;
    PyObject *itemValue;
    PyObject *itemName;
    PyObject *attrList;
    PyObject *iterator;

    double doubleValue;
    JSINT64 longValue;

    char *cStr;
    NpyArrContext *npyarr;
    PdBlockContext *pdblock;
    int transpose;
    char **rowLabels;
    char **columnLabels;
    npy_intp rowLabelsLen;
    npy_intp columnLabelsLen;
} TypeContext;

typedef struct __PyObjectEncoder {
    JSONObjectEncoder enc;

    // pass through the NpyArrContext when encoding multi-dimensional arrays
    NpyArrContext *npyCtxtPassthru;

    // pass through the PdBlockContext when encoding blocks
    PdBlockContext *blkCtxtPassthru;

    // pass-through to encode numpy data directly
    int npyType;
    void *npyValue;

    int datetimeIso;
    NPY_DATETIMEUNIT datetimeUnit;

    // output format style for pandas data types
    int outputFormat;
    int originalOutputFormat;

    PyObject *defaultHandler;
} PyObjectEncoder;

#define GET_TC(__ptrtc) ((TypeContext *)((__ptrtc)->prv))

enum PANDAS_FORMAT { SPLIT, RECORDS, INDEX, COLUMNS, VALUES };

int PdBlock_iterNext(JSOBJ, JSONTypeContext *);

void *initObjToJSON(void) {
    PyObject *mod_pandas;
    PyObject *mod_nattype;
    PyObject *mod_natype;
    PyObject *mod_decimal = PyImport_ImportModule("decimal");
    type_decimal =
        (PyTypeObject *)PyObject_GetAttrString(mod_decimal, "Decimal");
    Py_DECREF(mod_decimal);

    PyDateTime_IMPORT;

    mod_pandas = PyImport_ImportModule("pandas");
    if (mod_pandas) {
        cls_dataframe =
            (PyTypeObject *)PyObject_GetAttrString(mod_pandas, "DataFrame");
        cls_index = (PyTypeObject *)PyObject_GetAttrString(mod_pandas, "Index");
        cls_series =
            (PyTypeObject *)PyObject_GetAttrString(mod_pandas, "Series");
        Py_DECREF(mod_pandas);
    }

    mod_nattype = PyImport_ImportModule("pandas._libs.tslibs.nattype");
    if (mod_nattype) {
        cls_nat =
            (PyTypeObject *)PyObject_GetAttrString(mod_nattype, "NaTType");
        Py_DECREF(mod_nattype);
    }

    mod_natype = PyImport_ImportModule("pandas._libs.missing");
    if (mod_natype) {
        cls_na = (PyTypeObject *)PyObject_GetAttrString(mod_natype, "NAType");
        Py_DECREF(mod_natype);
    }

    // GH 31463
    return NULL;
}

static TypeContext *createTypeContext(void) {
    TypeContext *pc;

    pc = PyObject_Malloc(sizeof(TypeContext));
    if (!pc) {
        PyErr_NoMemory();
        return NULL;
    }
    pc->newObj = NULL;
    pc->dictObj = NULL;
    pc->itemValue = NULL;
    pc->itemName = NULL;
    pc->attrList = NULL;
    pc->index = 0;
    pc->size = 0;
    pc->longValue = 0;
    pc->doubleValue = 0.0;
    pc->cStr = NULL;
    pc->npyarr = NULL;
    pc->pdblock = NULL;
    pc->rowLabels = NULL;
    pc->columnLabels = NULL;
    pc->transpose = 0;
    pc->rowLabelsLen = 0;
    pc->columnLabelsLen = 0;

    return pc;
}

static PyObject *get_values(PyObject *obj) {
    PyObject *values = NULL;

    if (PyObject_TypeCheck(obj, cls_index) ||
        PyObject_TypeCheck(obj, cls_series)) {
        // The special cases to worry about are dt64tz and category[dt64tz].
        //  In both cases we want the UTC-localized datetime64 ndarray,
        //  without going through and object array of Timestamps.
        values = PyObject_GetAttrString(obj, "values");

        if (values == NULL) {
            // Clear so we can subsequently try another method
            PyErr_Clear();
        } else if (PyObject_HasAttrString(values, "__array__")) {
            // We may have gotten a Categorical or Sparse array so call np.array
            values = PyObject_CallMethod(values, "__array__", NULL);
        } else if (!PyArray_CheckExact(values)) {
            // Didn't get a numpy array, so keep trying
            Py_DECREF(values);
            values = NULL;
        }
    }

    if (values == NULL) {
        PyObject *typeRepr = PyObject_Repr((PyObject *)Py_TYPE(obj));
        PyObject *repr;
        if (PyObject_HasAttrString(obj, "dtype")) {
            PyObject *dtype = PyObject_GetAttrString(obj, "dtype");
            repr = PyObject_Repr(dtype);
            Py_DECREF(dtype);
        } else {
            repr = PyUnicode_FromString("<unknown dtype>");
        }

        PyErr_Format(PyExc_ValueError, "%R or %R are not JSON serializable yet",
                     repr, typeRepr);
        Py_DECREF(repr);
        Py_DECREF(typeRepr);

        return NULL;
    }

    return values;
}

static PyObject *get_sub_attr(PyObject *obj, char *attr, char *subAttr) {
    PyObject *tmp = PyObject_GetAttrString(obj, attr);
    PyObject *ret;

    if (tmp == 0) {
        return 0;
    }
    ret = PyObject_GetAttrString(tmp, subAttr);
    Py_DECREF(tmp);

    return ret;
}

static Py_ssize_t get_attr_length(PyObject *obj, char *attr) {
    PyObject *tmp = PyObject_GetAttrString(obj, attr);
    Py_ssize_t ret;

    if (tmp == 0) {
        return 0;
    }
    ret = PyObject_Length(tmp);
    Py_DECREF(tmp);

    if (ret == -1) {
        return 0;
    }

    return ret;
}

static int is_simple_frame(PyObject *obj) {
    PyObject *mgr = PyObject_GetAttrString(obj, "_mgr");
    if (!mgr) {
        return 0;
    }
    int ret;
    if (PyObject_HasAttrString(mgr, "blocks")) {
        ret = (get_attr_length(mgr, "blocks") <= 1);
    } else {
        ret = 0;
    }

    Py_DECREF(mgr);
    return ret;
}

static npy_int64 get_long_attr(PyObject *o, const char *attr) {
    npy_int64 long_val;
    PyObject *value = PyObject_GetAttrString(o, attr);
    long_val =
        (PyLong_Check(value) ? PyLong_AsLongLong(value) : PyLong_AsLong(value));
    Py_DECREF(value);
    return long_val;
}

static npy_float64 total_seconds(PyObject *td) {
    npy_float64 double_val;
    PyObject *value = PyObject_CallMethod(td, "total_seconds", NULL);
    double_val = PyFloat_AS_DOUBLE(value);
    Py_DECREF(value);
    return double_val;
}

static char *PyBytesToUTF8(JSOBJ _obj, JSONTypeContext *Py_UNUSED(tc),
                           size_t *_outLen) {
    PyObject *obj = (PyObject *)_obj;
    *_outLen = PyBytes_GET_SIZE(obj);
    return PyBytes_AS_STRING(obj);
}

static char *PyUnicodeToUTF8(JSOBJ _obj, JSONTypeContext *Py_UNUSED(tc),
                             size_t *_outLen) {
    return (char *)PyUnicode_AsUTF8AndSize(_obj, (Py_ssize_t *)_outLen);
}

/* JSON callback. returns a char* and mutates the pointer to *len */
static char *NpyDateTimeToIsoCallback(JSOBJ Py_UNUSED(unused),
                                      JSONTypeContext *tc, size_t *len) {
    NPY_DATETIMEUNIT base = ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
    return int64ToIso(GET_TC(tc)->longValue, base, len);
}

/* JSON callback. returns a char* and mutates the pointer to *len */
static char *NpyTimeDeltaToIsoCallback(JSOBJ Py_UNUSED(unused),
                                       JSONTypeContext *tc, size_t *len) {
    return int64ToIsoDuration(GET_TC(tc)->longValue, len);
}

/* JSON callback */
static char *PyDateTimeToIsoCallback(JSOBJ obj, JSONTypeContext *tc,
                                     size_t *len) {
    if (!PyDate_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected date object");
        return NULL;
    }

    NPY_DATETIMEUNIT base = ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
    return PyDateTimeToIso(obj, base, len);
}

static char *PyTimeToJSON(JSOBJ _obj, JSONTypeContext *tc, size_t *outLen) {
    PyObject *obj = (PyObject *)_obj;
    PyObject *str;
    PyObject *tmp;

    str = PyObject_CallMethod(obj, "isoformat", NULL);
    if (str == NULL) {
        *outLen = 0;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert time");
        }
        ((JSONObjectEncoder *)tc->encoder)->errorMsg = "";
        return NULL;
    }
    if (PyUnicode_Check(str)) {
        tmp = str;
        str = PyUnicode_AsUTF8String(str);
        Py_DECREF(tmp);
    }

    GET_TC(tc)->newObj = str;

    *outLen = PyBytes_GET_SIZE(str);
    char *outValue = PyBytes_AS_STRING(str);
    return outValue;
}

//=============================================================================
// Numpy array iteration functions
//=============================================================================

static void NpyArr_freeItemValue(JSOBJ Py_UNUSED(_obj), JSONTypeContext *tc) {
    if (GET_TC(tc)->npyarr &&
        GET_TC(tc)->itemValue != GET_TC(tc)->npyarr->array) {
        Py_XDECREF(GET_TC(tc)->itemValue);
        GET_TC(tc)->itemValue = NULL;
    }
}

int NpyArr_iterNextNone(JSOBJ Py_UNUSED(_obj), JSONTypeContext *Py_UNUSED(tc)) {
    return 0;
}

void NpyArr_iterBegin(JSOBJ _obj, JSONTypeContext *tc) {
    PyArrayObject *obj;
    NpyArrContext *npyarr;

    if (GET_TC(tc)->newObj) {
        obj = (PyArrayObject *)GET_TC(tc)->newObj;
    } else {
        obj = (PyArrayObject *)_obj;
    }

    npyarr = PyObject_Malloc(sizeof(NpyArrContext));
    GET_TC(tc)->npyarr = npyarr;

    if (!npyarr) {
        PyErr_NoMemory();
        GET_TC(tc)->iterNext = NpyArr_iterNextNone;
        return;
    }

    npyarr->array = (PyObject *)obj;
    npyarr->getitem = (PyArray_GetItemFunc *)PyArray_DESCR(obj)->f->getitem;
    npyarr->dataptr = PyArray_DATA(obj);
    npyarr->ndim = PyArray_NDIM(obj) - 1;
    npyarr->curdim = 0;
    npyarr->type_num = PyArray_DESCR(obj)->type_num;

    if (GET_TC(tc)->transpose) {
        npyarr->dim = PyArray_DIM(obj, npyarr->ndim);
        npyarr->stride = PyArray_STRIDE(obj, npyarr->ndim);
        npyarr->stridedim = npyarr->ndim;
        npyarr->index[npyarr->ndim] = 0;
        npyarr->inc = -1;
    } else {
        npyarr->dim = PyArray_DIM(obj, 0);
        npyarr->stride = PyArray_STRIDE(obj, 0);
        npyarr->stridedim = 0;
        npyarr->index[0] = 0;
        npyarr->inc = 1;
    }

    npyarr->columnLabels = GET_TC(tc)->columnLabels;
    npyarr->rowLabels = GET_TC(tc)->rowLabels;
}

void NpyArr_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
    NpyArrContext *npyarr = GET_TC(tc)->npyarr;

    if (npyarr) {
        NpyArr_freeItemValue(obj, tc);
        PyObject_Free(npyarr);
    }
}

void NpyArrPassThru_iterBegin(JSOBJ Py_UNUSED(obj),
                              JSONTypeContext *Py_UNUSED(tc)) {}

void NpyArrPassThru_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
    NpyArrContext *npyarr = GET_TC(tc)->npyarr;
    // finished this dimension, reset the data pointer
    npyarr->curdim--;
    npyarr->dataptr -= npyarr->stride * npyarr->index[npyarr->stridedim];
    npyarr->stridedim -= npyarr->inc;
    npyarr->dim = PyArray_DIM(npyarr->array, npyarr->stridedim);
    npyarr->stride = PyArray_STRIDE(npyarr->array, npyarr->stridedim);
    npyarr->dataptr += npyarr->stride;

    NpyArr_freeItemValue(obj, tc);
}

int NpyArr_iterNextItem(JSOBJ obj, JSONTypeContext *tc) {
    NpyArrContext *npyarr = GET_TC(tc)->npyarr;

    if (PyErr_Occurred()) {
        return 0;
    }

    if (npyarr->index[npyarr->stridedim] >= npyarr->dim) {
        return 0;
    }

    NpyArr_freeItemValue(obj, tc);

    if (PyArray_ISDATETIME(npyarr->array)) {
        GET_TC(tc)->itemValue = obj;
        Py_INCREF(obj);
        ((PyObjectEncoder *)tc->encoder)->npyType = PyArray_TYPE(npyarr->array);
        ((PyObjectEncoder *)tc->encoder)->npyValue = npyarr->dataptr;
        ((PyObjectEncoder *)tc->encoder)->npyCtxtPassthru = npyarr;
    } else {
        GET_TC(tc)->itemValue = npyarr->getitem(npyarr->dataptr, npyarr->array);
    }

    npyarr->dataptr += npyarr->stride;
    npyarr->index[npyarr->stridedim]++;
    return 1;
}

int NpyArr_iterNext(JSOBJ _obj, JSONTypeContext *tc) {
    NpyArrContext *npyarr = GET_TC(tc)->npyarr;

    if (PyErr_Occurred()) {
        return 0;
    }

    if (npyarr->curdim >= npyarr->ndim ||
        npyarr->index[npyarr->stridedim] >= npyarr->dim) {
        // innermost dimension, start retrieving item values
        GET_TC(tc)->iterNext = NpyArr_iterNextItem;
        return NpyArr_iterNextItem(_obj, tc);
    }

    // dig a dimension deeper
    npyarr->index[npyarr->stridedim]++;

    npyarr->curdim++;
    npyarr->stridedim += npyarr->inc;
    npyarr->dim = PyArray_DIM(npyarr->array, npyarr->stridedim);
    npyarr->stride = PyArray_STRIDE(npyarr->array, npyarr->stridedim);
    npyarr->index[npyarr->stridedim] = 0;

    ((PyObjectEncoder *)tc->encoder)->npyCtxtPassthru = npyarr;
    GET_TC(tc)->itemValue = npyarr->array;
    return 1;
}

JSOBJ NpyArr_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

char *NpyArr_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                         size_t *outLen) {
    NpyArrContext *npyarr = GET_TC(tc)->npyarr;
    npy_intp idx;
    char *cStr;

    if (GET_TC(tc)->iterNext == NpyArr_iterNextItem) {
        idx = npyarr->index[npyarr->stridedim] - 1;
        cStr = npyarr->columnLabels[idx];
    } else {
        idx = npyarr->index[npyarr->stridedim - npyarr->inc] - 1;
        cStr = npyarr->rowLabels[idx];
    }

    *outLen = strlen(cStr);

    return cStr;
}

//=============================================================================
// Pandas block iteration functions
//
// Serialises a DataFrame column by column to avoid unnecessary data copies and
// more representative serialisation when dealing with mixed dtypes.
//
// Uses a dedicated NpyArrContext for each column.
//=============================================================================

void PdBlockPassThru_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;

    if (blkCtxt->transpose) {
        blkCtxt->colIdx++;
    } else {
        blkCtxt->colIdx = 0;
    }

    NpyArr_freeItemValue(obj, tc);
}

int PdBlock_iterNextItem(JSOBJ obj, JSONTypeContext *tc) {
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;

    if (blkCtxt->colIdx >= blkCtxt->ncols) {
        return 0;
    }

    GET_TC(tc)->npyarr = blkCtxt->npyCtxts[blkCtxt->colIdx];
    blkCtxt->colIdx++;
    return NpyArr_iterNextItem(obj, tc);
}

char *PdBlock_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                          size_t *outLen) {
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;
    NpyArrContext *npyarr = blkCtxt->npyCtxts[0];
    npy_intp idx;
    char *cStr;

    if (GET_TC(tc)->iterNext == PdBlock_iterNextItem) {
        idx = blkCtxt->colIdx - 1;
        cStr = npyarr->columnLabels[idx];
    } else {
        idx = GET_TC(tc)->iterNext != PdBlock_iterNext
                  ? npyarr->index[npyarr->stridedim - npyarr->inc] - 1
                  : npyarr->index[npyarr->stridedim];

        cStr = npyarr->rowLabels[idx];
    }

    *outLen = strlen(cStr);
    return cStr;
}

char *PdBlock_iterGetName_Transpose(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                                    size_t *outLen) {
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;
    NpyArrContext *npyarr = blkCtxt->npyCtxts[blkCtxt->colIdx];
    npy_intp idx;
    char *cStr;

    if (GET_TC(tc)->iterNext == NpyArr_iterNextItem) {
        idx = npyarr->index[npyarr->stridedim] - 1;
        cStr = npyarr->columnLabels[idx];
    } else {
        idx = blkCtxt->colIdx;
        cStr = npyarr->rowLabels[idx];
    }

    *outLen = strlen(cStr);
    return cStr;
}

int PdBlock_iterNext(JSOBJ obj, JSONTypeContext *tc) {
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;
    NpyArrContext *npyarr;

    if (PyErr_Occurred() || ((JSONObjectEncoder *)tc->encoder)->errorMsg) {
        return 0;
    }

    if (blkCtxt->transpose) {
        if (blkCtxt->colIdx >= blkCtxt->ncols) {
            return 0;
        }
    } else {
        npyarr = blkCtxt->npyCtxts[0];
        if (npyarr->index[npyarr->stridedim] >= npyarr->dim) {
            return 0;
        }
    }

    ((PyObjectEncoder *)tc->encoder)->blkCtxtPassthru = blkCtxt;
    GET_TC(tc)->itemValue = obj;

    return 1;
}

void PdBlockPassThru_iterBegin(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    PdBlockContext *blkCtxt = GET_TC(tc)->pdblock;

    if (blkCtxt->transpose) {
        // if transposed we exhaust each column before moving to the next
        GET_TC(tc)->iterNext = NpyArr_iterNextItem;
        GET_TC(tc)->iterGetName = PdBlock_iterGetName_Transpose;
        GET_TC(tc)->npyarr = blkCtxt->npyCtxts[blkCtxt->colIdx];
    }
}

void PdBlock_iterBegin(JSOBJ _obj, JSONTypeContext *tc) {
    PyObject *obj, *values, *arrays, *array;
    PdBlockContext *blkCtxt;
    NpyArrContext *npyarr;
    Py_ssize_t i;

    obj = (PyObject *)_obj;

    GET_TC(tc)->iterGetName = GET_TC(tc)->transpose
                                  ? PdBlock_iterGetName_Transpose
                                  : PdBlock_iterGetName;

    blkCtxt = PyObject_Malloc(sizeof(PdBlockContext));
    if (!blkCtxt) {
        PyErr_NoMemory();
        GET_TC(tc)->iterNext = NpyArr_iterNextNone;
        return;
    }
    GET_TC(tc)->pdblock = blkCtxt;

    blkCtxt->colIdx = 0;
    blkCtxt->transpose = GET_TC(tc)->transpose;
    blkCtxt->ncols = get_attr_length(obj, "columns");

    if (blkCtxt->ncols == 0) {
        blkCtxt->npyCtxts = NULL;

        GET_TC(tc)->iterNext = NpyArr_iterNextNone;
        return;
    }

    blkCtxt->npyCtxts =
        PyObject_Malloc(sizeof(NpyArrContext *) * blkCtxt->ncols);
    if (!blkCtxt->npyCtxts) {
        PyErr_NoMemory();
        GET_TC(tc)->iterNext = NpyArr_iterNextNone;
        return;
    }

    arrays = get_sub_attr(obj, "_mgr", "column_arrays");
    if (!arrays) {
        GET_TC(tc)->iterNext = NpyArr_iterNextNone;
        return;
    }

    for (i = 0; i < PyObject_Length(arrays); i++) {
        array = PyList_GET_ITEM(arrays, i);
        if (!array) {
            GET_TC(tc)->iterNext = NpyArr_iterNextNone;
            goto ARR_RET;
        }

        // ensure we have a numpy array (i.e. np.asarray)
        values = PyObject_CallMethod(array, "__array__", NULL);
        if ((!values) || (!PyArray_CheckExact(values))) {
            // Didn't get a numpy array
            ((JSONObjectEncoder *)tc->encoder)->errorMsg = "";
            GET_TC(tc)->iterNext = NpyArr_iterNextNone;
            goto ARR_RET;
        }

        GET_TC(tc)->newObj = values;

        // init a dedicated context for this column
        NpyArr_iterBegin(obj, tc);
        npyarr = GET_TC(tc)->npyarr;

        GET_TC(tc)->itemValue = NULL;
        ((PyObjectEncoder *)tc->encoder)->npyCtxtPassthru = NULL;

        blkCtxt->npyCtxts[i] = npyarr;
        GET_TC(tc)->newObj = NULL;
    }
    GET_TC(tc)->npyarr = blkCtxt->npyCtxts[0];
    goto ARR_RET;

ARR_RET:
    Py_DECREF(arrays);
}

void PdBlock_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
    PdBlockContext *blkCtxt;
    NpyArrContext *npyarr;
    int i;

    GET_TC(tc)->itemValue = NULL;
    npyarr = GET_TC(tc)->npyarr;

    blkCtxt = GET_TC(tc)->pdblock;

    if (blkCtxt) {
        for (i = 0; i < blkCtxt->ncols; i++) {
            npyarr = blkCtxt->npyCtxts[i];
            if (npyarr) {
                if (npyarr->array) {
                    Py_DECREF(npyarr->array);
                    npyarr->array = NULL;
                }

                GET_TC(tc)->npyarr = npyarr;
                NpyArr_iterEnd(obj, tc);

                blkCtxt->npyCtxts[i] = NULL;
            }
        }

        if (blkCtxt->npyCtxts) {
            PyObject_Free(blkCtxt->npyCtxts);
        }
        PyObject_Free(blkCtxt);
    }
}

//=============================================================================
// Tuple iteration functions
// itemValue is borrowed reference, no ref counting
//=============================================================================
void Tuple_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
    GET_TC(tc)->index = 0;
    GET_TC(tc)->size = PyTuple_GET_SIZE((PyObject *)obj);
    GET_TC(tc)->itemValue = NULL;
}

int Tuple_iterNext(JSOBJ obj, JSONTypeContext *tc) {
    PyObject *item;

    if (GET_TC(tc)->index >= GET_TC(tc)->size) {
        return 0;
    }

    item = PyTuple_GET_ITEM(obj, GET_TC(tc)->index);

    GET_TC(tc)->itemValue = item;
    GET_TC(tc)->index++;
    return 1;
}

void Tuple_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *Py_UNUSED(tc)) {}

JSOBJ Tuple_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

char *Tuple_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *Py_UNUSED(tc),
                        size_t *Py_UNUSED(outLen)) {
    return NULL;
}

//=============================================================================
// Set iteration functions
// itemValue is borrowed reference, no ref counting
//=============================================================================
void Set_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
    GET_TC(tc)->itemValue = NULL;
    GET_TC(tc)->iterator = PyObject_GetIter(obj);
}

int Set_iterNext(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    PyObject *item;

    if (GET_TC(tc)->itemValue) {
        Py_DECREF(GET_TC(tc)->itemValue);
        GET_TC(tc)->itemValue = NULL;
    }

    item = PyIter_Next(GET_TC(tc)->iterator);

    if (item == NULL) {
        return 0;
    }

    GET_TC(tc)->itemValue = item;
    return 1;
}

void Set_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    if (GET_TC(tc)->itemValue) {
        Py_DECREF(GET_TC(tc)->itemValue);
        GET_TC(tc)->itemValue = NULL;
    }

    if (GET_TC(tc)->iterator) {
        Py_DECREF(GET_TC(tc)->iterator);
        GET_TC(tc)->iterator = NULL;
    }
}

JSOBJ Set_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

char *Set_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *Py_UNUSED(tc),
                      size_t *Py_UNUSED(outLen)) {
    return NULL;
}

//=============================================================================
// Dir iteration functions
// itemName ref is borrowed from PyObject_Dir (attrList). No refcount
// itemValue ref is from PyObject_GetAttr. Ref counted
//=============================================================================
void Dir_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
    GET_TC(tc)->attrList = PyObject_Dir(obj);
    GET_TC(tc)->index = 0;
    GET_TC(tc)->size = PyList_GET_SIZE(GET_TC(tc)->attrList);
}

void Dir_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    if (GET_TC(tc)->itemValue) {
        Py_DECREF(GET_TC(tc)->itemValue);
        GET_TC(tc)->itemValue = NULL;
    }

    if (GET_TC(tc)->itemName) {
        Py_DECREF(GET_TC(tc)->itemName);
        GET_TC(tc)->itemName = NULL;
    }

    Py_DECREF((PyObject *)GET_TC(tc)->attrList);
}

int Dir_iterNext(JSOBJ _obj, JSONTypeContext *tc) {
    PyObject *obj = (PyObject *)_obj;
    PyObject *itemValue = GET_TC(tc)->itemValue;
    PyObject *itemName = GET_TC(tc)->itemName;
    PyObject *attr;
    PyObject *attrName;
    char *attrStr;

    if (PyErr_Occurred() || ((JSONObjectEncoder *)tc->encoder)->errorMsg) {
        return 0;
    }

    if (itemValue) {
        Py_DECREF(GET_TC(tc)->itemValue);
        GET_TC(tc)->itemValue = itemValue = NULL;
    }

    if (itemName) {
        Py_DECREF(GET_TC(tc)->itemName);
        GET_TC(tc)->itemName = itemName = NULL;
    }

    for (; GET_TC(tc)->index < GET_TC(tc)->size; GET_TC(tc)->index++) {
        attrName = PyList_GET_ITEM(GET_TC(tc)->attrList, GET_TC(tc)->index);
        attr = PyUnicode_AsUTF8String(attrName);
        attrStr = PyBytes_AS_STRING(attr);

        if (attrStr[0] == '_') {
            Py_DECREF(attr);
            continue;
        }

        itemValue = PyObject_GetAttr(obj, attrName);
        if (itemValue == NULL) {
            PyErr_Clear();
            Py_DECREF(attr);
            continue;
        }

        if (PyCallable_Check(itemValue)) {
            Py_DECREF(itemValue);
            Py_DECREF(attr);
            continue;
        }

        GET_TC(tc)->itemName = itemName;
        GET_TC(tc)->itemValue = itemValue;
        GET_TC(tc)->index++;

        itemName = attr;
        break;
    }

    if (itemName == NULL) {
        GET_TC(tc)->index = GET_TC(tc)->size;
        GET_TC(tc)->itemValue = NULL;
        return 0;
    }

    GET_TC(tc)->itemName = itemName;
    GET_TC(tc)->itemValue = itemValue;
    GET_TC(tc)->index++;

    return 1;
}

JSOBJ Dir_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

char *Dir_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                      size_t *outLen) {
    *outLen = PyBytes_GET_SIZE(GET_TC(tc)->itemName);
    return PyBytes_AS_STRING(GET_TC(tc)->itemName);
}

//=============================================================================
// List iteration functions
// itemValue is borrowed from object (which is list). No refcounting
//=============================================================================
void List_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
    GET_TC(tc)->index = 0;
    GET_TC(tc)->size = PyList_GET_SIZE((PyObject *)obj);
}

int List_iterNext(JSOBJ obj, JSONTypeContext *tc) {
    if (GET_TC(tc)->index >= GET_TC(tc)->size) {
        return 0;
    }

    GET_TC(tc)->itemValue = PyList_GET_ITEM(obj, GET_TC(tc)->index);
    GET_TC(tc)->index++;
    return 1;
}

void List_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *Py_UNUSED(tc)) {}

JSOBJ List_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

char *List_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *Py_UNUSED(tc),
                       size_t *Py_UNUSED(outLen)) {
    return NULL;
}

//=============================================================================
// pandas Index iteration functions
//=============================================================================
void Index_iterBegin(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    GET_TC(tc)->index = 0;
    GET_TC(tc)->cStr = PyObject_Malloc(20 * sizeof(char));
    if (!GET_TC(tc)->cStr) {
        PyErr_NoMemory();
    }
}

int Index_iterNext(JSOBJ obj, JSONTypeContext *tc) {
    Py_ssize_t index;
    if (!GET_TC(tc)->cStr) {
        return 0;
    }

    index = GET_TC(tc)->index;
    Py_XDECREF(GET_TC(tc)->itemValue);
    if (index == 0) {
        memcpy(GET_TC(tc)->cStr, "name", sizeof(char) * 5);
        GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "name");
    } else if (index == 1) {
        memcpy(GET_TC(tc)->cStr, "data", sizeof(char) * 5);
        GET_TC(tc)->itemValue = get_values(obj);
        if (!GET_TC(tc)->itemValue) {
            return 0;
        }
    } else {
        return 0;
    }

    GET_TC(tc)->index++;
    return 1;
}

void Index_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *Py_UNUSED(tc)) {}

JSOBJ Index_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

char *Index_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                        size_t *outLen) {
    *outLen = strlen(GET_TC(tc)->cStr);
    return GET_TC(tc)->cStr;
}

//=============================================================================
// pandas Series iteration functions
//=============================================================================
void Series_iterBegin(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    PyObjectEncoder *enc = (PyObjectEncoder *)tc->encoder;
    GET_TC(tc)->index = 0;
    GET_TC(tc)->cStr = PyObject_Malloc(20 * sizeof(char));
    enc->outputFormat = VALUES;  // for contained series
    if (!GET_TC(tc)->cStr) {
        PyErr_NoMemory();
    }
}

int Series_iterNext(JSOBJ obj, JSONTypeContext *tc) {
    Py_ssize_t index;
    if (!GET_TC(tc)->cStr) {
        return 0;
    }

    index = GET_TC(tc)->index;
    Py_XDECREF(GET_TC(tc)->itemValue);
    if (index == 0) {
        memcpy(GET_TC(tc)->cStr, "name", sizeof(char) * 5);
        GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "name");
    } else if (index == 1) {
        memcpy(GET_TC(tc)->cStr, "index", sizeof(char) * 6);
        GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "index");
    } else if (index == 2) {
        memcpy(GET_TC(tc)->cStr, "data", sizeof(char) * 5);
        GET_TC(tc)->itemValue = get_values(obj);
        if (!GET_TC(tc)->itemValue) {
            return 0;
        }
    } else {
        return 0;
    }

    GET_TC(tc)->index++;
    return 1;
}

void Series_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    PyObjectEncoder *enc = (PyObjectEncoder *)tc->encoder;
    enc->outputFormat = enc->originalOutputFormat;
}

JSOBJ Series_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

char *Series_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                         size_t *outLen) {
    *outLen = strlen(GET_TC(tc)->cStr);
    return GET_TC(tc)->cStr;
}

//=============================================================================
// pandas DataFrame iteration functions
//=============================================================================
void DataFrame_iterBegin(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    PyObjectEncoder *enc = (PyObjectEncoder *)tc->encoder;
    GET_TC(tc)->index = 0;
    GET_TC(tc)->cStr = PyObject_Malloc(20 * sizeof(char));
    enc->outputFormat = VALUES;  // for contained series & index
    if (!GET_TC(tc)->cStr) {
        PyErr_NoMemory();
    }
}

int DataFrame_iterNext(JSOBJ obj, JSONTypeContext *tc) {
    Py_ssize_t index;
    if (!GET_TC(tc)->cStr) {
        return 0;
    }

    index = GET_TC(tc)->index;
    Py_XDECREF(GET_TC(tc)->itemValue);
    if (index == 0) {
        memcpy(GET_TC(tc)->cStr, "columns", sizeof(char) * 8);
        GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "columns");
    } else if (index == 1) {
        memcpy(GET_TC(tc)->cStr, "index", sizeof(char) * 6);
        GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "index");
    } else if (index == 2) {
        memcpy(GET_TC(tc)->cStr, "data", sizeof(char) * 5);
        if (is_simple_frame(obj)) {
            GET_TC(tc)->itemValue = PyObject_GetAttrString(obj, "values");
            if (!GET_TC(tc)->itemValue) {
                return 0;
            }
        } else {
            Py_INCREF(obj);
            GET_TC(tc)->itemValue = obj;
        }
    } else {
        return 0;
    }

    GET_TC(tc)->index++;
    return 1;
}

void DataFrame_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    PyObjectEncoder *enc = (PyObjectEncoder *)tc->encoder;
    enc->outputFormat = enc->originalOutputFormat;
}

JSOBJ DataFrame_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

char *DataFrame_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                            size_t *outLen) {
    *outLen = strlen(GET_TC(tc)->cStr);
    return GET_TC(tc)->cStr;
}

//=============================================================================
// Dict iteration functions
// itemName might converted to string (Python_Str). Do refCounting
// itemValue is borrowed from object (which is dict). No refCounting
//=============================================================================
void Dict_iterBegin(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    GET_TC(tc)->index = 0;
}

int Dict_iterNext(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    PyObject *itemNameTmp;

    if (GET_TC(tc)->itemName) {
        Py_DECREF(GET_TC(tc)->itemName);
        GET_TC(tc)->itemName = NULL;
    }

    if (!PyDict_Next((PyObject *)GET_TC(tc)->dictObj, &GET_TC(tc)->index,
                     &GET_TC(tc)->itemName, &GET_TC(tc)->itemValue)) {
        return 0;
    }

    if (PyUnicode_Check(GET_TC(tc)->itemName)) {
        GET_TC(tc)->itemName = PyUnicode_AsUTF8String(GET_TC(tc)->itemName);
    } else if (!PyBytes_Check(GET_TC(tc)->itemName)) {
        GET_TC(tc)->itemName = PyObject_Str(GET_TC(tc)->itemName);
        itemNameTmp = GET_TC(tc)->itemName;
        GET_TC(tc)->itemName = PyUnicode_AsUTF8String(GET_TC(tc)->itemName);
        Py_DECREF(itemNameTmp);
    } else {
        Py_INCREF(GET_TC(tc)->itemName);
    }
    return 1;
}

void Dict_iterEnd(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    if (GET_TC(tc)->itemName) {
        Py_DECREF(GET_TC(tc)->itemName);
        GET_TC(tc)->itemName = NULL;
    }
    Py_DECREF(GET_TC(tc)->dictObj);
}

JSOBJ Dict_iterGetValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->itemValue;
}

char *Dict_iterGetName(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc,
                       size_t *outLen) {
    *outLen = PyBytes_GET_SIZE(GET_TC(tc)->itemName);
    return PyBytes_AS_STRING(GET_TC(tc)->itemName);
}

void NpyArr_freeLabels(char **labels, npy_intp len) {
    npy_intp i;

    if (labels) {
        for (i = 0; i < len; i++) {
            PyObject_Free(labels[i]);
        }
        PyObject_Free(labels);
    }
}

/*
 * Function: NpyArr_encodeLabels
 * -----------------------------
 *
 * Builds an array of "encoded" labels.
 *
 * labels: PyArrayObject pointer for labels to be "encoded"
 * num : number of labels
 *
 * "encode" is quoted above because we aren't really doing encoding
 * For historical reasons this function would actually encode the entire
 * array into a separate buffer with a separate call to JSON_Encode
 * and would leave it to complex pointer manipulation from there to
 * unpack values as needed. To make things simpler and more idiomatic
 * this has instead just stringified any input save for datetime values,
 * which may need to be represented in various formats.
 */
char **NpyArr_encodeLabels(PyArrayObject *labels, PyObjectEncoder *enc,
                           npy_intp num) {
    // NOTE this function steals a reference to labels.
    PyObject *item = NULL;
    size_t len;
    npy_intp i, stride;
    char **ret;
    char *dataptr, *cLabel;
    int type_num;
    NPY_DATETIMEUNIT base = enc->datetimeUnit;

    if (!labels) {
        return 0;
    }

    if (PyArray_SIZE(labels) < num) {
        PyErr_SetString(
            PyExc_ValueError,
            "Label array sizes do not match corresponding data shape");
        Py_DECREF(labels);
        return 0;
    }

    ret = PyObject_Malloc(sizeof(char *) * num);
    if (!ret) {
        PyErr_NoMemory();
        Py_DECREF(labels);
        return 0;
    }

    for (i = 0; i < num; i++) {
        ret[i] = NULL;
    }

    stride = PyArray_STRIDE(labels, 0);
    dataptr = PyArray_DATA(labels);
    type_num = PyArray_TYPE(labels);

    for (i = 0; i < num; i++) {
        item = PyArray_GETITEM(labels, dataptr);
        if (!item) {
            NpyArr_freeLabels(ret, num);
            ret = 0;
            break;
        }

        int is_datetimelike = 0;
        npy_int64 nanosecVal;
        if (PyTypeNum_ISDATETIME(type_num)) {
            is_datetimelike = 1;
            PyArray_VectorUnaryFunc *castfunc =
                PyArray_GetCastFunc(PyArray_DescrFromType(type_num), NPY_INT64);
            if (!castfunc) {
                PyErr_Format(PyExc_ValueError,
                             "Cannot cast numpy dtype %d to long",
                             enc->npyType);
            }
            castfunc(dataptr, &nanosecVal, 1, NULL, NULL);
        } else if (PyDate_Check(item) || PyDelta_Check(item)) {
            is_datetimelike = 1;
            if (PyObject_HasAttrString(item, "value")) {
                nanosecVal = get_long_attr(item, "value");
            } else {
                if (PyDelta_Check(item)) {
                    nanosecVal = total_seconds(item) *
                                 1000000000LL;  // nanoseconds per second
                } else {
                    // datetime.* objects don't follow above rules
                    nanosecVal = PyDateTimeToEpoch(item, NPY_FR_ns);
                }
            }
        }

        if (is_datetimelike) {
            if (nanosecVal == get_nat()) {
                len = 4;
                cLabel = PyObject_Malloc(len + 1);
                strncpy(cLabel, "null", len + 1);
            } else {
                if (enc->datetimeIso) {
                    if ((type_num == NPY_TIMEDELTA) || (PyDelta_Check(item))) {
                        cLabel = int64ToIsoDuration(nanosecVal, &len);
                    } else {
                        if (type_num == NPY_DATETIME) {
                            cLabel = int64ToIso(nanosecVal, base, &len);
                        } else {
                            cLabel = PyDateTimeToIso(item, base, &len);
                        }
                    }
                    if (cLabel == NULL) {
                        Py_DECREF(item);
                        NpyArr_freeLabels(ret, num);
                        ret = 0;
                        break;
                    }
                } else {
                    int size_of_cLabel = 21;  // 21 chars for int 64
                    cLabel = PyObject_Malloc(size_of_cLabel);
                    snprintf(cLabel, size_of_cLabel, "%" NPY_DATETIME_FMT,
                            NpyDateTimeToEpoch(nanosecVal, base));
                    len = strlen(cLabel);
                }
            }
        } else {  // Fallback to string representation
            // Replace item with the string to keep it alive.
            Py_SETREF(item, PyObject_Str(item));
            if (item == NULL) {
                NpyArr_freeLabels(ret, num);
                ret = 0;
                break;
            }

            cLabel = (char *)PyUnicode_AsUTF8(item);
            len = strlen(cLabel);
        }

        // Add 1 to include NULL terminator
        ret[i] = PyObject_Malloc(len + 1);
        memcpy(ret[i], cLabel, len + 1);
        Py_DECREF(item);

        if (is_datetimelike) {
            PyObject_Free(cLabel);
        }

        if (PyErr_Occurred()) {
            NpyArr_freeLabels(ret, num);
            ret = 0;
            break;
        }

        if (!ret[i]) {
            PyErr_NoMemory();
            ret = 0;
            break;
        }

        dataptr += stride;
    }

    Py_DECREF(labels);
    return ret;
}

void Object_invokeDefaultHandler(PyObject *obj, PyObjectEncoder *enc) {
    PyObject *tmpObj = NULL;
    tmpObj = PyObject_CallFunctionObjArgs(enc->defaultHandler, obj, NULL);
    if (!PyErr_Occurred()) {
        if (tmpObj == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "Failed to execute default handler");
        } else {
            encode(tmpObj, (JSONObjectEncoder *)enc, NULL, 0);
        }
    }
    Py_XDECREF(tmpObj);
    return;
}

void Object_beginTypeContext(JSOBJ _obj, JSONTypeContext *tc) {
    PyObject *obj, *exc, *toDictFunc, *tmpObj, *values;
    TypeContext *pc;
    PyObjectEncoder *enc;
    double val;
    npy_int64 value;
    int unit;

    tc->prv = NULL;

    if (!_obj) {
        tc->type = JT_INVALID;
        return;
    }

    obj = (PyObject *)_obj;
    enc = (PyObjectEncoder *)tc->encoder;

    if (PyBool_Check(obj)) {
        tc->type = (obj == Py_True) ? JT_TRUE : JT_FALSE;
        return;
    } else if (obj == Py_None) {
        tc->type = JT_NULL;
        return;
    }

    pc = createTypeContext();
    if (!pc) {
        tc->type = JT_INVALID;
        return;
    }
    tc->prv = pc;

    if (PyTypeNum_ISDATETIME(enc->npyType)) {
        int64_t longVal;
        PyArray_VectorUnaryFunc *castfunc =
            PyArray_GetCastFunc(PyArray_DescrFromType(enc->npyType), NPY_INT64);
        if (!castfunc) {
            PyErr_Format(PyExc_ValueError, "Cannot cast numpy dtype %d to long",
                         enc->npyType);
        }
        castfunc(enc->npyValue, &longVal, 1, NULL, NULL);
        if (longVal == get_nat()) {
            tc->type = JT_NULL;
        } else {
            if (enc->datetimeIso) {
                if (enc->npyType == NPY_TIMEDELTA) {
                    pc->PyTypeToUTF8 = NpyTimeDeltaToIsoCallback;
                } else {
                    pc->PyTypeToUTF8 = NpyDateTimeToIsoCallback;
                }
                // Currently no way to pass longVal to iso function, so use
                // state management
                GET_TC(tc)->longValue = longVal;
                tc->type = JT_UTF8;
            } else {
                NPY_DATETIMEUNIT base =
                    ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
                GET_TC(tc)->longValue = NpyDateTimeToEpoch(longVal, base);
                tc->type = JT_LONG;
            }
        }

        // TODO(username): this prevents infinite loop with
        // mixed-type DataFrames;
        // refactor
        enc->npyCtxtPassthru = NULL;
        enc->npyType = -1;
        return;
    }

    if (PyIter_Check(obj) ||
        (PyArray_Check(obj) && !PyArray_CheckScalar(obj))) {
        goto ISITERABLE;
    }

    if (PyLong_Check(obj)) {
        tc->type = JT_LONG;
        int overflow = 0;
        GET_TC(tc)->longValue = PyLong_AsLongLongAndOverflow(obj, &overflow);
        int err;
        err = (GET_TC(tc)->longValue == -1) && PyErr_Occurred();

        if (overflow) {
            tc->type = JT_BIGNUM;
        } else if (err) {
            goto INVALID;
        }

        return;
    } else if (PyFloat_Check(obj)) {
        val = PyFloat_AS_DOUBLE(obj);
        if (npy_isnan(val) || npy_isinf(val)) {
            tc->type = JT_NULL;
        } else {
            GET_TC(tc)->doubleValue = val;
            tc->type = JT_DOUBLE;
        }
        return;
    } else if (PyBytes_Check(obj)) {
        pc->PyTypeToUTF8 = PyBytesToUTF8;
        tc->type = JT_UTF8;
        return;
    } else if (PyUnicode_Check(obj)) {
        pc->PyTypeToUTF8 = PyUnicodeToUTF8;
        tc->type = JT_UTF8;
        return;
    } else if (PyObject_TypeCheck(obj, type_decimal)) {
        GET_TC(tc)->doubleValue = PyFloat_AsDouble(obj);
        tc->type = JT_DOUBLE;
        return;
    } else if (PyDateTime_Check(obj) || PyDate_Check(obj)) {
        if (PyObject_TypeCheck(obj, cls_nat)) {
            tc->type = JT_NULL;
            return;
        }

        if (enc->datetimeIso) {
            pc->PyTypeToUTF8 = PyDateTimeToIsoCallback;
            tc->type = JT_UTF8;
        } else {
            NPY_DATETIMEUNIT base =
                ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
            GET_TC(tc)->longValue = PyDateTimeToEpoch(obj, base);
            tc->type = JT_LONG;
        }
        return;
    } else if (PyTime_Check(obj)) {
        pc->PyTypeToUTF8 = PyTimeToJSON;
        tc->type = JT_UTF8;
        return;
    } else if (PyArray_IsScalar(obj, Datetime)) {
        if (((PyDatetimeScalarObject *)obj)->obval == get_nat()) {
            tc->type = JT_NULL;
            return;
        }

        if (enc->datetimeIso) {
            pc->PyTypeToUTF8 = PyDateTimeToIsoCallback;
            tc->type = JT_UTF8;
        } else {
            NPY_DATETIMEUNIT base =
                ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
            GET_TC(tc)->longValue = PyDateTimeToEpoch(obj, base);
            tc->type = JT_LONG;
        }
        return;
    } else if (PyDelta_Check(obj)) {
        if (PyObject_HasAttrString(obj, "value")) {
            value = get_long_attr(obj, "value");
        } else {
            value = total_seconds(obj) * 1000000000LL;  // nanoseconds per sec
        }

        if (value == get_nat()) {
            tc->type = JT_NULL;
            return;
        } else if (enc->datetimeIso) {
            pc->PyTypeToUTF8 = NpyTimeDeltaToIsoCallback;
            tc->type = JT_UTF8;
        } else {
            unit = ((PyObjectEncoder *)tc->encoder)->datetimeUnit;
            if (scaleNanosecToUnit(&value, unit) != 0) {
                // TODO(username): Add some kind of error handling here
            }

            exc = PyErr_Occurred();

            if (exc && PyErr_ExceptionMatches(PyExc_OverflowError)) {
                goto INVALID;
            }

            tc->type = JT_LONG;
        }
        GET_TC(tc)->longValue = value;
        return;
    } else if (PyArray_IsScalar(obj, Integer)) {
        tc->type = JT_LONG;
        PyArray_CastScalarToCtype(obj, &(GET_TC(tc)->longValue),
                                  PyArray_DescrFromType(NPY_INT64));

        exc = PyErr_Occurred();

        if (exc && PyErr_ExceptionMatches(PyExc_OverflowError)) {
            goto INVALID;
        }

        return;
    } else if (PyArray_IsScalar(obj, Bool)) {
        PyArray_CastScalarToCtype(obj, &(GET_TC(tc)->longValue),
                                  PyArray_DescrFromType(NPY_BOOL));
        tc->type = (GET_TC(tc)->longValue) ? JT_TRUE : JT_FALSE;
        return;
    } else if (PyArray_IsScalar(obj, Float) || PyArray_IsScalar(obj, Double)) {
        PyArray_CastScalarToCtype(obj, &(GET_TC(tc)->doubleValue),
                                  PyArray_DescrFromType(NPY_DOUBLE));
        tc->type = JT_DOUBLE;
        return;
    } else if (PyArray_Check(obj) && PyArray_CheckScalar(obj)) {
        PyErr_Format(PyExc_TypeError,
                     "%R (0d array) is not JSON serializable at the moment",
                     obj);
        goto INVALID;
    } else if (PyObject_TypeCheck(obj, cls_na)) {
        tc->type = JT_NULL;
        return;
    }

ISITERABLE:

    if (PyObject_TypeCheck(obj, cls_index)) {
        if (enc->outputFormat == SPLIT) {
            tc->type = JT_OBJECT;
            pc->iterBegin = Index_iterBegin;
            pc->iterEnd = Index_iterEnd;
            pc->iterNext = Index_iterNext;
            pc->iterGetValue = Index_iterGetValue;
            pc->iterGetName = Index_iterGetName;
            return;
        }

        pc->newObj = get_values(obj);
        if (pc->newObj) {
            tc->type = JT_ARRAY;
            pc->iterBegin = NpyArr_iterBegin;
            pc->iterEnd = NpyArr_iterEnd;
            pc->iterNext = NpyArr_iterNext;
            pc->iterGetValue = NpyArr_iterGetValue;
            pc->iterGetName = NpyArr_iterGetName;
        } else {
            goto INVALID;
        }

        return;
    } else if (PyObject_TypeCheck(obj, cls_series)) {
        if (enc->outputFormat == SPLIT) {
            tc->type = JT_OBJECT;
            pc->iterBegin = Series_iterBegin;
            pc->iterEnd = Series_iterEnd;
            pc->iterNext = Series_iterNext;
            pc->iterGetValue = Series_iterGetValue;
            pc->iterGetName = Series_iterGetName;
            return;
        }

        pc->newObj = get_values(obj);
        if (!pc->newObj) {
            goto INVALID;
        }

        if (enc->outputFormat == INDEX || enc->outputFormat == COLUMNS) {
            tc->type = JT_OBJECT;
            tmpObj = PyObject_GetAttrString(obj, "index");
            if (!tmpObj) {
                goto INVALID;
            }
            values = get_values(tmpObj);
            Py_DECREF(tmpObj);
            if (!values) {
                goto INVALID;
            }
            pc->columnLabelsLen = PyArray_DIM(pc->newObj, 0);
            pc->columnLabels = NpyArr_encodeLabels((PyArrayObject *)values, enc,
                                                   pc->columnLabelsLen);
            if (!pc->columnLabels) {
                goto INVALID;
            }
        } else {
            tc->type = JT_ARRAY;
        }
        pc->iterBegin = NpyArr_iterBegin;
        pc->iterEnd = NpyArr_iterEnd;
        pc->iterNext = NpyArr_iterNext;
        pc->iterGetValue = NpyArr_iterGetValue;
        pc->iterGetName = NpyArr_iterGetName;
        return;
    } else if (PyArray_Check(obj)) {
        if (enc->npyCtxtPassthru) {
            pc->npyarr = enc->npyCtxtPassthru;
            tc->type = (pc->npyarr->columnLabels ? JT_OBJECT : JT_ARRAY);

            pc->iterBegin = NpyArrPassThru_iterBegin;
            pc->iterNext = NpyArr_iterNext;
            pc->iterEnd = NpyArrPassThru_iterEnd;
            pc->iterGetValue = NpyArr_iterGetValue;
            pc->iterGetName = NpyArr_iterGetName;

            enc->npyCtxtPassthru = NULL;
            return;
        }

        tc->type = JT_ARRAY;
        pc->iterBegin = NpyArr_iterBegin;
        pc->iterEnd = NpyArr_iterEnd;
        pc->iterNext = NpyArr_iterNext;
        pc->iterGetValue = NpyArr_iterGetValue;
        pc->iterGetName = NpyArr_iterGetName;
        return;
    } else if (PyObject_TypeCheck(obj, cls_dataframe)) {
        if (enc->blkCtxtPassthru) {
            pc->pdblock = enc->blkCtxtPassthru;
            tc->type =
                (pc->pdblock->npyCtxts[0]->columnLabels ? JT_OBJECT : JT_ARRAY);

            pc->iterBegin = PdBlockPassThru_iterBegin;
            pc->iterEnd = PdBlockPassThru_iterEnd;
            pc->iterNext = PdBlock_iterNextItem;
            pc->iterGetName = PdBlock_iterGetName;
            pc->iterGetValue = NpyArr_iterGetValue;

            enc->blkCtxtPassthru = NULL;
            return;
        }

        if (enc->outputFormat == SPLIT) {
            tc->type = JT_OBJECT;
            pc->iterBegin = DataFrame_iterBegin;
            pc->iterEnd = DataFrame_iterEnd;
            pc->iterNext = DataFrame_iterNext;
            pc->iterGetValue = DataFrame_iterGetValue;
            pc->iterGetName = DataFrame_iterGetName;
            return;
        }

        if (is_simple_frame(obj)) {
            pc->iterBegin = NpyArr_iterBegin;
            pc->iterEnd = NpyArr_iterEnd;
            pc->iterNext = NpyArr_iterNext;
            pc->iterGetName = NpyArr_iterGetName;

            pc->newObj = PyObject_GetAttrString(obj, "values");
            if (!pc->newObj) {
                goto INVALID;
            }
        } else {
            pc->iterBegin = PdBlock_iterBegin;
            pc->iterEnd = PdBlock_iterEnd;
            pc->iterNext = PdBlock_iterNext;
            pc->iterGetName = PdBlock_iterGetName;
        }
        pc->iterGetValue = NpyArr_iterGetValue;

        if (enc->outputFormat == VALUES) {
            tc->type = JT_ARRAY;
        } else if (enc->outputFormat == RECORDS) {
            tc->type = JT_ARRAY;
            tmpObj = PyObject_GetAttrString(obj, "columns");
            if (!tmpObj) {
                goto INVALID;
            }
            values = get_values(tmpObj);
            if (!values) {
                Py_DECREF(tmpObj);
                goto INVALID;
            }
            pc->columnLabelsLen = PyObject_Size(tmpObj);
            pc->columnLabels = NpyArr_encodeLabels((PyArrayObject *)values, enc,
                                                   pc->columnLabelsLen);
            Py_DECREF(tmpObj);
            if (!pc->columnLabels) {
                goto INVALID;
            }
        } else if (enc->outputFormat == INDEX || enc->outputFormat == COLUMNS) {
            tc->type = JT_OBJECT;
            tmpObj = (enc->outputFormat == INDEX
                          ? PyObject_GetAttrString(obj, "index")
                          : PyObject_GetAttrString(obj, "columns"));
            if (!tmpObj) {
                goto INVALID;
            }
            values = get_values(tmpObj);
            if (!values) {
                Py_DECREF(tmpObj);
                goto INVALID;
            }
            pc->rowLabelsLen = PyObject_Size(tmpObj);
            pc->rowLabels = NpyArr_encodeLabels((PyArrayObject *)values, enc,
                                                pc->rowLabelsLen);
            Py_DECREF(tmpObj);
            tmpObj = (enc->outputFormat == INDEX
                          ? PyObject_GetAttrString(obj, "columns")
                          : PyObject_GetAttrString(obj, "index"));
            if (!tmpObj) {
                NpyArr_freeLabels(pc->rowLabels, pc->rowLabelsLen);
                pc->rowLabels = NULL;
                goto INVALID;
            }
            values = get_values(tmpObj);
            if (!values) {
                Py_DECREF(tmpObj);
                NpyArr_freeLabels(pc->rowLabels, pc->rowLabelsLen);
                pc->rowLabels = NULL;
                goto INVALID;
            }
            pc->columnLabelsLen = PyObject_Size(tmpObj);
            pc->columnLabels = NpyArr_encodeLabels((PyArrayObject *)values, enc,
                                                   pc->columnLabelsLen);
            Py_DECREF(tmpObj);
            if (!pc->columnLabels) {
                NpyArr_freeLabels(pc->rowLabels, pc->rowLabelsLen);
                pc->rowLabels = NULL;
                goto INVALID;
            }

            if (enc->outputFormat == COLUMNS) {
                pc->transpose = 1;
            }
        } else {
            goto INVALID;
        }
        return;
    } else if (PyDict_Check(obj)) {
        tc->type = JT_OBJECT;
        pc->iterBegin = Dict_iterBegin;
        pc->iterEnd = Dict_iterEnd;
        pc->iterNext = Dict_iterNext;
        pc->iterGetValue = Dict_iterGetValue;
        pc->iterGetName = Dict_iterGetName;
        pc->dictObj = obj;
        Py_INCREF(obj);

        return;
    } else if (PyList_Check(obj)) {
        tc->type = JT_ARRAY;
        pc->iterBegin = List_iterBegin;
        pc->iterEnd = List_iterEnd;
        pc->iterNext = List_iterNext;
        pc->iterGetValue = List_iterGetValue;
        pc->iterGetName = List_iterGetName;
        return;
    } else if (PyTuple_Check(obj)) {
        tc->type = JT_ARRAY;
        pc->iterBegin = Tuple_iterBegin;
        pc->iterEnd = Tuple_iterEnd;
        pc->iterNext = Tuple_iterNext;
        pc->iterGetValue = Tuple_iterGetValue;
        pc->iterGetName = Tuple_iterGetName;
        return;
    } else if (PyAnySet_Check(obj)) {
        tc->type = JT_ARRAY;
        pc->iterBegin = Set_iterBegin;
        pc->iterEnd = Set_iterEnd;
        pc->iterNext = Set_iterNext;
        pc->iterGetValue = Set_iterGetValue;
        pc->iterGetName = Set_iterGetName;
        return;
    }

    toDictFunc = PyObject_GetAttrString(obj, "toDict");

    if (toDictFunc) {
        PyObject *tuple = PyTuple_New(0);
        PyObject *toDictResult = PyObject_Call(toDictFunc, tuple, NULL);
        Py_DECREF(tuple);
        Py_DECREF(toDictFunc);

        if (toDictResult == NULL) {
            PyErr_Clear();
            tc->type = JT_NULL;
            return;
        }

        if (!PyDict_Check(toDictResult)) {
            Py_DECREF(toDictResult);
            tc->type = JT_NULL;
            return;
        }

        tc->type = JT_OBJECT;
        pc->iterBegin = Dict_iterBegin;
        pc->iterEnd = Dict_iterEnd;
        pc->iterNext = Dict_iterNext;
        pc->iterGetValue = Dict_iterGetValue;
        pc->iterGetName = Dict_iterGetName;
        pc->dictObj = toDictResult;
        return;
    }

    PyErr_Clear();

    if (enc->defaultHandler) {
        Object_invokeDefaultHandler(obj, enc);
        goto INVALID;
    }

    tc->type = JT_OBJECT;
    pc->iterBegin = Dir_iterBegin;
    pc->iterEnd = Dir_iterEnd;
    pc->iterNext = Dir_iterNext;
    pc->iterGetValue = Dir_iterGetValue;
    pc->iterGetName = Dir_iterGetName;
    return;

INVALID:
    tc->type = JT_INVALID;
    PyObject_Free(tc->prv);
    tc->prv = NULL;
    return;
}

void Object_endTypeContext(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    if (tc->prv) {
        Py_XDECREF(GET_TC(tc)->newObj);
        GET_TC(tc)->newObj = NULL;
        NpyArr_freeLabels(GET_TC(tc)->rowLabels, GET_TC(tc)->rowLabelsLen);
        GET_TC(tc)->rowLabels = NULL;
        NpyArr_freeLabels(GET_TC(tc)->columnLabels,
                          GET_TC(tc)->columnLabelsLen);
        GET_TC(tc)->columnLabels = NULL;
        PyObject_Free(GET_TC(tc)->cStr);
        GET_TC(tc)->cStr = NULL;
        PyObject_Free(tc->prv);
        tc->prv = NULL;
    }
}

const char *Object_getStringValue(JSOBJ obj, JSONTypeContext *tc,
                                  size_t *_outLen) {
    return GET_TC(tc)->PyTypeToUTF8(obj, tc, _outLen);
}

JSINT64 Object_getLongValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->longValue;
}

double Object_getDoubleValue(JSOBJ Py_UNUSED(obj), JSONTypeContext *tc) {
    return GET_TC(tc)->doubleValue;
}

const char *Object_getBigNumStringValue(JSOBJ obj, JSONTypeContext *tc,
                                        size_t *_outLen) {
    PyObject *repr = PyObject_Str(obj);
    const char *str = PyUnicode_AsUTF8AndSize(repr, (Py_ssize_t *)_outLen);
    char *bytes = PyObject_Malloc(*_outLen + 1);
    memcpy(bytes, str, *_outLen + 1);
    GET_TC(tc)->cStr = bytes;

    Py_DECREF(repr);

    return GET_TC(tc)->cStr;
}

static void Object_releaseObject(JSOBJ _obj) { Py_DECREF((PyObject *)_obj); }

void Object_iterBegin(JSOBJ obj, JSONTypeContext *tc) {
    GET_TC(tc)->iterBegin(obj, tc);
}

int Object_iterNext(JSOBJ obj, JSONTypeContext *tc) {
    return GET_TC(tc)->iterNext(obj, tc);
}

void Object_iterEnd(JSOBJ obj, JSONTypeContext *tc) {
    GET_TC(tc)->iterEnd(obj, tc);
}

JSOBJ Object_iterGetValue(JSOBJ obj, JSONTypeContext *tc) {
    return GET_TC(tc)->iterGetValue(obj, tc);
}

char *Object_iterGetName(JSOBJ obj, JSONTypeContext *tc, size_t *outLen) {
    return GET_TC(tc)->iterGetName(obj, tc, outLen);
}

PyObject *objToJSON(PyObject *Py_UNUSED(self), PyObject *args,
                    PyObject *kwargs) {
    static char *kwlist[] = {"obj",
                             "ensure_ascii",
                             "double_precision",
                             "encode_html_chars",
                             "orient",
                             "date_unit",
                             "iso_dates",
                             "default_handler",
                             "indent",
                             NULL};

    char buffer[65536];
    char *ret;
    PyObject *newobj;
    PyObject *oinput = NULL;
    PyObject *oensureAscii = NULL;
    int idoublePrecision = 10;  // default double precision setting
    PyObject *oencodeHTMLChars = NULL;
    char *sOrient = NULL;
    char *sdateFormat = NULL;
    PyObject *oisoDates = 0;
    PyObject *odefHandler = 0;
    int indent = 0;

    PyObjectEncoder pyEncoder = {{
        Object_beginTypeContext,
        Object_endTypeContext,
        Object_getStringValue,
        Object_getLongValue,
        NULL,  // getIntValue is unused
        Object_getDoubleValue,
        Object_getBigNumStringValue,
        Object_iterBegin,
        Object_iterNext,
        Object_iterEnd,
        Object_iterGetValue,
        Object_iterGetName,
        Object_releaseObject,
        PyObject_Malloc,
        PyObject_Realloc,
        PyObject_Free,
        -1,  // recursionMax
        idoublePrecision,
        1,  // forceAscii
        0,  // encodeHTMLChars
        0,  // indent
    }};
    JSONObjectEncoder *encoder = (JSONObjectEncoder *)&pyEncoder;

    pyEncoder.npyCtxtPassthru = NULL;
    pyEncoder.blkCtxtPassthru = NULL;
    pyEncoder.npyType = -1;
    pyEncoder.npyValue = NULL;
    pyEncoder.datetimeIso = 0;
    pyEncoder.datetimeUnit = NPY_FR_ms;
    pyEncoder.outputFormat = COLUMNS;
    pyEncoder.defaultHandler = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OiOssOOi", kwlist,
                                     &oinput, &oensureAscii, &idoublePrecision,
                                     &oencodeHTMLChars, &sOrient, &sdateFormat,
                                     &oisoDates, &odefHandler, &indent)) {
        return NULL;
    }

    if (oensureAscii != NULL && !PyObject_IsTrue(oensureAscii)) {
        encoder->forceASCII = 0;
    }

    if (oencodeHTMLChars != NULL && PyObject_IsTrue(oencodeHTMLChars)) {
        encoder->encodeHTMLChars = 1;
    }

    if (idoublePrecision > JSON_DOUBLE_MAX_DECIMALS || idoublePrecision < 0) {
        PyErr_Format(
            PyExc_ValueError,
            "Invalid value '%d' for option 'double_precision', max is '%u'",
            idoublePrecision, JSON_DOUBLE_MAX_DECIMALS);
        return NULL;
    }
    encoder->doublePrecision = idoublePrecision;

    if (sOrient != NULL) {
        if (strcmp(sOrient, "records") == 0) {
            pyEncoder.outputFormat = RECORDS;
        } else if (strcmp(sOrient, "index") == 0) {
            pyEncoder.outputFormat = INDEX;
        } else if (strcmp(sOrient, "split") == 0) {
            pyEncoder.outputFormat = SPLIT;
        } else if (strcmp(sOrient, "values") == 0) {
            pyEncoder.outputFormat = VALUES;
        } else if (strcmp(sOrient, "columns") != 0) {
            PyErr_Format(PyExc_ValueError,
                         "Invalid value '%s' for option 'orient'", sOrient);
            return NULL;
        }
    }

    if (sdateFormat != NULL) {
        if (strcmp(sdateFormat, "s") == 0) {
            pyEncoder.datetimeUnit = NPY_FR_s;
        } else if (strcmp(sdateFormat, "ms") == 0) {
            pyEncoder.datetimeUnit = NPY_FR_ms;
        } else if (strcmp(sdateFormat, "us") == 0) {
            pyEncoder.datetimeUnit = NPY_FR_us;
        } else if (strcmp(sdateFormat, "ns") == 0) {
            pyEncoder.datetimeUnit = NPY_FR_ns;
        } else {
            PyErr_Format(PyExc_ValueError,
                         "Invalid value '%s' for option 'date_unit'",
                         sdateFormat);
            return NULL;
        }
    }

    if (oisoDates != NULL && PyObject_IsTrue(oisoDates)) {
        pyEncoder.datetimeIso = 1;
    }

    if (odefHandler != NULL && odefHandler != Py_None) {
        if (!PyCallable_Check(odefHandler)) {
            PyErr_SetString(PyExc_TypeError, "Default handler is not callable");
            return NULL;
        }
        pyEncoder.defaultHandler = odefHandler;
    }

    encoder->indent = indent;

    pyEncoder.originalOutputFormat = pyEncoder.outputFormat;
    ret = JSON_EncodeObject(oinput, encoder, buffer, sizeof(buffer));
    if (PyErr_Occurred()) {
        return NULL;
    }

    if (encoder->errorMsg) {
        if (ret != buffer) {
            encoder->free(ret);
        }
        PyErr_Format(PyExc_OverflowError, "%s", encoder->errorMsg);
        return NULL;
    }

    newobj = PyUnicode_FromString(ret);

    if (ret != buffer) {
        encoder->free(ret);
    }

    return newobj;
}
