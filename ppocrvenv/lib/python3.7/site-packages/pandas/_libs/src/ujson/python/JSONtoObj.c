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
DISCLAIMED. IN NO EVENT SHALL ESN SOCIAL SOFTWARE AB OR JONAS TARNSTROM BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Portions of code from MODP_ASCII - Ascii transformations (upper/lower, etc)
https://github.com/client9/stringencoders
Copyright (c) 2007  Nick Galbreath -- nickg [at] modp [dot] com. All rights reserved.

Numeric decoder derived from from TCL library
https://www.opensource.apple.com/source/tcl/tcl-14/tcl/license.terms
 * Copyright (c) 1988-1993 The Regents of the University of California.
 * Copyright (c) 1994 Sun Microsystems, Inc.
*/

#define PY_ARRAY_UNIQUE_SYMBOL UJSON_NUMPY
#define NO_IMPORT_ARRAY
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <ultrajson.h>

#define PRINTMARK()

typedef struct __PyObjectDecoder {
    JSONObjectDecoder dec;

    void *npyarr;       // Numpy context buffer
    void *npyarr_addr;  // Ref to npyarr ptr to track DECREF calls
    npy_intp curdim;    // Current array dimension

    PyArray_Descr *dtype;
} PyObjectDecoder;

typedef struct __NpyArrContext {
    PyObject *ret;
    PyObject *labels[2];
    PyArray_Dims shape;

    PyObjectDecoder *dec;

    npy_intp i;
    npy_intp elsize;
    npy_intp elcount;
} NpyArrContext;

// Numpy handling based on numpy internal code, specifically the function
// PyArray_FromIter.

// numpy related functions are inter-dependent so declare them all here,
// to ensure the compiler catches any errors

// standard numpy array handling
JSOBJ Object_npyNewArray(void *prv, void *decoder);
JSOBJ Object_npyEndArray(void *prv, JSOBJ obj);
int Object_npyArrayAddItem(void *prv, JSOBJ obj, JSOBJ value);

// for more complex dtypes (object and string) fill a standard Python list
// and convert to a numpy array when done.
JSOBJ Object_npyNewArrayList(void *prv, void *decoder);
JSOBJ Object_npyEndArrayList(void *prv, JSOBJ obj);
int Object_npyArrayListAddItem(void *prv, JSOBJ obj, JSOBJ value);

// labelled support, encode keys and values of JS object into separate numpy
// arrays
JSOBJ Object_npyNewObject(void *prv, void *decoder);
JSOBJ Object_npyEndObject(void *prv, JSOBJ obj);
int Object_npyObjectAddKey(void *prv, JSOBJ obj, JSOBJ name, JSOBJ value);

// free the numpy context buffer
void Npy_releaseContext(NpyArrContext *npyarr) {
    PRINTMARK();
    if (npyarr) {
        if (npyarr->shape.ptr) {
            PyObject_Free(npyarr->shape.ptr);
        }
        if (npyarr->dec) {
            npyarr->dec->npyarr = NULL;
            npyarr->dec->curdim = 0;
        }
        Py_XDECREF(npyarr->labels[0]);
        Py_XDECREF(npyarr->labels[1]);
        Py_XDECREF(npyarr->ret);
        PyObject_Free(npyarr);
    }
}

JSOBJ Object_npyNewArray(void *prv, void *_decoder) {
    NpyArrContext *npyarr;
    PyObjectDecoder *decoder = (PyObjectDecoder *)_decoder;
    PRINTMARK();
    if (decoder->curdim <= 0) {
        // start of array - initialise the context buffer
        npyarr = decoder->npyarr = PyObject_Malloc(sizeof(NpyArrContext));
        decoder->npyarr_addr = npyarr;

        if (!npyarr) {
            PyErr_NoMemory();
            return NULL;
        }

        npyarr->dec = decoder;
        npyarr->labels[0] = npyarr->labels[1] = NULL;

        npyarr->shape.ptr = PyObject_Malloc(sizeof(npy_intp) * NPY_MAXDIMS);
        npyarr->shape.len = 1;
        npyarr->ret = NULL;

        npyarr->elsize = 0;
        npyarr->elcount = 4;
        npyarr->i = 0;
    } else {
        // starting a new dimension continue the current array (and reshape
        // after)
        npyarr = (NpyArrContext *)decoder->npyarr;
        if (decoder->curdim >= npyarr->shape.len) {
            npyarr->shape.len++;
        }
    }

    npyarr->shape.ptr[decoder->curdim] = 0;
    decoder->curdim++;
    return npyarr;
}

PyObject *Npy_returnLabelled(NpyArrContext *npyarr) {
    PyObject *ret = npyarr->ret;
    npy_intp i;

    if (npyarr->labels[0] || npyarr->labels[1]) {
        // finished decoding, build tuple with values and labels
        ret = PyTuple_New(npyarr->shape.len + 1);
        for (i = 0; i < npyarr->shape.len; i++) {
            if (npyarr->labels[i]) {
                PyTuple_SET_ITEM(ret, i + 1, npyarr->labels[i]);
                npyarr->labels[i] = NULL;
            } else {
                Py_INCREF(Py_None);
                PyTuple_SET_ITEM(ret, i + 1, Py_None);
            }
        }
        PyTuple_SET_ITEM(ret, 0, npyarr->ret);
    }

    return ret;
}

JSOBJ Object_npyEndArray(void *prv, JSOBJ obj) {
    PyObject *ret;
    char *new_data;
    NpyArrContext *npyarr = (NpyArrContext *)obj;
    int emptyType = NPY_DEFAULT_TYPE;
    npy_intp i;
    PRINTMARK();
    if (!npyarr) {
        return NULL;
    }

    ret = npyarr->ret;
    i = npyarr->i;

    npyarr->dec->curdim--;

    if (i == 0 || !npyarr->ret) {
        // empty array would not have been initialised so do it now.
        if (npyarr->dec->dtype) {
            emptyType = npyarr->dec->dtype->type_num;
        }
        npyarr->ret = ret =
            PyArray_EMPTY(npyarr->shape.len, npyarr->shape.ptr, emptyType, 0);
    } else if (npyarr->dec->curdim <= 0) {
        // realloc to final size
        new_data = PyDataMem_RENEW(PyArray_DATA(ret), i * npyarr->elsize);
        if (new_data == NULL) {
            PyErr_NoMemory();
            Npy_releaseContext(npyarr);
            return NULL;
        }
        ((PyArrayObject *)ret)->data = (void *)new_data;
        // PyArray_BYTES(ret) = new_data;
    }

    if (npyarr->dec->curdim <= 0) {
        // finished decoding array, reshape if necessary
        if (npyarr->shape.len > 1) {
            npyarr->ret = PyArray_Newshape((PyArrayObject *)ret, &npyarr->shape,
                                           NPY_ANYORDER);
            Py_DECREF(ret);
        }

        ret = Npy_returnLabelled(npyarr);

        npyarr->ret = NULL;
        Npy_releaseContext(npyarr);
    }

    return ret;
}

int Object_npyArrayAddItem(void *prv, JSOBJ obj, JSOBJ value) {
    PyObject *type;
    PyArray_Descr *dtype;
    npy_intp i;
    char *new_data, *item;
    NpyArrContext *npyarr = (NpyArrContext *)obj;
    PRINTMARK();
    if (!npyarr) {
        return 0;
    }

    i = npyarr->i;

    npyarr->shape.ptr[npyarr->dec->curdim - 1]++;

    if (PyArray_Check((PyObject *)value)) {
        // multidimensional array, keep decoding values.
        return 1;
    }

    if (!npyarr->ret) {
        // Array not initialised yet.
        // We do it here so we can 'sniff' the data type if none was provided
        if (!npyarr->dec->dtype) {
            type = PyObject_Type(value);
            if (!PyArray_DescrConverter(type, &dtype)) {
                Py_DECREF(type);
                goto fail;
            }
            Py_INCREF(dtype);
            Py_DECREF(type);
        } else {
            dtype = PyArray_DescrNew(npyarr->dec->dtype);
        }

        // If it's an object or string then fill a Python list and subsequently
        // convert. Otherwise we would need to somehow mess about with
        // reference counts when renewing memory.
        npyarr->elsize = dtype->elsize;
        if (PyDataType_REFCHK(dtype) || npyarr->elsize == 0) {
            Py_XDECREF(dtype);

            if (npyarr->dec->curdim > 1) {
                PyErr_SetString(PyExc_ValueError,
                                "Cannot decode multidimensional arrays with "
                                "variable length elements to numpy");
                goto fail;
            }
            npyarr->elcount = 0;
            npyarr->ret = PyList_New(0);
            if (!npyarr->ret) {
                goto fail;
            }
            ((JSONObjectDecoder *)npyarr->dec)->newArray =
                Object_npyNewArrayList;
            ((JSONObjectDecoder *)npyarr->dec)->arrayAddItem =
                Object_npyArrayListAddItem;
            ((JSONObjectDecoder *)npyarr->dec)->endArray =
                Object_npyEndArrayList;
            return Object_npyArrayListAddItem(prv, obj, value);
        }

        npyarr->ret = PyArray_NewFromDescr(
            &PyArray_Type, dtype, 1, &npyarr->elcount, NULL, NULL, 0, NULL);

        if (!npyarr->ret) {
            goto fail;
        }
    }

    if (i >= npyarr->elcount) {
        // Grow PyArray_DATA(ret):
        // this is similar for the strategy for PyListObject, but we use
        // 50% overallocation => 0, 4, 8, 14, 23, 36, 56, 86 ...
        if (npyarr->elsize == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "Cannot decode multidimensional arrays with "
                            "variable length elements to numpy");
            goto fail;
        }

        npyarr->elcount = (i >> 1) + (i < 4 ? 4 : 2) + i;
        if (npyarr->elcount <= NPY_MAX_INTP / npyarr->elsize) {
            new_data = PyDataMem_RENEW(PyArray_DATA(npyarr->ret),
                                       npyarr->elcount * npyarr->elsize);
        } else {
            PyErr_NoMemory();
            goto fail;
        }
        ((PyArrayObject *)npyarr->ret)->data = (void *)new_data;

        // PyArray_BYTES(npyarr->ret) = new_data;
    }

    PyArray_DIMS(npyarr->ret)[0] = i + 1;

    if ((item = PyArray_GETPTR1(npyarr->ret, i)) == NULL ||
        PyArray_SETITEM(npyarr->ret, item, value) == -1) {
        goto fail;
    }

    Py_DECREF((PyObject *)value);
    npyarr->i++;
    return 1;

fail:

    Npy_releaseContext(npyarr);
    return 0;
}

JSOBJ Object_npyNewArrayList(void *prv, void *_decoder) {
    PyObjectDecoder *decoder = (PyObjectDecoder *)_decoder;
    PRINTMARK();
    PyErr_SetString(
        PyExc_ValueError,
        "nesting not supported for object or variable length dtypes");
    Npy_releaseContext(decoder->npyarr);
    return NULL;
}

JSOBJ Object_npyEndArrayList(void *prv, JSOBJ obj) {
    PyObject *list, *ret;
    NpyArrContext *npyarr = (NpyArrContext *)obj;
    PRINTMARK();
    if (!npyarr) {
        return NULL;
    }

    // convert decoded list to numpy array
    list = (PyObject *)npyarr->ret;
    npyarr->ret = PyArray_FROM_O(list);

    ret = Npy_returnLabelled(npyarr);
    npyarr->ret = list;

    ((JSONObjectDecoder *)npyarr->dec)->newArray = Object_npyNewArray;
    ((JSONObjectDecoder *)npyarr->dec)->arrayAddItem = Object_npyArrayAddItem;
    ((JSONObjectDecoder *)npyarr->dec)->endArray = Object_npyEndArray;
    Npy_releaseContext(npyarr);
    return ret;
}

int Object_npyArrayListAddItem(void *prv, JSOBJ obj, JSOBJ value) {
    NpyArrContext *npyarr = (NpyArrContext *)obj;
    PRINTMARK();
    if (!npyarr) {
        return 0;
    }
    PyList_Append((PyObject *)npyarr->ret, value);
    Py_DECREF((PyObject *)value);
    npyarr->elcount++;
    return 1;
}

JSOBJ Object_npyNewObject(void *prv, void *_decoder) {
    PyObjectDecoder *decoder = (PyObjectDecoder *)_decoder;
    PRINTMARK();
    if (decoder->curdim > 1) {
        PyErr_SetString(PyExc_ValueError,
                        "labels only supported up to 2 dimensions");
        return NULL;
    }

    return ((JSONObjectDecoder *)decoder)->newArray(prv, decoder);
}

JSOBJ Object_npyEndObject(void *prv, JSOBJ obj) {
    PyObject *list;
    npy_intp labelidx;
    NpyArrContext *npyarr = (NpyArrContext *)obj;
    PRINTMARK();
    if (!npyarr) {
        return NULL;
    }

    labelidx = npyarr->dec->curdim - 1;

    list = npyarr->labels[labelidx];
    if (list) {
        npyarr->labels[labelidx] = PyArray_FROM_O(list);
        Py_DECREF(list);
    }

    return (PyObject *)((JSONObjectDecoder *)npyarr->dec)->endArray(prv, obj);
}

int Object_npyObjectAddKey(void *prv, JSOBJ obj, JSOBJ name, JSOBJ value) {
    PyObject *label, *labels;
    npy_intp labelidx;
    // add key to label array, value to values array
    NpyArrContext *npyarr = (NpyArrContext *)obj;
    PRINTMARK();
    if (!npyarr) {
        return 0;
    }

    label = (PyObject *)name;
    labelidx = npyarr->dec->curdim - 1;

    if (!npyarr->labels[labelidx]) {
        npyarr->labels[labelidx] = PyList_New(0);
    }
    labels = npyarr->labels[labelidx];
    // only fill label array once, assumes all column labels are the same
    // for 2-dimensional arrays.
    if (PyList_Check(labels) && PyList_GET_SIZE(labels) <= npyarr->elcount) {
        PyList_Append(labels, label);
    }

    if (((JSONObjectDecoder *)npyarr->dec)->arrayAddItem(prv, obj, value)) {
        Py_DECREF(label);
        return 1;
    }
    return 0;
}

int Object_objectAddKey(void *prv, JSOBJ obj, JSOBJ name, JSOBJ value) {
    int ret = PyDict_SetItem(obj, name, value);
    Py_DECREF((PyObject *)name);
    Py_DECREF((PyObject *)value);
    return ret == 0 ? 1 : 0;
}

int Object_arrayAddItem(void *prv, JSOBJ obj, JSOBJ value) {
    int ret = PyList_Append(obj, value);
    Py_DECREF((PyObject *)value);
    return ret == 0 ? 1 : 0;
}

JSOBJ Object_newString(void *prv, wchar_t *start, wchar_t *end) {
    return PyUnicode_FromWideChar(start, (end - start));
}

JSOBJ Object_newTrue(void *prv) { Py_RETURN_TRUE; }

JSOBJ Object_newFalse(void *prv) { Py_RETURN_FALSE; }

JSOBJ Object_newNull(void *prv) { Py_RETURN_NONE; }

JSOBJ Object_newPosInf(void *prv) { return PyFloat_FromDouble(Py_HUGE_VAL); }

JSOBJ Object_newNegInf(void *prv) { return PyFloat_FromDouble(-Py_HUGE_VAL); }

JSOBJ Object_newObject(void *prv, void *decoder) { return PyDict_New(); }

JSOBJ Object_endObject(void *prv, JSOBJ obj) { return obj; }

JSOBJ Object_newArray(void *prv, void *decoder) { return PyList_New(0); }

JSOBJ Object_endArray(void *prv, JSOBJ obj) { return obj; }

JSOBJ Object_newInteger(void *prv, JSINT32 value) {
    return PyLong_FromLong((long)value);
}

JSOBJ Object_newLong(void *prv, JSINT64 value) {
    return PyLong_FromLongLong(value);
}

JSOBJ Object_newDouble(void *prv, double value) {
    return PyFloat_FromDouble(value);
}

static void Object_releaseObject(void *prv, JSOBJ obj, void *_decoder) {
    PyObjectDecoder *decoder = (PyObjectDecoder *)_decoder;
    if (obj != decoder->npyarr_addr) {
        Py_XDECREF(((PyObject *)obj));
    }
}

static char *g_kwlist[] = {"obj",      "precise_float", "numpy",
                           "labelled", "dtype",         NULL};

PyObject *JSONToObj(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *ret;
    PyObject *sarg;
    PyObject *arg;
    PyObject *opreciseFloat = NULL;
    JSONObjectDecoder *decoder;
    PyObjectDecoder pyDecoder;
    PyArray_Descr *dtype = NULL;
    int numpy = 0, labelled = 0;

    JSONObjectDecoder dec = {
        Object_newString, Object_objectAddKey,  Object_arrayAddItem,
        Object_newTrue,   Object_newFalse,      Object_newNull,
        Object_newPosInf, Object_newNegInf,     Object_newObject,
        Object_endObject,     Object_newArray,  Object_endArray,
        Object_newInteger,    Object_newLong,   Object_newDouble,
        Object_releaseObject, PyObject_Malloc, PyObject_Free,
        PyObject_Realloc};

    dec.preciseFloat = 0;
    dec.prv = NULL;

    pyDecoder.dec = dec;
    pyDecoder.curdim = 0;
    pyDecoder.npyarr = NULL;
    pyDecoder.npyarr_addr = NULL;

    decoder = (JSONObjectDecoder *)&pyDecoder;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OiiO&", g_kwlist, &arg,
                                     &opreciseFloat, &numpy, &labelled,
                                     PyArray_DescrConverter2, &dtype)) {
        Npy_releaseContext(pyDecoder.npyarr);
        return NULL;
    }

    if (opreciseFloat && PyObject_IsTrue(opreciseFloat)) {
        decoder->preciseFloat = 1;
    }

    if (PyBytes_Check(arg)) {
        sarg = arg;
    } else if (PyUnicode_Check(arg)) {
        sarg = PyUnicode_AsUTF8String(arg);
        if (sarg == NULL) {
            // Exception raised above us by codec according to docs
            return NULL;
        }
    } else {
        PyErr_Format(PyExc_TypeError, "Expected 'str' or 'bytes'");
        return NULL;
    }

    decoder->errorStr = NULL;
    decoder->errorOffset = NULL;

    if (numpy) {
        pyDecoder.dtype = dtype;
        decoder->newArray = Object_npyNewArray;
        decoder->endArray = Object_npyEndArray;
        decoder->arrayAddItem = Object_npyArrayAddItem;

        if (labelled) {
            decoder->newObject = Object_npyNewObject;
            decoder->endObject = Object_npyEndObject;
            decoder->objectAddKey = Object_npyObjectAddKey;
        }
    }

    ret = JSON_DecodeObject(decoder, PyBytes_AS_STRING(sarg),
                            PyBytes_GET_SIZE(sarg));

    if (sarg != arg) {
        Py_DECREF(sarg);
    }

    if (PyErr_Occurred()) {
        if (ret) {
            Py_DECREF((PyObject *)ret);
        }
        Npy_releaseContext(pyDecoder.npyarr);
        return NULL;
    }

    if (decoder->errorStr) {
        /*
        FIXME: It's possible to give a much nicer error message here with actual
        failing element in input etc*/

        PyErr_Format(PyExc_ValueError, "%s", decoder->errorStr);

        if (ret) {
            Py_DECREF((PyObject *)ret);
        }
        Npy_releaseContext(pyDecoder.npyarr);

        return NULL;
    }

    return ret;
}
