cimport numpy as cnp
from numpy cimport ndarray

from pandas._libs.tslibs.util cimport *


cdef extern from "numpy/ndarraytypes.h":
    void PyArray_CLEARFLAGS(ndarray arr, int flags) nogil


cdef extern from "numpy/arrayobject.h":
    enum:
        NPY_ARRAY_C_CONTIGUOUS
        NPY_ARRAY_F_CONTIGUOUS


cdef extern from "src/headers/stdint.h":
    enum: UINT8_MAX
    enum: UINT16_MAX
    enum: UINT32_MAX
    enum: UINT64_MAX
    enum: INT8_MIN
    enum: INT8_MAX
    enum: INT16_MIN
    enum: INT16_MAX
    enum: INT32_MAX
    enum: INT32_MIN
    enum: INT64_MAX
    enum: INT64_MIN


ctypedef fused numeric:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t

    cnp.float32_t
    cnp.float64_t


cdef inline void set_array_not_contiguous(ndarray ao) nogil:
    # Numpy>=1.8-compliant equivalent to:
    # ao->flags &= ~(NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    PyArray_CLEARFLAGS(ao,
                       (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))
