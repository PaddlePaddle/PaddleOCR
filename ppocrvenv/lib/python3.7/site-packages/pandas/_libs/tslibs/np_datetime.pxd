from cpython.datetime cimport (
    date,
    datetime,
)
from numpy cimport (
    int32_t,
    int64_t,
)


cdef extern from "numpy/ndarrayobject.h":
    ctypedef int64_t npy_timedelta
    ctypedef int64_t npy_datetime

cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct PyArray_DatetimeMetaData:
        NPY_DATETIMEUNIT base
        int64_t num

cdef extern from "numpy/arrayscalars.h":
    ctypedef struct PyDatetimeScalarObject:
        # PyObject_HEAD
        npy_datetime obval
        PyArray_DatetimeMetaData obmeta

    ctypedef struct PyTimedeltaScalarObject:
        # PyObject_HEAD
        npy_timedelta obval
        PyArray_DatetimeMetaData obmeta

cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct npy_datetimestruct:
        int64_t year
        int32_t month, day, hour, min, sec, us, ps, as

    ctypedef enum NPY_DATETIMEUNIT:
        NPY_FR_Y
        NPY_FR_M
        NPY_FR_W
        NPY_FR_D
        NPY_FR_B
        NPY_FR_h
        NPY_FR_m
        NPY_FR_s
        NPY_FR_ms
        NPY_FR_us
        NPY_FR_ns
        NPY_FR_ps
        NPY_FR_fs
        NPY_FR_as
        NPY_FR_GENERIC

cdef extern from "src/datetime/np_datetime.h":
    ctypedef struct pandas_timedeltastruct:
        int64_t days
        int32_t hrs, min, sec, ms, us, ns, seconds, microseconds, nanoseconds

    void pandas_datetime_to_datetimestruct(npy_datetime val,
                                           NPY_DATETIMEUNIT fr,
                                           npy_datetimestruct *result) nogil


cdef bint cmp_scalar(int64_t lhs, int64_t rhs, int op) except -1

cdef check_dts_bounds(npy_datetimestruct *dts)

cdef int64_t dtstruct_to_dt64(npy_datetimestruct* dts) nogil
cdef void dt64_to_dtstruct(int64_t dt64, npy_datetimestruct* out) nogil
cdef void td64_to_tdstruct(int64_t td64, pandas_timedeltastruct* out) nogil

cdef int64_t pydatetime_to_dt64(datetime val, npy_datetimestruct *dts)
cdef int64_t pydate_to_dt64(date val, npy_datetimestruct *dts)
cdef void pydate_to_dtstruct(date val, npy_datetimestruct *dts)

cdef npy_datetime get_datetime64_value(object obj) nogil
cdef npy_timedelta get_timedelta64_value(object obj) nogil
cdef NPY_DATETIMEUNIT get_datetime64_unit(object obj) nogil

cdef int _string_to_dts(str val, npy_datetimestruct* dts,
                        int* out_local, int* out_tzoffset,
                        bint want_exc) except? -1
