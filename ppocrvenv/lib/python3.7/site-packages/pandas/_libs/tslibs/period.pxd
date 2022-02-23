from numpy cimport int64_t

from .np_datetime cimport npy_datetimestruct


cdef bint is_period_object(object obj)
cdef int64_t get_period_ordinal(npy_datetimestruct *dts, int freq) nogil
