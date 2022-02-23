from cpython.datetime cimport timedelta
from numpy cimport int64_t


# Exposed for tslib, not intended for outside use.
cpdef int64_t delta_to_nanoseconds(delta) except? -1
cdef convert_to_timedelta64(object ts, str unit)
cdef bint is_any_td_scalar(object obj)


cdef class _Timedelta(timedelta):
    cdef readonly:
        int64_t value      # nanoseconds
        object freq        # frequency reference
        bint is_populated  # are my components populated
        int64_t _d, _h, _m, _s, _ms, _us, _ns

    cpdef timedelta to_pytimedelta(_Timedelta self)
    cpdef bint _has_ns(self)
