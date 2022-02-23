from numpy cimport (
    ndarray,
    uint8_t,
)


cpdef bint is_matching_na(object left, object right, bint nan_matches_none=*)

cpdef bint checknull(object val)
cpdef bint checknull_old(object val)
cpdef ndarray[uint8_t] isnaobj(ndarray arr)

cdef bint is_null_datetime64(v)
cdef bint is_null_timedelta64(v)
cdef bint checknull_with_nat_and_na(object obj)

cdef class C_NAType:
    pass

cdef C_NAType C_NA
