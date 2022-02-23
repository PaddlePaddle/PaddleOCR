from numpy cimport int64_t


cpdef to_offset(object obj)
cdef bint is_offset_object(object obj)
cdef bint is_tick_object(object obj)

cdef class BaseOffset:
    cdef readonly:
        int64_t n
        bint normalize
        dict _cache
