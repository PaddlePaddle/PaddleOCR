
from numpy cimport ndarray


cdef class NDArrayBacked:
    cdef:
        readonly ndarray _ndarray
        readonly object _dtype

    cpdef NDArrayBacked _from_backing_data(self, ndarray values)
    cpdef __setstate__(self, state)
