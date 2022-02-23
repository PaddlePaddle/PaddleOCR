from cython cimport floating
cimport numpy as np


cdef floating _euclidean_dense_dense(floating*, floating*, int, bint) nogil

cdef floating _euclidean_sparse_dense(floating[::1], int[::1], floating[::1],
                                      floating, bint) nogil

cpdef void _relocate_empty_clusters_dense(
    floating[:, ::1], floating[::1], floating[:, ::1],
    floating[:, ::1], floating[::1], int[::1])

cpdef void _relocate_empty_clusters_sparse(
    floating[::1], int[::1], int[::1], floating[::1], floating[:, ::1],
    floating[:, ::1], floating[::1], int[::1])

cdef void _average_centers(floating[:, ::1], floating[::1])

cdef void _center_shift(floating[:, ::1], floating[:, ::1], floating[::1])
