cimport numpy as np
from libc.math cimport sqrt, exp

from ..utils._typedefs cimport DTYPE_t, ITYPE_t

######################################################################
# Inline distance functions
#
#  We use these for the default (euclidean) case so that they can be
#  inlined.  This leads to faster computation for the most common case
cdef inline DTYPE_t euclidean_dist(const DTYPE_t* x1, const DTYPE_t* x2,
                                   ITYPE_t size) nogil except -1:
    cdef DTYPE_t tmp, d=0
    cdef np.intp_t j
    for j in range(size):
        tmp = x1[j] - x2[j]
        d += tmp * tmp
    return sqrt(d)


cdef inline DTYPE_t euclidean_rdist(const DTYPE_t* x1, const DTYPE_t* x2,
                                    ITYPE_t size) nogil except -1:
    cdef DTYPE_t tmp, d=0
    cdef np.intp_t j
    for j in range(size):
        tmp = x1[j] - x2[j]
        d += tmp * tmp
    return d


cdef inline DTYPE_t euclidean_dist_to_rdist(const DTYPE_t dist) nogil except -1:
    return dist * dist


cdef inline DTYPE_t euclidean_rdist_to_dist(const DTYPE_t dist) nogil except -1:
    return sqrt(dist)


######################################################################
# DistanceMetric base class
cdef class DistanceMetric:
    # The following attributes are required for a few of the subclasses.
    # we must define them here so that cython's limited polymorphism will work.
    # Because we don't expect to instantiate a lot of these objects, the
    # extra memory overhead of this setup should not be an issue.
    cdef DTYPE_t p
    cdef DTYPE_t[::1] vec
    cdef DTYPE_t[:, ::1] mat
    cdef ITYPE_t size
    cdef object func
    cdef object kwargs

    cdef DTYPE_t dist(self, const DTYPE_t* x1, const DTYPE_t* x2,
                      ITYPE_t size) nogil except -1

    cdef DTYPE_t rdist(self, const DTYPE_t* x1, const DTYPE_t* x2,
                       ITYPE_t size) nogil except -1

    cdef int pdist(self, const DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] D) except -1

    cdef int cdist(self, const DTYPE_t[:, ::1] X, const DTYPE_t[:, ::1] Y,
                   DTYPE_t[:, ::1] D) except -1

    cdef DTYPE_t _rdist_to_dist(self, DTYPE_t rdist) nogil except -1

    cdef DTYPE_t _dist_to_rdist(self, DTYPE_t dist) nogil except -1
