cimport cython
from cpython.mem cimport (
    PyMem_Free,
    PyMem_Malloc,
)
from cpython.ref cimport (
    Py_INCREF,
    PyObject,
)
from libc.stdlib cimport (
    free,
    malloc,
)

import numpy as np

cimport numpy as cnp
from numpy cimport (
    float64_t,
    ndarray,
    uint8_t,
    uint32_t,
)
from numpy.math cimport NAN

cnp.import_array()


from pandas._libs cimport util
from pandas._libs.khash cimport (
    KHASH_TRACE_DOMAIN,
    are_equivalent_float32_t,
    are_equivalent_float64_t,
    are_equivalent_khcomplex64_t,
    are_equivalent_khcomplex128_t,
    kh_needed_n_buckets,
    kh_python_hash_equal,
    kh_python_hash_func,
    kh_str_t,
    khcomplex64_t,
    khcomplex128_t,
    khiter_t,
)
from pandas._libs.missing cimport checknull


def get_hashtable_trace_domain():
    return KHASH_TRACE_DOMAIN


def object_hash(obj):
    return kh_python_hash_func(obj)


def objects_are_equal(a, b):
    return kh_python_hash_equal(a, b)


cdef int64_t NPY_NAT = util.get_nat()
SIZE_HINT_LIMIT = (1 << 20) + 7


cdef Py_ssize_t _INIT_VEC_CAP = 128

include "hashtable_class_helper.pxi"
include "hashtable_func_helper.pxi"

cdef class Factorizer:
    cdef readonly:
        Py_ssize_t count

    def __cinit__(self, size_hint: int):
        self.count = 0

    def get_count(self) -> int:
        return self.count


cdef class ObjectFactorizer(Factorizer):
    cdef public:
        PyObjectHashTable table
        ObjectVector uniques

    def __cinit__(self, size_hint: int):
        self.table = PyObjectHashTable(size_hint)
        self.uniques = ObjectVector()

    def factorize(
        self, ndarray[object] values, sort=False, na_sentinel=-1, na_value=None
    ) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray[np.intp]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> factorize(np.array([1,2,np.nan], dtype='O'), na_sentinel=20)
        array([ 0,  1, 20])
        """
        cdef:
            ndarray[intp_t] labels

        if self.uniques.external_view_exists:
            uniques = ObjectVector()
            uniques.extend(self.uniques.to_array())
            self.uniques = uniques
        labels = self.table.get_labels(values, self.uniques,
                                       self.count, na_sentinel, na_value)
        mask = (labels == na_sentinel)
        # sort on
        if sort:
            sorter = self.uniques.to_array().argsort()
            reverse_indexer = np.empty(len(sorter), dtype=np.intp)
            reverse_indexer.put(sorter, np.arange(len(sorter)))
            labels = reverse_indexer.take(labels, mode='clip')
            labels[mask] = na_sentinel
        self.count = len(self.uniques)
        return labels


cdef class Int64Factorizer(Factorizer):
    cdef public:
        Int64HashTable table
        Int64Vector uniques

    def __cinit__(self, size_hint: int):
        self.table = Int64HashTable(size_hint)
        self.uniques = Int64Vector()

    def factorize(self, const int64_t[:] values, sort=False,
                  na_sentinel=-1, na_value=None) -> np.ndarray:
        """
        Returns
        -------
        ndarray[intp_t]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> factorize(np.array([1,2,np.nan], dtype='O'), na_sentinel=20)
        array([ 0,  1, 20])
        """
        cdef:
            ndarray[intp_t] labels

        if self.uniques.external_view_exists:
            uniques = Int64Vector()
            uniques.extend(self.uniques.to_array())
            self.uniques = uniques
        labels = self.table.get_labels(values, self.uniques,
                                       self.count, na_sentinel,
                                       na_value=na_value)

        # sort on
        if sort:
            sorter = self.uniques.to_array().argsort()
            reverse_indexer = np.empty(len(sorter), dtype=np.intp)
            reverse_indexer.put(sorter, np.arange(len(sorter)))

            labels = reverse_indexer.take(labels)

        self.count = len(self.uniques)
        return labels


@cython.wraparound(False)
@cython.boundscheck(False)
def unique_label_indices(const int64_t[:] labels) -> ndarray:
    """
    Indices of the first occurrences of the unique labels
    *excluding* -1. equivalent to:
        np.unique(labels, return_index=True)[1]
    """
    cdef:
        int ret = 0
        Py_ssize_t i, n = len(labels)
        kh_int64_t *table = kh_init_int64()
        Int64Vector idx = Int64Vector()
        ndarray[int64_t, ndim=1] arr
        Int64VectorData *ud = idx.data

    kh_resize_int64(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    with nogil:
        for i in range(n):
            kh_put_int64(table, labels[i], &ret)
            if ret != 0:
                if needs_resize(ud):
                    with gil:
                        idx.resize()
                append_data_int64(ud, i)

    kh_destroy_int64(table)

    arr = idx.to_array()
    arr = arr[np.asarray(labels)[arr].argsort()]

    return arr[1:] if arr.size != 0 and labels[arr[0]] == -1 else arr
