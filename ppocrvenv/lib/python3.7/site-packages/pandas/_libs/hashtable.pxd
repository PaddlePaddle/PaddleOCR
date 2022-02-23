from numpy cimport (
    intp_t,
    ndarray,
)

from pandas._libs.khash cimport (
    complex64_t,
    complex128_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    kh_complex64_t,
    kh_complex128_t,
    kh_float32_t,
    kh_float64_t,
    kh_int8_t,
    kh_int16_t,
    kh_int32_t,
    kh_int64_t,
    kh_pymap_t,
    kh_str_t,
    kh_uint8_t,
    kh_uint16_t,
    kh_uint32_t,
    kh_uint64_t,
    khcomplex64_t,
    khcomplex128_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

# prototypes for sharing

cdef class HashTable:
    pass

cdef class UInt64HashTable(HashTable):
    cdef kh_uint64_t *table

    cpdef get_item(self, uint64_t val)
    cpdef set_item(self, uint64_t key, Py_ssize_t val)

cdef class Int64HashTable(HashTable):
    cdef kh_int64_t *table

    cpdef get_item(self, int64_t val)
    cpdef set_item(self, int64_t key, Py_ssize_t val)

cdef class UInt32HashTable(HashTable):
    cdef kh_uint32_t *table

    cpdef get_item(self, uint32_t val)
    cpdef set_item(self, uint32_t key, Py_ssize_t val)

cdef class Int32HashTable(HashTable):
    cdef kh_int32_t *table

    cpdef get_item(self, int32_t val)
    cpdef set_item(self, int32_t key, Py_ssize_t val)

cdef class UInt16HashTable(HashTable):
    cdef kh_uint16_t *table

    cpdef get_item(self, uint16_t val)
    cpdef set_item(self, uint16_t key, Py_ssize_t val)

cdef class Int16HashTable(HashTable):
    cdef kh_int16_t *table

    cpdef get_item(self, int16_t val)
    cpdef set_item(self, int16_t key, Py_ssize_t val)

cdef class UInt8HashTable(HashTable):
    cdef kh_uint8_t *table

    cpdef get_item(self, uint8_t val)
    cpdef set_item(self, uint8_t key, Py_ssize_t val)

cdef class Int8HashTable(HashTable):
    cdef kh_int8_t *table

    cpdef get_item(self, int8_t val)
    cpdef set_item(self, int8_t key, Py_ssize_t val)

cdef class Float64HashTable(HashTable):
    cdef kh_float64_t *table

    cpdef get_item(self, float64_t val)
    cpdef set_item(self, float64_t key, Py_ssize_t val)

cdef class Float32HashTable(HashTable):
    cdef kh_float32_t *table

    cpdef get_item(self, float32_t val)
    cpdef set_item(self, float32_t key, Py_ssize_t val)

cdef class Complex64HashTable(HashTable):
    cdef kh_complex64_t *table

    cpdef get_item(self, complex64_t val)
    cpdef set_item(self, complex64_t key, Py_ssize_t val)

cdef class Complex128HashTable(HashTable):
    cdef kh_complex128_t *table

    cpdef get_item(self, complex128_t val)
    cpdef set_item(self, complex128_t key, Py_ssize_t val)

cdef class PyObjectHashTable(HashTable):
    cdef kh_pymap_t *table

    cpdef get_item(self, object val)
    cpdef set_item(self, object key, Py_ssize_t val)


cdef class StringHashTable(HashTable):
    cdef kh_str_t *table

    cpdef get_item(self, str val)
    cpdef set_item(self, str key, Py_ssize_t val)

cdef struct Int64VectorData:
    int64_t *data
    Py_ssize_t n, m

cdef class Vector:
    cdef bint external_view_exists

cdef class Int64Vector(Vector):
    cdef Int64VectorData *data
    cdef ndarray ao

    cdef resize(self)
    cpdef ndarray to_array(self)
    cdef inline void append(self, int64_t x)
    cdef extend(self, int64_t[:] x)
