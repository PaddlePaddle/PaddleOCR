from cpython.object cimport PyObject
from numpy cimport (
    complex64_t,
    complex128_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)


cdef extern from "khash_python.h":
    const int KHASH_TRACE_DOMAIN

    ctypedef uint32_t khuint_t
    ctypedef khuint_t khiter_t

    ctypedef struct khcomplex128_t:
        double real
        double imag

    bint are_equivalent_khcomplex128_t \
    "kh_complex_hash_equal" (khcomplex128_t a, khcomplex128_t b) nogil

    ctypedef struct khcomplex64_t:
        float real
        float imag

    bint are_equivalent_khcomplex64_t \
    "kh_complex_hash_equal" (khcomplex64_t a, khcomplex64_t b) nogil

    bint are_equivalent_float64_t \
    "kh_floats_hash_equal" (float64_t a, float64_t b) nogil

    bint are_equivalent_float32_t \
    "kh_floats_hash_equal" (float32_t a, float32_t b) nogil

    uint32_t kh_python_hash_func(object key)
    bint kh_python_hash_equal(object a, object b)

    ctypedef struct kh_pymap_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        PyObject **keys
        size_t *vals

    kh_pymap_t* kh_init_pymap()
    void kh_destroy_pymap(kh_pymap_t*)
    void kh_clear_pymap(kh_pymap_t*)
    khuint_t kh_get_pymap(kh_pymap_t*, PyObject*)
    void kh_resize_pymap(kh_pymap_t*, khuint_t)
    khuint_t kh_put_pymap(kh_pymap_t*, PyObject*, int*)
    void kh_del_pymap(kh_pymap_t*, khuint_t)

    bint kh_exist_pymap(kh_pymap_t*, khiter_t)

    ctypedef struct kh_pyset_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        PyObject **keys
        size_t *vals

    kh_pyset_t* kh_init_pyset()
    void kh_destroy_pyset(kh_pyset_t*)
    void kh_clear_pyset(kh_pyset_t*)
    khuint_t kh_get_pyset(kh_pyset_t*, PyObject*)
    void kh_resize_pyset(kh_pyset_t*, khuint_t)
    khuint_t kh_put_pyset(kh_pyset_t*, PyObject*, int*)
    void kh_del_pyset(kh_pyset_t*, khuint_t)

    bint kh_exist_pyset(kh_pyset_t*, khiter_t)

    ctypedef char* kh_cstr_t

    ctypedef struct kh_str_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        kh_cstr_t *keys
        size_t *vals

    kh_str_t* kh_init_str() nogil
    void kh_destroy_str(kh_str_t*) nogil
    void kh_clear_str(kh_str_t*) nogil
    khuint_t kh_get_str(kh_str_t*, kh_cstr_t) nogil
    void kh_resize_str(kh_str_t*, khuint_t) nogil
    khuint_t kh_put_str(kh_str_t*, kh_cstr_t, int*) nogil
    void kh_del_str(kh_str_t*, khuint_t) nogil

    bint kh_exist_str(kh_str_t*, khiter_t) nogil

    ctypedef struct kh_str_starts_t:
        kh_str_t *table
        int starts[256]

    kh_str_starts_t* kh_init_str_starts() nogil
    khuint_t kh_put_str_starts_item(kh_str_starts_t* table, char* key,
                                    int* ret) nogil
    khuint_t kh_get_str_starts_item(kh_str_starts_t* table, char* key) nogil
    void kh_destroy_str_starts(kh_str_starts_t*) nogil
    void kh_resize_str_starts(kh_str_starts_t*, khuint_t) nogil

    # sweep factorize

    ctypedef struct kh_strbox_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        kh_cstr_t *keys
        PyObject **vals

    kh_strbox_t* kh_init_strbox() nogil
    void kh_destroy_strbox(kh_strbox_t*) nogil
    void kh_clear_strbox(kh_strbox_t*) nogil
    khuint_t kh_get_strbox(kh_strbox_t*, kh_cstr_t) nogil
    void kh_resize_strbox(kh_strbox_t*, khuint_t) nogil
    khuint_t kh_put_strbox(kh_strbox_t*, kh_cstr_t, int*) nogil
    void kh_del_strbox(kh_strbox_t*, khuint_t) nogil

    bint kh_exist_strbox(kh_strbox_t*, khiter_t) nogil

    khuint_t kh_needed_n_buckets(khuint_t element_n) nogil


include "khash_for_primitive_helper.pxi"
