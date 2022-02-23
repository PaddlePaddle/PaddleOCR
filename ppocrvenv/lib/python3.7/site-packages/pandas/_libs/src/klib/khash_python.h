#include <string.h>
#include <Python.h>


// use numpy's definitions for complex
#include <numpy/arrayobject.h>
typedef npy_complex64 khcomplex64_t;
typedef npy_complex128 khcomplex128_t;



// khash should report usage to tracemalloc
#if PY_VERSION_HEX >= 0x03060000
#include <pymem.h>
#if PY_VERSION_HEX < 0x03070000
#define PyTraceMalloc_Track _PyTraceMalloc_Track
#define PyTraceMalloc_Untrack _PyTraceMalloc_Untrack
#endif
#else
#define PyTraceMalloc_Track(...)
#define PyTraceMalloc_Untrack(...)
#endif


static const int KHASH_TRACE_DOMAIN = 424242;
void *traced_malloc(size_t size){
    void * ptr = malloc(size);
    if(ptr!=NULL){
        PyTraceMalloc_Track(KHASH_TRACE_DOMAIN, (uintptr_t)ptr, size);
    }
    return ptr;
}

void *traced_calloc(size_t num, size_t size){
    void * ptr = calloc(num, size);
    if(ptr!=NULL){
        PyTraceMalloc_Track(KHASH_TRACE_DOMAIN, (uintptr_t)ptr, num*size);
    }
    return ptr;
}

void *traced_realloc(void* old_ptr, size_t size){
    void * ptr = realloc(old_ptr, size);
    if(ptr!=NULL){
        if(old_ptr != ptr){
            PyTraceMalloc_Untrack(KHASH_TRACE_DOMAIN, (uintptr_t)old_ptr);
        }
        PyTraceMalloc_Track(KHASH_TRACE_DOMAIN, (uintptr_t)ptr, size);
    }
    return ptr;
}

void traced_free(void* ptr){
    if(ptr!=NULL){
        PyTraceMalloc_Untrack(KHASH_TRACE_DOMAIN, (uintptr_t)ptr);
    }
    free(ptr);
}


#define KHASH_MALLOC traced_malloc
#define KHASH_REALLOC traced_realloc
#define KHASH_CALLOC traced_calloc
#define KHASH_FREE traced_free
#include "khash.h"

// Previously we were using the built in cpython hash function for doubles
// python 2.7 https://github.com/python/cpython/blob/2.7/Objects/object.c#L1021
// python 3.5 https://github.com/python/cpython/blob/3.5/Python/pyhash.c#L85

// The python 3 hash function has the invariant hash(x) == hash(int(x)) == hash(decimal(x))
// and the size of hash may be different by platform / version (long in py2, Py_ssize_t in py3).
// We don't need those invariants because types will be cast before hashing, and if Py_ssize_t
// is 64 bits the truncation causes collission issues.  Given all that, we use our own
// simple hash, viewing the double bytes as an int64 and using khash's default
// hash for 64 bit integers.
// GH 13436 showed that _Py_HashDouble doesn't work well with khash
// GH 28303 showed, that the simple xoring-version isn't good enough
// See GH 36729 for evaluation of the currently used murmur2-hash version
// An interesting alternative to expensive murmur2-hash would be to change
// the probing strategy and use e.g. the probing strategy from CPython's
// implementation of dicts, which shines for smaller sizes but is more
// predisposed to superlinear running times (see GH 36729 for comparison)


khuint64_t PANDAS_INLINE asuint64(double key) {
    khuint64_t val;
    memcpy(&val, &key, sizeof(double));
    return val;
}

khuint32_t PANDAS_INLINE asuint32(float key) {
    khuint32_t val;
    memcpy(&val, &key, sizeof(float));
    return val;
}

#define ZERO_HASH 0
#define NAN_HASH  0

khuint32_t PANDAS_INLINE kh_float64_hash_func(double val){
    // 0.0 and -0.0 should have the same hash:
    if (val == 0.0){
        return ZERO_HASH;
    }
    // all nans should have the same hash:
    if ( val!=val ){
        return NAN_HASH;
    }
    khuint64_t as_int = asuint64(val);
    return murmur2_64to32(as_int);
}

khuint32_t PANDAS_INLINE kh_float32_hash_func(float val){
    // 0.0 and -0.0 should have the same hash:
    if (val == 0.0f){
        return ZERO_HASH;
    }
    // all nans should have the same hash:
    if ( val!=val ){
        return NAN_HASH;
    }
    khuint32_t as_int = asuint32(val);
    return murmur2_32to32(as_int);
}

#define kh_floats_hash_equal(a, b) ((a) == (b) || ((b) != (b) && (a) != (a)))

#define KHASH_MAP_INIT_FLOAT64(name, khval_t)								\
	KHASH_INIT(name, khfloat64_t, khval_t, 1, kh_float64_hash_func, kh_floats_hash_equal)

KHASH_MAP_INIT_FLOAT64(float64, size_t)

#define KHASH_MAP_INIT_FLOAT32(name, khval_t)								\
	KHASH_INIT(name, khfloat32_t, khval_t, 1, kh_float32_hash_func, kh_floats_hash_equal)

KHASH_MAP_INIT_FLOAT32(float32, size_t)

khint32_t PANDAS_INLINE kh_complex128_hash_func(khcomplex128_t val){
    return kh_float64_hash_func(val.real)^kh_float64_hash_func(val.imag);
}
khint32_t PANDAS_INLINE kh_complex64_hash_func(khcomplex64_t val){
    return kh_float32_hash_func(val.real)^kh_float32_hash_func(val.imag);
}

#define kh_complex_hash_equal(a, b) \
  (kh_floats_hash_equal(a.real, b.real) && kh_floats_hash_equal(a.imag, b.imag))


#define KHASH_MAP_INIT_COMPLEX64(name, khval_t)								\
	KHASH_INIT(name, khcomplex64_t, khval_t, 1, kh_complex64_hash_func, kh_complex_hash_equal)

KHASH_MAP_INIT_COMPLEX64(complex64, size_t)


#define KHASH_MAP_INIT_COMPLEX128(name, khval_t)								\
	KHASH_INIT(name, khcomplex128_t, khval_t, 1, kh_complex128_hash_func, kh_complex_hash_equal)

KHASH_MAP_INIT_COMPLEX128(complex128, size_t)


#define kh_exist_complex64(h, k) (kh_exist(h, k))
#define kh_exist_complex128(h, k) (kh_exist(h, k))


// NaN-floats should be in the same equivalency class, see GH 22119
int PANDAS_INLINE floatobject_cmp(PyFloatObject* a, PyFloatObject* b){
    return (
             Py_IS_NAN(PyFloat_AS_DOUBLE(a)) &&
             Py_IS_NAN(PyFloat_AS_DOUBLE(b))
           )
           ||
           ( PyFloat_AS_DOUBLE(a) == PyFloat_AS_DOUBLE(b) );
}


// NaNs should be in the same equivalency class, see GH 41836
// PyObject_RichCompareBool for complexobjects has a different behavior
// needs to be replaced
int PANDAS_INLINE complexobject_cmp(PyComplexObject* a, PyComplexObject* b){
    return (
                Py_IS_NAN(a->cval.real) &&
                Py_IS_NAN(b->cval.real) &&
                Py_IS_NAN(a->cval.imag) &&
                Py_IS_NAN(b->cval.imag)
           )
           ||
           (
                Py_IS_NAN(a->cval.real) &&
                Py_IS_NAN(b->cval.real) &&
                a->cval.imag == b->cval.imag
           )
           ||
           (
                a->cval.real == b->cval.real &&
                Py_IS_NAN(a->cval.imag) &&
                Py_IS_NAN(b->cval.imag)
           )
           ||
           (
                a->cval.real == b->cval.real &&
                a->cval.imag == b->cval.imag
           );
}

int PANDAS_INLINE pyobject_cmp(PyObject* a, PyObject* b);


// replacing PyObject_RichCompareBool (NaN!=NaN) with pyobject_cmp (NaN==NaN),
// which treats NaNs as equivalent
// see GH 41836
int PANDAS_INLINE tupleobject_cmp(PyTupleObject* a, PyTupleObject* b){
    Py_ssize_t i;

    if (Py_SIZE(a) != Py_SIZE(b)) {
        return 0;
    }

    for (i = 0; i < Py_SIZE(a); ++i) {
        if (!pyobject_cmp(PyTuple_GET_ITEM(a, i), PyTuple_GET_ITEM(b, i))) {
            return 0;
        }
    }
    return 1;
}


int PANDAS_INLINE pyobject_cmp(PyObject* a, PyObject* b) {
    if (a == b) {
        return 1;
    }
    if (Py_TYPE(a) == Py_TYPE(b)) {
        // special handling for some built-in types which could have NaNs
        // as we would like to have them equivalent, but the usual
        // PyObject_RichCompareBool would return False
        if (PyFloat_CheckExact(a)) {
            return floatobject_cmp((PyFloatObject*)a, (PyFloatObject*)b);
        }
        if (PyComplex_CheckExact(a)) {
            return complexobject_cmp((PyComplexObject*)a, (PyComplexObject*)b);
        }
        if (PyTuple_CheckExact(a)) {
            return tupleobject_cmp((PyTupleObject*)a, (PyTupleObject*)b);
        }
        // frozenset isn't yet supported
    }

	int result = PyObject_RichCompareBool(a, b, Py_EQ);
	if (result < 0) {
		PyErr_Clear();
		return 0;
	}
	return result;
}


Py_hash_t PANDAS_INLINE _Pandas_HashDouble(double val) {
    //Since Python3.10, nan is no longer has hash 0
    if (Py_IS_NAN(val)) {
        return 0;
    }
#if PY_VERSION_HEX < 0x030A0000
    return _Py_HashDouble(val);
#else
    return _Py_HashDouble(NULL, val);
#endif
}


Py_hash_t PANDAS_INLINE floatobject_hash(PyFloatObject* key) {
    return _Pandas_HashDouble(PyFloat_AS_DOUBLE(key));
}


#define _PandasHASH_IMAG 1000003UL

// replaces _Py_HashDouble with _Pandas_HashDouble
Py_hash_t PANDAS_INLINE complexobject_hash(PyComplexObject* key) {
    Py_uhash_t realhash = (Py_uhash_t)_Pandas_HashDouble(key->cval.real);
    Py_uhash_t imaghash = (Py_uhash_t)_Pandas_HashDouble(key->cval.imag);
    if (realhash == (Py_uhash_t)-1 || imaghash == (Py_uhash_t)-1) {
        return -1;
    }
    Py_uhash_t combined = realhash + _PandasHASH_IMAG * imaghash;
    if (combined == (Py_uhash_t)-1) {
        return -2;
    }
    return (Py_hash_t)combined;
}


khuint32_t PANDAS_INLINE kh_python_hash_func(PyObject* key);

//we could use any hashing algorithm, this is the original CPython's for tuples

#if SIZEOF_PY_UHASH_T > 4
#define _PandasHASH_XXPRIME_1 ((Py_uhash_t)11400714785074694791ULL)
#define _PandasHASH_XXPRIME_2 ((Py_uhash_t)14029467366897019727ULL)
#define _PandasHASH_XXPRIME_5 ((Py_uhash_t)2870177450012600261ULL)
#define _PandasHASH_XXROTATE(x) ((x << 31) | (x >> 33))  /* Rotate left 31 bits */
#else
#define _PandasHASH_XXPRIME_1 ((Py_uhash_t)2654435761UL)
#define _PandasHASH_XXPRIME_2 ((Py_uhash_t)2246822519UL)
#define _PandasHASH_XXPRIME_5 ((Py_uhash_t)374761393UL)
#define _PandasHASH_XXROTATE(x) ((x << 13) | (x >> 19))  /* Rotate left 13 bits */
#endif

Py_hash_t PANDAS_INLINE tupleobject_hash(PyTupleObject* key) {
    Py_ssize_t i, len = Py_SIZE(key);
    PyObject **item = key->ob_item;

    Py_uhash_t acc = _PandasHASH_XXPRIME_5;
    for (i = 0; i < len; i++) {
        Py_uhash_t lane = kh_python_hash_func(item[i]);
        if (lane == (Py_uhash_t)-1) {
            return -1;
        }
        acc += lane * _PandasHASH_XXPRIME_2;
        acc = _PandasHASH_XXROTATE(acc);
        acc *= _PandasHASH_XXPRIME_1;
    }

    /* Add input length, mangled to keep the historical value of hash(()). */
    acc += len ^ (_PandasHASH_XXPRIME_5 ^ 3527539UL);

    if (acc == (Py_uhash_t)-1) {
        return 1546275796;
    }
    return acc;
}


khuint32_t PANDAS_INLINE kh_python_hash_func(PyObject* key) {
    Py_hash_t hash;
    // For PyObject_Hash holds:
    //    hash(0.0) == 0 == hash(-0.0)
    //    yet for different nan-objects different hash-values
    //    are possible
    if (PyFloat_CheckExact(key)) {
        // we cannot use kh_float64_hash_func
        // becase float(k) == k holds for any int-object k
        // and kh_float64_hash_func doesn't respect it
        hash = floatobject_hash((PyFloatObject*)key);
    }
    else if (PyComplex_CheckExact(key)) {
        // we cannot use kh_complex128_hash_func
        // becase complex(k,0) == k holds for any int-object k
        // and kh_complex128_hash_func doesn't respect it
        hash = complexobject_hash((PyComplexObject*)key);
    }
    else if (PyTuple_CheckExact(key)) {
        hash = tupleobject_hash((PyTupleObject*)key);
    }
    else {
        hash = PyObject_Hash(key);
    }

	if (hash == -1) {
		PyErr_Clear();
		return 0;
	}
    #if SIZEOF_PY_HASH_T == 4
        // it is already 32bit value
        return hash;
    #else
        // for 64bit builds,
        // we need information of the upper 32bits as well
        // see GH 37615
        khuint64_t as_uint = (khuint64_t) hash;
        // uints avoid undefined behavior of signed ints
        return (as_uint>>32)^as_uint;
    #endif
}


#define kh_python_hash_equal(a, b) (pyobject_cmp(a, b))


// Python object

typedef PyObject* kh_pyobject_t;

#define KHASH_MAP_INIT_PYOBJECT(name, khval_t)							\
	KHASH_INIT(name, kh_pyobject_t, khval_t, 1,						\
			   kh_python_hash_func, kh_python_hash_equal)

KHASH_MAP_INIT_PYOBJECT(pymap, Py_ssize_t)

#define KHASH_SET_INIT_PYOBJECT(name)                                  \
	KHASH_INIT(name, kh_pyobject_t, char, 0,     \
			   kh_python_hash_func, kh_python_hash_equal)

KHASH_SET_INIT_PYOBJECT(pyset)

#define kh_exist_pymap(h, k) (kh_exist(h, k))
#define kh_exist_pyset(h, k) (kh_exist(h, k))

KHASH_MAP_INIT_STR(strbox, kh_pyobject_t)

typedef struct {
	kh_str_t *table;
	int starts[256];
} kh_str_starts_t;

typedef kh_str_starts_t* p_kh_str_starts_t;

p_kh_str_starts_t PANDAS_INLINE kh_init_str_starts(void) {
	kh_str_starts_t *result = (kh_str_starts_t*)KHASH_CALLOC(1, sizeof(kh_str_starts_t));
	result->table = kh_init_str();
	return result;
}

khuint_t PANDAS_INLINE kh_put_str_starts_item(kh_str_starts_t* table, char* key, int* ret) {
    khuint_t result = kh_put_str(table->table, key, ret);
	if (*ret != 0) {
		table->starts[(unsigned char)key[0]] = 1;
	}
    return result;
}

khuint_t PANDAS_INLINE kh_get_str_starts_item(const kh_str_starts_t* table, const char* key) {
    unsigned char ch = *key;
	if (table->starts[ch]) {
		if (ch == '\0' || kh_get_str(table->table, key) != table->table->n_buckets) return 1;
	}
    return 0;
}

void PANDAS_INLINE kh_destroy_str_starts(kh_str_starts_t* table) {
	kh_destroy_str(table->table);
	KHASH_FREE(table);
}

void PANDAS_INLINE kh_resize_str_starts(kh_str_starts_t* table, khuint_t val) {
	kh_resize_str(table->table, val);
}

// utility function: given the number of elements
// returns number of necessary buckets
khuint_t PANDAS_INLINE kh_needed_n_buckets(khuint_t n_elements){
    khuint_t candidate = n_elements;
    kroundup32(candidate);
    khuint_t upper_bound = (khuint_t)(candidate * __ac_HASH_UPPER + 0.5);
    return (upper_bound < n_elements) ? 2*candidate : candidate;

}
