/*
 * Optional optimisations of built-in functions and methods.
 *
 * Required replacements of builtins are in Builtins.c.
 *
 * General object operations and protocols are in ObjectHandling.c.
 */

/////////////// append.proto ///////////////

static CYTHON_INLINE int __Pyx_PyObject_Append(PyObject* L, PyObject* x); /*proto*/

/////////////// append ///////////////
//@requires: ListAppend
//@requires: ObjectHandling.c::PyObjectCallMethod1

static CYTHON_INLINE int __Pyx_PyObject_Append(PyObject* L, PyObject* x) {
    if (likely(PyList_CheckExact(L))) {
        if (unlikely(__Pyx_PyList_Append(L, x) < 0)) return -1;
    } else {
        PyObject* retval = __Pyx_PyObject_CallMethod1(L, PYIDENT("append"), x);
        if (unlikely(!retval))
            return -1;
        Py_DECREF(retval);
    }
    return 0;
}

/////////////// ListAppend.proto ///////////////

#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static CYTHON_INLINE int __Pyx_PyList_Append(PyObject* list, PyObject* x) {
    PyListObject* L = (PyListObject*) list;
    Py_ssize_t len = Py_SIZE(list);
    if (likely(L->allocated > len) & likely(len > (L->allocated >> 1))) {
        Py_INCREF(x);
        PyList_SET_ITEM(list, len, x);
        __Pyx_SET_SIZE(list, len + 1);
        return 0;
    }
    return PyList_Append(list, x);
}
#else
#define __Pyx_PyList_Append(L,x) PyList_Append(L,x)
#endif

/////////////// ListCompAppend.proto ///////////////

#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static CYTHON_INLINE int __Pyx_ListComp_Append(PyObject* list, PyObject* x) {
    PyListObject* L = (PyListObject*) list;
    Py_ssize_t len = Py_SIZE(list);
    if (likely(L->allocated > len)) {
        Py_INCREF(x);
        PyList_SET_ITEM(list, len, x);
        __Pyx_SET_SIZE(list, len + 1);
        return 0;
    }
    return PyList_Append(list, x);
}
#else
#define __Pyx_ListComp_Append(L,x) PyList_Append(L,x)
#endif

//////////////////// ListExtend.proto ////////////////////

static CYTHON_INLINE int __Pyx_PyList_Extend(PyObject* L, PyObject* v) {
#if CYTHON_COMPILING_IN_CPYTHON
    PyObject* none = _PyList_Extend((PyListObject*)L, v);
    if (unlikely(!none))
        return -1;
    Py_DECREF(none);
    return 0;
#else
    return PyList_SetSlice(L, PY_SSIZE_T_MAX, PY_SSIZE_T_MAX, v);
#endif
}

/////////////// pop.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx__PyObject_Pop(PyObject* L); /*proto*/

#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static CYTHON_INLINE PyObject* __Pyx_PyList_Pop(PyObject* L); /*proto*/
#define __Pyx_PyObject_Pop(L) (likely(PyList_CheckExact(L)) ? \
    __Pyx_PyList_Pop(L) : __Pyx__PyObject_Pop(L))

#else
#define __Pyx_PyList_Pop(L)  __Pyx__PyObject_Pop(L)
#define __Pyx_PyObject_Pop(L)  __Pyx__PyObject_Pop(L)
#endif

/////////////// pop ///////////////
//@requires: ObjectHandling.c::PyObjectCallMethod0

static CYTHON_INLINE PyObject* __Pyx__PyObject_Pop(PyObject* L) {
    if (Py_TYPE(L) == &PySet_Type) {
        return PySet_Pop(L);
    }
    return __Pyx_PyObject_CallMethod0(L, PYIDENT("pop"));
}

#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static CYTHON_INLINE PyObject* __Pyx_PyList_Pop(PyObject* L) {
    /* Check that both the size is positive and no reallocation shrinking needs to be done. */
    if (likely(PyList_GET_SIZE(L) > (((PyListObject*)L)->allocated >> 1))) {
        __Pyx_SET_SIZE(L, Py_SIZE(L) - 1);
        return PyList_GET_ITEM(L, PyList_GET_SIZE(L));
    }
    return CALL_UNBOUND_METHOD(PyList_Type, "pop", L);
}
#endif


/////////////// pop_index.proto ///////////////

static PyObject* __Pyx__PyObject_PopNewIndex(PyObject* L, PyObject* py_ix); /*proto*/
static PyObject* __Pyx__PyObject_PopIndex(PyObject* L, PyObject* py_ix); /*proto*/

#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static PyObject* __Pyx__PyList_PopIndex(PyObject* L, PyObject* py_ix, Py_ssize_t ix); /*proto*/

#define __Pyx_PyObject_PopIndex(L, py_ix, ix, is_signed, type, to_py_func) ( \
    (likely(PyList_CheckExact(L) && __Pyx_fits_Py_ssize_t(ix, type, is_signed))) ? \
        __Pyx__PyList_PopIndex(L, py_ix, ix) : ( \
        (unlikely((py_ix) == Py_None)) ? __Pyx__PyObject_PopNewIndex(L, to_py_func(ix)) : \
            __Pyx__PyObject_PopIndex(L, py_ix)))

#define __Pyx_PyList_PopIndex(L, py_ix, ix, is_signed, type, to_py_func) ( \
    __Pyx_fits_Py_ssize_t(ix, type, is_signed) ? \
        __Pyx__PyList_PopIndex(L, py_ix, ix) : ( \
        (unlikely((py_ix) == Py_None)) ? __Pyx__PyObject_PopNewIndex(L, to_py_func(ix)) : \
            __Pyx__PyObject_PopIndex(L, py_ix)))

#else

#define __Pyx_PyList_PopIndex(L, py_ix, ix, is_signed, type, to_py_func) \
    __Pyx_PyObject_PopIndex(L, py_ix, ix, is_signed, type, to_py_func)

#define __Pyx_PyObject_PopIndex(L, py_ix, ix, is_signed, type, to_py_func) ( \
    (unlikely((py_ix) == Py_None)) ? __Pyx__PyObject_PopNewIndex(L, to_py_func(ix)) : \
        __Pyx__PyObject_PopIndex(L, py_ix))
#endif

/////////////// pop_index ///////////////
//@requires: ObjectHandling.c::PyObjectCallMethod1

static PyObject* __Pyx__PyObject_PopNewIndex(PyObject* L, PyObject* py_ix) {
    PyObject *r;
    if (unlikely(!py_ix)) return NULL;
    r = __Pyx__PyObject_PopIndex(L, py_ix);
    Py_DECREF(py_ix);
    return r;
}

static PyObject* __Pyx__PyObject_PopIndex(PyObject* L, PyObject* py_ix) {
    return __Pyx_PyObject_CallMethod1(L, PYIDENT("pop"), py_ix);
}

#if CYTHON_USE_PYLIST_INTERNALS && CYTHON_ASSUME_SAFE_MACROS
static PyObject* __Pyx__PyList_PopIndex(PyObject* L, PyObject* py_ix, Py_ssize_t ix) {
    Py_ssize_t size = PyList_GET_SIZE(L);
    if (likely(size > (((PyListObject*)L)->allocated >> 1))) {
        Py_ssize_t cix = ix;
        if (cix < 0) {
            cix += size;
        }
        if (likely(__Pyx_is_valid_index(cix, size))) {
            PyObject* v = PyList_GET_ITEM(L, cix);
            __Pyx_SET_SIZE(L, Py_SIZE(L) - 1);
            size -= 1;
            memmove(&PyList_GET_ITEM(L, cix), &PyList_GET_ITEM(L, cix+1), (size_t)(size-cix)*sizeof(PyObject*));
            return v;
        }
    }
    if (py_ix == Py_None) {
        return __Pyx__PyObject_PopNewIndex(L, PyInt_FromSsize_t(ix));
    } else {
        return __Pyx__PyObject_PopIndex(L, py_ix);
    }
}
#endif


/////////////// dict_getitem_default.proto ///////////////

static PyObject* __Pyx_PyDict_GetItemDefault(PyObject* d, PyObject* key, PyObject* default_value); /*proto*/

/////////////// dict_getitem_default ///////////////

static PyObject* __Pyx_PyDict_GetItemDefault(PyObject* d, PyObject* key, PyObject* default_value) {
    PyObject* value;
#if PY_MAJOR_VERSION >= 3 && !CYTHON_COMPILING_IN_PYPY
    value = PyDict_GetItemWithError(d, key);
    if (unlikely(!value)) {
        if (unlikely(PyErr_Occurred()))
            return NULL;
        value = default_value;
    }
    Py_INCREF(value);
    // avoid C compiler warning about unused utility functions
    if ((1));
#else
    if (PyString_CheckExact(key) || PyUnicode_CheckExact(key) || PyInt_CheckExact(key)) {
        /* these presumably have safe hash functions */
        value = PyDict_GetItem(d, key);
        if (unlikely(!value)) {
            value = default_value;
        }
        Py_INCREF(value);
    }
#endif
    else {
        if (default_value == Py_None)
            value = CALL_UNBOUND_METHOD(PyDict_Type, "get", d, key);
        else
            value = CALL_UNBOUND_METHOD(PyDict_Type, "get", d, key, default_value);
    }
    return value;
}


/////////////// dict_setdefault.proto ///////////////

static CYTHON_INLINE PyObject *__Pyx_PyDict_SetDefault(PyObject *d, PyObject *key, PyObject *default_value, int is_safe_type); /*proto*/

/////////////// dict_setdefault ///////////////

static CYTHON_INLINE PyObject *__Pyx_PyDict_SetDefault(PyObject *d, PyObject *key, PyObject *default_value,
                                                       CYTHON_UNUSED int is_safe_type) {
    PyObject* value;
#if PY_VERSION_HEX >= 0x030400A0
    // we keep the method call at the end to avoid "unused" C compiler warnings
    if ((1)) {
        value = PyDict_SetDefault(d, key, default_value);
        if (unlikely(!value)) return NULL;
        Py_INCREF(value);
#else
    if (is_safe_type == 1 || (is_safe_type == -1 &&
        /* the following builtins presumably have repeatably safe and fast hash functions */
#if PY_MAJOR_VERSION >= 3 && !CYTHON_COMPILING_IN_PYPY
            (PyUnicode_CheckExact(key) || PyString_CheckExact(key) || PyLong_CheckExact(key)))) {
        value = PyDict_GetItemWithError(d, key);
        if (unlikely(!value)) {
            if (unlikely(PyErr_Occurred()))
                return NULL;
            if (unlikely(PyDict_SetItem(d, key, default_value) == -1))
                return NULL;
            value = default_value;
        }
        Py_INCREF(value);
#else
            (PyString_CheckExact(key) || PyUnicode_CheckExact(key) || PyInt_CheckExact(key) || PyLong_CheckExact(key)))) {
        value = PyDict_GetItem(d, key);
        if (unlikely(!value)) {
            if (unlikely(PyDict_SetItem(d, key, default_value) == -1))
                return NULL;
            value = default_value;
        }
        Py_INCREF(value);
#endif
#endif
    } else {
        value = CALL_UNBOUND_METHOD(PyDict_Type, "setdefault", d, key, default_value);
    }
    return value;
}


/////////////// py_dict_clear.proto ///////////////

#define __Pyx_PyDict_Clear(d) (PyDict_Clear(d), 0)


/////////////// py_dict_pop.proto ///////////////

static CYTHON_INLINE PyObject *__Pyx_PyDict_Pop(PyObject *d, PyObject *key, PyObject *default_value); /*proto*/

/////////////// py_dict_pop ///////////////

static CYTHON_INLINE PyObject *__Pyx_PyDict_Pop(PyObject *d, PyObject *key, PyObject *default_value) {
#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX > 0x030600B3
    if ((1)) {
        return _PyDict_Pop(d, key, default_value);
    } else
    // avoid "function unused" warnings
#endif
    if (default_value) {
        return CALL_UNBOUND_METHOD(PyDict_Type, "pop", d, key, default_value);
    } else {
        return CALL_UNBOUND_METHOD(PyDict_Type, "pop", d, key);
    }
}


/////////////// dict_iter.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_dict_iterator(PyObject* dict, int is_dict, PyObject* method_name,
                                                   Py_ssize_t* p_orig_length, int* p_is_dict);
static CYTHON_INLINE int __Pyx_dict_iter_next(PyObject* dict_or_iter, Py_ssize_t orig_length, Py_ssize_t* ppos,
                                              PyObject** pkey, PyObject** pvalue, PyObject** pitem, int is_dict);

/////////////// dict_iter ///////////////
//@requires: ObjectHandling.c::UnpackTuple2
//@requires: ObjectHandling.c::IterFinish
//@requires: ObjectHandling.c::PyObjectCallMethod0

static CYTHON_INLINE PyObject* __Pyx_dict_iterator(PyObject* iterable, int is_dict, PyObject* method_name,
                                                   Py_ssize_t* p_orig_length, int* p_source_is_dict) {
    is_dict = is_dict || likely(PyDict_CheckExact(iterable));
    *p_source_is_dict = is_dict;
    if (is_dict) {
#if !CYTHON_COMPILING_IN_PYPY
        *p_orig_length = PyDict_Size(iterable);
        Py_INCREF(iterable);
        return iterable;
#elif PY_MAJOR_VERSION >= 3
        // On PyPy3, we need to translate manually a few method names.
        // This logic is not needed on CPython thanks to the fast case above.
        static PyObject *py_items = NULL, *py_keys = NULL, *py_values = NULL;
        PyObject **pp = NULL;
        if (method_name) {
            const char *name = PyUnicode_AsUTF8(method_name);
            if (strcmp(name, "iteritems") == 0) pp = &py_items;
            else if (strcmp(name, "iterkeys") == 0) pp = &py_keys;
            else if (strcmp(name, "itervalues") == 0) pp = &py_values;
            if (pp) {
                if (!*pp) {
                    *pp = PyUnicode_FromString(name + 4);
                    if (!*pp)
                        return NULL;
                }
                method_name = *pp;
            }
        }
#endif
    }
    *p_orig_length = 0;
    if (method_name) {
        PyObject* iter;
        iterable = __Pyx_PyObject_CallMethod0(iterable, method_name);
        if (!iterable)
            return NULL;
#if !CYTHON_COMPILING_IN_PYPY
        if (PyTuple_CheckExact(iterable) || PyList_CheckExact(iterable))
            return iterable;
#endif
        iter = PyObject_GetIter(iterable);
        Py_DECREF(iterable);
        return iter;
    }
    return PyObject_GetIter(iterable);
}

static CYTHON_INLINE int __Pyx_dict_iter_next(
        PyObject* iter_obj, CYTHON_NCP_UNUSED Py_ssize_t orig_length, CYTHON_NCP_UNUSED Py_ssize_t* ppos,
        PyObject** pkey, PyObject** pvalue, PyObject** pitem, int source_is_dict) {
    PyObject* next_item;
#if !CYTHON_COMPILING_IN_PYPY
    if (source_is_dict) {
        PyObject *key, *value;
        if (unlikely(orig_length != PyDict_Size(iter_obj))) {
            PyErr_SetString(PyExc_RuntimeError, "dictionary changed size during iteration");
            return -1;
        }
        if (unlikely(!PyDict_Next(iter_obj, ppos, &key, &value))) {
            return 0;
        }
        if (pitem) {
            PyObject* tuple = PyTuple_New(2);
            if (unlikely(!tuple)) {
                return -1;
            }
            Py_INCREF(key);
            Py_INCREF(value);
            PyTuple_SET_ITEM(tuple, 0, key);
            PyTuple_SET_ITEM(tuple, 1, value);
            *pitem = tuple;
        } else {
            if (pkey) {
                Py_INCREF(key);
                *pkey = key;
            }
            if (pvalue) {
                Py_INCREF(value);
                *pvalue = value;
            }
        }
        return 1;
    } else if (PyTuple_CheckExact(iter_obj)) {
        Py_ssize_t pos = *ppos;
        if (unlikely(pos >= PyTuple_GET_SIZE(iter_obj))) return 0;
        *ppos = pos + 1;
        next_item = PyTuple_GET_ITEM(iter_obj, pos);
        Py_INCREF(next_item);
    } else if (PyList_CheckExact(iter_obj)) {
        Py_ssize_t pos = *ppos;
        if (unlikely(pos >= PyList_GET_SIZE(iter_obj))) return 0;
        *ppos = pos + 1;
        next_item = PyList_GET_ITEM(iter_obj, pos);
        Py_INCREF(next_item);
    } else
#endif
    {
        next_item = PyIter_Next(iter_obj);
        if (unlikely(!next_item)) {
            return __Pyx_IterFinish();
        }
    }
    if (pitem) {
        *pitem = next_item;
    } else if (pkey && pvalue) {
        if (__Pyx_unpack_tuple2(next_item, pkey, pvalue, source_is_dict, source_is_dict, 1))
            return -1;
    } else if (pkey) {
        *pkey = next_item;
    } else {
        *pvalue = next_item;
    }
    return 1;
}


/////////////// set_iter.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_set_iterator(PyObject* iterable, int is_set,
                                                  Py_ssize_t* p_orig_length, int* p_source_is_set); /*proto*/
static CYTHON_INLINE int __Pyx_set_iter_next(
        PyObject* iter_obj, Py_ssize_t orig_length,
        Py_ssize_t* ppos, PyObject **value,
        int source_is_set); /*proto*/

/////////////// set_iter ///////////////
//@requires: ObjectHandling.c::IterFinish

static CYTHON_INLINE PyObject* __Pyx_set_iterator(PyObject* iterable, int is_set,
                                                  Py_ssize_t* p_orig_length, int* p_source_is_set) {
#if CYTHON_COMPILING_IN_CPYTHON
    is_set = is_set || likely(PySet_CheckExact(iterable) || PyFrozenSet_CheckExact(iterable));
    *p_source_is_set = is_set;
    if (likely(is_set)) {
        *p_orig_length = PySet_Size(iterable);
        Py_INCREF(iterable);
        return iterable;
    }
#else
    (void)is_set;
    *p_source_is_set = 0;
#endif
    *p_orig_length = 0;
    return PyObject_GetIter(iterable);
}

static CYTHON_INLINE int __Pyx_set_iter_next(
        PyObject* iter_obj, Py_ssize_t orig_length,
        Py_ssize_t* ppos, PyObject **value,
        int source_is_set) {
    if (!CYTHON_COMPILING_IN_CPYTHON || unlikely(!source_is_set)) {
        *value = PyIter_Next(iter_obj);
        if (unlikely(!*value)) {
            return __Pyx_IterFinish();
        }
        (void)orig_length;
        (void)ppos;
        return 1;
    }
#if CYTHON_COMPILING_IN_CPYTHON
    if (unlikely(PySet_GET_SIZE(iter_obj) != orig_length)) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "set changed size during iteration");
        return -1;
    }
    {
        Py_hash_t hash;
        int ret = _PySet_NextEntry(iter_obj, ppos, value, &hash);
        // CPython does not raise errors here, only if !isinstance(iter_obj, set/frozenset)
        assert (ret != -1);
        if (likely(ret)) {
            Py_INCREF(*value);
            return 1;
        }
    }
#endif
    return 0;
}

/////////////// py_set_discard_unhashable ///////////////
//@requires: Builtins.c::pyfrozenset_new

static int __Pyx_PySet_DiscardUnhashable(PyObject *set, PyObject *key) {
    PyObject *tmpkey;
    int rv;

    if (likely(!PySet_Check(key) || !PyErr_ExceptionMatches(PyExc_TypeError)))
        return -1;
    PyErr_Clear();
    tmpkey = __Pyx_PyFrozenSet_New(key);
    if (tmpkey == NULL)
        return -1;
    rv = PySet_Discard(set, tmpkey);
    Py_DECREF(tmpkey);
    return rv;
}


/////////////// py_set_discard.proto ///////////////

static CYTHON_INLINE int __Pyx_PySet_Discard(PyObject *set, PyObject *key); /*proto*/

/////////////// py_set_discard ///////////////
//@requires: py_set_discard_unhashable

static CYTHON_INLINE int __Pyx_PySet_Discard(PyObject *set, PyObject *key) {
    int found = PySet_Discard(set, key);
    // Convert *key* to frozenset if necessary
    if (unlikely(found < 0)) {
        found = __Pyx_PySet_DiscardUnhashable(set, key);
    }
    // note: returns -1 on error, 0 (not found) or 1 (found) otherwise => error check for -1 or < 0 works
    return found;
}


/////////////// py_set_remove.proto ///////////////

static CYTHON_INLINE int __Pyx_PySet_Remove(PyObject *set, PyObject *key); /*proto*/

/////////////// py_set_remove ///////////////
//@requires: py_set_discard_unhashable

static int __Pyx_PySet_RemoveNotFound(PyObject *set, PyObject *key, int found) {
    // Convert *key* to frozenset if necessary
    if (unlikely(found < 0)) {
        found = __Pyx_PySet_DiscardUnhashable(set, key);
    }
    if (likely(found == 0)) {
        // Not found
        PyObject *tup;
        tup = PyTuple_Pack(1, key);
        if (!tup)
            return -1;
        PyErr_SetObject(PyExc_KeyError, tup);
        Py_DECREF(tup);
        return -1;
    }
    // note: returns -1 on error, 0 (not found) or 1 (found) otherwise => error check for -1 or < 0 works
    return found;
}

static CYTHON_INLINE int __Pyx_PySet_Remove(PyObject *set, PyObject *key) {
    int found = PySet_Discard(set, key);
    if (unlikely(found != 1)) {
        // note: returns -1 on error, 0 (not found) or 1 (found) otherwise => error check for -1 or < 0 works
        return __Pyx_PySet_RemoveNotFound(set, key, found);
    }
    return 0;
}


/////////////// unicode_iter.proto ///////////////

static CYTHON_INLINE int __Pyx_init_unicode_iteration(
    PyObject* ustring, Py_ssize_t *length, void** data, int *kind); /* proto */

/////////////// unicode_iter ///////////////

static CYTHON_INLINE int __Pyx_init_unicode_iteration(
    PyObject* ustring, Py_ssize_t *length, void** data, int *kind) {
#if CYTHON_PEP393_ENABLED
    if (unlikely(__Pyx_PyUnicode_READY(ustring) < 0)) return -1;
    *kind   = PyUnicode_KIND(ustring);
    *length = PyUnicode_GET_LENGTH(ustring);
    *data   = PyUnicode_DATA(ustring);
#else
    *kind   = 0;
    *length = PyUnicode_GET_SIZE(ustring);
    *data   = (void*)PyUnicode_AS_UNICODE(ustring);
#endif
    return 0;
}

/////////////// pyobject_as_double.proto ///////////////

static double __Pyx__PyObject_AsDouble(PyObject* obj); /* proto */

#if CYTHON_COMPILING_IN_PYPY
#define __Pyx_PyObject_AsDouble(obj) \
(likely(PyFloat_CheckExact(obj)) ? PyFloat_AS_DOUBLE(obj) : \
 likely(PyInt_CheckExact(obj)) ? \
 PyFloat_AsDouble(obj) : __Pyx__PyObject_AsDouble(obj))
#else
#define __Pyx_PyObject_AsDouble(obj) \
((likely(PyFloat_CheckExact(obj))) ? \
 PyFloat_AS_DOUBLE(obj) : __Pyx__PyObject_AsDouble(obj))
#endif

/////////////// pyobject_as_double ///////////////

static double __Pyx__PyObject_AsDouble(PyObject* obj) {
    PyObject* float_value;
#if !CYTHON_USE_TYPE_SLOTS
    float_value = PyNumber_Float(obj);  if ((0)) goto bad;
#else
    PyNumberMethods *nb = Py_TYPE(obj)->tp_as_number;
    if (likely(nb) && likely(nb->nb_float)) {
        float_value = nb->nb_float(obj);
        if (likely(float_value) && unlikely(!PyFloat_Check(float_value))) {
            PyErr_Format(PyExc_TypeError,
                "__float__ returned non-float (type %.200s)",
                Py_TYPE(float_value)->tp_name);
            Py_DECREF(float_value);
            goto bad;
        }
    } else if (PyUnicode_CheckExact(obj) || PyBytes_CheckExact(obj)) {
#if PY_MAJOR_VERSION >= 3
        float_value = PyFloat_FromString(obj);
#else
        float_value = PyFloat_FromString(obj, 0);
#endif
    } else {
        PyObject* args = PyTuple_New(1);
        if (unlikely(!args)) goto bad;
        PyTuple_SET_ITEM(args, 0, obj);
        float_value = PyObject_Call((PyObject*)&PyFloat_Type, args, 0);
        PyTuple_SET_ITEM(args, 0, 0);
        Py_DECREF(args);
    }
#endif
    if (likely(float_value)) {
        double value = PyFloat_AS_DOUBLE(float_value);
        Py_DECREF(float_value);
        return value;
    }
bad:
    return (double)-1;
}


/////////////// PyNumberPow2.proto ///////////////

#define __Pyx_PyNumber_InPlacePowerOf2(a, b, c) __Pyx__PyNumber_PowerOf2(a, b, c, 1)
#define __Pyx_PyNumber_PowerOf2(a, b, c) __Pyx__PyNumber_PowerOf2(a, b, c, 0)

static PyObject* __Pyx__PyNumber_PowerOf2(PyObject *two, PyObject *exp, PyObject *none, int inplace); /*proto*/

/////////////// PyNumberPow2 ///////////////

static PyObject* __Pyx__PyNumber_PowerOf2(PyObject *two, PyObject *exp, PyObject *none, int inplace) {
// in CPython, 1<<N is substantially faster than 2**N
// see http://bugs.python.org/issue21420
#if !CYTHON_COMPILING_IN_PYPY
    Py_ssize_t shiftby;
#if PY_MAJOR_VERSION < 3
    if (likely(PyInt_CheckExact(exp))) {
        shiftby = PyInt_AS_LONG(exp);
    } else
#endif
    if (likely(PyLong_CheckExact(exp))) {
        #if CYTHON_USE_PYLONG_INTERNALS
        const Py_ssize_t size = Py_SIZE(exp);
        // tuned to optimise branch prediction
        if (likely(size == 1)) {
            shiftby = ((PyLongObject*)exp)->ob_digit[0];
        } else if (size == 0) {
            return PyInt_FromLong(1L);
        } else if (unlikely(size < 0)) {
            goto fallback;
        } else {
            shiftby = PyLong_AsSsize_t(exp);
        }
        #else
        shiftby = PyLong_AsSsize_t(exp);
        #endif
    } else {
        goto fallback;
    }
    if (likely(shiftby >= 0)) {
        if ((size_t)shiftby <= sizeof(long) * 8 - 2) {
            long value = 1L << shiftby;
            return PyInt_FromLong(value);
#ifdef HAVE_LONG_LONG
        } else if ((size_t)shiftby <= sizeof(unsigned PY_LONG_LONG) * 8 - 1) {
            unsigned PY_LONG_LONG value = ((unsigned PY_LONG_LONG)1) << shiftby;
            return PyLong_FromUnsignedLongLong(value);
#endif
        } else {
            PyObject *result, *one = PyInt_FromLong(1L);
            if (unlikely(!one)) return NULL;
            result = PyNumber_Lshift(one, exp);
            Py_DECREF(one);
            return result;
        }
    } else if (shiftby == -1 && PyErr_Occurred()) {
        PyErr_Clear();
    }
fallback:
#endif
    return (inplace ? PyNumber_InPlacePower : PyNumber_Power)(two, exp, none);
}


/////////////// PyIntCompare.proto ///////////////

{{py: c_ret_type = 'PyObject*' if ret_type.is_pyobject else 'int'}}
static CYTHON_INLINE {{c_ret_type}} __Pyx_PyInt_{{'' if ret_type.is_pyobject else 'Bool'}}{{op}}{{order}}(PyObject *op1, PyObject *op2, long intval, long inplace); /*proto*/

/////////////// PyIntCompare ///////////////

{{py: pyval, ival = ('op2', 'b') if order == 'CObj' else ('op1', 'a') }}
{{py: c_ret_type = 'PyObject*' if ret_type.is_pyobject else 'int'}}
{{py: return_true = 'Py_RETURN_TRUE' if ret_type.is_pyobject else 'return 1'}}
{{py: return_false = 'Py_RETURN_FALSE' if ret_type.is_pyobject else 'return 0'}}
{{py: slot_name = op.lower() }}
{{py: c_op = {'Eq': '==', 'Ne': '!='}[op] }}
{{py:
return_compare = (
    (lambda a,b,c_op, return_true=return_true, return_false=return_false: "if ({a} {c_op} {b}) {return_true}; else {return_false};".format(
        a=a, b=b, c_op=c_op, return_true=return_true, return_false=return_false))
    if ret_type.is_pyobject else
    (lambda a,b,c_op: "return ({a} {c_op} {b});".format(a=a, b=b, c_op=c_op))
    )
}}

static CYTHON_INLINE {{c_ret_type}} __Pyx_PyInt_{{'' if ret_type.is_pyobject else 'Bool'}}{{op}}{{order}}(PyObject *op1, PyObject *op2, CYTHON_UNUSED long intval, CYTHON_UNUSED long inplace) {
    if (op1 == op2) {
        {{return_true if op == 'Eq' else return_false}};
    }

    #if PY_MAJOR_VERSION < 3
    if (likely(PyInt_CheckExact({{pyval}}))) {
        const long {{'a' if order == 'CObj' else 'b'}} = intval;
        long {{ival}} = PyInt_AS_LONG({{pyval}});
        {{return_compare('a', 'b', c_op)}}
    }
    #endif

    #if CYTHON_USE_PYLONG_INTERNALS
    if (likely(PyLong_CheckExact({{pyval}}))) {
        int unequal;
        unsigned long uintval;
        Py_ssize_t size = Py_SIZE({{pyval}});
        const digit* digits = ((PyLongObject*){{pyval}})->ob_digit;
        if (intval == 0) {
            // == 0  =>  Py_SIZE(pyval) == 0
            {{return_compare('size', '0', c_op)}}
        } else if (intval < 0) {
            // < 0  =>  Py_SIZE(pyval) < 0
            if (size >= 0)
                {{return_false if op == 'Eq' else return_true}};
            // both are negative => can use absolute values now.
            intval = -intval;
            size = -size;
        } else {
            // > 0  =>  Py_SIZE(pyval) > 0
            if (size <= 0)
                {{return_false if op == 'Eq' else return_true}};
        }
        // After checking that the sign is the same (and excluding 0), now compare the absolute values.
        // When inlining, the C compiler should select exactly one line from this unrolled loop.
        uintval = (unsigned long) intval;
        {{for _size in range(4, 0, -1)}}
#if PyLong_SHIFT * {{_size}} < SIZEOF_LONG*8
        if (uintval >> (PyLong_SHIFT * {{_size}})) {
            // The C integer value is between (PyLong_BASE ** _size) and MIN(PyLong_BASE ** _size, LONG_MAX).
            unequal = (size != {{_size+1}}) || (digits[0] != (uintval & (unsigned long) PyLong_MASK))
                {{for _i in range(1, _size+1)}} | (digits[{{_i}}] != ((uintval >> ({{_i}} * PyLong_SHIFT)) & (unsigned long) PyLong_MASK)){{endfor}};
        } else
#endif
        {{endfor}}
            unequal = (size != 1) || (((unsigned long) digits[0]) != (uintval & (unsigned long) PyLong_MASK));

        {{return_compare('unequal', '0', c_op)}}
    }
    #endif

    if (PyFloat_CheckExact({{pyval}})) {
        const long {{'a' if order == 'CObj' else 'b'}} = intval;
        double {{ival}} = PyFloat_AS_DOUBLE({{pyval}});
        {{return_compare('(double)a', '(double)b', c_op)}}
    }

    return {{'' if ret_type.is_pyobject else '__Pyx_PyObject_IsTrueAndDecref'}}(
        PyObject_RichCompare(op1, op2, Py_{{op.upper()}}));
}


/////////////// PyIntBinop.proto ///////////////

{{py: c_ret_type = 'PyObject*' if ret_type.is_pyobject else 'int'}}
#if !CYTHON_COMPILING_IN_PYPY
static {{c_ret_type}} __Pyx_PyInt_{{'' if ret_type.is_pyobject else 'Bool'}}{{op}}{{order}}(PyObject *op1, PyObject *op2, long intval, int inplace, int zerodivision_check); /*proto*/
#else
#define __Pyx_PyInt_{{'' if ret_type.is_pyobject else 'Bool'}}{{op}}{{order}}(op1, op2, intval, inplace, zerodivision_check) \
    {{if op in ('Eq', 'Ne')}}{{'' if ret_type.is_pyobject else '__Pyx_PyObject_IsTrueAndDecref'}}(PyObject_RichCompare(op1, op2, Py_{{op.upper()}}))
    {{else}}(inplace ? PyNumber_InPlace{{op}}(op1, op2) : PyNumber_{{op}}(op1, op2))
    {{endif}}
#endif

/////////////// PyIntBinop ///////////////

#if !CYTHON_COMPILING_IN_PYPY
{{py: from Cython.Utility import pylong_join }}
{{py: pyval, ival = ('op2', 'b') if order == 'CObj' else ('op1', 'a') }}
{{py: c_ret_type = 'PyObject*' if ret_type.is_pyobject else 'int'}}
{{py: return_true = 'Py_RETURN_TRUE' if ret_type.is_pyobject else 'return 1'}}
{{py: return_false = 'Py_RETURN_FALSE' if ret_type.is_pyobject else 'return 0'}}
{{py: slot_name = {'TrueDivide': 'true_divide', 'FloorDivide': 'floor_divide'}.get(op, op.lower()) }}
{{py: cfunc_name = '__Pyx_PyInt_%s%s%s' % ('' if ret_type.is_pyobject else 'Bool', op, order)}}
{{py: zerodiv_check = lambda operand, _cfunc_name=cfunc_name: '%s_ZeroDivisionError(%s)' % (_cfunc_name, operand)}}
{{py:
c_op = {
    'Add': '+', 'Subtract': '-', 'Remainder': '%', 'TrueDivide': '/', 'FloorDivide': '/',
    'Or': '|', 'Xor': '^', 'And': '&', 'Rshift': '>>', 'Lshift': '<<',
    'Eq': '==', 'Ne': '!=',
    }[op]
}}

{{if op in ('TrueDivide', 'FloorDivide', 'Remainder')}}
#if PY_MAJOR_VERSION < 3 || CYTHON_USE_PYLONG_INTERNALS
#define {{zerodiv_check('operand')}} \
    if (unlikely(zerodivision_check && ((operand) == 0))) { \
        PyErr_SetString(PyExc_ZeroDivisionError, "integer division{{if op == 'Remainder'}} or modulo{{endif}} by zero"); \
        return NULL; \
    }
#endif
{{endif}}

static {{c_ret_type}} {{cfunc_name}}(PyObject *op1, PyObject *op2, CYTHON_UNUSED long intval, int inplace, int zerodivision_check) {
    // Prevent "unused" warnings.
    (void)inplace;
    (void)zerodivision_check;

    {{if op in ('Eq', 'Ne')}}
    if (op1 == op2) {
        {{return_true if op == 'Eq' else return_false}};
    }
    {{endif}}

    #if PY_MAJOR_VERSION < 3
    if (likely(PyInt_CheckExact({{pyval}}))) {
        const long {{'a' if order == 'CObj' else 'b'}} = intval;
        {{if c_op in '+-%' or op == 'FloorDivide'}}
        long x;
        {{endif}}
        long {{ival}} = PyInt_AS_LONG({{pyval}});

        {{if op in ('Eq', 'Ne')}}
        if (a {{c_op}} b) {
            {{return_true}};
        } else {
            {{return_false}};
        }
        {{elif c_op in '+-'}}
            // adapted from intobject.c in Py2.7:
            // casts in the line below avoid undefined behaviour on overflow
            x = (long)((unsigned long)a {{c_op}} b);
            if (likely((x^a) >= 0 || (x^{{ '~' if op == 'Subtract' else '' }}b) >= 0))
                return PyInt_FromLong(x);
            return PyLong_Type.tp_as_number->nb_{{slot_name}}(op1, op2);
        {{elif c_op == '%'}}
            {{zerodiv_check('b')}}
            // see ExprNodes.py :: mod_int_utility_code
            x = a % b;
            x += ((x != 0) & ((x ^ b) < 0)) * b;
            return PyInt_FromLong(x);
        {{elif op == 'TrueDivide'}}
            {{zerodiv_check('b')}}
            if (8 * sizeof(long) <= 53 || likely(labs({{ival}}) <= ((PY_LONG_LONG)1 << 53))) {
                return PyFloat_FromDouble((double)a / (double)b);
            }
            // let Python do the rounding
            return PyInt_Type.tp_as_number->nb_{{slot_name}}(op1, op2);
        {{elif op == 'FloorDivide'}}
            // INT_MIN / -1  is the only case that overflows, b == 0 is an error case
            {{zerodiv_check('b')}}
            if (unlikely(b == -1 && ((unsigned long)a) == 0-(unsigned long)a))
                return PyInt_Type.tp_as_number->nb_{{slot_name}}(op1, op2);
            else {
                long q, r;
                // see ExprNodes.py :: div_int_utility_code
                q = a / b;
                r = a - q*b;
                q -= ((r != 0) & ((r ^ b) < 0));
                x = q;
            }
            return PyInt_FromLong(x);
        {{elif op == 'Lshift'}}
            if (likely(b < (long) (sizeof(long)*8) && a == (a << b) >> b) || !a) {
                return PyInt_FromLong(a {{c_op}} b);
            }
        {{else}}
            // other operations are safe, no overflow
            return PyInt_FromLong(a {{c_op}} b);
        {{endif}}
    }
    #endif

    #if CYTHON_USE_PYLONG_INTERNALS
    if (likely(PyLong_CheckExact({{pyval}}))) {
        const long {{'a' if order == 'CObj' else 'b'}} = intval;
        long {{ival}}{{if op not in ('Eq', 'Ne')}}, x{{endif}};
        {{if op not in ('Eq', 'Ne', 'TrueDivide')}}
#ifdef HAVE_LONG_LONG
        const PY_LONG_LONG ll{{'a' if order == 'CObj' else 'b'}} = intval;
        PY_LONG_LONG ll{{ival}}, llx;
#endif
        {{endif}}
        const digit* digits = ((PyLongObject*){{pyval}})->ob_digit;
        const Py_ssize_t size = Py_SIZE({{pyval}});
        // handle most common case first to avoid indirect branch and optimise branch prediction
        if (likely(__Pyx_sst_abs(size) <= 1)) {
            {{ival}} = likely(size) ? digits[0] : 0;
            if (size == -1) {{ival}} = -{{ival}};
        } else {
            switch (size) {
                {{for _size in range(2, 5)}}
                {{for _case in (-_size, _size)}}
                case {{_case}}:
                    if (8 * sizeof(long) - 1 > {{_size}} * PyLong_SHIFT{{if op == 'TrueDivide'}} && {{_size-1}} * PyLong_SHIFT < 53{{endif}}) {
                        {{ival}} = {{'-' if _case < 0 else ''}}(long) {{pylong_join(_size, 'digits')}};
                        break;
                    {{if op not in ('Eq', 'Ne', 'TrueDivide')}}
#ifdef HAVE_LONG_LONG
                    } else if (8 * sizeof(PY_LONG_LONG) - 1 > {{_size}} * PyLong_SHIFT) {
                        ll{{ival}} = {{'-' if _case < 0 else ''}}(PY_LONG_LONG) {{pylong_join(_size, 'digits', 'unsigned PY_LONG_LONG')}};
                        goto long_long;
#endif
                    {{endif}}
                    }
                    // if size doesn't fit into a long or PY_LONG_LONG anymore, fall through to default
                    CYTHON_FALLTHROUGH;
                {{endfor}}
                {{endfor}}

                {{if op in ('Eq', 'Ne')}}
                #if PyLong_SHIFT < 30 && PyLong_SHIFT != 15
                // unusual setup - your fault
                default: return {{'' if ret_type.is_pyobject else '__Pyx_PyObject_IsTrueAndDecref'}}(
                    PyLong_Type.tp_richcompare({{'op1, op2' if order == 'ObjC' else 'op2, op1'}}, Py_{{op.upper()}}));
                #else
                // too large for the long values we allow => definitely not equal
                default: {{return_false if op == 'Eq' else return_true}};
                #endif
                {{else}}
                default: return PyLong_Type.tp_as_number->nb_{{slot_name}}(op1, op2);
                {{endif}}
            }
        }
        {{if op in ('Eq', 'Ne')}}
            if (a {{c_op}} b) {
                {{return_true}};
            } else {
                {{return_false}};
            }
        {{else}}
            {{if c_op == '%'}}
                {{zerodiv_check('b')}}
                // see ExprNodes.py :: mod_int_utility_code
                x = a % b;
                x += ((x != 0) & ((x ^ b) < 0)) * b;
            {{elif op == 'TrueDivide'}}
                {{zerodiv_check('b')}}
                if ((8 * sizeof(long) <= 53 || likely(labs({{ival}}) <= ((PY_LONG_LONG)1 << 53)))
                        || __Pyx_sst_abs(size) <= 52 / PyLong_SHIFT) {
                    return PyFloat_FromDouble((double)a / (double)b);
                }
                return PyLong_Type.tp_as_number->nb_{{slot_name}}(op1, op2);
            {{elif op == 'FloorDivide'}}
                {{zerodiv_check('b')}}
                {
                    long q, r;
                    // see ExprNodes.py :: div_int_utility_code
                    q = a / b;
                    r = a - q*b;
                    q -= ((r != 0) & ((r ^ b) < 0));
                    x = q;
                }
            {{else}}
                x = a {{c_op}} b;
                {{if op == 'Lshift'}}
#ifdef HAVE_LONG_LONG
                if (unlikely(!(b < (long) (sizeof(long)*8) && a == x >> b)) && a) {
                    ll{{ival}} = {{ival}};
                    goto long_long;
                }
#else
                if (likely(b < (long) (sizeof(long)*8) && a == x >> b) || !a) /* execute return statement below */
#endif
                {{endif}}
            {{endif}}
            return PyLong_FromLong(x);

        {{if op != 'TrueDivide'}}
#ifdef HAVE_LONG_LONG
        long_long:
            {{if c_op == '%'}}
                // see ExprNodes.py :: mod_int_utility_code
                llx = lla % llb;
                llx += ((llx != 0) & ((llx ^ llb) < 0)) * llb;
            {{elif op == 'FloorDivide'}}
                {
                    PY_LONG_LONG q, r;
                    // see ExprNodes.py :: div_int_utility_code
                    q = lla / llb;
                    r = lla - q*llb;
                    q -= ((r != 0) & ((r ^ llb) < 0));
                    llx = q;
                }
            {{else}}
                llx = lla {{c_op}} llb;
                {{if op == 'Lshift'}}
                if (likely(lla == llx >> llb)) /* then execute 'return' below */
                {{endif}}
            {{endif}}
            return PyLong_FromLongLong(llx);
#endif
        {{endif}}{{# if op != 'TrueDivide' #}}
        {{endif}}{{# if op in ('Eq', 'Ne') #}}
    }
    #endif

    {{if c_op in '+-' or op in ('TrueDivide', 'Eq', 'Ne')}}
    if (PyFloat_CheckExact({{pyval}})) {
        const long {{'a' if order == 'CObj' else 'b'}} = intval;
        double {{ival}} = PyFloat_AS_DOUBLE({{pyval}});
        {{if op in ('Eq', 'Ne')}}
            if ((double)a {{c_op}} (double)b) {
                {{return_true}};
            } else {
                {{return_false}};
            }
        {{else}}
            double result;
            {{if op == 'TrueDivide'}}
            if (unlikely(zerodivision_check && b == 0)) {
                PyErr_SetString(PyExc_ZeroDivisionError, "float division by zero");
                return NULL;
            }
            {{endif}}
            // copied from floatobject.c in Py3.5:
            PyFPE_START_PROTECT("{{op.lower() if not op.endswith('Divide') else 'divide'}}", return NULL)
            result = ((double)a) {{c_op}} (double)b;
            PyFPE_END_PROTECT(result)
            return PyFloat_FromDouble(result);
        {{endif}}
    }
    {{endif}}

    {{if op in ('Eq', 'Ne')}}
    return {{'' if ret_type.is_pyobject else '__Pyx_PyObject_IsTrueAndDecref'}}(
        PyObject_RichCompare(op1, op2, Py_{{op.upper()}}));
    {{else}}
    return (inplace ? PyNumber_InPlace{{op}} : PyNumber_{{op}})(op1, op2);
    {{endif}}
}
#endif

/////////////// PyFloatBinop.proto ///////////////

{{py: c_ret_type = 'PyObject*' if ret_type.is_pyobject else 'int'}}
#if !CYTHON_COMPILING_IN_PYPY
static {{c_ret_type}} __Pyx_PyFloat_{{'' if ret_type.is_pyobject else 'Bool'}}{{op}}{{order}}(PyObject *op1, PyObject *op2, double floatval, int inplace, int zerodivision_check); /*proto*/
#else
#define __Pyx_PyFloat_{{'' if ret_type.is_pyobject else 'Bool'}}{{op}}{{order}}(op1, op2, floatval, inplace, zerodivision_check) \
    {{if op in ('Eq', 'Ne')}}{{'' if ret_type.is_pyobject else '__Pyx_PyObject_IsTrueAndDecref'}}(PyObject_RichCompare(op1, op2, Py_{{op.upper()}}))
    {{elif op == 'Divide'}}((inplace ? __Pyx_PyNumber_InPlaceDivide(op1, op2) : __Pyx_PyNumber_Divide(op1, op2)))
    {{else}}(inplace ? PyNumber_InPlace{{op}}(op1, op2) : PyNumber_{{op}}(op1, op2))
    {{endif}}
#endif

/////////////// PyFloatBinop ///////////////

#if !CYTHON_COMPILING_IN_PYPY
{{py: from Cython.Utility import pylong_join }}
{{py: c_ret_type = 'PyObject*' if ret_type.is_pyobject else 'int'}}
{{py: return_true = 'Py_RETURN_TRUE' if ret_type.is_pyobject else 'return 1'}}
{{py: return_false = 'Py_RETURN_FALSE' if ret_type.is_pyobject else 'return 0'}}
{{py: pyval, fval = ('op2', 'b') if order == 'CObj' else ('op1', 'a') }}
{{py: cfunc_name = '__Pyx_PyFloat_%s%s%s' % ('' if ret_type.is_pyobject else 'Bool', op, order) }}
{{py: zerodiv_check = lambda operand, _cfunc_name=cfunc_name: '%s_ZeroDivisionError(%s)' % (_cfunc_name, operand)}}
{{py:
c_op = {
    'Add': '+', 'Subtract': '-', 'TrueDivide': '/', 'Divide': '/', 'Remainder': '%',
    'Eq': '==', 'Ne': '!=',
    }[op]
}}

{{if order == 'CObj' and c_op in '%/'}}
#define {{zerodiv_check('operand')}} if (unlikely(zerodivision_check && ((operand) == 0))) { \
    PyErr_SetString(PyExc_ZeroDivisionError, "float division{{if op == 'Remainder'}} or modulo{{endif}} by zero"); \
    return NULL; \
}
{{endif}}

static {{c_ret_type}} {{cfunc_name}}(PyObject *op1, PyObject *op2, double floatval, int inplace, int zerodivision_check) {
    const double {{'a' if order == 'CObj' else 'b'}} = floatval;
    double {{fval}}{{if op not in ('Eq', 'Ne')}}, result{{endif}};
    // Prevent "unused" warnings.
    (void)inplace;
    (void)zerodivision_check;

    {{if op in ('Eq', 'Ne')}}
    if (op1 == op2) {
        {{return_true if op == 'Eq' else return_false}};
    }
    {{endif}}

    if (likely(PyFloat_CheckExact({{pyval}}))) {
        {{fval}} = PyFloat_AS_DOUBLE({{pyval}});
        {{if order == 'CObj' and c_op in '%/'}}{{zerodiv_check(fval)}}{{endif}}
    } else

    #if PY_MAJOR_VERSION < 3
    if (likely(PyInt_CheckExact({{pyval}}))) {
        {{fval}} = (double) PyInt_AS_LONG({{pyval}});
        {{if order == 'CObj' and c_op in '%/'}}{{zerodiv_check(fval)}}{{endif}}
    } else
    #endif

    if (likely(PyLong_CheckExact({{pyval}}))) {
        #if CYTHON_USE_PYLONG_INTERNALS
        const digit* digits = ((PyLongObject*){{pyval}})->ob_digit;
        const Py_ssize_t size = Py_SIZE({{pyval}});
        switch (size) {
            case  0: {{if order == 'CObj' and c_op in '%/'}}{{zerodiv_check('0')}}{{else}}{{fval}} = 0.0;{{endif}} break;
            case -1: {{fval}} = -(double) digits[0]; break;
            case  1: {{fval}} = (double) digits[0]; break;
            {{for _size in (2, 3, 4)}}
            case -{{_size}}:
            case {{_size}}:
                if (8 * sizeof(unsigned long) > {{_size}} * PyLong_SHIFT && ((8 * sizeof(unsigned long) < 53) || ({{_size-1}} * PyLong_SHIFT < 53))) {
                    {{fval}} = (double) {{pylong_join(_size, 'digits')}};
                    // let CPython do its own float rounding from 2**53 on (max. consecutive integer in double float)
                    if ((8 * sizeof(unsigned long) < 53) || ({{_size}} * PyLong_SHIFT < 53) || ({{fval}} < (double) ((PY_LONG_LONG)1 << 53))) {
                        if (size == {{-_size}})
                            {{fval}} = -{{fval}};
                        break;
                    }
                }
                // Fall through if size doesn't fit safely into a double anymore.
                // It may not be obvious that this is a safe fall-through given the "fval < 2**53"
                // check above.  However, the number of digits that CPython uses for a given PyLong
                // value is minimal, and together with the "(size-1) * SHIFT < 53" check above,
                // this should make it safe.
                CYTHON_FALLTHROUGH;
            {{endfor}}
            default:
        #else
        {
        #endif
        {{if op in ('Eq', 'Ne')}}
            return {{'' if ret_type.is_pyobject else '__Pyx_PyObject_IsTrueAndDecref'}}(
                PyFloat_Type.tp_richcompare({{'op1, op2' if order == 'CObj' else 'op2, op1'}}, Py_{{op.upper()}}));
        {{else}}
            {{fval}} = PyLong_AsDouble({{pyval}});
            if (unlikely({{fval}} == -1.0 && PyErr_Occurred())) return NULL;
            {{if order == 'CObj' and c_op in '%/'}}{{zerodiv_check(fval)}}{{endif}}
        {{endif}}
        }
    } else {
        {{if op in ('Eq', 'Ne')}}
        return {{'' if ret_type.is_pyobject else '__Pyx_PyObject_IsTrueAndDecref'}}(
            PyObject_RichCompare(op1, op2, Py_{{op.upper()}}));
        {{elif op == 'Divide'}}
        return (inplace ? __Pyx_PyNumber_InPlaceDivide(op1, op2) : __Pyx_PyNumber_Divide(op1, op2));
        {{else}}
        return (inplace ? PyNumber_InPlace{{op}} : PyNumber_{{op}})(op1, op2);
        {{endif}}
    }

    {{if op in ('Eq', 'Ne')}}
        if (a {{c_op}} b) {
            {{return_true}};
        } else {
            {{return_false}};
        }
    {{else}}
        // copied from floatobject.c in Py3.5:
        {{if order == 'CObj' and c_op in '%/'}}{{zerodiv_check('b')}}{{endif}}
        PyFPE_START_PROTECT("{{op.lower() if not op.endswith('Divide') else 'divide'}}", return NULL)
        {{if c_op == '%'}}
        result = fmod(a, b);
        if (result)
            result += ((result < 0) ^ (b < 0)) * b;
        else
            result = copysign(0.0, b);
        {{else}}
        result = a {{c_op}} b;
        {{endif}}
        PyFPE_END_PROTECT(result)
        return PyFloat_FromDouble(result);
    {{endif}}
}
#endif
