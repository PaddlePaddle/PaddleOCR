/*
 * Special implementations of built-in functions and methods.
 *
 * Optional optimisations for builtins are in Optimize.c.
 *
 * General object operations and protocols are in ObjectHandling.c.
 */

//////////////////// Globals.proto ////////////////////

static PyObject* __Pyx_Globals(void); /*proto*/

//////////////////// Globals ////////////////////
//@substitute: naming
//@requires: ObjectHandling.c::GetAttr

// This is a stub implementation until we have something more complete.
// Currently, we only handle the most common case of a read-only dict
// of Python names.  Supporting cdef names in the module and write
// access requires a rewrite as a dedicated class.

static PyObject* __Pyx_Globals(void) {
    Py_ssize_t i;
    PyObject *names;
    PyObject *globals = $moddict_cname;
    Py_INCREF(globals);
    names = PyObject_Dir($module_cname);
    if (!names)
        goto bad;
    for (i = PyList_GET_SIZE(names)-1; i >= 0; i--) {
#if CYTHON_COMPILING_IN_PYPY
        PyObject* name = PySequence_ITEM(names, i);
        if (!name)
            goto bad;
#else
        PyObject* name = PyList_GET_ITEM(names, i);
#endif
        if (!PyDict_Contains(globals, name)) {
            PyObject* value = __Pyx_GetAttr($module_cname, name);
            if (!value) {
#if CYTHON_COMPILING_IN_PYPY
                Py_DECREF(name);
#endif
                goto bad;
            }
            if (PyDict_SetItem(globals, name, value) < 0) {
#if CYTHON_COMPILING_IN_PYPY
                Py_DECREF(name);
#endif
                Py_DECREF(value);
                goto bad;
            }
        }
#if CYTHON_COMPILING_IN_PYPY
        Py_DECREF(name);
#endif
    }
    Py_DECREF(names);
    return globals;
bad:
    Py_XDECREF(names);
    Py_XDECREF(globals);
    return NULL;
}

//////////////////// PyExecGlobals.proto ////////////////////

static PyObject* __Pyx_PyExecGlobals(PyObject*);

//////////////////// PyExecGlobals ////////////////////
//@requires: Globals
//@requires: PyExec

static PyObject* __Pyx_PyExecGlobals(PyObject* code) {
    PyObject* result;
    PyObject* globals = __Pyx_Globals();
    if (unlikely(!globals))
        return NULL;
    result = __Pyx_PyExec2(code, globals);
    Py_DECREF(globals);
    return result;
}

//////////////////// PyExec.proto ////////////////////

static PyObject* __Pyx_PyExec3(PyObject*, PyObject*, PyObject*);
static CYTHON_INLINE PyObject* __Pyx_PyExec2(PyObject*, PyObject*);

//////////////////// PyExec ////////////////////
//@substitute: naming

static CYTHON_INLINE PyObject* __Pyx_PyExec2(PyObject* o, PyObject* globals) {
    return __Pyx_PyExec3(o, globals, NULL);
}

static PyObject* __Pyx_PyExec3(PyObject* o, PyObject* globals, PyObject* locals) {
    PyObject* result;
    PyObject* s = 0;
    char *code = 0;

    if (!globals || globals == Py_None) {
        globals = $moddict_cname;
    } else if (!PyDict_Check(globals)) {
        PyErr_Format(PyExc_TypeError, "exec() arg 2 must be a dict, not %.200s",
                     Py_TYPE(globals)->tp_name);
        goto bad;
    }
    if (!locals || locals == Py_None) {
        locals = globals;
    }

    if (__Pyx_PyDict_GetItemStr(globals, PYIDENT("__builtins__")) == NULL) {
        if (PyDict_SetItem(globals, PYIDENT("__builtins__"), PyEval_GetBuiltins()) < 0)
            goto bad;
    }

    if (PyCode_Check(o)) {
        if (__Pyx_PyCode_HasFreeVars((PyCodeObject *)o)) {
            PyErr_SetString(PyExc_TypeError,
                "code object passed to exec() may not contain free variables");
            goto bad;
        }
        #if CYTHON_COMPILING_IN_PYPY || PY_VERSION_HEX < 0x030200B1
        result = PyEval_EvalCode((PyCodeObject *)o, globals, locals);
        #else
        result = PyEval_EvalCode(o, globals, locals);
        #endif
    } else {
        PyCompilerFlags cf;
        cf.cf_flags = 0;
#if PY_VERSION_HEX >= 0x030800A3
        cf.cf_feature_version = PY_MINOR_VERSION;
#endif
        if (PyUnicode_Check(o)) {
            cf.cf_flags = PyCF_SOURCE_IS_UTF8;
            s = PyUnicode_AsUTF8String(o);
            if (!s) goto bad;
            o = s;
        #if PY_MAJOR_VERSION >= 3
        } else if (!PyBytes_Check(o)) {
        #else
        } else if (!PyString_Check(o)) {
        #endif
            PyErr_Format(PyExc_TypeError,
                "exec: arg 1 must be string, bytes or code object, got %.200s",
                Py_TYPE(o)->tp_name);
            goto bad;
        }
        #if PY_MAJOR_VERSION >= 3
        code = PyBytes_AS_STRING(o);
        #else
        code = PyString_AS_STRING(o);
        #endif
        if (PyEval_MergeCompilerFlags(&cf)) {
            result = PyRun_StringFlags(code, Py_file_input, globals, locals, &cf);
        } else {
            result = PyRun_String(code, Py_file_input, globals, locals);
        }
        Py_XDECREF(s);
    }

    return result;
bad:
    Py_XDECREF(s);
    return 0;
}

//////////////////// GetAttr3.proto ////////////////////

static CYTHON_INLINE PyObject *__Pyx_GetAttr3(PyObject *, PyObject *, PyObject *); /*proto*/

//////////////////// GetAttr3 ////////////////////
//@requires: ObjectHandling.c::GetAttr
//@requires: Exceptions.c::PyThreadStateGet
//@requires: Exceptions.c::PyErrFetchRestore
//@requires: Exceptions.c::PyErrExceptionMatches

static PyObject *__Pyx_GetAttr3Default(PyObject *d) {
    __Pyx_PyThreadState_declare
    __Pyx_PyThreadState_assign
    if (unlikely(!__Pyx_PyErr_ExceptionMatches(PyExc_AttributeError)))
        return NULL;
    __Pyx_PyErr_Clear();
    Py_INCREF(d);
    return d;
}

static CYTHON_INLINE PyObject *__Pyx_GetAttr3(PyObject *o, PyObject *n, PyObject *d) {
    PyObject *r = __Pyx_GetAttr(o, n);
    return (likely(r)) ? r : __Pyx_GetAttr3Default(d);
}

//////////////////// HasAttr.proto ////////////////////

static CYTHON_INLINE int __Pyx_HasAttr(PyObject *, PyObject *); /*proto*/

//////////////////// HasAttr ////////////////////
//@requires: ObjectHandling.c::GetAttr

static CYTHON_INLINE int __Pyx_HasAttr(PyObject *o, PyObject *n) {
    PyObject *r;
    if (unlikely(!__Pyx_PyBaseString_Check(n))) {
        PyErr_SetString(PyExc_TypeError,
                        "hasattr(): attribute name must be string");
        return -1;
    }
    r = __Pyx_GetAttr(o, n);
    if (unlikely(!r)) {
        PyErr_Clear();
        return 0;
    } else {
        Py_DECREF(r);
        return 1;
    }
}

//////////////////// Intern.proto ////////////////////

static PyObject* __Pyx_Intern(PyObject* s); /* proto */

//////////////////// Intern ////////////////////

static PyObject* __Pyx_Intern(PyObject* s) {
    if (!(likely(PyString_CheckExact(s)))) {
        PyErr_Format(PyExc_TypeError, "Expected %.16s, got %.200s", "str", Py_TYPE(s)->tp_name);
        return 0;
    }
    Py_INCREF(s);
    #if PY_MAJOR_VERSION >= 3
    PyUnicode_InternInPlace(&s);
    #else
    PyString_InternInPlace(&s);
    #endif
    return s;
}

//////////////////// abs_longlong.proto ////////////////////

static CYTHON_INLINE PY_LONG_LONG __Pyx_abs_longlong(PY_LONG_LONG x) {
#if defined (__cplusplus) && __cplusplus >= 201103L
    return std::abs(x);
#elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    return llabs(x);
#elif defined (_MSC_VER)
    // abs() is defined for long, but 64-bits type on MSVC is long long.
    // Use MS-specific _abs64() instead, which returns the original (negative) value for abs(-MAX-1)
    return _abs64(x);
#elif defined (__GNUC__)
    // gcc or clang on 64 bit windows.
    return __builtin_llabs(x);
#else
    if (sizeof(PY_LONG_LONG) <= sizeof(Py_ssize_t))
        return __Pyx_sst_abs(x);
    return (x<0) ? -x : x;
#endif
}


//////////////////// py_abs.proto ////////////////////

#if CYTHON_USE_PYLONG_INTERNALS
static PyObject *__Pyx_PyLong_AbsNeg(PyObject *num);/*proto*/

#define __Pyx_PyNumber_Absolute(x) \
    ((likely(PyLong_CheckExact(x))) ? \
         (likely(Py_SIZE(x) >= 0) ? (Py_INCREF(x), (x)) : __Pyx_PyLong_AbsNeg(x)) : \
         PyNumber_Absolute(x))

#else
#define __Pyx_PyNumber_Absolute(x)  PyNumber_Absolute(x)
#endif

//////////////////// py_abs ////////////////////

#if CYTHON_USE_PYLONG_INTERNALS
static PyObject *__Pyx_PyLong_AbsNeg(PyObject *n) {
    if (likely(Py_SIZE(n) == -1)) {
        // digits are unsigned
        return PyLong_FromLong(((PyLongObject*)n)->ob_digit[0]);
    }
#if CYTHON_COMPILING_IN_CPYTHON
    {
        PyObject *copy = _PyLong_Copy((PyLongObject*)n);
        if (likely(copy)) {
            // negate the size to swap the sign
            __Pyx_SET_SIZE(copy, -Py_SIZE(copy));
        }
        return copy;
    }
#else
    return PyNumber_Negative(n);
#endif
}
#endif


//////////////////// pow2.proto ////////////////////

#define __Pyx_PyNumber_Power2(a, b) PyNumber_Power(a, b, Py_None)


//////////////////// object_ord.proto ////////////////////
//@requires: TypeConversion.c::UnicodeAsUCS4

#if PY_MAJOR_VERSION >= 3
#define __Pyx_PyObject_Ord(c) \
    (likely(PyUnicode_Check(c)) ? (long)__Pyx_PyUnicode_AsPy_UCS4(c) : __Pyx__PyObject_Ord(c))
#else
#define __Pyx_PyObject_Ord(c) __Pyx__PyObject_Ord(c)
#endif
static long __Pyx__PyObject_Ord(PyObject* c); /*proto*/

//////////////////// object_ord ////////////////////

static long __Pyx__PyObject_Ord(PyObject* c) {
    Py_ssize_t size;
    if (PyBytes_Check(c)) {
        size = PyBytes_GET_SIZE(c);
        if (likely(size == 1)) {
            return (unsigned char) PyBytes_AS_STRING(c)[0];
        }
#if PY_MAJOR_VERSION < 3
    } else if (PyUnicode_Check(c)) {
        return (long)__Pyx_PyUnicode_AsPy_UCS4(c);
#endif
#if (!CYTHON_COMPILING_IN_PYPY) || (defined(PyByteArray_AS_STRING) && defined(PyByteArray_GET_SIZE))
    } else if (PyByteArray_Check(c)) {
        size = PyByteArray_GET_SIZE(c);
        if (likely(size == 1)) {
            return (unsigned char) PyByteArray_AS_STRING(c)[0];
        }
#endif
    } else {
        // FIXME: support character buffers - but CPython doesn't support them either
        PyErr_Format(PyExc_TypeError,
            "ord() expected string of length 1, but %.200s found", Py_TYPE(c)->tp_name);
        return (long)(Py_UCS4)-1;
    }
    PyErr_Format(PyExc_TypeError,
        "ord() expected a character, but string of length %zd found", size);
    return (long)(Py_UCS4)-1;
}


//////////////////// py_dict_keys.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_Keys(PyObject* d); /*proto*/

//////////////////// py_dict_keys ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_Keys(PyObject* d) {
    if (PY_MAJOR_VERSION >= 3)
        return CALL_UNBOUND_METHOD(PyDict_Type, "keys", d);
    else
        return PyDict_Keys(d);
}

//////////////////// py_dict_values.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_Values(PyObject* d); /*proto*/

//////////////////// py_dict_values ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_Values(PyObject* d) {
    if (PY_MAJOR_VERSION >= 3)
        return CALL_UNBOUND_METHOD(PyDict_Type, "values", d);
    else
        return PyDict_Values(d);
}

//////////////////// py_dict_items.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_Items(PyObject* d); /*proto*/

//////////////////// py_dict_items ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_Items(PyObject* d) {
    if (PY_MAJOR_VERSION >= 3)
        return CALL_UNBOUND_METHOD(PyDict_Type, "items", d);
    else
        return PyDict_Items(d);
}

//////////////////// py_dict_iterkeys.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_IterKeys(PyObject* d); /*proto*/

//////////////////// py_dict_iterkeys ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_IterKeys(PyObject* d) {
    if (PY_MAJOR_VERSION >= 3)
        return CALL_UNBOUND_METHOD(PyDict_Type, "keys", d);
    else
        return CALL_UNBOUND_METHOD(PyDict_Type, "iterkeys", d);
}

//////////////////// py_dict_itervalues.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_IterValues(PyObject* d); /*proto*/

//////////////////// py_dict_itervalues ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_IterValues(PyObject* d) {
    if (PY_MAJOR_VERSION >= 3)
        return CALL_UNBOUND_METHOD(PyDict_Type, "values", d);
    else
        return CALL_UNBOUND_METHOD(PyDict_Type, "itervalues", d);
}

//////////////////// py_dict_iteritems.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_IterItems(PyObject* d); /*proto*/

//////////////////// py_dict_iteritems ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_IterItems(PyObject* d) {
    if (PY_MAJOR_VERSION >= 3)
        return CALL_UNBOUND_METHOD(PyDict_Type, "items", d);
    else
        return CALL_UNBOUND_METHOD(PyDict_Type, "iteritems", d);
}

//////////////////// py_dict_viewkeys.proto ////////////////////

#if PY_VERSION_HEX < 0x02070000
#error This module uses dict views, which require Python 2.7 or later
#endif
static CYTHON_INLINE PyObject* __Pyx_PyDict_ViewKeys(PyObject* d); /*proto*/

//////////////////// py_dict_viewkeys ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_ViewKeys(PyObject* d) {
    if (PY_MAJOR_VERSION >= 3)
        return CALL_UNBOUND_METHOD(PyDict_Type, "keys", d);
    else
        return CALL_UNBOUND_METHOD(PyDict_Type, "viewkeys", d);
}

//////////////////// py_dict_viewvalues.proto ////////////////////

#if PY_VERSION_HEX < 0x02070000
#error This module uses dict views, which require Python 2.7 or later
#endif
static CYTHON_INLINE PyObject* __Pyx_PyDict_ViewValues(PyObject* d); /*proto*/

//////////////////// py_dict_viewvalues ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_ViewValues(PyObject* d) {
    if (PY_MAJOR_VERSION >= 3)
        return CALL_UNBOUND_METHOD(PyDict_Type, "values", d);
    else
        return CALL_UNBOUND_METHOD(PyDict_Type, "viewvalues", d);
}

//////////////////// py_dict_viewitems.proto ////////////////////

#if PY_VERSION_HEX < 0x02070000
#error This module uses dict views, which require Python 2.7 or later
#endif
static CYTHON_INLINE PyObject* __Pyx_PyDict_ViewItems(PyObject* d); /*proto*/

//////////////////// py_dict_viewitems ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyDict_ViewItems(PyObject* d) {
    if (PY_MAJOR_VERSION >= 3)
        return CALL_UNBOUND_METHOD(PyDict_Type, "items", d);
    else
        return CALL_UNBOUND_METHOD(PyDict_Type, "viewitems", d);
}


//////////////////// pyfrozenset_new.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyFrozenSet_New(PyObject* it);

//////////////////// pyfrozenset_new ////////////////////
//@substitute: naming

static CYTHON_INLINE PyObject* __Pyx_PyFrozenSet_New(PyObject* it) {
    if (it) {
        PyObject* result;
#if CYTHON_COMPILING_IN_PYPY
        // PyPy currently lacks PyFrozenSet_CheckExact() and PyFrozenSet_New()
        PyObject* args;
        args = PyTuple_Pack(1, it);
        if (unlikely(!args))
            return NULL;
        result = PyObject_Call((PyObject*)&PyFrozenSet_Type, args, NULL);
        Py_DECREF(args);
        return result;
#else
        if (PyFrozenSet_CheckExact(it)) {
            Py_INCREF(it);
            return it;
        }
        result = PyFrozenSet_New(it);
        if (unlikely(!result))
            return NULL;
        if ((PY_VERSION_HEX >= 0x031000A1) || likely(PySet_GET_SIZE(result)))
            return result;
        // empty frozenset is a singleton (on Python <3.10)
        // seems wasteful, but CPython does the same
        Py_DECREF(result);
#endif
    }
#if CYTHON_USE_TYPE_SLOTS
    return PyFrozenSet_Type.tp_new(&PyFrozenSet_Type, $empty_tuple, NULL);
#else
    return PyObject_Call((PyObject*)&PyFrozenSet_Type, $empty_tuple, NULL);
#endif
}


//////////////////// PySet_Update.proto ////////////////////

static CYTHON_INLINE int __Pyx_PySet_Update(PyObject* set, PyObject* it); /*proto*/

//////////////////// PySet_Update ////////////////////

static CYTHON_INLINE int __Pyx_PySet_Update(PyObject* set, PyObject* it) {
    PyObject *retval;
    #if CYTHON_USE_TYPE_SLOTS && !CYTHON_COMPILING_IN_PYPY
    if (PyAnySet_Check(it)) {
        if (PySet_GET_SIZE(it) == 0)
            return 0;
        // fast and safe case: CPython will update our result set and return it
        retval = PySet_Type.tp_as_number->nb_inplace_or(set, it);
        if (likely(retval == set)) {
            Py_DECREF(retval);
            return 0;
        }
        if (unlikely(!retval))
            return -1;
        // unusual result, fall through to set.update() call below
        Py_DECREF(retval);
    }
    #endif
    retval = CALL_UNBOUND_METHOD(PySet_Type, "update", set, it);
    if (unlikely(!retval)) return -1;
    Py_DECREF(retval);
    return 0;
}
