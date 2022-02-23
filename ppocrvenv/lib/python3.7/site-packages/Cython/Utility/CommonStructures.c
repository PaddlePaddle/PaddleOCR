/////////////// FetchCommonType.proto ///////////////

static PyTypeObject* __Pyx_FetchCommonType(PyTypeObject* type);

/////////////// FetchCommonType ///////////////

static PyTypeObject* __Pyx_FetchCommonType(PyTypeObject* type) {
    PyObject* fake_module;
    PyTypeObject* cached_type = NULL;

    fake_module = PyImport_AddModule((char*) "_cython_" CYTHON_ABI);
    if (!fake_module) return NULL;
    Py_INCREF(fake_module);

    cached_type = (PyTypeObject*) PyObject_GetAttrString(fake_module, type->tp_name);
    if (cached_type) {
        if (!PyType_Check((PyObject*)cached_type)) {
            PyErr_Format(PyExc_TypeError,
                "Shared Cython type %.200s is not a type object",
                type->tp_name);
            goto bad;
        }
        if (cached_type->tp_basicsize != type->tp_basicsize) {
            PyErr_Format(PyExc_TypeError,
                "Shared Cython type %.200s has the wrong size, try recompiling",
                type->tp_name);
            goto bad;
        }
    } else {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError)) goto bad;
        PyErr_Clear();
        if (PyType_Ready(type) < 0) goto bad;
        if (PyObject_SetAttrString(fake_module, type->tp_name, (PyObject*) type) < 0)
            goto bad;
        Py_INCREF(type);
        cached_type = type;
    }

done:
    Py_DECREF(fake_module);
    // NOTE: always returns owned reference, or NULL on error
    return cached_type;

bad:
    Py_XDECREF(cached_type);
    cached_type = NULL;
    goto done;
}


/////////////// FetchCommonPointer.proto ///////////////

static void* __Pyx_FetchCommonPointer(void* pointer, const char* name);

/////////////// FetchCommonPointer ///////////////


static void* __Pyx_FetchCommonPointer(void* pointer, const char* name) {
#if PY_VERSION_HEX >= 0x02070000
    PyObject* fake_module = NULL;
    PyObject* capsule = NULL;
    void* value = NULL;

    fake_module = PyImport_AddModule((char*) "_cython_" CYTHON_ABI);
    if (!fake_module) return NULL;
    Py_INCREF(fake_module);

    capsule = PyObject_GetAttrString(fake_module, name);
    if (!capsule) {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError)) goto bad;
        PyErr_Clear();
        capsule = PyCapsule_New(pointer, name, NULL);
        if (!capsule) goto bad;
        if (PyObject_SetAttrString(fake_module, name, capsule) < 0)
            goto bad;
    }
    value = PyCapsule_GetPointer(capsule, name);

bad:
    Py_XDECREF(capsule);
    Py_DECREF(fake_module);
    return value;
#else
    return pointer;
#endif
}
