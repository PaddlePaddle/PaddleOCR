/////////////// PyIdentifierFromString.proto ///////////////

#if !defined(__Pyx_PyIdentifier_FromString)
#if PY_MAJOR_VERSION < 3
  #define __Pyx_PyIdentifier_FromString(s) PyString_FromString(s)
#else
  #define __Pyx_PyIdentifier_FromString(s) PyUnicode_FromString(s)
#endif
#endif


/////////////// Import.proto ///////////////

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level); /*proto*/

/////////////// Import ///////////////
//@requires: ObjectHandling.c::PyObjectGetAttrStr
//@substitute: naming

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, int level) {
    PyObject *empty_list = 0;
    PyObject *module = 0;
    PyObject *global_dict = 0;
    PyObject *empty_dict = 0;
    PyObject *list;
    #if PY_MAJOR_VERSION < 3
    PyObject *py_import;
    py_import = __Pyx_PyObject_GetAttrStr($builtins_cname, PYIDENT("__import__"));
    if (!py_import)
        goto bad;
    #endif
    if (from_list)
        list = from_list;
    else {
        empty_list = PyList_New(0);
        if (!empty_list)
            goto bad;
        list = empty_list;
    }
    global_dict = PyModule_GetDict($module_cname);
    if (!global_dict)
        goto bad;
    empty_dict = PyDict_New();
    if (!empty_dict)
        goto bad;
    {
        #if PY_MAJOR_VERSION >= 3
        if (level == -1) {
            // Avoid C compiler warning if strchr() evaluates to false at compile time.
            if ((1) && (strchr(__Pyx_MODULE_NAME, '.'))) {
                /* try package relative import first */
                module = PyImport_ImportModuleLevelObject(
                    name, global_dict, empty_dict, list, 1);
                if (!module) {
                    if (!PyErr_ExceptionMatches(PyExc_ImportError))
                        goto bad;
                    PyErr_Clear();
                }
            }
            level = 0; /* try absolute import on failure */
        }
        #endif
        if (!module) {
            #if PY_MAJOR_VERSION < 3
            PyObject *py_level = PyInt_FromLong(level);
            if (!py_level)
                goto bad;
            module = PyObject_CallFunctionObjArgs(py_import,
                name, global_dict, empty_dict, list, py_level, (PyObject *)NULL);
            Py_DECREF(py_level);
            #else
            module = PyImport_ImportModuleLevelObject(
                name, global_dict, empty_dict, list, level);
            #endif
        }
    }
bad:
    #if PY_MAJOR_VERSION < 3
    Py_XDECREF(py_import);
    #endif
    Py_XDECREF(empty_list);
    Py_XDECREF(empty_dict);
    return module;
}


/////////////// ImportFrom.proto ///////////////

static PyObject* __Pyx_ImportFrom(PyObject* module, PyObject* name); /*proto*/

/////////////// ImportFrom ///////////////
//@requires: ObjectHandling.c::PyObjectGetAttrStr

static PyObject* __Pyx_ImportFrom(PyObject* module, PyObject* name) {
    PyObject* value = __Pyx_PyObject_GetAttrStr(module, name);
    if (unlikely(!value) && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Format(PyExc_ImportError,
        #if PY_MAJOR_VERSION < 3
            "cannot import name %.230s", PyString_AS_STRING(name));
        #else
            "cannot import name %S", name);
        #endif
    }
    return value;
}


/////////////// ImportStar ///////////////
//@substitute: naming

/* import_all_from is an unexposed function from ceval.c */

static int
__Pyx_import_all_from(PyObject *locals, PyObject *v)
{
    PyObject *all = PyObject_GetAttrString(v, "__all__");
    PyObject *dict, *name, *value;
    int skip_leading_underscores = 0;
    int pos, err;

    if (all == NULL) {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError))
            return -1; /* Unexpected error */
        PyErr_Clear();
        dict = PyObject_GetAttrString(v, "__dict__");
        if (dict == NULL) {
            if (!PyErr_ExceptionMatches(PyExc_AttributeError))
                return -1;
            PyErr_SetString(PyExc_ImportError,
            "from-import-* object has no __dict__ and no __all__");
            return -1;
        }
#if PY_MAJOR_VERSION < 3
        all = PyObject_CallMethod(dict, (char *)"keys", NULL);
#else
        all = PyMapping_Keys(dict);
#endif
        Py_DECREF(dict);
        if (all == NULL)
            return -1;
        skip_leading_underscores = 1;
    }

    for (pos = 0, err = 0; ; pos++) {
        name = PySequence_GetItem(all, pos);
        if (name == NULL) {
            if (!PyErr_ExceptionMatches(PyExc_IndexError))
                err = -1;
            else
                PyErr_Clear();
            break;
        }
        if (skip_leading_underscores &&
#if PY_MAJOR_VERSION < 3
            likely(PyString_Check(name)) &&
            PyString_AS_STRING(name)[0] == '_')
#else
            likely(PyUnicode_Check(name)) &&
            likely(__Pyx_PyUnicode_GET_LENGTH(name)) &&
            __Pyx_PyUnicode_READ_CHAR(name, 0) == '_')
#endif
        {
            Py_DECREF(name);
            continue;
        }
        value = PyObject_GetAttr(v, name);
        if (value == NULL)
            err = -1;
        else if (PyDict_CheckExact(locals))
            err = PyDict_SetItem(locals, name, value);
        else
            err = PyObject_SetItem(locals, name, value);
        Py_DECREF(name);
        Py_XDECREF(value);
        if (err != 0)
            break;
    }
    Py_DECREF(all);
    return err;
}


static int ${import_star}(PyObject* m) {

    int i;
    int ret = -1;
    char* s;
    PyObject *locals = 0;
    PyObject *list = 0;
#if PY_MAJOR_VERSION >= 3
    PyObject *utf8_name = 0;
#endif
    PyObject *name;
    PyObject *item;

    locals = PyDict_New();              if (!locals) goto bad;
    if (__Pyx_import_all_from(locals, m) < 0) goto bad;
    list = PyDict_Items(locals);        if (!list) goto bad;

    for(i=0; i<PyList_GET_SIZE(list); i++) {
        name = PyTuple_GET_ITEM(PyList_GET_ITEM(list, i), 0);
        item = PyTuple_GET_ITEM(PyList_GET_ITEM(list, i), 1);
#if PY_MAJOR_VERSION >= 3
        utf8_name = PyUnicode_AsUTF8String(name);
        if (!utf8_name) goto bad;
        s = PyBytes_AS_STRING(utf8_name);
        if (${import_star_set}(item, name, s) < 0) goto bad;
        Py_DECREF(utf8_name); utf8_name = 0;
#else
        s = PyString_AsString(name);
        if (!s) goto bad;
        if (${import_star_set}(item, name, s) < 0) goto bad;
#endif
    }
    ret = 0;

bad:
    Py_XDECREF(locals);
    Py_XDECREF(list);
#if PY_MAJOR_VERSION >= 3
    Py_XDECREF(utf8_name);
#endif
    return ret;
}


/////////////// SetPackagePathFromImportLib.proto ///////////////

// PY_VERSION_HEX >= 0x03030000
#if PY_MAJOR_VERSION >= 3 && !CYTHON_PEP489_MULTI_PHASE_INIT
static int __Pyx_SetPackagePathFromImportLib(const char* parent_package_name, PyObject *module_name);
#else
#define __Pyx_SetPackagePathFromImportLib(a, b) 0
#endif

/////////////// SetPackagePathFromImportLib ///////////////
//@requires: ObjectHandling.c::PyObjectGetAttrStr
//@substitute: naming

// PY_VERSION_HEX >= 0x03030000
#if PY_MAJOR_VERSION >= 3 && !CYTHON_PEP489_MULTI_PHASE_INIT
static int __Pyx_SetPackagePathFromImportLib(const char* parent_package_name, PyObject *module_name) {
    PyObject *importlib, *loader, *osmod, *ossep, *parts, *package_path;
    PyObject *path = NULL, *file_path = NULL;
    int result;
    if (parent_package_name) {
        PyObject *package = PyImport_ImportModule(parent_package_name);
        if (unlikely(!package))
            goto bad;
        path = PyObject_GetAttrString(package, "__path__");
        Py_DECREF(package);
        if (unlikely(!path) || unlikely(path == Py_None))
            goto bad;
    } else {
        path = Py_None; Py_INCREF(Py_None);
    }
    // package_path = [importlib.find_loader(module_name, path).path.rsplit(os.sep, 1)[0]]
    importlib = PyImport_ImportModule("importlib");
    if (unlikely(!importlib))
        goto bad;
    loader = PyObject_CallMethod(importlib, "find_loader", "(OO)", module_name, path);
    Py_DECREF(importlib);
    Py_DECREF(path); path = NULL;
    if (unlikely(!loader))
        goto bad;
    file_path = PyObject_GetAttrString(loader, "path");
    Py_DECREF(loader);
    if (unlikely(!file_path))
        goto bad;

    if (unlikely(PyObject_SetAttrString($module_cname, "__file__", file_path) < 0))
        goto bad;

    osmod = PyImport_ImportModule("os");
    if (unlikely(!osmod))
        goto bad;
    ossep = PyObject_GetAttrString(osmod, "sep");
    Py_DECREF(osmod);
    if (unlikely(!ossep))
        goto bad;
    parts = PyObject_CallMethod(file_path, "rsplit", "(Oi)", ossep, 1);
    Py_DECREF(file_path); file_path = NULL;
    Py_DECREF(ossep);
    if (unlikely(!parts))
        goto bad;
    package_path = Py_BuildValue("[O]", PyList_GET_ITEM(parts, 0));
    Py_DECREF(parts);
    if (unlikely(!package_path))
        goto bad;
    goto set_path;

bad:
    PyErr_WriteUnraisable(module_name);
    Py_XDECREF(path);
    Py_XDECREF(file_path);

    // set an empty path list on failure
    PyErr_Clear();
    package_path = PyList_New(0);
    if (unlikely(!package_path))
        return -1;

set_path:
    result = PyObject_SetAttrString($module_cname, "__path__", package_path);
    Py_DECREF(package_path);
    return result;
}
#endif


/////////////// TypeImport.proto ///////////////

#ifndef __PYX_HAVE_RT_ImportType_proto
#define __PYX_HAVE_RT_ImportType_proto

enum __Pyx_ImportType_CheckSize {
   __Pyx_ImportType_CheckSize_Error = 0,
   __Pyx_ImportType_CheckSize_Warn = 1,
   __Pyx_ImportType_CheckSize_Ignore = 2
};

static PyTypeObject *__Pyx_ImportType(PyObject* module, const char *module_name, const char *class_name, size_t size, enum __Pyx_ImportType_CheckSize check_size);  /*proto*/

#endif

/////////////// TypeImport ///////////////

#ifndef __PYX_HAVE_RT_ImportType
#define __PYX_HAVE_RT_ImportType
static PyTypeObject *__Pyx_ImportType(PyObject *module, const char *module_name, const char *class_name,
    size_t size, enum __Pyx_ImportType_CheckSize check_size)
{
    PyObject *result = 0;
    char warning[200];
    Py_ssize_t basicsize;
#ifdef Py_LIMITED_API
    PyObject *py_basicsize;
#endif

    result = PyObject_GetAttrString(module, class_name);
    if (!result)
        goto bad;
    if (!PyType_Check(result)) {
        PyErr_Format(PyExc_TypeError,
            "%.200s.%.200s is not a type object",
            module_name, class_name);
        goto bad;
    }
#ifndef Py_LIMITED_API
    basicsize = ((PyTypeObject *)result)->tp_basicsize;
#else
    py_basicsize = PyObject_GetAttrString(result, "__basicsize__");
    if (!py_basicsize)
        goto bad;
    basicsize = PyLong_AsSsize_t(py_basicsize);
    Py_DECREF(py_basicsize);
    py_basicsize = 0;
    if (basicsize == (Py_ssize_t)-1 && PyErr_Occurred())
        goto bad;
#endif
    if ((size_t)basicsize < size) {
        PyErr_Format(PyExc_ValueError,
            "%.200s.%.200s size changed, may indicate binary incompatibility. "
            "Expected %zd from C header, got %zd from PyObject",
            module_name, class_name, size, basicsize);
        goto bad;
    }
    if (check_size == __Pyx_ImportType_CheckSize_Error && (size_t)basicsize != size) {
        PyErr_Format(PyExc_ValueError,
            "%.200s.%.200s size changed, may indicate binary incompatibility. "
            "Expected %zd from C header, got %zd from PyObject",
            module_name, class_name, size, basicsize);
        goto bad;
    }
    else if (check_size == __Pyx_ImportType_CheckSize_Warn && (size_t)basicsize > size) {
        PyOS_snprintf(warning, sizeof(warning),
            "%s.%s size changed, may indicate binary incompatibility. "
            "Expected %zd from C header, got %zd from PyObject",
            module_name, class_name, size, basicsize);
        if (PyErr_WarnEx(NULL, warning, 0) < 0) goto bad;
    }
    /* check_size == __Pyx_ImportType_CheckSize_Ignore does not warn nor error */
    return (PyTypeObject *)result;
bad:
    Py_XDECREF(result);
    return NULL;
}
#endif

/////////////// FunctionImport.proto ///////////////

static int __Pyx_ImportFunction(PyObject *module, const char *funcname, void (**f)(void), const char *sig); /*proto*/

/////////////// FunctionImport ///////////////
//@substitute: naming

#ifndef __PYX_HAVE_RT_ImportFunction
#define __PYX_HAVE_RT_ImportFunction
static int __Pyx_ImportFunction(PyObject *module, const char *funcname, void (**f)(void), const char *sig) {
    PyObject *d = 0;
    PyObject *cobj = 0;
    union {
        void (*fp)(void);
        void *p;
    } tmp;

    d = PyObject_GetAttrString(module, (char *)"$api_name");
    if (!d)
        goto bad;
    cobj = PyDict_GetItemString(d, funcname);
    if (!cobj) {
        PyErr_Format(PyExc_ImportError,
            "%.200s does not export expected C function %.200s",
                PyModule_GetName(module), funcname);
        goto bad;
    }
#if PY_VERSION_HEX >= 0x02070000
    if (!PyCapsule_IsValid(cobj, sig)) {
        PyErr_Format(PyExc_TypeError,
            "C function %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), funcname, sig, PyCapsule_GetName(cobj));
        goto bad;
    }
    tmp.p = PyCapsule_GetPointer(cobj, sig);
#else
    {const char *desc, *s1, *s2;
    desc = (const char *)PyCObject_GetDesc(cobj);
    if (!desc)
        goto bad;
    s1 = desc; s2 = sig;
    while (*s1 != '\0' && *s1 == *s2) { s1++; s2++; }
    if (*s1 != *s2) {
        PyErr_Format(PyExc_TypeError,
            "C function %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), funcname, sig, desc);
        goto bad;
    }
    tmp.p = PyCObject_AsVoidPtr(cobj);}
#endif
    *f = tmp.fp;
    if (!(*f))
        goto bad;
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(d);
    return -1;
}
#endif

/////////////// FunctionExport.proto ///////////////

static int __Pyx_ExportFunction(const char *name, void (*f)(void), const char *sig); /*proto*/

/////////////// FunctionExport ///////////////
//@substitute: naming

static int __Pyx_ExportFunction(const char *name, void (*f)(void), const char *sig) {
    PyObject *d = 0;
    PyObject *cobj = 0;
    union {
        void (*fp)(void);
        void *p;
    } tmp;

    d = PyObject_GetAttrString($module_cname, (char *)"$api_name");
    if (!d) {
        PyErr_Clear();
        d = PyDict_New();
        if (!d)
            goto bad;
        Py_INCREF(d);
        if (PyModule_AddObject($module_cname, (char *)"$api_name", d) < 0)
            goto bad;
    }
    tmp.fp = f;
#if PY_VERSION_HEX >= 0x02070000
    cobj = PyCapsule_New(tmp.p, sig, 0);
#else
    cobj = PyCObject_FromVoidPtrAndDesc(tmp.p, (void *)sig, 0);
#endif
    if (!cobj)
        goto bad;
    if (PyDict_SetItemString(d, name, cobj) < 0)
        goto bad;
    Py_DECREF(cobj);
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(cobj);
    Py_XDECREF(d);
    return -1;
}

/////////////// VoidPtrImport.proto ///////////////

static int __Pyx_ImportVoidPtr(PyObject *module, const char *name, void **p, const char *sig); /*proto*/

/////////////// VoidPtrImport ///////////////
//@substitute: naming

#ifndef __PYX_HAVE_RT_ImportVoidPtr
#define __PYX_HAVE_RT_ImportVoidPtr
static int __Pyx_ImportVoidPtr(PyObject *module, const char *name, void **p, const char *sig) {
    PyObject *d = 0;
    PyObject *cobj = 0;

    d = PyObject_GetAttrString(module, (char *)"$api_name");
    if (!d)
        goto bad;
    cobj = PyDict_GetItemString(d, name);
    if (!cobj) {
        PyErr_Format(PyExc_ImportError,
            "%.200s does not export expected C variable %.200s",
                PyModule_GetName(module), name);
        goto bad;
    }
#if PY_VERSION_HEX >= 0x02070000
    if (!PyCapsule_IsValid(cobj, sig)) {
        PyErr_Format(PyExc_TypeError,
            "C variable %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), name, sig, PyCapsule_GetName(cobj));
        goto bad;
    }
    *p = PyCapsule_GetPointer(cobj, sig);
#else
    {const char *desc, *s1, *s2;
    desc = (const char *)PyCObject_GetDesc(cobj);
    if (!desc)
        goto bad;
    s1 = desc; s2 = sig;
    while (*s1 != '\0' && *s1 == *s2) { s1++; s2++; }
    if (*s1 != *s2) {
        PyErr_Format(PyExc_TypeError,
            "C variable %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), name, sig, desc);
        goto bad;
    }
    *p = PyCObject_AsVoidPtr(cobj);}
#endif
    if (!(*p))
        goto bad;
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(d);
    return -1;
}
#endif

/////////////// VoidPtrExport.proto ///////////////

static int __Pyx_ExportVoidPtr(PyObject *name, void *p, const char *sig); /*proto*/

/////////////// VoidPtrExport ///////////////
//@substitute: naming
//@requires: ObjectHandling.c::PyObjectSetAttrStr

static int __Pyx_ExportVoidPtr(PyObject *name, void *p, const char *sig) {
    PyObject *d;
    PyObject *cobj = 0;

    d = PyDict_GetItem($moddict_cname, PYIDENT("$api_name"));
    Py_XINCREF(d);
    if (!d) {
        d = PyDict_New();
        if (!d)
            goto bad;
        if (__Pyx_PyObject_SetAttrStr($module_cname, PYIDENT("$api_name"), d) < 0)
            goto bad;
    }
#if PY_VERSION_HEX >= 0x02070000
    cobj = PyCapsule_New(p, sig, 0);
#else
    cobj = PyCObject_FromVoidPtrAndDesc(p, (void *)sig, 0);
#endif
    if (!cobj)
        goto bad;
    if (PyDict_SetItem(d, name, cobj) < 0)
        goto bad;
    Py_DECREF(cobj);
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(cobj);
    Py_XDECREF(d);
    return -1;
}


/////////////// SetVTable.proto ///////////////

static int __Pyx_SetVtable(PyObject *dict, void *vtable); /*proto*/

/////////////// SetVTable ///////////////

static int __Pyx_SetVtable(PyObject *dict, void *vtable) {
#if PY_VERSION_HEX >= 0x02070000
    PyObject *ob = PyCapsule_New(vtable, 0, 0);
#else
    PyObject *ob = PyCObject_FromVoidPtr(vtable, 0);
#endif
    if (!ob)
        goto bad;
    if (PyDict_SetItem(dict, PYIDENT("__pyx_vtable__"), ob) < 0)
        goto bad;
    Py_DECREF(ob);
    return 0;
bad:
    Py_XDECREF(ob);
    return -1;
}


/////////////// GetVTable.proto ///////////////

static void* __Pyx_GetVtable(PyObject *dict); /*proto*/

/////////////// GetVTable ///////////////

static void* __Pyx_GetVtable(PyObject *dict) {
    void* ptr;
    PyObject *ob = PyObject_GetItem(dict, PYIDENT("__pyx_vtable__"));
    if (!ob)
        goto bad;
#if PY_VERSION_HEX >= 0x02070000
    ptr = PyCapsule_GetPointer(ob, 0);
#else
    ptr = PyCObject_AsVoidPtr(ob);
#endif
    if (!ptr && !PyErr_Occurred())
        PyErr_SetString(PyExc_RuntimeError, "invalid vtable found for imported type");
    Py_DECREF(ob);
    return ptr;
bad:
    Py_XDECREF(ob);
    return NULL;
}


/////////////// MergeVTables.proto ///////////////
//@requires: GetVTable

static int __Pyx_MergeVtables(PyTypeObject *type); /*proto*/

/////////////// MergeVTables ///////////////

static int __Pyx_MergeVtables(PyTypeObject *type) {
    int i;
    void** base_vtables;
    void* unknown = (void*)-1;
    PyObject* bases = type->tp_bases;
    int base_depth = 0;
    {
        PyTypeObject* base = type->tp_base;
        while (base) {
            base_depth += 1;
            base = base->tp_base;
        }
    }
    base_vtables = (void**) malloc(sizeof(void*) * (size_t)(base_depth + 1));
    base_vtables[0] = unknown;
    // Could do MRO resolution of individual methods in the future, assuming
    // compatible vtables, but for now simply require a common vtable base.
    // Note that if the vtables of various bases are extended separately,
    // resolution isn't possible and we must reject it just as when the
    // instance struct is so extended.  (It would be good to also do this
    // check when a multiple-base class is created in pure Python as well.)
    for (i = 1; i < PyTuple_GET_SIZE(bases); i++) {
        void* base_vtable = __Pyx_GetVtable(((PyTypeObject*)PyTuple_GET_ITEM(bases, i))->tp_dict);
        if (base_vtable != NULL) {
            int j;
            PyTypeObject* base = type->tp_base;
            for (j = 0; j < base_depth; j++) {
                if (base_vtables[j] == unknown) {
                    base_vtables[j] = __Pyx_GetVtable(base->tp_dict);
                    base_vtables[j + 1] = unknown;
                }
                if (base_vtables[j] == base_vtable) {
                    break;
                } else if (base_vtables[j] == NULL) {
                    // No more potential matching bases (with vtables).
                    goto bad;
                }
                base = base->tp_base;
            }
        }
    }
    PyErr_Clear();
    free(base_vtables);
    return 0;
bad:
    PyErr_Format(
        PyExc_TypeError,
        "multiple bases have vtable conflict: '%s' and '%s'",
        type->tp_base->tp_name, ((PyTypeObject*)PyTuple_GET_ITEM(bases, i))->tp_name);
    free(base_vtables);
    return -1;
}


/////////////// ImportNumPyArray.proto ///////////////

static PyObject *__pyx_numpy_ndarray = NULL;

static PyObject* __Pyx_ImportNumPyArrayTypeIfAvailable(void); /*proto*/

/////////////// ImportNumPyArray.cleanup ///////////////
Py_CLEAR(__pyx_numpy_ndarray);

/////////////// ImportNumPyArray ///////////////
//@requires: ImportExport.c::Import

static PyObject* __Pyx__ImportNumPyArray(void) {
    PyObject *numpy_module, *ndarray_object = NULL;
    numpy_module = __Pyx_Import(PYIDENT("numpy"), NULL, 0);
    if (likely(numpy_module)) {
        ndarray_object = PyObject_GetAttrString(numpy_module, "ndarray");
        Py_DECREF(numpy_module);
    }
    if (unlikely(!ndarray_object)) {
        // ImportError, AttributeError, ...
        PyErr_Clear();
    }
    if (unlikely(!ndarray_object || !PyObject_TypeCheck(ndarray_object, &PyType_Type))) {
        Py_XDECREF(ndarray_object);
        Py_INCREF(Py_None);
        ndarray_object = Py_None;
    }
    return ndarray_object;
}

static CYTHON_INLINE PyObject* __Pyx_ImportNumPyArrayTypeIfAvailable(void) {
    if (unlikely(!__pyx_numpy_ndarray)) {
        __pyx_numpy_ndarray = __Pyx__ImportNumPyArray();
    }
    Py_INCREF(__pyx_numpy_ndarray);
    return __pyx_numpy_ndarray;
}
