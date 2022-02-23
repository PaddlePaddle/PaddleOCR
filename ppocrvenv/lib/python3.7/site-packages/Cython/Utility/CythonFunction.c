
//////////////////// CythonFunctionShared.proto ////////////////////

#define __Pyx_CyFunction_USED 1

#define __Pyx_CYFUNCTION_STATICMETHOD  0x01
#define __Pyx_CYFUNCTION_CLASSMETHOD   0x02
#define __Pyx_CYFUNCTION_CCLASS        0x04

#define __Pyx_CyFunction_GetClosure(f) \
    (((__pyx_CyFunctionObject *) (f))->func_closure)
#define __Pyx_CyFunction_GetClassObj(f) \
    (((__pyx_CyFunctionObject *) (f))->func_classobj)

#define __Pyx_CyFunction_Defaults(type, f) \
    ((type *)(((__pyx_CyFunctionObject *) (f))->defaults))
#define __Pyx_CyFunction_SetDefaultsGetter(f, g) \
    ((__pyx_CyFunctionObject *) (f))->defaults_getter = (g)


typedef struct {
    PyCFunctionObject func;
#if PY_VERSION_HEX < 0x030500A0
    PyObject *func_weakreflist;
#endif
    PyObject *func_dict;
    PyObject *func_name;
    PyObject *func_qualname;
    PyObject *func_doc;
    PyObject *func_globals;
    PyObject *func_code;
    PyObject *func_closure;
    // No-args super() class cell
    PyObject *func_classobj;

    // Dynamic default args and annotations
    void *defaults;
    int defaults_pyobjects;
    size_t defaults_size;  // used by FusedFunction for copying defaults
    int flags;

    // Defaults info
    PyObject *defaults_tuple;   /* Const defaults tuple */
    PyObject *defaults_kwdict;  /* Const kwonly defaults dict */
    PyObject *(*defaults_getter)(PyObject *);
    PyObject *func_annotations; /* function annotations dict */
} __pyx_CyFunctionObject;

static PyTypeObject *__pyx_CyFunctionType = 0;

#define __Pyx_CyFunction_Check(obj)  (__Pyx_TypeCheck(obj, __pyx_CyFunctionType))

static PyObject *__Pyx_CyFunction_Init(__pyx_CyFunctionObject* op, PyMethodDef *ml,
                                      int flags, PyObject* qualname,
                                      PyObject *self,
                                      PyObject *module, PyObject *globals,
                                      PyObject* code);

static CYTHON_INLINE void *__Pyx_CyFunction_InitDefaults(PyObject *m,
                                                         size_t size,
                                                         int pyobjects);
static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsTuple(PyObject *m,
                                                            PyObject *tuple);
static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsKwDict(PyObject *m,
                                                             PyObject *dict);
static CYTHON_INLINE void __Pyx_CyFunction_SetAnnotationsDict(PyObject *m,
                                                              PyObject *dict);


static int __pyx_CyFunction_init(void);


//////////////////// CythonFunctionShared ////////////////////
//@substitute: naming
//@requires: CommonStructures.c::FetchCommonType
////@requires: ObjectHandling.c::PyObjectGetAttrStr

#include <structmember.h>

static PyObject *
__Pyx_CyFunction_get_doc(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *closure)
{
    if (unlikely(op->func_doc == NULL)) {
        if (op->func.m_ml->ml_doc) {
#if PY_MAJOR_VERSION >= 3
            op->func_doc = PyUnicode_FromString(op->func.m_ml->ml_doc);
#else
            op->func_doc = PyString_FromString(op->func.m_ml->ml_doc);
#endif
            if (unlikely(op->func_doc == NULL))
                return NULL;
        } else {
            Py_INCREF(Py_None);
            return Py_None;
        }
    }
    Py_INCREF(op->func_doc);
    return op->func_doc;
}

static int
__Pyx_CyFunction_set_doc(__pyx_CyFunctionObject *op, PyObject *value, CYTHON_UNUSED void *context)
{
    PyObject *tmp = op->func_doc;
    if (value == NULL) {
        // Mark as deleted
        value = Py_None;
    }
    Py_INCREF(value);
    op->func_doc = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_name(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *context)
{
    if (unlikely(op->func_name == NULL)) {
#if PY_MAJOR_VERSION >= 3
        op->func_name = PyUnicode_InternFromString(op->func.m_ml->ml_name);
#else
        op->func_name = PyString_InternFromString(op->func.m_ml->ml_name);
#endif
        if (unlikely(op->func_name == NULL))
            return NULL;
    }
    Py_INCREF(op->func_name);
    return op->func_name;
}

static int
__Pyx_CyFunction_set_name(__pyx_CyFunctionObject *op, PyObject *value, CYTHON_UNUSED void *context)
{
    PyObject *tmp;

#if PY_MAJOR_VERSION >= 3
    if (unlikely(value == NULL || !PyUnicode_Check(value)))
#else
    if (unlikely(value == NULL || !PyString_Check(value)))
#endif
    {
        PyErr_SetString(PyExc_TypeError,
                        "__name__ must be set to a string object");
        return -1;
    }
    tmp = op->func_name;
    Py_INCREF(value);
    op->func_name = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_qualname(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *context)
{
    Py_INCREF(op->func_qualname);
    return op->func_qualname;
}

static int
__Pyx_CyFunction_set_qualname(__pyx_CyFunctionObject *op, PyObject *value, CYTHON_UNUSED void *context)
{
    PyObject *tmp;

#if PY_MAJOR_VERSION >= 3
    if (unlikely(value == NULL || !PyUnicode_Check(value)))
#else
    if (unlikely(value == NULL || !PyString_Check(value)))
#endif
    {
        PyErr_SetString(PyExc_TypeError,
                        "__qualname__ must be set to a string object");
        return -1;
    }
    tmp = op->func_qualname;
    Py_INCREF(value);
    op->func_qualname = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_self(__pyx_CyFunctionObject *m, CYTHON_UNUSED void *closure)
{
    PyObject *self;

    self = m->func_closure;
    if (self == NULL)
        self = Py_None;
    Py_INCREF(self);
    return self;
}

static PyObject *
__Pyx_CyFunction_get_dict(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *context)
{
    if (unlikely(op->func_dict == NULL)) {
        op->func_dict = PyDict_New();
        if (unlikely(op->func_dict == NULL))
            return NULL;
    }
    Py_INCREF(op->func_dict);
    return op->func_dict;
}

static int
__Pyx_CyFunction_set_dict(__pyx_CyFunctionObject *op, PyObject *value, CYTHON_UNUSED void *context)
{
    PyObject *tmp;

    if (unlikely(value == NULL)) {
        PyErr_SetString(PyExc_TypeError,
               "function's dictionary may not be deleted");
        return -1;
    }
    if (unlikely(!PyDict_Check(value))) {
        PyErr_SetString(PyExc_TypeError,
               "setting function's dictionary to a non-dict");
        return -1;
    }
    tmp = op->func_dict;
    Py_INCREF(value);
    op->func_dict = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_globals(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *context)
{
    Py_INCREF(op->func_globals);
    return op->func_globals;
}

static PyObject *
__Pyx_CyFunction_get_closure(CYTHON_UNUSED __pyx_CyFunctionObject *op, CYTHON_UNUSED void *context)
{
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
__Pyx_CyFunction_get_code(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *context)
{
    PyObject* result = (op->func_code) ? op->func_code : Py_None;
    Py_INCREF(result);
    return result;
}

static int
__Pyx_CyFunction_init_defaults(__pyx_CyFunctionObject *op) {
    int result = 0;
    PyObject *res = op->defaults_getter((PyObject *) op);
    if (unlikely(!res))
        return -1;

    // Cache result
    #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
    op->defaults_tuple = PyTuple_GET_ITEM(res, 0);
    Py_INCREF(op->defaults_tuple);
    op->defaults_kwdict = PyTuple_GET_ITEM(res, 1);
    Py_INCREF(op->defaults_kwdict);
    #else
    op->defaults_tuple = PySequence_ITEM(res, 0);
    if (unlikely(!op->defaults_tuple)) result = -1;
    else {
        op->defaults_kwdict = PySequence_ITEM(res, 1);
        if (unlikely(!op->defaults_kwdict)) result = -1;
    }
    #endif
    Py_DECREF(res);
    return result;
}

static int
__Pyx_CyFunction_set_defaults(__pyx_CyFunctionObject *op, PyObject* value, CYTHON_UNUSED void *context) {
    PyObject* tmp;
    if (!value) {
        // del => explicit None to prevent rebuilding
        value = Py_None;
    } else if (value != Py_None && !PyTuple_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "__defaults__ must be set to a tuple object");
        return -1;
    }
    Py_INCREF(value);
    tmp = op->defaults_tuple;
    op->defaults_tuple = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_defaults(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *context) {
    PyObject* result = op->defaults_tuple;
    if (unlikely(!result)) {
        if (op->defaults_getter) {
            if (__Pyx_CyFunction_init_defaults(op) < 0) return NULL;
            result = op->defaults_tuple;
        } else {
            result = Py_None;
        }
    }
    Py_INCREF(result);
    return result;
}

static int
__Pyx_CyFunction_set_kwdefaults(__pyx_CyFunctionObject *op, PyObject* value, CYTHON_UNUSED void *context) {
    PyObject* tmp;
    if (!value) {
        // del => explicit None to prevent rebuilding
        value = Py_None;
    } else if (value != Py_None && !PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "__kwdefaults__ must be set to a dict object");
        return -1;
    }
    Py_INCREF(value);
    tmp = op->defaults_kwdict;
    op->defaults_kwdict = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_kwdefaults(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *context) {
    PyObject* result = op->defaults_kwdict;
    if (unlikely(!result)) {
        if (op->defaults_getter) {
            if (__Pyx_CyFunction_init_defaults(op) < 0) return NULL;
            result = op->defaults_kwdict;
        } else {
            result = Py_None;
        }
    }
    Py_INCREF(result);
    return result;
}

static int
__Pyx_CyFunction_set_annotations(__pyx_CyFunctionObject *op, PyObject* value, CYTHON_UNUSED void *context) {
    PyObject* tmp;
    if (!value || value == Py_None) {
        value = NULL;
    } else if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "__annotations__ must be set to a dict object");
        return -1;
    }
    Py_XINCREF(value);
    tmp = op->func_annotations;
    op->func_annotations = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
__Pyx_CyFunction_get_annotations(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *context) {
    PyObject* result = op->func_annotations;
    if (unlikely(!result)) {
        result = PyDict_New();
        if (unlikely(!result)) return NULL;
        op->func_annotations = result;
    }
    Py_INCREF(result);
    return result;
}

//#if PY_VERSION_HEX >= 0x030400C1
//static PyObject *
//__Pyx_CyFunction_get_signature(__pyx_CyFunctionObject *op, CYTHON_UNUSED void *context) {
//    PyObject *inspect_module, *signature_class, *signature;
//    // from inspect import Signature
//    inspect_module = PyImport_ImportModuleLevelObject(PYIDENT("inspect"), NULL, NULL, NULL, 0);
//    if (unlikely(!inspect_module))
//        goto bad;
//    signature_class = __Pyx_PyObject_GetAttrStr(inspect_module, PYIDENT("Signature"));
//    Py_DECREF(inspect_module);
//    if (unlikely(!signature_class))
//        goto bad;
//    // return Signature.from_function(op)
//    signature = PyObject_CallMethodObjArgs(signature_class, PYIDENT("from_function"), op, NULL);
//    Py_DECREF(signature_class);
//    if (likely(signature))
//        return signature;
//bad:
//    // make sure we raise an AttributeError from this property on any errors
//    if (!PyErr_ExceptionMatches(PyExc_AttributeError))
//        PyErr_SetString(PyExc_AttributeError, "failed to calculate __signature__");
//    return NULL;
//}
//#endif

static PyGetSetDef __pyx_CyFunction_getsets[] = {
    {(char *) "func_doc", (getter)__Pyx_CyFunction_get_doc, (setter)__Pyx_CyFunction_set_doc, 0, 0},
    {(char *) "__doc__",  (getter)__Pyx_CyFunction_get_doc, (setter)__Pyx_CyFunction_set_doc, 0, 0},
    {(char *) "func_name", (getter)__Pyx_CyFunction_get_name, (setter)__Pyx_CyFunction_set_name, 0, 0},
    {(char *) "__name__", (getter)__Pyx_CyFunction_get_name, (setter)__Pyx_CyFunction_set_name, 0, 0},
    {(char *) "__qualname__", (getter)__Pyx_CyFunction_get_qualname, (setter)__Pyx_CyFunction_set_qualname, 0, 0},
    {(char *) "__self__", (getter)__Pyx_CyFunction_get_self, 0, 0, 0},
    {(char *) "func_dict", (getter)__Pyx_CyFunction_get_dict, (setter)__Pyx_CyFunction_set_dict, 0, 0},
    {(char *) "__dict__", (getter)__Pyx_CyFunction_get_dict, (setter)__Pyx_CyFunction_set_dict, 0, 0},
    {(char *) "func_globals", (getter)__Pyx_CyFunction_get_globals, 0, 0, 0},
    {(char *) "__globals__", (getter)__Pyx_CyFunction_get_globals, 0, 0, 0},
    {(char *) "func_closure", (getter)__Pyx_CyFunction_get_closure, 0, 0, 0},
    {(char *) "__closure__", (getter)__Pyx_CyFunction_get_closure, 0, 0, 0},
    {(char *) "func_code", (getter)__Pyx_CyFunction_get_code, 0, 0, 0},
    {(char *) "__code__", (getter)__Pyx_CyFunction_get_code, 0, 0, 0},
    {(char *) "func_defaults", (getter)__Pyx_CyFunction_get_defaults, (setter)__Pyx_CyFunction_set_defaults, 0, 0},
    {(char *) "__defaults__", (getter)__Pyx_CyFunction_get_defaults, (setter)__Pyx_CyFunction_set_defaults, 0, 0},
    {(char *) "__kwdefaults__", (getter)__Pyx_CyFunction_get_kwdefaults, (setter)__Pyx_CyFunction_set_kwdefaults, 0, 0},
    {(char *) "__annotations__", (getter)__Pyx_CyFunction_get_annotations, (setter)__Pyx_CyFunction_set_annotations, 0, 0},
//#if PY_VERSION_HEX >= 0x030400C1
//    {(char *) "__signature__", (getter)__Pyx_CyFunction_get_signature, 0, 0, 0},
//#endif
    {0, 0, 0, 0, 0}
};

static PyMemberDef __pyx_CyFunction_members[] = {
    {(char *) "__module__", T_OBJECT, offsetof(PyCFunctionObject, m_module), PY_WRITE_RESTRICTED, 0},
    {0, 0, 0,  0, 0}
};

static PyObject *
__Pyx_CyFunction_reduce(__pyx_CyFunctionObject *m, CYTHON_UNUSED PyObject *args)
{
#if PY_MAJOR_VERSION >= 3
    Py_INCREF(m->func_qualname);
    return m->func_qualname;
#else
    return PyString_FromString(m->func.m_ml->ml_name);
#endif
}

static PyMethodDef __pyx_CyFunction_methods[] = {
    {"__reduce__", (PyCFunction)__Pyx_CyFunction_reduce, METH_VARARGS, 0},
    {0, 0, 0, 0}
};


#if PY_VERSION_HEX < 0x030500A0
#define __Pyx_CyFunction_weakreflist(cyfunc) ((cyfunc)->func_weakreflist)
#else
#define __Pyx_CyFunction_weakreflist(cyfunc) ((cyfunc)->func.m_weakreflist)
#endif

static PyObject *__Pyx_CyFunction_Init(__pyx_CyFunctionObject *op, PyMethodDef *ml, int flags, PyObject* qualname,
                                       PyObject *closure, PyObject *module, PyObject* globals, PyObject* code) {
    if (unlikely(op == NULL))
        return NULL;
    op->flags = flags;
    __Pyx_CyFunction_weakreflist(op) = NULL;
    op->func.m_ml = ml;
    op->func.m_self = (PyObject *) op;
    Py_XINCREF(closure);
    op->func_closure = closure;
    Py_XINCREF(module);
    op->func.m_module = module;
    op->func_dict = NULL;
    op->func_name = NULL;
    Py_INCREF(qualname);
    op->func_qualname = qualname;
    op->func_doc = NULL;
    op->func_classobj = NULL;
    op->func_globals = globals;
    Py_INCREF(op->func_globals);
    Py_XINCREF(code);
    op->func_code = code;
    // Dynamic Default args
    op->defaults_pyobjects = 0;
    op->defaults_size = 0;
    op->defaults = NULL;
    op->defaults_tuple = NULL;
    op->defaults_kwdict = NULL;
    op->defaults_getter = NULL;
    op->func_annotations = NULL;
    return (PyObject *) op;
}

static int
__Pyx_CyFunction_clear(__pyx_CyFunctionObject *m)
{
    Py_CLEAR(m->func_closure);
    Py_CLEAR(m->func.m_module);
    Py_CLEAR(m->func_dict);
    Py_CLEAR(m->func_name);
    Py_CLEAR(m->func_qualname);
    Py_CLEAR(m->func_doc);
    Py_CLEAR(m->func_globals);
    Py_CLEAR(m->func_code);
    Py_CLEAR(m->func_classobj);
    Py_CLEAR(m->defaults_tuple);
    Py_CLEAR(m->defaults_kwdict);
    Py_CLEAR(m->func_annotations);

    if (m->defaults) {
        PyObject **pydefaults = __Pyx_CyFunction_Defaults(PyObject *, m);
        int i;

        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_XDECREF(pydefaults[i]);

        PyObject_Free(m->defaults);
        m->defaults = NULL;
    }

    return 0;
}

static void __Pyx__CyFunction_dealloc(__pyx_CyFunctionObject *m)
{
    if (__Pyx_CyFunction_weakreflist(m) != NULL)
        PyObject_ClearWeakRefs((PyObject *) m);
    __Pyx_CyFunction_clear(m);
    PyObject_GC_Del(m);
}

static void __Pyx_CyFunction_dealloc(__pyx_CyFunctionObject *m)
{
    PyObject_GC_UnTrack(m);
    __Pyx__CyFunction_dealloc(m);
}

static int __Pyx_CyFunction_traverse(__pyx_CyFunctionObject *m, visitproc visit, void *arg)
{
    Py_VISIT(m->func_closure);
    Py_VISIT(m->func.m_module);
    Py_VISIT(m->func_dict);
    Py_VISIT(m->func_name);
    Py_VISIT(m->func_qualname);
    Py_VISIT(m->func_doc);
    Py_VISIT(m->func_globals);
    Py_VISIT(m->func_code);
    Py_VISIT(m->func_classobj);
    Py_VISIT(m->defaults_tuple);
    Py_VISIT(m->defaults_kwdict);

    if (m->defaults) {
        PyObject **pydefaults = __Pyx_CyFunction_Defaults(PyObject *, m);
        int i;

        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_VISIT(pydefaults[i]);
    }

    return 0;
}

static PyObject *__Pyx_CyFunction_descr_get(PyObject *func, PyObject *obj, PyObject *type)
{
#if PY_MAJOR_VERSION < 3
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;

    if (m->flags & __Pyx_CYFUNCTION_STATICMETHOD) {
        Py_INCREF(func);
        return func;
    }

    if (m->flags & __Pyx_CYFUNCTION_CLASSMETHOD) {
        if (type == NULL)
            type = (PyObject *)(Py_TYPE(obj));
        return __Pyx_PyMethod_New(func, type, (PyObject *)(Py_TYPE(type)));
    }

    if (obj == Py_None)
        obj = NULL;
#endif
    return __Pyx_PyMethod_New(func, obj, type);
}

static PyObject*
__Pyx_CyFunction_repr(__pyx_CyFunctionObject *op)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromFormat("<cyfunction %U at %p>",
                                op->func_qualname, (void *)op);
#else
    return PyString_FromFormat("<cyfunction %s at %p>",
                               PyString_AsString(op->func_qualname), (void *)op);
#endif
}

static PyObject * __Pyx_CyFunction_CallMethod(PyObject *func, PyObject *self, PyObject *arg, PyObject *kw) {
    // originally copied from PyCFunction_Call() in CPython's Objects/methodobject.c
    PyCFunctionObject* f = (PyCFunctionObject*)func;
    PyCFunction meth = f->m_ml->ml_meth;
    Py_ssize_t size;

    switch (f->m_ml->ml_flags & (METH_VARARGS | METH_KEYWORDS | METH_NOARGS | METH_O)) {
    case METH_VARARGS:
        if (likely(kw == NULL || PyDict_Size(kw) == 0))
            return (*meth)(self, arg);
        break;
    case METH_VARARGS | METH_KEYWORDS:
        return (*(PyCFunctionWithKeywords)(void*)meth)(self, arg, kw);
    case METH_NOARGS:
        if (likely(kw == NULL || PyDict_Size(kw) == 0)) {
            size = PyTuple_GET_SIZE(arg);
            if (likely(size == 0))
                return (*meth)(self, NULL);
            PyErr_Format(PyExc_TypeError,
                "%.200s() takes no arguments (%" CYTHON_FORMAT_SSIZE_T "d given)",
                f->m_ml->ml_name, size);
            return NULL;
        }
        break;
    case METH_O:
        if (likely(kw == NULL || PyDict_Size(kw) == 0)) {
            size = PyTuple_GET_SIZE(arg);
            if (likely(size == 1)) {
                PyObject *result, *arg0;
                #if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
                arg0 = PyTuple_GET_ITEM(arg, 0);
                #else
                arg0 = PySequence_ITEM(arg, 0); if (unlikely(!arg0)) return NULL;
                #endif
                result = (*meth)(self, arg0);
                #if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
                Py_DECREF(arg0);
                #endif
                return result;
            }
            PyErr_Format(PyExc_TypeError,
                "%.200s() takes exactly one argument (%" CYTHON_FORMAT_SSIZE_T "d given)",
                f->m_ml->ml_name, size);
            return NULL;
        }
        break;
    default:
        PyErr_SetString(PyExc_SystemError, "Bad call flags in "
                        "__Pyx_CyFunction_Call. METH_OLDARGS is no "
                        "longer supported!");

        return NULL;
    }
    PyErr_Format(PyExc_TypeError, "%.200s() takes no keyword arguments",
                 f->m_ml->ml_name);
    return NULL;
}

static CYTHON_INLINE PyObject *__Pyx_CyFunction_Call(PyObject *func, PyObject *arg, PyObject *kw) {
    return __Pyx_CyFunction_CallMethod(func, ((PyCFunctionObject*)func)->m_self, arg, kw);
}

static PyObject *__Pyx_CyFunction_CallAsMethod(PyObject *func, PyObject *args, PyObject *kw) {
    PyObject *result;
    __pyx_CyFunctionObject *cyfunc = (__pyx_CyFunctionObject *) func;
    if ((cyfunc->flags & __Pyx_CYFUNCTION_CCLASS) && !(cyfunc->flags & __Pyx_CYFUNCTION_STATICMETHOD)) {
        Py_ssize_t argc;
        PyObject *new_args;
        PyObject *self;

        argc = PyTuple_GET_SIZE(args);
        new_args = PyTuple_GetSlice(args, 1, argc);

        if (unlikely(!new_args))
            return NULL;

        self = PyTuple_GetItem(args, 0);
        if (unlikely(!self)) {
            Py_DECREF(new_args);
            return NULL;
        }

        result = __Pyx_CyFunction_CallMethod(func, self, new_args, kw);
        Py_DECREF(new_args);
    } else {
        result = __Pyx_CyFunction_Call(func, args, kw);
    }
    return result;
}

static PyTypeObject __pyx_CyFunctionType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "cython_function_or_method",      /*tp_name*/
    sizeof(__pyx_CyFunctionObject),   /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) __Pyx_CyFunction_dealloc, /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
#if PY_MAJOR_VERSION < 3
    0,                                  /*tp_compare*/
#else
    0,                                  /*reserved*/
#endif
    (reprfunc) __Pyx_CyFunction_repr,   /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    __Pyx_CyFunction_CallAsMethod,      /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /*tp_flags*/
    0,                                  /*tp_doc*/
    (traverseproc) __Pyx_CyFunction_traverse,   /*tp_traverse*/
    (inquiry) __Pyx_CyFunction_clear,   /*tp_clear*/
    0,                                  /*tp_richcompare*/
#if PY_VERSION_HEX < 0x030500A0
    offsetof(__pyx_CyFunctionObject, func_weakreflist), /*tp_weaklistoffset*/
#else
    offsetof(PyCFunctionObject, m_weakreflist),         /*tp_weaklistoffset*/
#endif
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    __pyx_CyFunction_methods,           /*tp_methods*/
    __pyx_CyFunction_members,           /*tp_members*/
    __pyx_CyFunction_getsets,           /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    __Pyx_CyFunction_descr_get,         /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    offsetof(__pyx_CyFunctionObject, func_dict),/*tp_dictoffset*/
    0,                                  /*tp_init*/
    0,                                  /*tp_alloc*/
    0,                                  /*tp_new*/
    0,                                  /*tp_free*/
    0,                                  /*tp_is_gc*/
    0,                                  /*tp_bases*/
    0,                                  /*tp_mro*/
    0,                                  /*tp_cache*/
    0,                                  /*tp_subclasses*/
    0,                                  /*tp_weaklist*/
    0,                                  /*tp_del*/
    0,                                  /*tp_version_tag*/
#if PY_VERSION_HEX >= 0x030400a1
    0,                                  /*tp_finalize*/
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                  /*tp_vectorcall*/
#endif
#if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
    0,                                  /*tp_print*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000
    0,                                          /*tp_pypy_flags*/
#endif
};


static int __pyx_CyFunction_init(void) {
    __pyx_CyFunctionType = __Pyx_FetchCommonType(&__pyx_CyFunctionType_type);
    if (unlikely(__pyx_CyFunctionType == NULL)) {
        return -1;
    }
    return 0;
}

static CYTHON_INLINE void *__Pyx_CyFunction_InitDefaults(PyObject *func, size_t size, int pyobjects) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;

    m->defaults = PyObject_Malloc(size);
    if (unlikely(!m->defaults))
        return PyErr_NoMemory();
    memset(m->defaults, 0, size);
    m->defaults_pyobjects = pyobjects;
    m->defaults_size = size;
    return m->defaults;
}

static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsTuple(PyObject *func, PyObject *tuple) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->defaults_tuple = tuple;
    Py_INCREF(tuple);
}

static CYTHON_INLINE void __Pyx_CyFunction_SetDefaultsKwDict(PyObject *func, PyObject *dict) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->defaults_kwdict = dict;
    Py_INCREF(dict);
}

static CYTHON_INLINE void __Pyx_CyFunction_SetAnnotationsDict(PyObject *func, PyObject *dict) {
    __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *) func;
    m->func_annotations = dict;
    Py_INCREF(dict);
}


//////////////////// CythonFunction.proto ////////////////////

static PyObject *__Pyx_CyFunction_New(PyMethodDef *ml,
                                      int flags, PyObject* qualname,
                                      PyObject *closure,
                                      PyObject *module, PyObject *globals,
                                      PyObject* code);

//////////////////// CythonFunction ////////////////////
//@requires: CythonFunctionShared

static PyObject *__Pyx_CyFunction_New(PyMethodDef *ml, int flags, PyObject* qualname,
                                      PyObject *closure, PyObject *module, PyObject* globals, PyObject* code) {
    PyObject *op = __Pyx_CyFunction_Init(
        PyObject_GC_New(__pyx_CyFunctionObject, __pyx_CyFunctionType),
        ml, flags, qualname, closure, module, globals, code
    );
    if (likely(op)) {
        PyObject_GC_Track(op);
    }
    return op;
}


//////////////////// CyFunctionClassCell.proto ////////////////////
static int __Pyx_CyFunction_InitClassCell(PyObject *cyfunctions, PyObject *classobj);/*proto*/

//////////////////// CyFunctionClassCell ////////////////////
//@requires: CythonFunctionShared

static int __Pyx_CyFunction_InitClassCell(PyObject *cyfunctions, PyObject *classobj) {
    Py_ssize_t i, count = PyList_GET_SIZE(cyfunctions);

    for (i = 0; i < count; i++) {
        __pyx_CyFunctionObject *m = (__pyx_CyFunctionObject *)
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
            PyList_GET_ITEM(cyfunctions, i);
#else
            PySequence_ITEM(cyfunctions, i);
        if (unlikely(!m))
            return -1;
#endif
        Py_INCREF(classobj);
        m->func_classobj = classobj;
#if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
        Py_DECREF((PyObject*)m);
#endif
    }
    return 0;
}


//////////////////// FusedFunction.proto ////////////////////

typedef struct {
    __pyx_CyFunctionObject func;
    PyObject *__signatures__;
    PyObject *type;
    PyObject *self;
} __pyx_FusedFunctionObject;

static PyObject *__pyx_FusedFunction_New(PyMethodDef *ml, int flags,
                                         PyObject *qualname, PyObject *closure,
                                         PyObject *module, PyObject *globals,
                                         PyObject *code);

static int __pyx_FusedFunction_clear(__pyx_FusedFunctionObject *self);
static PyTypeObject *__pyx_FusedFunctionType = NULL;
static int __pyx_FusedFunction_init(void);

#define __Pyx_FusedFunction_USED

//////////////////// FusedFunction ////////////////////
//@requires: CythonFunctionShared

static PyObject *
__pyx_FusedFunction_New(PyMethodDef *ml, int flags,
                        PyObject *qualname, PyObject *closure,
                        PyObject *module, PyObject *globals,
                        PyObject *code)
{
    PyObject *op = __Pyx_CyFunction_Init(
        // __pyx_CyFunctionObject is correct below since that's the cast that we want.
        PyObject_GC_New(__pyx_CyFunctionObject, __pyx_FusedFunctionType),
        ml, flags, qualname, closure, module, globals, code
    );
    if (likely(op)) {
        __pyx_FusedFunctionObject *fusedfunc = (__pyx_FusedFunctionObject *) op;
        fusedfunc->__signatures__ = NULL;
        fusedfunc->type = NULL;
        fusedfunc->self = NULL;
        PyObject_GC_Track(op);
    }
    return op;
}

static void
__pyx_FusedFunction_dealloc(__pyx_FusedFunctionObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->self);
    Py_CLEAR(self->type);
    Py_CLEAR(self->__signatures__);
    __Pyx__CyFunction_dealloc((__pyx_CyFunctionObject *) self);
}

static int
__pyx_FusedFunction_traverse(__pyx_FusedFunctionObject *self,
                             visitproc visit,
                             void *arg)
{
    Py_VISIT(self->self);
    Py_VISIT(self->type);
    Py_VISIT(self->__signatures__);
    return __Pyx_CyFunction_traverse((__pyx_CyFunctionObject *) self, visit, arg);
}

static int
__pyx_FusedFunction_clear(__pyx_FusedFunctionObject *self)
{
    Py_CLEAR(self->self);
    Py_CLEAR(self->type);
    Py_CLEAR(self->__signatures__);
    return __Pyx_CyFunction_clear((__pyx_CyFunctionObject *) self);
}


static PyObject *
__pyx_FusedFunction_descr_get(PyObject *self, PyObject *obj, PyObject *type)
{
    __pyx_FusedFunctionObject *func, *meth;

    func = (__pyx_FusedFunctionObject *) self;

    if (func->self || func->func.flags & __Pyx_CYFUNCTION_STATICMETHOD) {
        // Do not allow rebinding and don't do anything for static methods
        Py_INCREF(self);
        return self;
    }

    if (obj == Py_None)
        obj = NULL;

    meth = (__pyx_FusedFunctionObject *) __pyx_FusedFunction_New(
                    ((PyCFunctionObject *) func)->m_ml,
                    ((__pyx_CyFunctionObject *) func)->flags,
                    ((__pyx_CyFunctionObject *) func)->func_qualname,
                    ((__pyx_CyFunctionObject *) func)->func_closure,
                    ((PyCFunctionObject *) func)->m_module,
                    ((__pyx_CyFunctionObject *) func)->func_globals,
                    ((__pyx_CyFunctionObject *) func)->func_code);
    if (!meth)
        return NULL;

    // defaults needs copying fully rather than just copying the pointer
    // since otherwise it will be freed on destruction of meth despite
    // belonging to func rather than meth
    if (func->func.defaults) {
        PyObject **pydefaults;
        int i;

        if (!__Pyx_CyFunction_InitDefaults((PyObject*)meth,
                                      func->func.defaults_size,
                                      func->func.defaults_pyobjects)) {
            Py_XDECREF((PyObject*)meth);
            return NULL;
        }
        memcpy(meth->func.defaults, func->func.defaults, func->func.defaults_size);

        pydefaults = __Pyx_CyFunction_Defaults(PyObject *, meth);
        for (i = 0; i < meth->func.defaults_pyobjects; i++)
            Py_XINCREF(pydefaults[i]);
    }

    Py_XINCREF(func->func.func_classobj);
    meth->func.func_classobj = func->func.func_classobj;

    Py_XINCREF(func->__signatures__);
    meth->__signatures__ = func->__signatures__;

    Py_XINCREF(type);
    meth->type = type;

    Py_XINCREF(func->func.defaults_tuple);
    meth->func.defaults_tuple = func->func.defaults_tuple;

    if (func->func.flags & __Pyx_CYFUNCTION_CLASSMETHOD)
        obj = type;

    Py_XINCREF(obj);
    meth->self = obj;

    return (PyObject *) meth;
}

static PyObject *
_obj_to_str(PyObject *obj)
{
    if (PyType_Check(obj))
        return PyObject_GetAttr(obj, PYIDENT("__name__"));
    else
        return PyObject_Str(obj);
}

static PyObject *
__pyx_FusedFunction_getitem(__pyx_FusedFunctionObject *self, PyObject *idx)
{
    PyObject *signature = NULL;
    PyObject *unbound_result_func;
    PyObject *result_func = NULL;

    if (self->__signatures__ == NULL) {
        PyErr_SetString(PyExc_TypeError, "Function is not fused");
        return NULL;
    }

    if (PyTuple_Check(idx)) {
        PyObject *list = PyList_New(0);
        Py_ssize_t n = PyTuple_GET_SIZE(idx);
        PyObject *sep = NULL;
        int i;

        if (unlikely(!list))
            return NULL;

        for (i = 0; i < n; i++) {
            int ret;
            PyObject *string;
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
            PyObject *item = PyTuple_GET_ITEM(idx, i);
#else
            PyObject *item = PySequence_ITEM(idx, i);  if (unlikely(!item)) goto __pyx_err;
#endif
            string = _obj_to_str(item);
#if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
            Py_DECREF(item);
#endif
            if (unlikely(!string)) goto __pyx_err;
            ret = PyList_Append(list, string);
            Py_DECREF(string);
            if (unlikely(ret < 0)) goto __pyx_err;
        }

        sep = PyUnicode_FromString("|");
        if (likely(sep))
            signature = PyUnicode_Join(sep, list);
__pyx_err:
;
        Py_DECREF(list);
        Py_XDECREF(sep);
    } else {
        signature = _obj_to_str(idx);
    }

    if (!signature)
        return NULL;

    unbound_result_func = PyObject_GetItem(self->__signatures__, signature);

    if (unbound_result_func) {
        if (self->self || self->type) {
            __pyx_FusedFunctionObject *unbound = (__pyx_FusedFunctionObject *) unbound_result_func;

            // TODO: move this to InitClassCell
            Py_CLEAR(unbound->func.func_classobj);
            Py_XINCREF(self->func.func_classobj);
            unbound->func.func_classobj = self->func.func_classobj;

            result_func = __pyx_FusedFunction_descr_get(unbound_result_func,
                                                        self->self, self->type);
        } else {
            result_func = unbound_result_func;
            Py_INCREF(result_func);
        }
    }

    Py_DECREF(signature);
    Py_XDECREF(unbound_result_func);

    return result_func;
}

static PyObject *
__pyx_FusedFunction_callfunction(PyObject *func, PyObject *args, PyObject *kw)
{
     __pyx_CyFunctionObject *cyfunc = (__pyx_CyFunctionObject *) func;
    int static_specialized = (cyfunc->flags & __Pyx_CYFUNCTION_STATICMETHOD &&
                              !((__pyx_FusedFunctionObject *) func)->__signatures__);

    if (cyfunc->flags & __Pyx_CYFUNCTION_CCLASS && !static_specialized) {
        return __Pyx_CyFunction_CallAsMethod(func, args, kw);
    } else {
        return __Pyx_CyFunction_Call(func, args, kw);
    }
}

// Note: the 'self' from method binding is passed in in the args tuple,
//       whereas PyCFunctionObject's m_self is passed in as the first
//       argument to the C function. For extension methods we need
//       to pass 'self' as 'm_self' and not as the first element of the
//       args tuple.

static PyObject *
__pyx_FusedFunction_call(PyObject *func, PyObject *args, PyObject *kw)
{
    __pyx_FusedFunctionObject *binding_func = (__pyx_FusedFunctionObject *) func;
    Py_ssize_t argc = PyTuple_GET_SIZE(args);
    PyObject *new_args = NULL;
    __pyx_FusedFunctionObject *new_func = NULL;
    PyObject *result = NULL;
    PyObject *self = NULL;
    int is_staticmethod = binding_func->func.flags & __Pyx_CYFUNCTION_STATICMETHOD;
    int is_classmethod = binding_func->func.flags & __Pyx_CYFUNCTION_CLASSMETHOD;

    if (binding_func->self) {
        // Bound method call, put 'self' in the args tuple
        Py_ssize_t i;
        new_args = PyTuple_New(argc + 1);
        if (!new_args)
            return NULL;

        self = binding_func->self;
#if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
        Py_INCREF(self);
#endif
        Py_INCREF(self);
        PyTuple_SET_ITEM(new_args, 0, self);

        for (i = 0; i < argc; i++) {
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
            PyObject *item = PyTuple_GET_ITEM(args, i);
            Py_INCREF(item);
#else
            PyObject *item = PySequence_ITEM(args, i);  if (unlikely(!item)) goto bad;
#endif
            PyTuple_SET_ITEM(new_args, i + 1, item);
        }

        args = new_args;
    } else if (binding_func->type) {
        // Unbound method call
        if (argc < 1) {
            PyErr_SetString(PyExc_TypeError, "Need at least one argument, 0 given.");
            return NULL;
        }
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        self = PyTuple_GET_ITEM(args, 0);
#else
        self = PySequence_ITEM(args, 0);  if (unlikely(!self)) return NULL;
#endif
    }

    if (self && !is_classmethod && !is_staticmethod) {
        int is_instance = PyObject_IsInstance(self, binding_func->type);
        if (unlikely(!is_instance)) {
            PyErr_Format(PyExc_TypeError,
                         "First argument should be of type %.200s, got %.200s.",
                         ((PyTypeObject *) binding_func->type)->tp_name,
                         Py_TYPE(self)->tp_name);
            goto bad;
        } else if (unlikely(is_instance == -1)) {
            goto bad;
        }
    }
#if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
    Py_XDECREF(self);
    self = NULL;
#endif

    if (binding_func->__signatures__) {
        PyObject *tup;
        if (is_staticmethod && binding_func->func.flags & __Pyx_CYFUNCTION_CCLASS) {
            // FIXME: this seems wrong, but we must currently pass the signatures dict as 'self' argument
            tup = PyTuple_Pack(3, args,
                               kw == NULL ? Py_None : kw,
                               binding_func->func.defaults_tuple);
            if (unlikely(!tup)) goto bad;
            new_func = (__pyx_FusedFunctionObject *) __Pyx_CyFunction_CallMethod(
                func, binding_func->__signatures__, tup, NULL);
        } else {
            tup = PyTuple_Pack(4, binding_func->__signatures__, args,
                               kw == NULL ? Py_None : kw,
                               binding_func->func.defaults_tuple);
            if (unlikely(!tup)) goto bad;
            new_func = (__pyx_FusedFunctionObject *) __pyx_FusedFunction_callfunction(func, tup, NULL);
        }
        Py_DECREF(tup);

        if (unlikely(!new_func))
            goto bad;

        Py_XINCREF(binding_func->func.func_classobj);
        Py_CLEAR(new_func->func.func_classobj);
        new_func->func.func_classobj = binding_func->func.func_classobj;

        func = (PyObject *) new_func;
    }

    result = __pyx_FusedFunction_callfunction(func, args, kw);
bad:
#if !(CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS)
    Py_XDECREF(self);
#endif
    Py_XDECREF(new_args);
    Py_XDECREF((PyObject *) new_func);
    return result;
}

static PyMemberDef __pyx_FusedFunction_members[] = {
    {(char *) "__signatures__",
     T_OBJECT,
     offsetof(__pyx_FusedFunctionObject, __signatures__),
     READONLY,
     0},
    {0, 0, 0, 0, 0},
};

static PyMappingMethods __pyx_FusedFunction_mapping_methods = {
    0,
    (binaryfunc) __pyx_FusedFunction_getitem,
    0,
};

static PyTypeObject __pyx_FusedFunctionType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "fused_cython_function",           /*tp_name*/
    sizeof(__pyx_FusedFunctionObject), /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) __pyx_FusedFunction_dealloc, /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
#if PY_MAJOR_VERSION < 3
    0,                                  /*tp_compare*/
#else
    0,                                  /*reserved*/
#endif
    0,                                  /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    &__pyx_FusedFunction_mapping_methods, /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    (ternaryfunc) __pyx_FusedFunction_call, /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    0,                                  /*tp_doc*/
    (traverseproc) __pyx_FusedFunction_traverse,   /*tp_traverse*/
    (inquiry) __pyx_FusedFunction_clear,/*tp_clear*/
    0,                                  /*tp_richcompare*/
    0,                                  /*tp_weaklistoffset*/
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    0,                                  /*tp_methods*/
    __pyx_FusedFunction_members,        /*tp_members*/
    // __doc__ is None for the fused function type, but we need it to be
    // a descriptor for the instance's __doc__, so rebuild descriptors in our subclass
    __pyx_CyFunction_getsets,           /*tp_getset*/
    // NOTE: tp_base may be changed later during module initialisation when importing CyFunction across modules.
    &__pyx_CyFunctionType_type,         /*tp_base*/
    0,                                  /*tp_dict*/
    __pyx_FusedFunction_descr_get,      /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    0,                                  /*tp_init*/
    0,                                  /*tp_alloc*/
    0,                                  /*tp_new*/
    0,                                  /*tp_free*/
    0,                                  /*tp_is_gc*/
    0,                                  /*tp_bases*/
    0,                                  /*tp_mro*/
    0,                                  /*tp_cache*/
    0,                                  /*tp_subclasses*/
    0,                                  /*tp_weaklist*/
    0,                                  /*tp_del*/
    0,                                  /*tp_version_tag*/
#if PY_VERSION_HEX >= 0x030400a1
    0,                                  /*tp_finalize*/
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                  /*tp_vectorcall*/
#endif
#if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
    0,                                  /*tp_print*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000
    0,                                          /*tp_pypy_flags*/
#endif
};

static int __pyx_FusedFunction_init(void) {
    // Set base from __Pyx_FetchCommonTypeFromSpec, in case it's different from the local static value.
    __pyx_FusedFunctionType_type.tp_base = __pyx_CyFunctionType;
    __pyx_FusedFunctionType = __Pyx_FetchCommonType(&__pyx_FusedFunctionType_type);
    if (__pyx_FusedFunctionType == NULL) {
        return -1;
    }
    return 0;
}

//////////////////// ClassMethod.proto ////////////////////

#include "descrobject.h"
static CYTHON_UNUSED PyObject* __Pyx_Method_ClassMethod(PyObject *method); /*proto*/

//////////////////// ClassMethod ////////////////////

static PyObject* __Pyx_Method_ClassMethod(PyObject *method) {
#if CYTHON_COMPILING_IN_PYPY && PYPY_VERSION_NUM <= 0x05080000
    if (PyObject_TypeCheck(method, &PyWrapperDescr_Type)) {
        // cdef classes
        return PyClassMethod_New(method);
    }
#else
#if CYTHON_COMPILING_IN_PYSTON || CYTHON_COMPILING_IN_PYPY
    // special C-API function only in Pyston and PyPy >= 5.9
    if (PyMethodDescr_Check(method))
#else
    #if PY_MAJOR_VERSION == 2
    // PyMethodDescr_Type is not exposed in the CPython C-API in Py2.
    static PyTypeObject *methoddescr_type = NULL;
    if (methoddescr_type == NULL) {
       PyObject *meth = PyObject_GetAttrString((PyObject*)&PyList_Type, "append");
       if (!meth) return NULL;
       methoddescr_type = Py_TYPE(meth);
       Py_DECREF(meth);
    }
    #else
    PyTypeObject *methoddescr_type = &PyMethodDescr_Type;
    #endif
    if (__Pyx_TypeCheck(method, methoddescr_type))
#endif
    {
        // cdef classes
        PyMethodDescrObject *descr = (PyMethodDescrObject *)method;
        #if PY_VERSION_HEX < 0x03020000
        PyTypeObject *d_type = descr->d_type;
        #else
        PyTypeObject *d_type = descr->d_common.d_type;
        #endif
        return PyDescr_NewClassMethod(d_type, descr->d_method);
    }
#endif
    else if (PyMethod_Check(method)) {
        // python classes
        return PyClassMethod_New(PyMethod_GET_FUNCTION(method));
    }
    else {
        return PyClassMethod_New(method);
    }
}
