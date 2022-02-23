// This is copied from genobject.c in CPython 3.6.
// Try to keep it in sync by doing this from time to time:
//    sed -e 's|__pyx_||ig'  Cython/Utility/AsyncGen.c | diff -udw - cpython/Objects/genobject.c | less

//////////////////// AsyncGenerator.proto ////////////////////
//@requires: Coroutine.c::Coroutine

#define __Pyx_AsyncGen_USED
typedef struct {
    __pyx_CoroutineObject coro;
    PyObject *ag_finalizer;
    int ag_hooks_inited;
    int ag_closed;
} __pyx_PyAsyncGenObject;

static PyTypeObject *__pyx__PyAsyncGenWrappedValueType = 0;
static PyTypeObject *__pyx__PyAsyncGenASendType = 0;
static PyTypeObject *__pyx__PyAsyncGenAThrowType = 0;
static PyTypeObject *__pyx_AsyncGenType = 0;

#define __Pyx_AsyncGen_CheckExact(obj) (Py_TYPE(obj) == __pyx_AsyncGenType)
#define __pyx_PyAsyncGenASend_CheckExact(o) \
                    (Py_TYPE(o) == __pyx__PyAsyncGenASendType)
#define __pyx_PyAsyncGenAThrow_CheckExact(o) \
                    (Py_TYPE(o) == __pyx__PyAsyncGenAThrowType)

static PyObject *__Pyx_async_gen_anext(PyObject *o);
static CYTHON_INLINE PyObject *__Pyx_async_gen_asend_iternext(PyObject *o);
static PyObject *__Pyx_async_gen_asend_send(PyObject *o, PyObject *arg);
static PyObject *__Pyx_async_gen_asend_close(PyObject *o, PyObject *args);
static PyObject *__Pyx_async_gen_athrow_close(PyObject *o, PyObject *args);

static PyObject *__Pyx__PyAsyncGenValueWrapperNew(PyObject *val);


static __pyx_CoroutineObject *__Pyx_AsyncGen_New(
            __pyx_coroutine_body_t body, PyObject *code, PyObject *closure,
            PyObject *name, PyObject *qualname, PyObject *module_name) {
    __pyx_PyAsyncGenObject *gen = PyObject_GC_New(__pyx_PyAsyncGenObject, __pyx_AsyncGenType);
    if (unlikely(!gen))
        return NULL;
    gen->ag_finalizer = NULL;
    gen->ag_closed = 0;
    gen->ag_hooks_inited = 0;
    return __Pyx__Coroutine_NewInit((__pyx_CoroutineObject*)gen, body, code, closure, name, qualname, module_name);
}

static int __pyx_AsyncGen_init(void);
static void __Pyx_PyAsyncGen_Fini(void);

//////////////////// AsyncGenerator.cleanup ////////////////////

__Pyx_PyAsyncGen_Fini();

//////////////////// AsyncGeneratorInitFinalizer ////////////////////

// this is separated out because it needs more adaptation

#if PY_VERSION_HEX < 0x030600B0
static int __Pyx_async_gen_init_hooks(__pyx_PyAsyncGenObject *o) {
#if 0
    // TODO: implement finalizer support in older Python versions
    PyThreadState *tstate;
    PyObject *finalizer;
    PyObject *firstiter;
#endif

    if (likely(o->ag_hooks_inited)) {
        return 0;
    }

    o->ag_hooks_inited = 1;

#if 0
    tstate = __Pyx_PyThreadState_Current;

    finalizer = tstate->async_gen_finalizer;
    if (finalizer) {
        Py_INCREF(finalizer);
        o->ag_finalizer = finalizer;
    }

    firstiter = tstate->async_gen_firstiter;
    if (firstiter) {
        PyObject *res;

        Py_INCREF(firstiter);
        res = __Pyx_PyObject_CallOneArg(firstiter, (PyObject*)o);
        Py_DECREF(firstiter);
        if (res == NULL) {
            return 1;
        }
        Py_DECREF(res);
    }
#endif

    return 0;
}
#endif


//////////////////// AsyncGenerator ////////////////////
//@requires: AsyncGeneratorInitFinalizer
//@requires: Coroutine.c::Coroutine
//@requires: Coroutine.c::ReturnWithStopIteration
//@requires: ObjectHandling.c::PyObjectCall2Args
//@requires: ObjectHandling.c::PyObject_GenericGetAttrNoDict

PyDoc_STRVAR(__Pyx_async_gen_send_doc,
"send(arg) -> send 'arg' into generator,\n\
return next yielded value or raise StopIteration.");

PyDoc_STRVAR(__Pyx_async_gen_close_doc,
"close() -> raise GeneratorExit inside generator.");

PyDoc_STRVAR(__Pyx_async_gen_throw_doc,
"throw(typ[,val[,tb]]) -> raise exception in generator,\n\
return next yielded value or raise StopIteration.");

PyDoc_STRVAR(__Pyx_async_gen_await_doc,
"__await__() -> return a representation that can be passed into the 'await' expression.");

// COPY STARTS HERE:

static PyObject *__Pyx_async_gen_asend_new(__pyx_PyAsyncGenObject *, PyObject *);
static PyObject *__Pyx_async_gen_athrow_new(__pyx_PyAsyncGenObject *, PyObject *);

static const char *__Pyx_NON_INIT_CORO_MSG = "can't send non-None value to a just-started coroutine";
static const char *__Pyx_ASYNC_GEN_IGNORED_EXIT_MSG = "async generator ignored GeneratorExit";

typedef enum {
    __PYX_AWAITABLE_STATE_INIT,   /* new awaitable, has not yet been iterated */
    __PYX_AWAITABLE_STATE_ITER,   /* being iterated */
    __PYX_AWAITABLE_STATE_CLOSED, /* closed */
} __pyx_AwaitableState;

typedef struct {
    PyObject_HEAD
    __pyx_PyAsyncGenObject *ags_gen;

    /* Can be NULL, when in the __anext__() mode (equivalent of "asend(None)") */
    PyObject *ags_sendval;

    __pyx_AwaitableState ags_state;
} __pyx_PyAsyncGenASend;


typedef struct {
    PyObject_HEAD
    __pyx_PyAsyncGenObject *agt_gen;

    /* Can be NULL, when in the "aclose()" mode (equivalent of "athrow(GeneratorExit)") */
    PyObject *agt_args;

    __pyx_AwaitableState agt_state;
} __pyx_PyAsyncGenAThrow;


typedef struct {
    PyObject_HEAD
    PyObject *agw_val;
} __pyx__PyAsyncGenWrappedValue;


#ifndef _PyAsyncGen_MAXFREELIST
#define _PyAsyncGen_MAXFREELIST 80
#endif

// Freelists boost performance 6-10%; they also reduce memory
// fragmentation, as _PyAsyncGenWrappedValue and PyAsyncGenASend
// are short-living objects that are instantiated for every
// __anext__ call.

static __pyx__PyAsyncGenWrappedValue *__Pyx_ag_value_freelist[_PyAsyncGen_MAXFREELIST];
static int __Pyx_ag_value_freelist_free = 0;

static __pyx_PyAsyncGenASend *__Pyx_ag_asend_freelist[_PyAsyncGen_MAXFREELIST];
static int __Pyx_ag_asend_freelist_free = 0;

#define __pyx__PyAsyncGenWrappedValue_CheckExact(o) \
                    (Py_TYPE(o) == __pyx__PyAsyncGenWrappedValueType)


static int
__Pyx_async_gen_traverse(__pyx_PyAsyncGenObject *gen, visitproc visit, void *arg)
{
    Py_VISIT(gen->ag_finalizer);
    return __Pyx_Coroutine_traverse((__pyx_CoroutineObject*)gen, visit, arg);
}


static PyObject *
__Pyx_async_gen_repr(__pyx_CoroutineObject *o)
{
    // avoid NULL pointer dereference for qualname during garbage collection
    return PyUnicode_FromFormat("<async_generator object %S at %p>",
                                o->gi_qualname ? o->gi_qualname : Py_None, o);
}


#if PY_VERSION_HEX >= 0x030600B0
static int
__Pyx_async_gen_init_hooks(__pyx_PyAsyncGenObject *o)
{
    PyThreadState *tstate;
    PyObject *finalizer;
    PyObject *firstiter;

    if (o->ag_hooks_inited) {
        return 0;
    }

    o->ag_hooks_inited = 1;

    tstate = __Pyx_PyThreadState_Current;

    finalizer = tstate->async_gen_finalizer;
    if (finalizer) {
        Py_INCREF(finalizer);
        o->ag_finalizer = finalizer;
    }

    firstiter = tstate->async_gen_firstiter;
    if (firstiter) {
        PyObject *res;
#if CYTHON_UNPACK_METHODS
        PyObject *self;
#endif

        Py_INCREF(firstiter);
        // at least asyncio stores methods here => optimise the call
#if CYTHON_UNPACK_METHODS
        if (likely(PyMethod_Check(firstiter)) && likely((self = PyMethod_GET_SELF(firstiter)) != NULL)) {
            PyObject *function = PyMethod_GET_FUNCTION(firstiter);
            res = __Pyx_PyObject_Call2Args(function, self, (PyObject*)o);
        } else
#endif
        res = __Pyx_PyObject_CallOneArg(firstiter, (PyObject*)o);

        Py_DECREF(firstiter);
        if (unlikely(res == NULL)) {
            return 1;
        }
        Py_DECREF(res);
    }

    return 0;
}
#endif


static PyObject *
__Pyx_async_gen_anext(PyObject *g)
{
    __pyx_PyAsyncGenObject *o = (__pyx_PyAsyncGenObject*) g;
    if (__Pyx_async_gen_init_hooks(o)) {
        return NULL;
    }
    return __Pyx_async_gen_asend_new(o, NULL);
}

static PyObject *
__Pyx_async_gen_anext_method(PyObject *g, CYTHON_UNUSED PyObject *arg) {
    return __Pyx_async_gen_anext(g);
}


static PyObject *
__Pyx_async_gen_asend(__pyx_PyAsyncGenObject *o, PyObject *arg)
{
    if (__Pyx_async_gen_init_hooks(o)) {
        return NULL;
    }
    return __Pyx_async_gen_asend_new(o, arg);
}


static PyObject *
__Pyx_async_gen_aclose(__pyx_PyAsyncGenObject *o, CYTHON_UNUSED PyObject *arg)
{
    if (__Pyx_async_gen_init_hooks(o)) {
        return NULL;
    }
    return __Pyx_async_gen_athrow_new(o, NULL);
}


static PyObject *
__Pyx_async_gen_athrow(__pyx_PyAsyncGenObject *o, PyObject *args)
{
    if (__Pyx_async_gen_init_hooks(o)) {
        return NULL;
    }
    return __Pyx_async_gen_athrow_new(o, args);
}


static PyObject *
__Pyx_async_gen_self_method(PyObject *g, CYTHON_UNUSED PyObject *arg) {
    return __Pyx_NewRef(g);
}


static PyGetSetDef __Pyx_async_gen_getsetlist[] = {
    {(char*) "__name__", (getter)__Pyx_Coroutine_get_name, (setter)__Pyx_Coroutine_set_name,
     (char*) PyDoc_STR("name of the async generator"), 0},
    {(char*) "__qualname__", (getter)__Pyx_Coroutine_get_qualname, (setter)__Pyx_Coroutine_set_qualname,
     (char*) PyDoc_STR("qualified name of the async generator"), 0},
    //REMOVED: {(char*) "ag_await", (getter)coro_get_cr_await, NULL,
    //REMOVED:  (char*) PyDoc_STR("object being awaited on, or None")},
    {0, 0, 0, 0, 0} /* Sentinel */
};

static PyMemberDef __Pyx_async_gen_memberlist[] = {
    //REMOVED: {(char*) "ag_frame",   T_OBJECT, offsetof(__pyx_PyAsyncGenObject, ag_frame),   READONLY},
    {(char*) "ag_running", T_BOOL,   offsetof(__pyx_CoroutineObject, is_running), READONLY, NULL},
    //REMOVED: {(char*) "ag_code",    T_OBJECT, offsetof(__pyx_PyAsyncGenObject, ag_code),    READONLY},
    //ADDED: "ag_await"
    {(char*) "ag_await", T_OBJECT, offsetof(__pyx_CoroutineObject, yieldfrom), READONLY,
     (char*) PyDoc_STR("object being awaited on, or None")},
    {0, 0, 0, 0, 0}      /* Sentinel */
};

PyDoc_STRVAR(__Pyx_async_aclose_doc,
"aclose() -> raise GeneratorExit inside generator.");

PyDoc_STRVAR(__Pyx_async_asend_doc,
"asend(v) -> send 'v' in generator.");

PyDoc_STRVAR(__Pyx_async_athrow_doc,
"athrow(typ[,val[,tb]]) -> raise exception in generator.");

PyDoc_STRVAR(__Pyx_async_aiter_doc,
"__aiter__(v) -> return an asynchronous iterator.");

PyDoc_STRVAR(__Pyx_async_anext_doc,
"__anext__(v) -> continue asynchronous iteration and return the next element.");

static PyMethodDef __Pyx_async_gen_methods[] = {
    {"asend", (PyCFunction)__Pyx_async_gen_asend, METH_O, __Pyx_async_asend_doc},
    {"athrow",(PyCFunction)__Pyx_async_gen_athrow, METH_VARARGS, __Pyx_async_athrow_doc},
    {"aclose", (PyCFunction)__Pyx_async_gen_aclose, METH_NOARGS, __Pyx_async_aclose_doc},
    {"__aiter__", (PyCFunction)__Pyx_async_gen_self_method, METH_NOARGS, __Pyx_async_aiter_doc},
    {"__anext__", (PyCFunction)__Pyx_async_gen_anext_method, METH_NOARGS, __Pyx_async_anext_doc},
    {0, 0, 0, 0}        /* Sentinel */
};


#if CYTHON_USE_ASYNC_SLOTS
static __Pyx_PyAsyncMethodsStruct __Pyx_async_gen_as_async = {
    0,                                          /* am_await */
    PyObject_SelfIter,                          /* am_aiter */
    (unaryfunc)__Pyx_async_gen_anext,           /* am_anext */
#if PY_VERSION_HEX >= 0x030A00A3
    0, /*am_send*/
#endif
};
#endif

static PyTypeObject __pyx_AsyncGenType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "async_generator",                          /* tp_name */
    sizeof(__pyx_PyAsyncGenObject),             /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)__Pyx_Coroutine_dealloc,        /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if CYTHON_USE_ASYNC_SLOTS
    &__Pyx_async_gen_as_async,                        /* tp_as_async */
#else
    0,                                          /*tp_reserved*/
#endif
    (reprfunc)__Pyx_async_gen_repr,                   /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
        Py_TPFLAGS_HAVE_FINALIZE,               /* tp_flags */
    0,                                          /* tp_doc */
    (traverseproc)__Pyx_async_gen_traverse,           /* tp_traverse */
    0,                                          /* tp_clear */
#if CYTHON_USE_ASYNC_SLOTS && CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3 && PY_VERSION_HEX < 0x030500B1
    // in order to (mis-)use tp_reserved above, we must also implement tp_richcompare
    __Pyx_Coroutine_compare,            /*tp_richcompare*/
#else
    0,                                  /*tp_richcompare*/
#endif
    offsetof(__pyx_CoroutineObject, gi_weakreflist), /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    __Pyx_async_gen_methods,                          /* tp_methods */
    __Pyx_async_gen_memberlist,                       /* tp_members */
    __Pyx_async_gen_getsetlist,                       /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
#if CYTHON_USE_TP_FINALIZE
    0,                                  /*tp_del*/
#else
    __Pyx_Coroutine_del,                /*tp_del*/
#endif
    0,                                          /* tp_version_tag */
#if CYTHON_USE_TP_FINALIZE
    __Pyx_Coroutine_del,                        /* tp_finalize */
#elif PY_VERSION_HEX >= 0x030400a1
    0,                                          /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                          /*tp_vectorcall*/
#endif
#if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
    0,                                          /*tp_print*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000
    0,                                          /*tp_pypy_flags*/
#endif
};


static int
__Pyx_PyAsyncGen_ClearFreeLists(void)
{
    int ret = __Pyx_ag_value_freelist_free + __Pyx_ag_asend_freelist_free;

    while (__Pyx_ag_value_freelist_free) {
        __pyx__PyAsyncGenWrappedValue *o;
        o = __Pyx_ag_value_freelist[--__Pyx_ag_value_freelist_free];
        assert(__pyx__PyAsyncGenWrappedValue_CheckExact(o));
        PyObject_GC_Del(o);
    }

    while (__Pyx_ag_asend_freelist_free) {
        __pyx_PyAsyncGenASend *o;
        o = __Pyx_ag_asend_freelist[--__Pyx_ag_asend_freelist_free];
        assert(Py_TYPE(o) == __pyx__PyAsyncGenASendType);
        PyObject_GC_Del(o);
    }

    return ret;
}

static void
__Pyx_PyAsyncGen_Fini(void)
{
    __Pyx_PyAsyncGen_ClearFreeLists();
}


static PyObject *
__Pyx_async_gen_unwrap_value(__pyx_PyAsyncGenObject *gen, PyObject *result)
{
    if (result == NULL) {
        PyObject *exc_type = PyErr_Occurred();
        if (!exc_type) {
            PyErr_SetNone(__Pyx_PyExc_StopAsyncIteration);
            gen->ag_closed = 1;
        } else if (__Pyx_PyErr_GivenExceptionMatches2(exc_type, __Pyx_PyExc_StopAsyncIteration, PyExc_GeneratorExit)) {
            gen->ag_closed = 1;
        }

        return NULL;
    }

    if (__pyx__PyAsyncGenWrappedValue_CheckExact(result)) {
        /* async yield */
        __Pyx_ReturnWithStopIteration(((__pyx__PyAsyncGenWrappedValue*)result)->agw_val);
        Py_DECREF(result);
        return NULL;
    }

    return result;
}


/* ---------- Async Generator ASend Awaitable ------------ */


static void
__Pyx_async_gen_asend_dealloc(__pyx_PyAsyncGenASend *o)
{
    PyObject_GC_UnTrack((PyObject *)o);
    Py_CLEAR(o->ags_gen);
    Py_CLEAR(o->ags_sendval);
    if (__Pyx_ag_asend_freelist_free < _PyAsyncGen_MAXFREELIST) {
        assert(__pyx_PyAsyncGenASend_CheckExact(o));
        __Pyx_ag_asend_freelist[__Pyx_ag_asend_freelist_free++] = o;
    } else {
        PyObject_GC_Del(o);
    }
}

static int
__Pyx_async_gen_asend_traverse(__pyx_PyAsyncGenASend *o, visitproc visit, void *arg)
{
    Py_VISIT(o->ags_gen);
    Py_VISIT(o->ags_sendval);
    return 0;
}


static PyObject *
__Pyx_async_gen_asend_send(PyObject *g, PyObject *arg)
{
    __pyx_PyAsyncGenASend *o = (__pyx_PyAsyncGenASend*) g;
    PyObject *result;

    if (unlikely(o->ags_state == __PYX_AWAITABLE_STATE_CLOSED)) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    if (o->ags_state == __PYX_AWAITABLE_STATE_INIT) {
        if (arg == NULL || arg == Py_None) {
            arg = o->ags_sendval ? o->ags_sendval : Py_None;
        }
        o->ags_state = __PYX_AWAITABLE_STATE_ITER;
    }

    result = __Pyx_Coroutine_Send((PyObject*)o->ags_gen, arg);
    result = __Pyx_async_gen_unwrap_value(o->ags_gen, result);

    if (result == NULL) {
        o->ags_state = __PYX_AWAITABLE_STATE_CLOSED;
    }

    return result;
}


static CYTHON_INLINE PyObject *
__Pyx_async_gen_asend_iternext(PyObject *o)
{
    return __Pyx_async_gen_asend_send(o, Py_None);
}


static PyObject *
__Pyx_async_gen_asend_throw(__pyx_PyAsyncGenASend *o, PyObject *args)
{
    PyObject *result;

    if (unlikely(o->ags_state == __PYX_AWAITABLE_STATE_CLOSED)) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    result = __Pyx_Coroutine_Throw((PyObject*)o->ags_gen, args);
    result = __Pyx_async_gen_unwrap_value(o->ags_gen, result);

    if (result == NULL) {
        o->ags_state = __PYX_AWAITABLE_STATE_CLOSED;
    }

    return result;
}


static PyObject *
__Pyx_async_gen_asend_close(PyObject *g, CYTHON_UNUSED PyObject *args)
{
    __pyx_PyAsyncGenASend *o = (__pyx_PyAsyncGenASend*) g;
    o->ags_state = __PYX_AWAITABLE_STATE_CLOSED;
    Py_RETURN_NONE;
}


static PyMethodDef __Pyx_async_gen_asend_methods[] = {
    {"send", (PyCFunction)__Pyx_async_gen_asend_send, METH_O, __Pyx_async_gen_send_doc},
    {"throw", (PyCFunction)__Pyx_async_gen_asend_throw, METH_VARARGS, __Pyx_async_gen_throw_doc},
    {"close", (PyCFunction)__Pyx_async_gen_asend_close, METH_NOARGS, __Pyx_async_gen_close_doc},
    {"__await__", (PyCFunction)__Pyx_async_gen_self_method, METH_NOARGS, __Pyx_async_gen_await_doc},
    {0, 0, 0, 0}        /* Sentinel */
};


#if CYTHON_USE_ASYNC_SLOTS
static __Pyx_PyAsyncMethodsStruct __Pyx_async_gen_asend_as_async = {
    PyObject_SelfIter,                          /* am_await */
    0,                                          /* am_aiter */
    0,                                          /* am_anext */
#if PY_VERSION_HEX >= 0x030A00A3
    0, /*am_send*/
#endif
};
#endif


static PyTypeObject __pyx__PyAsyncGenASendType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "async_generator_asend",                    /* tp_name */
    sizeof(__pyx_PyAsyncGenASend),                    /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)__Pyx_async_gen_asend_dealloc,        /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if CYTHON_USE_ASYNC_SLOTS
    &__Pyx_async_gen_asend_as_async,                  /* tp_as_async */
#else
    0,                                          /*tp_reserved*/
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,    /* tp_flags */
    0,                                          /* tp_doc */
    (traverseproc)__Pyx_async_gen_asend_traverse,  /* tp_traverse */
    0,                                          /* tp_clear */
#if CYTHON_USE_ASYNC_SLOTS && CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3 && PY_VERSION_HEX < 0x030500B1
    // in order to (mis-)use tp_reserved above, we must also implement tp_richcompare
    __Pyx_Coroutine_compare,            /*tp_richcompare*/
#else
    0,                                  /*tp_richcompare*/
#endif
    0,                                          /* tp_weaklistoffset */
    PyObject_SelfIter,                          /* tp_iter */
    (iternextfunc)__Pyx_async_gen_asend_iternext,     /* tp_iternext */
    __Pyx_async_gen_asend_methods,                    /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
#if PY_VERSION_HEX >= 0x030400a1
    0,                                          /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                          /*tp_vectorcall*/
#endif
#if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
    0,                                          /*tp_print*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000
    0,                                          /*tp_pypy_flags*/
#endif
};


static PyObject *
__Pyx_async_gen_asend_new(__pyx_PyAsyncGenObject *gen, PyObject *sendval)
{
    __pyx_PyAsyncGenASend *o;
    if (__Pyx_ag_asend_freelist_free) {
        __Pyx_ag_asend_freelist_free--;
        o = __Pyx_ag_asend_freelist[__Pyx_ag_asend_freelist_free];
        _Py_NewReference((PyObject *)o);
    } else {
        o = PyObject_GC_New(__pyx_PyAsyncGenASend, __pyx__PyAsyncGenASendType);
        if (o == NULL) {
            return NULL;
        }
    }

    Py_INCREF(gen);
    o->ags_gen = gen;

    Py_XINCREF(sendval);
    o->ags_sendval = sendval;

    o->ags_state = __PYX_AWAITABLE_STATE_INIT;

    PyObject_GC_Track((PyObject*)o);
    return (PyObject*)o;
}


/* ---------- Async Generator Value Wrapper ------------ */


static void
__Pyx_async_gen_wrapped_val_dealloc(__pyx__PyAsyncGenWrappedValue *o)
{
    PyObject_GC_UnTrack((PyObject *)o);
    Py_CLEAR(o->agw_val);
    if (__Pyx_ag_value_freelist_free < _PyAsyncGen_MAXFREELIST) {
        assert(__pyx__PyAsyncGenWrappedValue_CheckExact(o));
        __Pyx_ag_value_freelist[__Pyx_ag_value_freelist_free++] = o;
    } else {
        PyObject_GC_Del(o);
    }
}


static int
__Pyx_async_gen_wrapped_val_traverse(__pyx__PyAsyncGenWrappedValue *o,
                                     visitproc visit, void *arg)
{
    Py_VISIT(o->agw_val);
    return 0;
}


static PyTypeObject __pyx__PyAsyncGenWrappedValueType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "async_generator_wrapped_value",            /* tp_name */
    sizeof(__pyx__PyAsyncGenWrappedValue),            /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)__Pyx_async_gen_wrapped_val_dealloc,  /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,    /* tp_flags */
    0,                                          /* tp_doc */
    (traverseproc)__Pyx_async_gen_wrapped_val_traverse,    /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
#if PY_VERSION_HEX >= 0x030400a1
    0,                                          /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                          /*tp_vectorcall*/
#endif
#if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
    0,                                          /*tp_print*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000
    0,                                          /*tp_pypy_flags*/
#endif
};


static PyObject *
__Pyx__PyAsyncGenValueWrapperNew(PyObject *val)
{
    // NOTE: steals a reference to val !
    __pyx__PyAsyncGenWrappedValue *o;
    assert(val);

    if (__Pyx_ag_value_freelist_free) {
        __Pyx_ag_value_freelist_free--;
        o = __Pyx_ag_value_freelist[__Pyx_ag_value_freelist_free];
        assert(__pyx__PyAsyncGenWrappedValue_CheckExact(o));
        _Py_NewReference((PyObject*)o);
    } else {
        o = PyObject_GC_New(__pyx__PyAsyncGenWrappedValue, __pyx__PyAsyncGenWrappedValueType);
        if (unlikely(!o)) {
            Py_DECREF(val);
            return NULL;
        }
    }
    o->agw_val = val;
    // no Py_INCREF(val) - steals reference!
    PyObject_GC_Track((PyObject*)o);
    return (PyObject*)o;
}


/* ---------- Async Generator AThrow awaitable ------------ */


static void
__Pyx_async_gen_athrow_dealloc(__pyx_PyAsyncGenAThrow *o)
{
    PyObject_GC_UnTrack((PyObject *)o);
    Py_CLEAR(o->agt_gen);
    Py_CLEAR(o->agt_args);
    PyObject_GC_Del(o);
}


static int
__Pyx_async_gen_athrow_traverse(__pyx_PyAsyncGenAThrow *o, visitproc visit, void *arg)
{
    Py_VISIT(o->agt_gen);
    Py_VISIT(o->agt_args);
    return 0;
}


static PyObject *
__Pyx_async_gen_athrow_send(__pyx_PyAsyncGenAThrow *o, PyObject *arg)
{
    __pyx_CoroutineObject *gen = (__pyx_CoroutineObject*)o->agt_gen;
    PyObject *retval;

    if (o->agt_state == __PYX_AWAITABLE_STATE_CLOSED) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    if (o->agt_state == __PYX_AWAITABLE_STATE_INIT) {
        if (o->agt_gen->ag_closed) {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }

        if (arg != Py_None) {
            PyErr_SetString(PyExc_RuntimeError, __Pyx_NON_INIT_CORO_MSG);
            return NULL;
        }

        o->agt_state = __PYX_AWAITABLE_STATE_ITER;

        if (o->agt_args == NULL) {
            /* aclose() mode */
            o->agt_gen->ag_closed = 1;

            retval = __Pyx__Coroutine_Throw((PyObject*)gen,
                                /* Do not close generator when
                                   PyExc_GeneratorExit is passed */
                                PyExc_GeneratorExit, NULL, NULL, NULL, 0);

            if (retval && __pyx__PyAsyncGenWrappedValue_CheckExact(retval)) {
                Py_DECREF(retval);
                goto yield_close;
            }
        } else {
            PyObject *typ;
            PyObject *tb = NULL;
            PyObject *val = NULL;

            if (!PyArg_UnpackTuple(o->agt_args, "athrow", 1, 3,
                                   &typ, &val, &tb)) {
                return NULL;
            }

            retval = __Pyx__Coroutine_Throw((PyObject*)gen,
                                /* Do not close generator when PyExc_GeneratorExit is passed */
                                typ, val, tb, o->agt_args, 0);
            retval = __Pyx_async_gen_unwrap_value(o->agt_gen, retval);
        }
        if (retval == NULL) {
            goto check_error;
        }
        return retval;
    }

    assert (o->agt_state == __PYX_AWAITABLE_STATE_ITER);

    retval = __Pyx_Coroutine_Send((PyObject *)gen, arg);
    if (o->agt_args) {
        return __Pyx_async_gen_unwrap_value(o->agt_gen, retval);
    } else {
        /* aclose() mode */
        if (retval) {
            if (__pyx__PyAsyncGenWrappedValue_CheckExact(retval)) {
                Py_DECREF(retval);
                goto yield_close;
            }
            else {
                return retval;
            }
        }
        else {
            goto check_error;
        }
    }

yield_close:
    PyErr_SetString(
        PyExc_RuntimeError, __Pyx_ASYNC_GEN_IGNORED_EXIT_MSG);
    return NULL;

check_error:
    if (PyErr_ExceptionMatches(__Pyx_PyExc_StopAsyncIteration)) {
        o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
        if (o->agt_args == NULL) {
            // when aclose() is called we don't want to propagate
            // StopAsyncIteration; just raise StopIteration, signalling
            // that 'aclose()' is done.
            PyErr_Clear();
            PyErr_SetNone(PyExc_StopIteration);
        }
    }
    else if (PyErr_ExceptionMatches(PyExc_GeneratorExit)) {
        o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
        PyErr_Clear();          /* ignore these errors */
        PyErr_SetNone(PyExc_StopIteration);
    }
    return NULL;
}


static PyObject *
__Pyx_async_gen_athrow_throw(__pyx_PyAsyncGenAThrow *o, PyObject *args)
{
    PyObject *retval;

    if (o->agt_state == __PYX_AWAITABLE_STATE_INIT) {
        PyErr_SetString(PyExc_RuntimeError, __Pyx_NON_INIT_CORO_MSG);
        return NULL;
    }

    if (o->agt_state == __PYX_AWAITABLE_STATE_CLOSED) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    retval = __Pyx_Coroutine_Throw((PyObject*)o->agt_gen, args);
    if (o->agt_args) {
        return __Pyx_async_gen_unwrap_value(o->agt_gen, retval);
    } else {
        /* aclose() mode */
        if (retval && __pyx__PyAsyncGenWrappedValue_CheckExact(retval)) {
            Py_DECREF(retval);
            PyErr_SetString(PyExc_RuntimeError, __Pyx_ASYNC_GEN_IGNORED_EXIT_MSG);
            return NULL;
        }
        return retval;
    }
}


static PyObject *
__Pyx_async_gen_athrow_iternext(__pyx_PyAsyncGenAThrow *o)
{
    return __Pyx_async_gen_athrow_send(o, Py_None);
}


static PyObject *
__Pyx_async_gen_athrow_close(PyObject *g, CYTHON_UNUSED PyObject *args)
{
    __pyx_PyAsyncGenAThrow *o = (__pyx_PyAsyncGenAThrow*) g;
    o->agt_state = __PYX_AWAITABLE_STATE_CLOSED;
    Py_RETURN_NONE;
}


static PyMethodDef __Pyx_async_gen_athrow_methods[] = {
    {"send", (PyCFunction)__Pyx_async_gen_athrow_send, METH_O, __Pyx_async_gen_send_doc},
    {"throw", (PyCFunction)__Pyx_async_gen_athrow_throw, METH_VARARGS, __Pyx_async_gen_throw_doc},
    {"close", (PyCFunction)__Pyx_async_gen_athrow_close, METH_NOARGS, __Pyx_async_gen_close_doc},
    {"__await__", (PyCFunction)__Pyx_async_gen_self_method, METH_NOARGS, __Pyx_async_gen_await_doc},
    {0, 0, 0, 0}        /* Sentinel */
};


#if CYTHON_USE_ASYNC_SLOTS
static __Pyx_PyAsyncMethodsStruct __Pyx_async_gen_athrow_as_async = {
    PyObject_SelfIter,                          /* am_await */
    0,                                          /* am_aiter */
    0,                                          /* am_anext */
#if PY_VERSION_HEX >= 0x030A00A3
    0, /*am_send*/
#endif
};
#endif


static PyTypeObject __pyx__PyAsyncGenAThrowType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "async_generator_athrow",                   /* tp_name */
    sizeof(__pyx_PyAsyncGenAThrow),                   /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)__Pyx_async_gen_athrow_dealloc,       /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if CYTHON_USE_ASYNC_SLOTS
    &__Pyx_async_gen_athrow_as_async,                 /* tp_as_async */
#else
    0,                                          /*tp_reserved*/
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,    /* tp_flags */
    0,                                          /* tp_doc */
    (traverseproc)__Pyx_async_gen_athrow_traverse,    /* tp_traverse */
    0,                                          /* tp_clear */
#if CYTHON_USE_ASYNC_SLOTS && CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3 && PY_VERSION_HEX < 0x030500B1
    // in order to (mis-)use tp_reserved above, we must also implement tp_richcompare
    __Pyx_Coroutine_compare,            /*tp_richcompare*/
#else
    0,                                  /*tp_richcompare*/
#endif
    0,                                          /* tp_weaklistoffset */
    PyObject_SelfIter,                          /* tp_iter */
    (iternextfunc)__Pyx_async_gen_athrow_iternext,    /* tp_iternext */
    __Pyx_async_gen_athrow_methods,                   /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
#if PY_VERSION_HEX >= 0x030400a1
    0,                                          /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x030800b1 && (!CYTHON_COMPILING_IN_PYPY || PYPY_VERSION_NUM >= 0x07030800)
    0,                                          /*tp_vectorcall*/
#endif
#if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
    0,                                          /*tp_print*/
#endif
#if CYTHON_COMPILING_IN_PYPY && PY_VERSION_HEX >= 0x03090000
    0,                                          /*tp_pypy_flags*/
#endif
};


static PyObject *
__Pyx_async_gen_athrow_new(__pyx_PyAsyncGenObject *gen, PyObject *args)
{
    __pyx_PyAsyncGenAThrow *o;
    o = PyObject_GC_New(__pyx_PyAsyncGenAThrow, __pyx__PyAsyncGenAThrowType);
    if (o == NULL) {
        return NULL;
    }
    o->agt_gen = gen;
    o->agt_args = args;
    o->agt_state = __PYX_AWAITABLE_STATE_INIT;
    Py_INCREF(gen);
    Py_XINCREF(args);
    PyObject_GC_Track((PyObject*)o);
    return (PyObject*)o;
}


/* ---------- global type sharing ------------ */

static int __pyx_AsyncGen_init(void) {
    // on Windows, C-API functions can't be used in slots statically
    __pyx_AsyncGenType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx__PyAsyncGenWrappedValueType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx__PyAsyncGenAThrowType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx__PyAsyncGenASendType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;

    __pyx_AsyncGenType = __Pyx_FetchCommonType(&__pyx_AsyncGenType_type);
    if (unlikely(!__pyx_AsyncGenType))
        return -1;

    __pyx__PyAsyncGenAThrowType = __Pyx_FetchCommonType(&__pyx__PyAsyncGenAThrowType_type);
    if (unlikely(!__pyx__PyAsyncGenAThrowType))
        return -1;

    __pyx__PyAsyncGenWrappedValueType = __Pyx_FetchCommonType(&__pyx__PyAsyncGenWrappedValueType_type);
    if (unlikely(!__pyx__PyAsyncGenWrappedValueType))
        return -1;

    __pyx__PyAsyncGenASendType = __Pyx_FetchCommonType(&__pyx__PyAsyncGenASendType_type);
    if (unlikely(!__pyx__PyAsyncGenASendType))
        return -1;

    return 0;
}
