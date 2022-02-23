//////////////////// GeneratorYieldFrom.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_Generator_Yield_From(__pyx_CoroutineObject *gen, PyObject *source);

//////////////////// GeneratorYieldFrom ////////////////////
//@requires: Generator

static void __PyxPyIter_CheckErrorAndDecref(PyObject *source) {
    PyErr_Format(PyExc_TypeError,
                 "iter() returned non-iterator of type '%.100s'",
                 Py_TYPE(source)->tp_name);
    Py_DECREF(source);
}

static CYTHON_INLINE PyObject* __Pyx_Generator_Yield_From(__pyx_CoroutineObject *gen, PyObject *source) {
    PyObject *source_gen, *retval;
#ifdef __Pyx_Coroutine_USED
    if (__Pyx_Coroutine_Check(source)) {
        // TODO: this should only happen for types.coroutine()ed generators, but we can't determine that here
        Py_INCREF(source);
        source_gen = source;
        retval = __Pyx_Generator_Next(source);
    } else
#endif
    {
#if CYTHON_USE_TYPE_SLOTS
        if (likely(Py_TYPE(source)->tp_iter)) {
            source_gen = Py_TYPE(source)->tp_iter(source);
            if (unlikely(!source_gen))
                return NULL;
            if (unlikely(!PyIter_Check(source_gen))) {
                __PyxPyIter_CheckErrorAndDecref(source_gen);
                return NULL;
            }
        } else
        // CPython also allows non-iterable sequences to be iterated over
#endif
        {
            source_gen = PyObject_GetIter(source);
            if (unlikely(!source_gen))
                return NULL;
        }
        // source_gen is now the iterator, make the first next() call
#if CYTHON_USE_TYPE_SLOTS
        retval = Py_TYPE(source_gen)->tp_iternext(source_gen);
#else
        retval = PyIter_Next(source_gen);
#endif
    }
    if (likely(retval)) {
        gen->yieldfrom = source_gen;
        return retval;
    }
    Py_DECREF(source_gen);
    return NULL;
}


//////////////////// CoroutineYieldFrom.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_Coroutine_Yield_From(__pyx_CoroutineObject *gen, PyObject *source);

//////////////////// CoroutineYieldFrom ////////////////////
//@requires: Coroutine
//@requires: GetAwaitIter

static PyObject* __Pyx__Coroutine_Yield_From_Generic(__pyx_CoroutineObject *gen, PyObject *source) {
    PyObject *retval;
    PyObject *source_gen = __Pyx__Coroutine_GetAwaitableIter(source);
    if (unlikely(!source_gen)) {
        return NULL;
    }
    // source_gen is now the iterator, make the first next() call
    if (__Pyx_Coroutine_Check(source_gen)) {
        retval = __Pyx_Generator_Next(source_gen);
    } else {
#if CYTHON_USE_TYPE_SLOTS
        retval = Py_TYPE(source_gen)->tp_iternext(source_gen);
#else
        retval = PyIter_Next(source_gen);
#endif
    }
    if (retval) {
        gen->yieldfrom = source_gen;
        return retval;
    }
    Py_DECREF(source_gen);
    return NULL;
}

static CYTHON_INLINE PyObject* __Pyx_Coroutine_Yield_From(__pyx_CoroutineObject *gen, PyObject *source) {
    PyObject *retval;
    if (__Pyx_Coroutine_Check(source)) {
        if (unlikely(((__pyx_CoroutineObject*)source)->yieldfrom)) {
            PyErr_SetString(
                PyExc_RuntimeError,
                "coroutine is being awaited already");
            return NULL;
        }
        retval = __Pyx_Generator_Next(source);
#ifdef __Pyx_AsyncGen_USED
    // inlined "__pyx_PyAsyncGenASend" handling to avoid the series of generic calls
    } else if (__pyx_PyAsyncGenASend_CheckExact(source)) {
        retval = __Pyx_async_gen_asend_iternext(source);
#endif
    } else {
        return __Pyx__Coroutine_Yield_From_Generic(gen, source);
    }
    if (retval) {
        Py_INCREF(source);
        gen->yieldfrom = source;
    }
    return retval;
}


//////////////////// GetAwaitIter.proto ////////////////////

static CYTHON_INLINE PyObject *__Pyx_Coroutine_GetAwaitableIter(PyObject *o); /*proto*/
static PyObject *__Pyx__Coroutine_GetAwaitableIter(PyObject *o); /*proto*/

//////////////////// GetAwaitIter ////////////////////
//@requires: ObjectHandling.c::PyObjectGetMethod
//@requires: ObjectHandling.c::PyObjectCallNoArg
//@requires: ObjectHandling.c::PyObjectCallOneArg

static CYTHON_INLINE PyObject *__Pyx_Coroutine_GetAwaitableIter(PyObject *o) {
#ifdef __Pyx_Coroutine_USED
    if (__Pyx_Coroutine_Check(o)) {
        return __Pyx_NewRef(o);
    }
#endif
    return __Pyx__Coroutine_GetAwaitableIter(o);
}


static void __Pyx_Coroutine_AwaitableIterError(PyObject *source) {
#if PY_VERSION_HEX >= 0x030600B3 || defined(_PyErr_FormatFromCause)
    _PyErr_FormatFromCause(
        PyExc_TypeError,
        "'async for' received an invalid object "
        "from __anext__: %.100s",
        Py_TYPE(source)->tp_name);
#elif PY_MAJOR_VERSION >= 3
    PyObject *exc, *val, *val2, *tb;
    assert(PyErr_Occurred());
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_NormalizeException(&exc, &val, &tb);
    if (tb != NULL) {
        PyException_SetTraceback(val, tb);
        Py_DECREF(tb);
    }
    Py_DECREF(exc);
    assert(!PyErr_Occurred());
    PyErr_Format(
        PyExc_TypeError,
        "'async for' received an invalid object "
        "from __anext__: %.100s",
        Py_TYPE(source)->tp_name);

    PyErr_Fetch(&exc, &val2, &tb);
    PyErr_NormalizeException(&exc, &val2, &tb);
    Py_INCREF(val);
    PyException_SetCause(val2, val);
    PyException_SetContext(val2, val);
    PyErr_Restore(exc, val2, tb);
#else
    // since Py2 does not have exception chaining, it's better to avoid shadowing exceptions there
    source++;
#endif
}

// adapted from genobject.c in Py3.5
static PyObject *__Pyx__Coroutine_GetAwaitableIter(PyObject *obj) {
    PyObject *res;
#if CYTHON_USE_ASYNC_SLOTS
    __Pyx_PyAsyncMethodsStruct* am = __Pyx_PyType_AsAsync(obj);
    if (likely(am && am->am_await)) {
        res = (*am->am_await)(obj);
    } else
#endif
#if PY_VERSION_HEX >= 0x030500B2 || defined(PyCoro_CheckExact)
    if (PyCoro_CheckExact(obj)) {
        return __Pyx_NewRef(obj);
    } else
#endif
#if CYTHON_COMPILING_IN_CPYTHON && defined(CO_ITERABLE_COROUTINE)
    if (PyGen_CheckExact(obj) && ((PyGenObject*)obj)->gi_code && ((PyCodeObject *)((PyGenObject*)obj)->gi_code)->co_flags & CO_ITERABLE_COROUTINE) {
        // Python generator marked with "@types.coroutine" decorator
        return __Pyx_NewRef(obj);
    } else
#endif
    {
        PyObject *method = NULL;
        int is_method = __Pyx_PyObject_GetMethod(obj, PYIDENT("__await__"), &method);
        if (likely(is_method)) {
            res = __Pyx_PyObject_CallOneArg(method, obj);
        } else if (likely(method)) {
            res = __Pyx_PyObject_CallNoArg(method);
        } else
            goto slot_error;
        Py_DECREF(method);
    }
    if (unlikely(!res)) {
        // surprisingly, CPython replaces the exception here...
        __Pyx_Coroutine_AwaitableIterError(obj);
        goto bad;
    }
    if (unlikely(!PyIter_Check(res))) {
        PyErr_Format(PyExc_TypeError,
                     "__await__() returned non-iterator of type '%.100s'",
                     Py_TYPE(res)->tp_name);
        Py_CLEAR(res);
    } else {
        int is_coroutine = 0;
        #ifdef __Pyx_Coroutine_USED
        is_coroutine |= __Pyx_Coroutine_Check(res);
        #endif
        #if PY_VERSION_HEX >= 0x030500B2 || defined(PyCoro_CheckExact)
        is_coroutine |= PyCoro_CheckExact(res);
        #endif
        if (unlikely(is_coroutine)) {
            /* __await__ must return an *iterator*, not
               a coroutine or another awaitable (see PEP 492) */
            PyErr_SetString(PyExc_TypeError,
                            "__await__() returned a coroutine");
            Py_CLEAR(res);
        }
    }
    return res;
slot_error:
    PyErr_Format(PyExc_TypeError,
                 "object %.100s can't be used in 'await' expression",
                 Py_TYPE(obj)->tp_name);
bad:
    return NULL;
}


//////////////////// AsyncIter.proto ////////////////////

static CYTHON_INLINE PyObject *__Pyx_Coroutine_GetAsyncIter(PyObject *o); /*proto*/
static CYTHON_INLINE PyObject *__Pyx_Coroutine_AsyncIterNext(PyObject *o); /*proto*/

//////////////////// AsyncIter ////////////////////
//@requires: GetAwaitIter
//@requires: ObjectHandling.c::PyObjectCallMethod0

static PyObject *__Pyx_Coroutine_GetAsyncIter_Generic(PyObject *obj) {
#if PY_VERSION_HEX < 0x030500B1
    {
        PyObject *iter = __Pyx_PyObject_CallMethod0(obj, PYIDENT("__aiter__"));
        if (likely(iter))
            return iter;
        // FIXME: for the sake of a nicely conforming exception message, assume any AttributeError meant '__aiter__'
        if (!PyErr_ExceptionMatches(PyExc_AttributeError))
            return NULL;
    }
#else
    // avoid C warning about 'unused function'
    if ((0)) (void) __Pyx_PyObject_CallMethod0(obj, PYIDENT("__aiter__"));
#endif

    PyErr_Format(PyExc_TypeError, "'async for' requires an object with __aiter__ method, got %.100s",
                 Py_TYPE(obj)->tp_name);
    return NULL;
}


static CYTHON_INLINE PyObject *__Pyx_Coroutine_GetAsyncIter(PyObject *obj) {
#ifdef __Pyx_AsyncGen_USED
    if (__Pyx_AsyncGen_CheckExact(obj)) {
        return __Pyx_NewRef(obj);
    }
#endif
#if CYTHON_USE_ASYNC_SLOTS
    {
        __Pyx_PyAsyncMethodsStruct* am = __Pyx_PyType_AsAsync(obj);
        if (likely(am && am->am_aiter)) {
            return (*am->am_aiter)(obj);
        }
    }
#endif
    return __Pyx_Coroutine_GetAsyncIter_Generic(obj);
}


static PyObject *__Pyx__Coroutine_AsyncIterNext(PyObject *obj) {
#if PY_VERSION_HEX < 0x030500B1
    {
        PyObject *value = __Pyx_PyObject_CallMethod0(obj, PYIDENT("__anext__"));
        if (likely(value))
            return value;
    }
    // FIXME: for the sake of a nicely conforming exception message, assume any AttributeError meant '__anext__'
    if (PyErr_ExceptionMatches(PyExc_AttributeError))
#endif
        PyErr_Format(PyExc_TypeError, "'async for' requires an object with __anext__ method, got %.100s",
                     Py_TYPE(obj)->tp_name);
    return NULL;
}


static CYTHON_INLINE PyObject *__Pyx_Coroutine_AsyncIterNext(PyObject *obj) {
#ifdef __Pyx_AsyncGen_USED
    if (__Pyx_AsyncGen_CheckExact(obj)) {
        return __Pyx_async_gen_anext(obj);
    }
#endif
#if CYTHON_USE_ASYNC_SLOTS
    {
        __Pyx_PyAsyncMethodsStruct* am = __Pyx_PyType_AsAsync(obj);
        if (likely(am && am->am_anext)) {
            return (*am->am_anext)(obj);
        }
    }
#endif
    return __Pyx__Coroutine_AsyncIterNext(obj);
}


//////////////////// pep479.proto ////////////////////

static void __Pyx_Generator_Replace_StopIteration(int in_async_gen); /*proto*/

//////////////////// pep479 ////////////////////
//@requires: Exceptions.c::GetException

static void __Pyx_Generator_Replace_StopIteration(CYTHON_UNUSED int in_async_gen) {
    PyObject *exc, *val, *tb, *cur_exc;
    __Pyx_PyThreadState_declare
    #ifdef __Pyx_StopAsyncIteration_USED
    int is_async_stopiteration = 0;
    #endif

    cur_exc = PyErr_Occurred();
    if (likely(!__Pyx_PyErr_GivenExceptionMatches(cur_exc, PyExc_StopIteration))) {
        #ifdef __Pyx_StopAsyncIteration_USED
        if (in_async_gen && unlikely(__Pyx_PyErr_GivenExceptionMatches(cur_exc, __Pyx_PyExc_StopAsyncIteration))) {
            is_async_stopiteration = 1;
        } else
        #endif
            return;
    }

    __Pyx_PyThreadState_assign
    // Chain exceptions by moving Stop(Async)Iteration to exc_info before creating the RuntimeError.
    // In Py2.x, no chaining happens, but the exception still stays visible in exc_info.
    __Pyx_GetException(&exc, &val, &tb);
    Py_XDECREF(exc);
    Py_XDECREF(val);
    Py_XDECREF(tb);
    PyErr_SetString(PyExc_RuntimeError,
        #ifdef __Pyx_StopAsyncIteration_USED
        is_async_stopiteration ? "async generator raised StopAsyncIteration" :
        in_async_gen ? "async generator raised StopIteration" :
        #endif
        "generator raised StopIteration");
}


//////////////////// CoroutineBase.proto ////////////////////
//@substitute: naming

typedef PyObject *(*__pyx_coroutine_body_t)(PyObject *, PyThreadState *, PyObject *);

#if CYTHON_USE_EXC_INFO_STACK
// See  https://bugs.python.org/issue25612
#define __Pyx_ExcInfoStruct  _PyErr_StackItem
#else
// Minimal replacement struct for Py<3.7, without the Py3.7 exception state stack.
typedef struct {
    PyObject *exc_type;
    PyObject *exc_value;
    PyObject *exc_traceback;
} __Pyx_ExcInfoStruct;
#endif

typedef struct {
    PyObject_HEAD
    __pyx_coroutine_body_t body;
    PyObject *closure;
    __Pyx_ExcInfoStruct gi_exc_state;
    PyObject *gi_weakreflist;
    PyObject *classobj;
    PyObject *yieldfrom;
    PyObject *gi_name;
    PyObject *gi_qualname;
    PyObject *gi_modulename;
    PyObject *gi_code;
    PyObject *gi_frame;
    int resume_label;
    // using T_BOOL for property below requires char value
    char is_running;
} __pyx_CoroutineObject;

static __pyx_CoroutineObject *__Pyx__Coroutine_New(
    PyTypeObject *type, __pyx_coroutine_body_t body, PyObject *code, PyObject *closure,
    PyObject *name, PyObject *qualname, PyObject *module_name); /*proto*/

static __pyx_CoroutineObject *__Pyx__Coroutine_NewInit(
            __pyx_CoroutineObject *gen, __pyx_coroutine_body_t body, PyObject *code, PyObject *closure,
            PyObject *name, PyObject *qualname, PyObject *module_name); /*proto*/

static CYTHON_INLINE void __Pyx_Coroutine_ExceptionClear(__Pyx_ExcInfoStruct *self);
static int __Pyx_Coroutine_clear(PyObject *self); /*proto*/
static PyObject *__Pyx_Coroutine_Send(PyObject *self, PyObject *value); /*proto*/
static PyObject *__Pyx_Coroutine_Close(PyObject *self); /*proto*/
static PyObject *__Pyx_Coroutine_Throw(PyObject *gen, PyObject *args); /*proto*/

// macros for exception state swapping instead of inline functions to make use of the local thread state context
#if CYTHON_USE_EXC_INFO_STACK
#define __Pyx_Coroutine_SwapException(self)
#define __Pyx_Coroutine_ResetAndClearException(self)  __Pyx_Coroutine_ExceptionClear(&(self)->gi_exc_state)
#else
#define __Pyx_Coroutine_SwapException(self) { \
    __Pyx_ExceptionSwap(&(self)->gi_exc_state.exc_type, &(self)->gi_exc_state.exc_value, &(self)->gi_exc_state.exc_traceback); \
    __Pyx_Coroutine_ResetFrameBackpointer(&(self)->gi_exc_state); \
    }
#define __Pyx_Coroutine_ResetAndClearException(self) { \
    __Pyx_ExceptionReset((self)->gi_exc_state.exc_type, (self)->gi_exc_state.exc_value, (self)->gi_exc_state.exc_traceback); \
    (self)->gi_exc_state.exc_type = (self)->gi_exc_state.exc_value = (self)->gi_exc_state.exc_traceback = NULL; \
    }
#endif

#if CYTHON_FAST_THREAD_STATE
#define __Pyx_PyGen_FetchStopIterationValue(pvalue) \
    __Pyx_PyGen__FetchStopIterationValue($local_tstate_cname, pvalue)
#else
#define __Pyx_PyGen_FetchStopIterationValue(pvalue) \
    __Pyx_PyGen__FetchStopIterationValue(__Pyx_PyThreadState_Current, pvalue)
#endif
static int __Pyx_PyGen__FetchStopIterationValue(PyThreadState *tstate, PyObject **pvalue); /*proto*/
static CYTHON_INLINE void __Pyx_Coroutine_ResetFrameBackpointer(__Pyx_ExcInfoStruct *exc_state); /*proto*/


//////////////////// Coroutine.proto ////////////////////

#define __Pyx_Coroutine_USED
static PyTypeObject *__pyx_CoroutineType = 0;
static PyTypeObject *__pyx_CoroutineAwaitType = 0;
#define __Pyx_Coroutine_CheckExact(obj) (Py_TYPE(obj) == __pyx_CoroutineType)
// __Pyx_Coroutine_Check(obj): see override for IterableCoroutine below
#define __Pyx_Coroutine_Check(obj) __Pyx_Coroutine_CheckExact(obj)
#define __Pyx_CoroutineAwait_CheckExact(obj) (Py_TYPE(obj) == __pyx_CoroutineAwaitType)

#define __Pyx_Coroutine_New(body, code, closure, name, qualname, module_name)  \
    __Pyx__Coroutine_New(__pyx_CoroutineType, body, code, closure, name, qualname, module_name)

static int __pyx_Coroutine_init(void); /*proto*/
static PyObject *__Pyx__Coroutine_await(PyObject *coroutine); /*proto*/

typedef struct {
    PyObject_HEAD
    PyObject *coroutine;
} __pyx_CoroutineAwaitObject;

static PyObject *__Pyx_CoroutineAwait_Close(__pyx_CoroutineAwaitObject *self, PyObject *arg); /*proto*/
static PyObject *__Pyx_CoroutineAwait_Throw(__pyx_CoroutineAwaitObject *self, PyObject *args); /*proto*/


//////////////////// Generator.proto ////////////////////

#define __Pyx_Generator_USED
static PyTypeObject *__pyx_GeneratorType = 0;
#define __Pyx_Generator_CheckExact(obj) (Py_TYPE(obj) == __pyx_GeneratorType)

#define __Pyx_Generator_New(body, code, closure, name, qualname, module_name)  \
    __Pyx__Coroutine_New(__pyx_GeneratorType, body, code, closure, name, qualname, module_name)

static PyObject *__Pyx_Generator_Next(PyObject *self);
static int __pyx_Generator_init(void); /*proto*/


//////////////////// AsyncGen ////////////////////
//@requires: AsyncGen.c::AsyncGenerator
// -> empty, only delegates to separate file


//////////////////// CoroutineBase ////////////////////
//@substitute: naming
//@requires: Exceptions.c::PyErrFetchRestore
//@requires: Exceptions.c::PyThreadStateGet
//@requires: Exceptions.c::SwapException
//@requires: Exceptions.c::RaiseException
//@requires: Exceptions.c::SaveResetException
//@requires: ObjectHandling.c::PyObjectCallMethod1
//@requires: ObjectHandling.c::PyObjectGetAttrStr
//@requires: CommonStructures.c::FetchCommonType

#include <structmember.h>
#include <frameobject.h>

#define __Pyx_Coroutine_Undelegate(gen) Py_CLEAR((gen)->yieldfrom)

//   If StopIteration exception is set, fetches its 'value'
//   attribute if any, otherwise sets pvalue to None.
//
//   Returns 0 if no exception or StopIteration is set.
//   If any other exception is set, returns -1 and leaves
//   pvalue unchanged.
static int __Pyx_PyGen__FetchStopIterationValue(CYTHON_UNUSED PyThreadState *$local_tstate_cname, PyObject **pvalue) {
    PyObject *et, *ev, *tb;
    PyObject *value = NULL;

    __Pyx_ErrFetch(&et, &ev, &tb);

    if (!et) {
        Py_XDECREF(tb);
        Py_XDECREF(ev);
        Py_INCREF(Py_None);
        *pvalue = Py_None;
        return 0;
    }

    // most common case: plain StopIteration without or with separate argument
    if (likely(et == PyExc_StopIteration)) {
        if (!ev) {
            Py_INCREF(Py_None);
            value = Py_None;
        }
#if PY_VERSION_HEX >= 0x030300A0
        else if (Py_TYPE(ev) == (PyTypeObject*)PyExc_StopIteration) {
            value = ((PyStopIterationObject *)ev)->value;
            Py_INCREF(value);
            Py_DECREF(ev);
        }
#endif
        // PyErr_SetObject() and friends put the value directly into ev
        else if (unlikely(PyTuple_Check(ev))) {
            // if it's a tuple, it is interpreted as separate constructor arguments (surprise!)
            if (PyTuple_GET_SIZE(ev) >= 1) {
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
                value = PyTuple_GET_ITEM(ev, 0);
                Py_INCREF(value);
#else
                value = PySequence_ITEM(ev, 0);
#endif
            } else {
                Py_INCREF(Py_None);
                value = Py_None;
            }
            Py_DECREF(ev);
        }
        else if (!__Pyx_TypeCheck(ev, (PyTypeObject*)PyExc_StopIteration)) {
            // 'steal' reference to ev
            value = ev;
        }
        if (likely(value)) {
            Py_XDECREF(tb);
            Py_DECREF(et);
            *pvalue = value;
            return 0;
        }
    } else if (!__Pyx_PyErr_GivenExceptionMatches(et, PyExc_StopIteration)) {
        __Pyx_ErrRestore(et, ev, tb);
        return -1;
    }

    // otherwise: normalise and check what that gives us
    PyErr_NormalizeException(&et, &ev, &tb);
    if (unlikely(!PyObject_TypeCheck(ev, (PyTypeObject*)PyExc_StopIteration))) {
        // looks like normalisation failed - raise the new exception
        __Pyx_ErrRestore(et, ev, tb);
        return -1;
    }
    Py_XDECREF(tb);
    Py_DECREF(et);
#if PY_VERSION_HEX >= 0x030300A0
    value = ((PyStopIterationObject *)ev)->value;
    Py_INCREF(value);
    Py_DECREF(ev);
#else
    {
        PyObject* args = __Pyx_PyObject_GetAttrStr(ev, PYIDENT("args"));
        Py_DECREF(ev);
        if (likely(args)) {
            value = PySequence_GetItem(args, 0);
            Py_DECREF(args);
        }
        if (unlikely(!value)) {
            __Pyx_ErrRestore(NULL, NULL, NULL);
            Py_INCREF(Py_None);
            value = Py_None;
        }
    }
#endif
    *pvalue = value;
    return 0;
}

static CYTHON_INLINE
void __Pyx_Coroutine_ExceptionClear(__Pyx_ExcInfoStruct *exc_state) {
    PyObject *t, *v, *tb;
    t = exc_state->exc_type;
    v = exc_state->exc_value;
    tb = exc_state->exc_traceback;

    exc_state->exc_type = NULL;
    exc_state->exc_value = NULL;
    exc_state->exc_traceback = NULL;

    Py_XDECREF(t);
    Py_XDECREF(v);
    Py_XDECREF(tb);
}

#define __Pyx_Coroutine_AlreadyRunningError(gen)  (__Pyx__Coroutine_AlreadyRunningError(gen), (PyObject*)NULL)
static void __Pyx__Coroutine_AlreadyRunningError(CYTHON_UNUSED __pyx_CoroutineObject *gen) {
    const char *msg;
    if ((0)) {
    #ifdef __Pyx_Coroutine_USED
    } else if (__Pyx_Coroutine_Check((PyObject*)gen)) {
        msg = "coroutine already executing";
    #endif
    #ifdef __Pyx_AsyncGen_USED
    } else if (__Pyx_AsyncGen_CheckExact((PyObject*)gen)) {
        msg = "async generator already executing";
    #endif
    } else {
        msg = "generator already executing";
    }
    PyErr_SetString(PyExc_ValueError, msg);
}

#define __Pyx_Coroutine_NotStartedError(gen)  (__Pyx__Coroutine_NotStartedError(gen), (PyObject*)NULL)
static void __Pyx__Coroutine_NotStartedError(CYTHON_UNUSED PyObject *gen) {
    const char *msg;
    if ((0)) {
    #ifdef __Pyx_Coroutine_USED
    } else if (__Pyx_Coroutine_Check(gen)) {
        msg = "can't send non-None value to a just-started coroutine";
    #endif
    #ifdef __Pyx_AsyncGen_USED
    } else if (__Pyx_AsyncGen_CheckExact(gen)) {
        msg = "can't send non-None value to a just-started async generator";
    #endif
    } else {
        msg = "can't send non-None value to a just-started generator";
    }
    PyErr_SetString(PyExc_TypeError, msg);
}

#define __Pyx_Coroutine_AlreadyTerminatedError(gen, value, closing)  (__Pyx__Coroutine_AlreadyTerminatedError(gen, value, closing), (PyObject*)NULL)
static void __Pyx__Coroutine_AlreadyTerminatedError(CYTHON_UNUSED PyObject *gen, PyObject *value, CYTHON_UNUSED int closing) {
    #ifdef __Pyx_Coroutine_USED
    if (!closing && __Pyx_Coroutine_Check(gen)) {
        // `self` is an exhausted coroutine: raise an error,
        // except when called from gen_close(), which should
        // always be a silent method.
        PyErr_SetString(PyExc_RuntimeError, "cannot reuse already awaited coroutine");
    } else
    #endif
    if (value) {
        // `gen` is an exhausted generator:
        // only set exception if called from send().
        #ifdef __Pyx_AsyncGen_USED
        if (__Pyx_AsyncGen_CheckExact(gen))
            PyErr_SetNone(__Pyx_PyExc_StopAsyncIteration);
        else
        #endif
        PyErr_SetNone(PyExc_StopIteration);
    }
}

static
PyObject *__Pyx_Coroutine_SendEx(__pyx_CoroutineObject *self, PyObject *value, int closing) {
    __Pyx_PyThreadState_declare
    PyThreadState *tstate;
    __Pyx_ExcInfoStruct *exc_state;
    PyObject *retval;

    assert(!self->is_running);

    if (unlikely(self->resume_label == 0)) {
        if (unlikely(value && value != Py_None)) {
            return __Pyx_Coroutine_NotStartedError((PyObject*)self);
        }
    }

    if (unlikely(self->resume_label == -1)) {
        return __Pyx_Coroutine_AlreadyTerminatedError((PyObject*)self, value, closing);
    }

#if CYTHON_FAST_THREAD_STATE
    __Pyx_PyThreadState_assign
    tstate = $local_tstate_cname;
#else
    tstate = __Pyx_PyThreadState_Current;
#endif

    // Traceback/Frame rules pre-Py3.7:
    // - on entry, save external exception state in self->gi_exc_state, restore it on exit
    // - on exit, keep internally generated exceptions in self->gi_exc_state, clear everything else
    // - on entry, set "f_back" pointer of internal exception traceback to (current) outer call frame
    // - on exit, clear "f_back" of internal exception traceback
    // - do not touch external frames and tracebacks

    // Traceback/Frame rules for Py3.7+ (CYTHON_USE_EXC_INFO_STACK):
    // - on entry, push internal exception state in self->gi_exc_state on the exception stack
    // - on exit, keep internally generated exceptions in self->gi_exc_state, clear everything else
    // - on entry, set "f_back" pointer of internal exception traceback to (current) outer call frame
    // - on exit, clear "f_back" of internal exception traceback
    // - do not touch external frames and tracebacks

    exc_state = &self->gi_exc_state;
    if (exc_state->exc_type) {
        #if CYTHON_COMPILING_IN_PYPY || CYTHON_COMPILING_IN_PYSTON
        // FIXME: what to do in PyPy?
        #else
        // Generators always return to their most recent caller, not
        // necessarily their creator.
        if (exc_state->exc_traceback) {
            PyTracebackObject *tb = (PyTracebackObject *) exc_state->exc_traceback;
            PyFrameObject *f = tb->tb_frame;

            assert(f->f_back == NULL);
            #if PY_VERSION_HEX >= 0x030B00A1
            // PyThreadState_GetFrame returns NULL if there isn't a current frame
            // which is a valid state so no need to check
            f->f_back = PyThreadState_GetFrame(tstate);
            #else
            Py_XINCREF(tstate->frame);
            f->f_back = tstate->frame;
            #endif
        }
        #endif
    }

#if CYTHON_USE_EXC_INFO_STACK
    // See  https://bugs.python.org/issue25612
    exc_state->previous_item = tstate->exc_info;
    tstate->exc_info = exc_state;
#else
    if (exc_state->exc_type) {
        // We were in an except handler when we left,
        // restore the exception state which was put aside.
        __Pyx_ExceptionSwap(&exc_state->exc_type, &exc_state->exc_value, &exc_state->exc_traceback);
        // self->exc_* now holds the exception state of the caller
    } else {
        // save away the exception state of the caller
        __Pyx_Coroutine_ExceptionClear(exc_state);
        __Pyx_ExceptionSave(&exc_state->exc_type, &exc_state->exc_value, &exc_state->exc_traceback);
    }
#endif

    self->is_running = 1;
    retval = self->body((PyObject *) self, tstate, value);
    self->is_running = 0;

#if CYTHON_USE_EXC_INFO_STACK
    // See  https://bugs.python.org/issue25612
    exc_state = &self->gi_exc_state;
    tstate->exc_info = exc_state->previous_item;
    exc_state->previous_item = NULL;
    // Cut off the exception frame chain so that we can reconnect it on re-entry above.
    __Pyx_Coroutine_ResetFrameBackpointer(exc_state);
#endif

    return retval;
}

static CYTHON_INLINE void __Pyx_Coroutine_ResetFrameBackpointer(__Pyx_ExcInfoStruct *exc_state) {
    // Don't keep the reference to f_back any longer than necessary.  It
    // may keep a chain of frames alive or it could create a reference
    // cycle.
    PyObject *exc_tb = exc_state->exc_traceback;

    if (likely(exc_tb)) {
#if CYTHON_COMPILING_IN_PYPY || CYTHON_COMPILING_IN_PYSTON
    // FIXME: what to do in PyPy?
#else
        PyTracebackObject *tb = (PyTracebackObject *) exc_tb;
        PyFrameObject *f = tb->tb_frame;
        Py_CLEAR(f->f_back);
#endif
    }
}

static CYTHON_INLINE
PyObject *__Pyx_Coroutine_MethodReturn(CYTHON_UNUSED PyObject* gen, PyObject *retval) {
    if (unlikely(!retval)) {
        __Pyx_PyThreadState_declare
        __Pyx_PyThreadState_assign
        if (!__Pyx_PyErr_Occurred()) {
            // method call must not terminate with NULL without setting an exception
            PyObject *exc = PyExc_StopIteration;
            #ifdef __Pyx_AsyncGen_USED
            if (__Pyx_AsyncGen_CheckExact(gen))
                exc = __Pyx_PyExc_StopAsyncIteration;
            #endif
            __Pyx_PyErr_SetNone(exc);
        }
    }
    return retval;
}

#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x03030000 && (defined(__linux__) || PY_VERSION_HEX >= 0x030600B3)
static CYTHON_INLINE
PyObject *__Pyx_PyGen_Send(PyGenObject *gen, PyObject *arg) {
#if PY_VERSION_HEX <= 0x030A00A1
    return _PyGen_Send(gen, arg);
#else
    PyObject *result;
    // PyIter_Send() asserts non-NULL arg
    if (PyIter_Send((PyObject*)gen, arg ? arg : Py_None, &result) == PYGEN_RETURN) {
        if (PyAsyncGen_CheckExact(gen)) {
            assert(result == Py_None);
            PyErr_SetNone(PyExc_StopAsyncIteration);
        }
        else if (result == Py_None) {
            PyErr_SetNone(PyExc_StopIteration);
        }
        else {
            _PyGen_SetStopIterationValue(result);
        }
        Py_CLEAR(result);
    }
    return result;
#endif
}
#endif

static CYTHON_INLINE
PyObject *__Pyx_Coroutine_FinishDelegation(__pyx_CoroutineObject *gen) {
    PyObject *ret;
    PyObject *val = NULL;
    __Pyx_Coroutine_Undelegate(gen);
    __Pyx_PyGen__FetchStopIterationValue(__Pyx_PyThreadState_Current, &val);
    // val == NULL on failure => pass on exception
    ret = __Pyx_Coroutine_SendEx(gen, val, 0);
    Py_XDECREF(val);
    return ret;
}

static PyObject *__Pyx_Coroutine_Send(PyObject *self, PyObject *value) {
    PyObject *retval;
    __pyx_CoroutineObject *gen = (__pyx_CoroutineObject*) self;
    PyObject *yf = gen->yieldfrom;
    if (unlikely(gen->is_running))
        return __Pyx_Coroutine_AlreadyRunningError(gen);
    if (yf) {
        PyObject *ret;
        // FIXME: does this really need an INCREF() ?
        //Py_INCREF(yf);
        gen->is_running = 1;
        #ifdef __Pyx_Generator_USED
        if (__Pyx_Generator_CheckExact(yf)) {
            ret = __Pyx_Coroutine_Send(yf, value);
        } else
        #endif
        #ifdef __Pyx_Coroutine_USED
        if (__Pyx_Coroutine_Check(yf)) {
            ret = __Pyx_Coroutine_Send(yf, value);
        } else
        #endif
        #ifdef __Pyx_AsyncGen_USED
        if (__pyx_PyAsyncGenASend_CheckExact(yf)) {
            ret = __Pyx_async_gen_asend_send(yf, value);
        } else
        #endif
        #if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x03030000 && (defined(__linux__) || PY_VERSION_HEX >= 0x030600B3)
        // _PyGen_Send() is not exported before Py3.6
        if (PyGen_CheckExact(yf)) {
            ret = __Pyx_PyGen_Send((PyGenObject*)yf, value == Py_None ? NULL : value);
        } else
        #endif
        #if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x03050000 && defined(PyCoro_CheckExact) && (defined(__linux__) || PY_VERSION_HEX >= 0x030600B3)
        // _PyGen_Send() is not exported before Py3.6
        if (PyCoro_CheckExact(yf)) {
            ret = __Pyx_PyGen_Send((PyGenObject*)yf, value == Py_None ? NULL : value);
        } else
        #endif
        {
            if (value == Py_None)
                ret = Py_TYPE(yf)->tp_iternext(yf);
            else
                ret = __Pyx_PyObject_CallMethod1(yf, PYIDENT("send"), value);
        }
        gen->is_running = 0;
        //Py_DECREF(yf);
        if (likely(ret)) {
            return ret;
        }
        retval = __Pyx_Coroutine_FinishDelegation(gen);
    } else {
        retval = __Pyx_Coroutine_SendEx(gen, value, 0);
    }
    return __Pyx_Coroutine_MethodReturn(self, retval);
}

//   This helper function is used by gen_close and gen_throw to
//   close a subiterator being delegated to by yield-from.
static int __Pyx_Coroutine_CloseIter(__pyx_CoroutineObject *gen, PyObject *yf) {
    PyObject *retval = NULL;
    int err = 0;

    #ifdef __Pyx_Generator_USED
    if (__Pyx_Generator_CheckExact(yf)) {
        retval = __Pyx_Coroutine_Close(yf);
        if (!retval)
            return -1;
    } else
    #endif
    #ifdef __Pyx_Coroutine_USED
    if (__Pyx_Coroutine_Check(yf)) {
        retval = __Pyx_Coroutine_Close(yf);
        if (!retval)
            return -1;
    } else
    if (__Pyx_CoroutineAwait_CheckExact(yf)) {
        retval = __Pyx_CoroutineAwait_Close((__pyx_CoroutineAwaitObject*)yf, NULL);
        if (!retval)
            return -1;
    } else
    #endif
    #ifdef __Pyx_AsyncGen_USED
    if (__pyx_PyAsyncGenASend_CheckExact(yf)) {
        retval = __Pyx_async_gen_asend_close(yf, NULL);
        // cannot fail
    } else
    if (__pyx_PyAsyncGenAThrow_CheckExact(yf)) {
        retval = __Pyx_async_gen_athrow_close(yf, NULL);
        // cannot fail
    } else
    #endif
    {
        PyObject *meth;
        gen->is_running = 1;
        meth = __Pyx_PyObject_GetAttrStr(yf, PYIDENT("close"));
        if (unlikely(!meth)) {
            if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
                PyErr_WriteUnraisable(yf);
            }
            PyErr_Clear();
        } else {
            retval = PyObject_CallFunction(meth, NULL);
            Py_DECREF(meth);
            if (!retval)
                err = -1;
        }
        gen->is_running = 0;
    }
    Py_XDECREF(retval);
    return err;
}

static PyObject *__Pyx_Generator_Next(PyObject *self) {
    __pyx_CoroutineObject *gen = (__pyx_CoroutineObject*) self;
    PyObject *yf = gen->yieldfrom;
    if (unlikely(gen->is_running))
        return __Pyx_Coroutine_AlreadyRunningError(gen);
    if (yf) {
        PyObject *ret;
        // FIXME: does this really need an INCREF() ?
        //Py_INCREF(yf);
        // YieldFrom code ensures that yf is an iterator
        gen->is_running = 1;
        #ifdef __Pyx_Generator_USED
        if (__Pyx_Generator_CheckExact(yf)) {
            ret = __Pyx_Generator_Next(yf);
        } else
        #endif
        #if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x03030000 && (defined(__linux__) || PY_VERSION_HEX >= 0x030600B3)
        // _PyGen_Send() is not exported before Py3.6
        if (PyGen_CheckExact(yf)) {
            ret = __Pyx_PyGen_Send((PyGenObject*)yf, NULL);
        } else
        #endif
        #ifdef __Pyx_Coroutine_USED
        if (__Pyx_Coroutine_Check(yf)) {
            ret = __Pyx_Coroutine_Send(yf, Py_None);
        } else
        #endif
            ret = Py_TYPE(yf)->tp_iternext(yf);
        gen->is_running = 0;
        //Py_DECREF(yf);
        if (likely(ret)) {
            return ret;
        }
        return __Pyx_Coroutine_FinishDelegation(gen);
    }
    return __Pyx_Coroutine_SendEx(gen, Py_None, 0);
}

static PyObject *__Pyx_Coroutine_Close_Method(PyObject *self, CYTHON_UNUSED PyObject *arg) {
    return __Pyx_Coroutine_Close(self);
}

static PyObject *__Pyx_Coroutine_Close(PyObject *self) {
    __pyx_CoroutineObject *gen = (__pyx_CoroutineObject *) self;
    PyObject *retval, *raised_exception;
    PyObject *yf = gen->yieldfrom;
    int err = 0;

    if (unlikely(gen->is_running))
        return __Pyx_Coroutine_AlreadyRunningError(gen);

    if (yf) {
        Py_INCREF(yf);
        err = __Pyx_Coroutine_CloseIter(gen, yf);
        __Pyx_Coroutine_Undelegate(gen);
        Py_DECREF(yf);
    }
    if (err == 0)
        PyErr_SetNone(PyExc_GeneratorExit);
    retval = __Pyx_Coroutine_SendEx(gen, NULL, 1);
    if (unlikely(retval)) {
        const char *msg;
        Py_DECREF(retval);
        if ((0)) {
        #ifdef __Pyx_Coroutine_USED
        } else if (__Pyx_Coroutine_Check(self)) {
            msg = "coroutine ignored GeneratorExit";
        #endif
        #ifdef __Pyx_AsyncGen_USED
        } else if (__Pyx_AsyncGen_CheckExact(self)) {
#if PY_VERSION_HEX < 0x03060000
            msg = "async generator ignored GeneratorExit - might require Python 3.6+ finalisation (PEP 525)";
#else
            msg = "async generator ignored GeneratorExit";
#endif
        #endif
        } else {
            msg = "generator ignored GeneratorExit";
        }
        PyErr_SetString(PyExc_RuntimeError, msg);
        return NULL;
    }
    raised_exception = PyErr_Occurred();
    if (likely(!raised_exception || __Pyx_PyErr_GivenExceptionMatches2(raised_exception, PyExc_GeneratorExit, PyExc_StopIteration))) {
        // ignore these errors
        if (raised_exception) PyErr_Clear();
        Py_INCREF(Py_None);
        return Py_None;
    }
    return NULL;
}

static PyObject *__Pyx__Coroutine_Throw(PyObject *self, PyObject *typ, PyObject *val, PyObject *tb,
                                        PyObject *args, int close_on_genexit) {
    __pyx_CoroutineObject *gen = (__pyx_CoroutineObject *) self;
    PyObject *yf = gen->yieldfrom;

    if (unlikely(gen->is_running))
        return __Pyx_Coroutine_AlreadyRunningError(gen);

    if (yf) {
        PyObject *ret;
        Py_INCREF(yf);
        if (__Pyx_PyErr_GivenExceptionMatches(typ, PyExc_GeneratorExit) && close_on_genexit) {
            // Asynchronous generators *should not* be closed right away.
            // We have to allow some awaits to work it through, hence the
            // `close_on_genexit` parameter here.
            int err = __Pyx_Coroutine_CloseIter(gen, yf);
            Py_DECREF(yf);
            __Pyx_Coroutine_Undelegate(gen);
            if (err < 0)
                return __Pyx_Coroutine_MethodReturn(self, __Pyx_Coroutine_SendEx(gen, NULL, 0));
            goto throw_here;
        }
        gen->is_running = 1;
        if (0
        #ifdef __Pyx_Generator_USED
            || __Pyx_Generator_CheckExact(yf)
        #endif
        #ifdef __Pyx_Coroutine_USED
            || __Pyx_Coroutine_Check(yf)
        #endif
            ) {
            ret = __Pyx__Coroutine_Throw(yf, typ, val, tb, args, close_on_genexit);
        #ifdef __Pyx_Coroutine_USED
        } else if (__Pyx_CoroutineAwait_CheckExact(yf)) {
            ret = __Pyx__Coroutine_Throw(((__pyx_CoroutineAwaitObject*)yf)->coroutine, typ, val, tb, args, close_on_genexit);
        #endif
        } else {
            PyObject *meth = __Pyx_PyObject_GetAttrStr(yf, PYIDENT("throw"));
            if (unlikely(!meth)) {
                Py_DECREF(yf);
                if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
                    gen->is_running = 0;
                    return NULL;
                }
                PyErr_Clear();
                __Pyx_Coroutine_Undelegate(gen);
                gen->is_running = 0;
                goto throw_here;
            }
            if (likely(args)) {
                ret = PyObject_CallObject(meth, args);
            } else {
                // "tb" or even "val" might be NULL, but that also correctly terminates the argument list
                ret = PyObject_CallFunctionObjArgs(meth, typ, val, tb, NULL);
            }
            Py_DECREF(meth);
        }
        gen->is_running = 0;
        Py_DECREF(yf);
        if (!ret) {
            ret = __Pyx_Coroutine_FinishDelegation(gen);
        }
        return __Pyx_Coroutine_MethodReturn(self, ret);
    }
throw_here:
    __Pyx_Raise(typ, val, tb, NULL);
    return __Pyx_Coroutine_MethodReturn(self, __Pyx_Coroutine_SendEx(gen, NULL, 0));
}

static PyObject *__Pyx_Coroutine_Throw(PyObject *self, PyObject *args) {
    PyObject *typ;
    PyObject *val = NULL;
    PyObject *tb = NULL;

    if (!PyArg_UnpackTuple(args, (char *)"throw", 1, 3, &typ, &val, &tb))
        return NULL;

    return __Pyx__Coroutine_Throw(self, typ, val, tb, args, 1);
}

static CYTHON_INLINE int __Pyx_Coroutine_traverse_excstate(__Pyx_ExcInfoStruct *exc_state, visitproc visit, void *arg) {
    Py_VISIT(exc_state->exc_type);
    Py_VISIT(exc_state->exc_value);
    Py_VISIT(exc_state->exc_traceback);
    return 0;
}

static int __Pyx_Coroutine_traverse(__pyx_CoroutineObject *gen, visitproc visit, void *arg) {
    Py_VISIT(gen->closure);
    Py_VISIT(gen->classobj);
    Py_VISIT(gen->yieldfrom);
    return __Pyx_Coroutine_traverse_excstate(&gen->gi_exc_state, visit, arg);
}

static int __Pyx_Coroutine_clear(PyObject *self) {
    __pyx_CoroutineObject *gen = (__pyx_CoroutineObject *) self;

    Py_CLEAR(gen->closure);
    Py_CLEAR(gen->classobj);
    Py_CLEAR(gen->yieldfrom);
    __Pyx_Coroutine_ExceptionClear(&gen->gi_exc_state);
#ifdef __Pyx_AsyncGen_USED
    if (__Pyx_AsyncGen_CheckExact(self)) {
        Py_CLEAR(((__pyx_PyAsyncGenObject*)gen)->ag_finalizer);
    }
#endif
    Py_CLEAR(gen->gi_code);
    Py_CLEAR(gen->gi_frame);
    Py_CLEAR(gen->gi_name);
    Py_CLEAR(gen->gi_qualname);
    Py_CLEAR(gen->gi_modulename);
    return 0;
}

static void __Pyx_Coroutine_dealloc(PyObject *self) {
    __pyx_CoroutineObject *gen = (__pyx_CoroutineObject *) self;

    PyObject_GC_UnTrack(gen);
    if (gen->gi_weakreflist != NULL)
        PyObject_ClearWeakRefs(self);

    if (gen->resume_label >= 0) {
        // Generator is paused or unstarted, so we need to close
        PyObject_GC_Track(self);
#if PY_VERSION_HEX >= 0x030400a1 && CYTHON_USE_TP_FINALIZE
        if (PyObject_CallFinalizerFromDealloc(self))
#else
        Py_TYPE(gen)->tp_del(self);
        if (Py_REFCNT(self) > 0)
#endif
        {
            // resurrected.  :(
            return;
        }
        PyObject_GC_UnTrack(self);
    }

#ifdef __Pyx_AsyncGen_USED
    if (__Pyx_AsyncGen_CheckExact(self)) {
        /* We have to handle this case for asynchronous generators
           right here, because this code has to be between UNTRACK
           and GC_Del. */
        Py_CLEAR(((__pyx_PyAsyncGenObject*)self)->ag_finalizer);
    }
#endif
    __Pyx_Coroutine_clear(self);
    PyObject_GC_Del(gen);
}

static void __Pyx_Coroutine_del(PyObject *self) {
    PyObject *error_type, *error_value, *error_traceback;
    __pyx_CoroutineObject *gen = (__pyx_CoroutineObject *) self;
    __Pyx_PyThreadState_declare

    if (gen->resume_label < 0) {
        // already terminated => nothing to clean up
        return;
    }

#if !CYTHON_USE_TP_FINALIZE
    // Temporarily resurrect the object.
    assert(self->ob_refcnt == 0);
    __Pyx_SET_REFCNT(self, 1);
#endif

    __Pyx_PyThreadState_assign

    // Save the current exception, if any.
    __Pyx_ErrFetch(&error_type, &error_value, &error_traceback);

#ifdef __Pyx_AsyncGen_USED
    if (__Pyx_AsyncGen_CheckExact(self)) {
        __pyx_PyAsyncGenObject *agen = (__pyx_PyAsyncGenObject*)self;
        PyObject *finalizer = agen->ag_finalizer;
        if (finalizer && !agen->ag_closed) {
            PyObject *res = __Pyx_PyObject_CallOneArg(finalizer, self);
            if (unlikely(!res)) {
                PyErr_WriteUnraisable(self);
            } else {
                Py_DECREF(res);
            }
            // Restore the saved exception.
            __Pyx_ErrRestore(error_type, error_value, error_traceback);
            return;
        }
    }
#endif

    if (unlikely(gen->resume_label == 0 && !error_value)) {
#ifdef __Pyx_Coroutine_USED
#ifdef __Pyx_Generator_USED
    // only warn about (async) coroutines
    if (!__Pyx_Generator_CheckExact(self))
#endif
        {
        // untrack dead object as we are executing Python code (which might trigger GC)
        PyObject_GC_UnTrack(self);
#if PY_MAJOR_VERSION >= 3 /* PY_VERSION_HEX >= 0x03030000*/ || defined(PyErr_WarnFormat)
        if (unlikely(PyErr_WarnFormat(PyExc_RuntimeWarning, 1, "coroutine '%.50S' was never awaited", gen->gi_qualname) < 0))
            PyErr_WriteUnraisable(self);
#else
        {PyObject *msg;
        char *cmsg;
        #if CYTHON_COMPILING_IN_PYPY
        msg = NULL;
        cmsg = (char*) "coroutine was never awaited";
        #else
        char *cname;
        PyObject *qualname;
        qualname = gen->gi_qualname;
        cname = PyString_AS_STRING(qualname);
        msg = PyString_FromFormat("coroutine '%.50s' was never awaited", cname);

        if (unlikely(!msg)) {
            PyErr_Clear();
            cmsg = (char*) "coroutine was never awaited";
        } else {
            cmsg = PyString_AS_STRING(msg);
        }
        #endif
        if (unlikely(PyErr_WarnEx(PyExc_RuntimeWarning, cmsg, 1) < 0))
            PyErr_WriteUnraisable(self);
        Py_XDECREF(msg);}
#endif
        PyObject_GC_Track(self);
        }
#endif /*__Pyx_Coroutine_USED*/
    } else {
        PyObject *res = __Pyx_Coroutine_Close(self);
        if (unlikely(!res)) {
            if (PyErr_Occurred())
                PyErr_WriteUnraisable(self);
        } else {
            Py_DECREF(res);
        }
    }

    // Restore the saved exception.
    __Pyx_ErrRestore(error_type, error_value, error_traceback);

#if !CYTHON_USE_TP_FINALIZE
    // Undo the temporary resurrection; can't use DECREF here, it would
    // cause a recursive call.
    assert(Py_REFCNT(self) > 0);
    if (--self->ob_refcnt == 0) {
        // this is the normal path out
        return;
    }

    // close() resurrected it!  Make it look like the original Py_DECREF
    // never happened.
    {
        Py_ssize_t refcnt = Py_REFCNT(self);
        _Py_NewReference(self);
        __Pyx_SET_REFCNT(self, refcnt);
    }
#if CYTHON_COMPILING_IN_CPYTHON
    assert(PyType_IS_GC(Py_TYPE(self)) &&
           _Py_AS_GC(self)->gc.gc_refs != _PyGC_REFS_UNTRACKED);

    // If Py_REF_DEBUG, _Py_NewReference bumped _Py_RefTotal, so
    // we need to undo that.
    _Py_DEC_REFTOTAL;
#endif
    // If Py_TRACE_REFS, _Py_NewReference re-added self to the object
    // chain, so no more to do there.
    // If COUNT_ALLOCS, the original decref bumped tp_frees, and
    // _Py_NewReference bumped tp_allocs:  both of those need to be
    // undone.
#ifdef COUNT_ALLOCS
    --Py_TYPE(self)->tp_frees;
    --Py_TYPE(self)->tp_allocs;
#endif
#endif
}

static PyObject *
__Pyx_Coroutine_get_name(__pyx_CoroutineObject *self, CYTHON_UNUSED void *context)
{
    PyObject *name = self->gi_name;
    // avoid NULL pointer dereference during garbage collection
    if (unlikely(!name)) name = Py_None;
    Py_INCREF(name);
    return name;
}

static int
__Pyx_Coroutine_set_name(__pyx_CoroutineObject *self, PyObject *value, CYTHON_UNUSED void *context)
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
    tmp = self->gi_name;
    Py_INCREF(value);
    self->gi_name = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
__Pyx_Coroutine_get_qualname(__pyx_CoroutineObject *self, CYTHON_UNUSED void *context)
{
    PyObject *name = self->gi_qualname;
    // avoid NULL pointer dereference during garbage collection
    if (unlikely(!name)) name = Py_None;
    Py_INCREF(name);
    return name;
}

static int
__Pyx_Coroutine_set_qualname(__pyx_CoroutineObject *self, PyObject *value, CYTHON_UNUSED void *context)
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
    tmp = self->gi_qualname;
    Py_INCREF(value);
    self->gi_qualname = value;
    Py_XDECREF(tmp);
    return 0;
}


static PyObject *
__Pyx_Coroutine_get_frame(__pyx_CoroutineObject *self, CYTHON_UNUSED void *context)
{
    PyObject *frame = self->gi_frame;
    if (!frame) {
        if (unlikely(!self->gi_code)) {
            // Avoid doing something stupid, e.g. during garbage collection.
            Py_RETURN_NONE;
        }
        frame = (PyObject *) PyFrame_New(
            PyThreadState_Get(),            /*PyThreadState *tstate,*/
            (PyCodeObject*) self->gi_code,  /*PyCodeObject *code,*/
            $moddict_cname,                 /*PyObject *globals,*/
            0                               /*PyObject *locals*/
        );
        if (unlikely(!frame))
            return NULL;
        // keep the frame cached once it's created
        self->gi_frame = frame;
    }
    Py_INCREF(frame);
    return frame;
}

static __pyx_CoroutineObject *__Pyx__Coroutine_New(
            PyTypeObject* type, __pyx_coroutine_body_t body, PyObject *code, PyObject *closure,
            PyObject *name, PyObject *qualname, PyObject *module_name) {
    __pyx_CoroutineObject *gen = PyObject_GC_New(__pyx_CoroutineObject, type);
    if (unlikely(!gen))
        return NULL;
    return __Pyx__Coroutine_NewInit(gen, body, code, closure, name, qualname, module_name);
}

static __pyx_CoroutineObject *__Pyx__Coroutine_NewInit(
            __pyx_CoroutineObject *gen, __pyx_coroutine_body_t body, PyObject *code, PyObject *closure,
            PyObject *name, PyObject *qualname, PyObject *module_name) {
    gen->body = body;
    gen->closure = closure;
    Py_XINCREF(closure);
    gen->is_running = 0;
    gen->resume_label = 0;
    gen->classobj = NULL;
    gen->yieldfrom = NULL;
    gen->gi_exc_state.exc_type = NULL;
    gen->gi_exc_state.exc_value = NULL;
    gen->gi_exc_state.exc_traceback = NULL;
#if CYTHON_USE_EXC_INFO_STACK
    gen->gi_exc_state.previous_item = NULL;
#endif
    gen->gi_weakreflist = NULL;
    Py_XINCREF(qualname);
    gen->gi_qualname = qualname;
    Py_XINCREF(name);
    gen->gi_name = name;
    Py_XINCREF(module_name);
    gen->gi_modulename = module_name;
    Py_XINCREF(code);
    gen->gi_code = code;
    gen->gi_frame = NULL;

    PyObject_GC_Track(gen);
    return gen;
}


//////////////////// Coroutine ////////////////////
//@requires: CoroutineBase
//@requires: PatchGeneratorABC
//@requires: ObjectHandling.c::PyObject_GenericGetAttrNoDict

static void __Pyx_CoroutineAwait_dealloc(PyObject *self) {
    PyObject_GC_UnTrack(self);
    Py_CLEAR(((__pyx_CoroutineAwaitObject*)self)->coroutine);
    PyObject_GC_Del(self);
}

static int __Pyx_CoroutineAwait_traverse(__pyx_CoroutineAwaitObject *self, visitproc visit, void *arg) {
    Py_VISIT(self->coroutine);
    return 0;
}

static int __Pyx_CoroutineAwait_clear(__pyx_CoroutineAwaitObject *self) {
    Py_CLEAR(self->coroutine);
    return 0;
}

static PyObject *__Pyx_CoroutineAwait_Next(__pyx_CoroutineAwaitObject *self) {
    return __Pyx_Generator_Next(self->coroutine);
}

static PyObject *__Pyx_CoroutineAwait_Send(__pyx_CoroutineAwaitObject *self, PyObject *value) {
    return __Pyx_Coroutine_Send(self->coroutine, value);
}

static PyObject *__Pyx_CoroutineAwait_Throw(__pyx_CoroutineAwaitObject *self, PyObject *args) {
    return __Pyx_Coroutine_Throw(self->coroutine, args);
}

static PyObject *__Pyx_CoroutineAwait_Close(__pyx_CoroutineAwaitObject *self, CYTHON_UNUSED PyObject *arg) {
    return __Pyx_Coroutine_Close(self->coroutine);
}

static PyObject *__Pyx_CoroutineAwait_self(PyObject *self) {
    Py_INCREF(self);
    return self;
}

#if !CYTHON_COMPILING_IN_PYPY
static PyObject *__Pyx_CoroutineAwait_no_new(CYTHON_UNUSED PyTypeObject *type, CYTHON_UNUSED PyObject *args, CYTHON_UNUSED PyObject *kwargs) {
    PyErr_SetString(PyExc_TypeError, "cannot instantiate type, use 'await coroutine' instead");
    return NULL;
}
#endif

static PyMethodDef __pyx_CoroutineAwait_methods[] = {
    {"send", (PyCFunction) __Pyx_CoroutineAwait_Send, METH_O,
     (char*) PyDoc_STR("send(arg) -> send 'arg' into coroutine,\nreturn next yielded value or raise StopIteration.")},
    {"throw", (PyCFunction) __Pyx_CoroutineAwait_Throw, METH_VARARGS,
     (char*) PyDoc_STR("throw(typ[,val[,tb]]) -> raise exception in coroutine,\nreturn next yielded value or raise StopIteration.")},
    {"close", (PyCFunction) __Pyx_CoroutineAwait_Close, METH_NOARGS,
     (char*) PyDoc_STR("close() -> raise GeneratorExit inside coroutine.")},
    {0, 0, 0, 0}
};

static PyTypeObject __pyx_CoroutineAwaitType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "coroutine_wrapper",                /*tp_name*/
    sizeof(__pyx_CoroutineAwaitObject), /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) __Pyx_CoroutineAwait_dealloc,/*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*tp_as_async resp. tp_compare*/
    0,                                  /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    0,                                  /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /*tp_flags*/
    PyDoc_STR("A wrapper object implementing __await__ for coroutines."), /*tp_doc*/
    (traverseproc) __Pyx_CoroutineAwait_traverse,   /*tp_traverse*/
    (inquiry) __Pyx_CoroutineAwait_clear,           /*tp_clear*/
    0,                                  /*tp_richcompare*/
    0,                                  /*tp_weaklistoffset*/
    __Pyx_CoroutineAwait_self,          /*tp_iter*/
    (iternextfunc) __Pyx_CoroutineAwait_Next, /*tp_iternext*/
    __pyx_CoroutineAwait_methods,       /*tp_methods*/
    0                         ,         /*tp_members*/
    0                      ,            /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    0,                                  /*tp_dictoffset*/
    0,                                  /*tp_init*/
    0,                                  /*tp_alloc*/
#if !CYTHON_COMPILING_IN_PYPY
    __Pyx_CoroutineAwait_no_new,        /*tp_new*/
#else
    0,                                  /*tp_new*/
#endif
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

#if PY_VERSION_HEX < 0x030500B1 || defined(__Pyx_IterableCoroutine_USED) || CYTHON_USE_ASYNC_SLOTS
static CYTHON_INLINE PyObject *__Pyx__Coroutine_await(PyObject *coroutine) {
    __pyx_CoroutineAwaitObject *await = PyObject_GC_New(__pyx_CoroutineAwaitObject, __pyx_CoroutineAwaitType);
    if (unlikely(!await)) return NULL;
    Py_INCREF(coroutine);
    await->coroutine = coroutine;
    PyObject_GC_Track(await);
    return (PyObject*)await;
}
#endif

#if PY_VERSION_HEX < 0x030500B1
static PyObject *__Pyx_Coroutine_await_method(PyObject *coroutine, CYTHON_UNUSED PyObject *arg) {
    return __Pyx__Coroutine_await(coroutine);
}
#endif

#if defined(__Pyx_IterableCoroutine_USED) || CYTHON_USE_ASYNC_SLOTS
static PyObject *__Pyx_Coroutine_await(PyObject *coroutine) {
    if (unlikely(!coroutine || !__Pyx_Coroutine_Check(coroutine))) {
        PyErr_SetString(PyExc_TypeError, "invalid input, expected coroutine");
        return NULL;
    }
    return __Pyx__Coroutine_await(coroutine);
}
#endif

#if CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3 && PY_VERSION_HEX < 0x030500B1
static PyObject *__Pyx_Coroutine_compare(PyObject *obj, PyObject *other, int op) {
    PyObject* result;
    switch (op) {
        case Py_EQ: result = (other == obj) ? Py_True : Py_False; break;
        case Py_NE: result = (other != obj) ? Py_True : Py_False; break;
        default:
            result = Py_NotImplemented;
    }
    Py_INCREF(result);
    return result;
}
#endif

static PyMethodDef __pyx_Coroutine_methods[] = {
    {"send", (PyCFunction) __Pyx_Coroutine_Send, METH_O,
     (char*) PyDoc_STR("send(arg) -> send 'arg' into coroutine,\nreturn next iterated value or raise StopIteration.")},
    {"throw", (PyCFunction) __Pyx_Coroutine_Throw, METH_VARARGS,
     (char*) PyDoc_STR("throw(typ[,val[,tb]]) -> raise exception in coroutine,\nreturn next iterated value or raise StopIteration.")},
    {"close", (PyCFunction) __Pyx_Coroutine_Close_Method, METH_NOARGS,
     (char*) PyDoc_STR("close() -> raise GeneratorExit inside coroutine.")},
#if PY_VERSION_HEX < 0x030500B1
    {"__await__", (PyCFunction) __Pyx_Coroutine_await_method, METH_NOARGS,
     (char*) PyDoc_STR("__await__() -> return an iterator to be used in await expression.")},
#endif
    {0, 0, 0, 0}
};

static PyMemberDef __pyx_Coroutine_memberlist[] = {
    {(char *) "cr_running", T_BOOL, offsetof(__pyx_CoroutineObject, is_running), READONLY, NULL},
    {(char*) "cr_await", T_OBJECT, offsetof(__pyx_CoroutineObject, yieldfrom), READONLY,
     (char*) PyDoc_STR("object being awaited, or None")},
    {(char*) "cr_code", T_OBJECT, offsetof(__pyx_CoroutineObject, gi_code), READONLY, NULL},
    {(char *) "__module__", T_OBJECT, offsetof(__pyx_CoroutineObject, gi_modulename), PY_WRITE_RESTRICTED, 0},
    {0, 0, 0, 0, 0}
};

static PyGetSetDef __pyx_Coroutine_getsets[] = {
    {(char *) "__name__", (getter)__Pyx_Coroutine_get_name, (setter)__Pyx_Coroutine_set_name,
     (char*) PyDoc_STR("name of the coroutine"), 0},
    {(char *) "__qualname__", (getter)__Pyx_Coroutine_get_qualname, (setter)__Pyx_Coroutine_set_qualname,
     (char*) PyDoc_STR("qualified name of the coroutine"), 0},
    {(char *) "cr_frame", (getter)__Pyx_Coroutine_get_frame, NULL,
     (char*) PyDoc_STR("Frame of the coroutine"), 0},
    {0, 0, 0, 0, 0}
};

#if CYTHON_USE_ASYNC_SLOTS
static __Pyx_PyAsyncMethodsStruct __pyx_Coroutine_as_async = {
    __Pyx_Coroutine_await, /*am_await*/
    0, /*am_aiter*/
    0, /*am_anext*/
#if PY_VERSION_HEX >= 0x030A00A3
    0, /*am_send*/
#endif
};
#endif

static PyTypeObject __pyx_CoroutineType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "coroutine",                        /*tp_name*/
    sizeof(__pyx_CoroutineObject),      /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) __Pyx_Coroutine_dealloc,/*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
#if CYTHON_USE_ASYNC_SLOTS
    &__pyx_Coroutine_as_async,          /*tp_as_async (tp_reserved) - Py3 only! */
#else
    0,                                  /*tp_reserved*/
#endif
    0,                                  /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    0,                                  /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_FINALIZE, /*tp_flags*/
    0,                                  /*tp_doc*/
    (traverseproc) __Pyx_Coroutine_traverse,   /*tp_traverse*/
    0,                                  /*tp_clear*/
#if CYTHON_USE_ASYNC_SLOTS && CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3 && PY_VERSION_HEX < 0x030500B1
    // in order to (mis-)use tp_reserved above, we must also implement tp_richcompare
    __Pyx_Coroutine_compare,            /*tp_richcompare*/
#else
    0,                                  /*tp_richcompare*/
#endif
    offsetof(__pyx_CoroutineObject, gi_weakreflist), /*tp_weaklistoffset*/
    // no tp_iter() as iterator is only available through __await__()
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    __pyx_Coroutine_methods,            /*tp_methods*/
    __pyx_Coroutine_memberlist,         /*tp_members*/
    __pyx_Coroutine_getsets,            /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
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
#if CYTHON_USE_TP_FINALIZE
    0,                                  /*tp_del*/
#else
    __Pyx_Coroutine_del,                /*tp_del*/
#endif
    0,                                  /*tp_version_tag*/
#if CYTHON_USE_TP_FINALIZE
    __Pyx_Coroutine_del,                /*tp_finalize*/
#elif PY_VERSION_HEX >= 0x030400a1
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

static int __pyx_Coroutine_init(void) {
    // on Windows, C-API functions can't be used in slots statically
    __pyx_CoroutineType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx_CoroutineType = __Pyx_FetchCommonType(&__pyx_CoroutineType_type);
    if (unlikely(!__pyx_CoroutineType))
        return -1;

#ifdef __Pyx_IterableCoroutine_USED
    if (unlikely(__pyx_IterableCoroutine_init() == -1))
        return -1;
#endif

    __pyx_CoroutineAwaitType = __Pyx_FetchCommonType(&__pyx_CoroutineAwaitType_type);
    if (unlikely(!__pyx_CoroutineAwaitType))
        return -1;
    return 0;
}


//////////////////// IterableCoroutine.proto ////////////////////

#define __Pyx_IterableCoroutine_USED

static PyTypeObject *__pyx_IterableCoroutineType = 0;

#undef __Pyx_Coroutine_Check
#define __Pyx_Coroutine_Check(obj) (__Pyx_Coroutine_CheckExact(obj) || (Py_TYPE(obj) == __pyx_IterableCoroutineType))

#define __Pyx_IterableCoroutine_New(body, code, closure, name, qualname, module_name)  \
    __Pyx__Coroutine_New(__pyx_IterableCoroutineType, body, code, closure, name, qualname, module_name)

static int __pyx_IterableCoroutine_init(void);/*proto*/


//////////////////// IterableCoroutine ////////////////////
//@requires: Coroutine
//@requires: CommonStructures.c::FetchCommonType

static PyTypeObject __pyx_IterableCoroutineType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "iterable_coroutine",               /*tp_name*/
    sizeof(__pyx_CoroutineObject),      /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) __Pyx_Coroutine_dealloc,/*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
#if CYTHON_USE_ASYNC_SLOTS
    &__pyx_Coroutine_as_async,          /*tp_as_async (tp_reserved) - Py3 only! */
#else
    0,                                  /*tp_reserved*/
#endif
    0,                                  /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    0,                                  /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_FINALIZE, /*tp_flags*/
    0,                                  /*tp_doc*/
    (traverseproc) __Pyx_Coroutine_traverse,   /*tp_traverse*/
    0,                                  /*tp_clear*/
#if CYTHON_USE_ASYNC_SLOTS && CYTHON_COMPILING_IN_CPYTHON && PY_MAJOR_VERSION >= 3 && PY_VERSION_HEX < 0x030500B1
    // in order to (mis-)use tp_reserved above, we must also implement tp_richcompare
    __Pyx_Coroutine_compare,            /*tp_richcompare*/
#else
    0,                                  /*tp_richcompare*/
#endif
    offsetof(__pyx_CoroutineObject, gi_weakreflist), /*tp_weaklistoffset*/
    // enable iteration for legacy support of asyncio yield-from protocol
    __Pyx_Coroutine_await,              /*tp_iter*/
    (iternextfunc) __Pyx_Generator_Next, /*tp_iternext*/
    __pyx_Coroutine_methods,            /*tp_methods*/
    __pyx_Coroutine_memberlist,         /*tp_members*/
    __pyx_Coroutine_getsets,            /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
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
#if PY_VERSION_HEX >= 0x030400a1
    0,                                  /*tp_del*/
#else
    __Pyx_Coroutine_del,                /*tp_del*/
#endif
    0,                                  /*tp_version_tag*/
#if PY_VERSION_HEX >= 0x030400a1
    __Pyx_Coroutine_del,                /*tp_finalize*/
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


static int __pyx_IterableCoroutine_init(void) {
    __pyx_IterableCoroutineType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx_IterableCoroutineType = __Pyx_FetchCommonType(&__pyx_IterableCoroutineType_type);
    if (unlikely(!__pyx_IterableCoroutineType))
        return -1;
    return 0;
}


//////////////////// Generator ////////////////////
//@requires: CoroutineBase
//@requires: PatchGeneratorABC
//@requires: ObjectHandling.c::PyObject_GenericGetAttrNoDict

static PyMethodDef __pyx_Generator_methods[] = {
    {"send", (PyCFunction) __Pyx_Coroutine_Send, METH_O,
     (char*) PyDoc_STR("send(arg) -> send 'arg' into generator,\nreturn next yielded value or raise StopIteration.")},
    {"throw", (PyCFunction) __Pyx_Coroutine_Throw, METH_VARARGS,
     (char*) PyDoc_STR("throw(typ[,val[,tb]]) -> raise exception in generator,\nreturn next yielded value or raise StopIteration.")},
    {"close", (PyCFunction) __Pyx_Coroutine_Close_Method, METH_NOARGS,
     (char*) PyDoc_STR("close() -> raise GeneratorExit inside generator.")},
    {0, 0, 0, 0}
};

static PyMemberDef __pyx_Generator_memberlist[] = {
    {(char *) "gi_running", T_BOOL, offsetof(__pyx_CoroutineObject, is_running), READONLY, NULL},
    {(char*) "gi_yieldfrom", T_OBJECT, offsetof(__pyx_CoroutineObject, yieldfrom), READONLY,
     (char*) PyDoc_STR("object being iterated by 'yield from', or None")},
    {(char*) "gi_code", T_OBJECT, offsetof(__pyx_CoroutineObject, gi_code), READONLY, NULL},
    {0, 0, 0, 0, 0}
};

static PyGetSetDef __pyx_Generator_getsets[] = {
    {(char *) "__name__", (getter)__Pyx_Coroutine_get_name, (setter)__Pyx_Coroutine_set_name,
     (char*) PyDoc_STR("name of the generator"), 0},
    {(char *) "__qualname__", (getter)__Pyx_Coroutine_get_qualname, (setter)__Pyx_Coroutine_set_qualname,
     (char*) PyDoc_STR("qualified name of the generator"), 0},
    {(char *) "gi_frame", (getter)__Pyx_Coroutine_get_frame, NULL,
     (char*) PyDoc_STR("Frame of the generator"), 0},
    {0, 0, 0, 0, 0}
};

static PyTypeObject __pyx_GeneratorType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "generator",                        /*tp_name*/
    sizeof(__pyx_CoroutineObject),      /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) __Pyx_Coroutine_dealloc,/*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*tp_compare / tp_as_async*/
    0,                                  /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    0,                                  /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_FINALIZE, /*tp_flags*/
    0,                                  /*tp_doc*/
    (traverseproc) __Pyx_Coroutine_traverse,   /*tp_traverse*/
    0,                                  /*tp_clear*/
    0,                                  /*tp_richcompare*/
    offsetof(__pyx_CoroutineObject, gi_weakreflist), /*tp_weaklistoffset*/
    0,                                  /*tp_iter*/
    (iternextfunc) __Pyx_Generator_Next, /*tp_iternext*/
    __pyx_Generator_methods,            /*tp_methods*/
    __pyx_Generator_memberlist,         /*tp_members*/
    __pyx_Generator_getsets,            /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
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
#if CYTHON_USE_TP_FINALIZE
    0,                                  /*tp_del*/
#else
    __Pyx_Coroutine_del,                /*tp_del*/
#endif
    0,                                  /*tp_version_tag*/
#if CYTHON_USE_TP_FINALIZE
    __Pyx_Coroutine_del,                /*tp_finalize*/
#elif PY_VERSION_HEX >= 0x030400a1
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

static int __pyx_Generator_init(void) {
    // on Windows, C-API functions can't be used in slots statically
    __pyx_GeneratorType_type.tp_getattro = __Pyx_PyObject_GenericGetAttrNoDict;
    __pyx_GeneratorType_type.tp_iter = PyObject_SelfIter;

    __pyx_GeneratorType = __Pyx_FetchCommonType(&__pyx_GeneratorType_type);
    if (unlikely(!__pyx_GeneratorType)) {
        return -1;
    }
    return 0;
}


/////////////// ReturnWithStopIteration.proto ///////////////

#define __Pyx_ReturnWithStopIteration(value)  \
    if (value == Py_None) PyErr_SetNone(PyExc_StopIteration); else __Pyx__ReturnWithStopIteration(value)
static void __Pyx__ReturnWithStopIteration(PyObject* value); /*proto*/

/////////////// ReturnWithStopIteration ///////////////
//@requires: Exceptions.c::PyErrFetchRestore
//@requires: Exceptions.c::PyThreadStateGet
//@substitute: naming

// 1) Instantiating an exception just to pass back a value is costly.
// 2) CPython 3.3 <= x < 3.5b1 crash in yield-from when the StopIteration is not instantiated.
// 3) Passing a tuple as value into PyErr_SetObject() passes its items on as arguments.
// 4) Passing an exception as value will interpret it as an exception on unpacking and raise it (or unpack its value).
// 5) If there is currently an exception being handled, we need to chain it.

static void __Pyx__ReturnWithStopIteration(PyObject* value) {
    PyObject *exc, *args;
#if CYTHON_COMPILING_IN_CPYTHON || CYTHON_COMPILING_IN_PYSTON
    __Pyx_PyThreadState_declare
    if ((PY_VERSION_HEX >= 0x03030000 && PY_VERSION_HEX < 0x030500B1)
            || unlikely(PyTuple_Check(value) || PyExceptionInstance_Check(value))) {
        args = PyTuple_New(1);
        if (unlikely(!args)) return;
        Py_INCREF(value);
        PyTuple_SET_ITEM(args, 0, value);
        exc = PyType_Type.tp_call(PyExc_StopIteration, args, NULL);
        Py_DECREF(args);
        if (!exc) return;
    } else {
        // it's safe to avoid instantiating the exception
        Py_INCREF(value);
        exc = value;
    }
    #if CYTHON_FAST_THREAD_STATE
    __Pyx_PyThreadState_assign
    #if CYTHON_USE_EXC_INFO_STACK
    if (!$local_tstate_cname->exc_info->exc_type)
    #else
    if (!$local_tstate_cname->exc_type)
    #endif
    {
        // no chaining needed => avoid the overhead in PyErr_SetObject()
        Py_INCREF(PyExc_StopIteration);
        __Pyx_ErrRestore(PyExc_StopIteration, exc, NULL);
        return;
    }
    #endif
#else
    args = PyTuple_Pack(1, value);
    if (unlikely(!args)) return;
    exc = PyObject_Call(PyExc_StopIteration, args, NULL);
    Py_DECREF(args);
    if (unlikely(!exc)) return;
#endif
    PyErr_SetObject(PyExc_StopIteration, exc);
    Py_DECREF(exc);
}


//////////////////// PatchModuleWithCoroutine.proto ////////////////////

static PyObject* __Pyx_Coroutine_patch_module(PyObject* module, const char* py_code); /*proto*/

//////////////////// PatchModuleWithCoroutine ////////////////////
//@substitute: naming

static PyObject* __Pyx_Coroutine_patch_module(PyObject* module, const char* py_code) {
#if defined(__Pyx_Generator_USED) || defined(__Pyx_Coroutine_USED)
    int result;
    PyObject *globals, *result_obj;
    globals = PyDict_New();  if (unlikely(!globals)) goto ignore;
    result = PyDict_SetItemString(globals, "_cython_coroutine_type",
    #ifdef __Pyx_Coroutine_USED
        (PyObject*)__pyx_CoroutineType);
    #else
        Py_None);
    #endif
    if (unlikely(result < 0)) goto ignore;
    result = PyDict_SetItemString(globals, "_cython_generator_type",
    #ifdef __Pyx_Generator_USED
        (PyObject*)__pyx_GeneratorType);
    #else
        Py_None);
    #endif
    if (unlikely(result < 0)) goto ignore;
    if (unlikely(PyDict_SetItemString(globals, "_module", module) < 0)) goto ignore;
    if (unlikely(PyDict_SetItemString(globals, "__builtins__", $builtins_cname) < 0)) goto ignore;
    result_obj = PyRun_String(py_code, Py_file_input, globals, globals);
    if (unlikely(!result_obj)) goto ignore;
    Py_DECREF(result_obj);
    Py_DECREF(globals);
    return module;

ignore:
    Py_XDECREF(globals);
    PyErr_WriteUnraisable(module);
    if (unlikely(PyErr_WarnEx(PyExc_RuntimeWarning, "Cython module failed to patch module with custom type", 1) < 0)) {
        Py_DECREF(module);
        module = NULL;
    }
#else
    // avoid "unused" warning
    py_code++;
#endif
    return module;
}


//////////////////// PatchGeneratorABC.proto ////////////////////

// register with Generator/Coroutine ABCs in 'collections.abc'
// see https://bugs.python.org/issue24018
static int __Pyx_patch_abc(void); /*proto*/

//////////////////// PatchGeneratorABC ////////////////////
//@requires: PatchModuleWithCoroutine

#ifndef CYTHON_REGISTER_ABCS
#define CYTHON_REGISTER_ABCS 1
#endif

#if defined(__Pyx_Generator_USED) || defined(__Pyx_Coroutine_USED)
static PyObject* __Pyx_patch_abc_module(PyObject *module); /*proto*/
static PyObject* __Pyx_patch_abc_module(PyObject *module) {
    module = __Pyx_Coroutine_patch_module(
        module, CSTRING("""\
if _cython_generator_type is not None:
    try: Generator = _module.Generator
    except AttributeError: pass
    else: Generator.register(_cython_generator_type)
if _cython_coroutine_type is not None:
    try: Coroutine = _module.Coroutine
    except AttributeError: pass
    else: Coroutine.register(_cython_coroutine_type)
""")
    );
    return module;
}
#endif

static int __Pyx_patch_abc(void) {
#if defined(__Pyx_Generator_USED) || defined(__Pyx_Coroutine_USED)
    static int abc_patched = 0;
    if (CYTHON_REGISTER_ABCS && !abc_patched) {
        PyObject *module;
        module = PyImport_ImportModule((PY_MAJOR_VERSION >= 3) ? "collections.abc" : "collections");
        if (!module) {
            PyErr_WriteUnraisable(NULL);
            if (unlikely(PyErr_WarnEx(PyExc_RuntimeWarning,
                    ((PY_MAJOR_VERSION >= 3) ?
                        "Cython module failed to register with collections.abc module" :
                        "Cython module failed to register with collections module"), 1) < 0)) {
                return -1;
            }
        } else {
            module = __Pyx_patch_abc_module(module);
            abc_patched = 1;
            if (unlikely(!module))
                return -1;
            Py_DECREF(module);
        }
        // also register with "backports_abc" module if available, just in case
        module = PyImport_ImportModule("backports_abc");
        if (module) {
            module = __Pyx_patch_abc_module(module);
            Py_XDECREF(module);
        }
        if (!module) {
            PyErr_Clear();
        }
    }
#else
    // avoid "unused" warning for __Pyx_Coroutine_patch_module()
    if ((0)) __Pyx_Coroutine_patch_module(NULL, NULL);
#endif
    return 0;
}


//////////////////// PatchAsyncIO.proto ////////////////////

// run after importing "asyncio" to patch Cython generator support into it
static PyObject* __Pyx_patch_asyncio(PyObject* module); /*proto*/

//////////////////// PatchAsyncIO ////////////////////
//@requires: ImportExport.c::Import
//@requires: PatchModuleWithCoroutine
//@requires: PatchInspect

static PyObject* __Pyx_patch_asyncio(PyObject* module) {
#if PY_VERSION_HEX < 0x030500B2 && \
        (defined(__Pyx_Coroutine_USED) || defined(__Pyx_Generator_USED)) && \
        (!defined(CYTHON_PATCH_ASYNCIO) || CYTHON_PATCH_ASYNCIO)
    PyObject *patch_module = NULL;
    static int asyncio_patched = 0;
    if (unlikely((!asyncio_patched) && module)) {
        PyObject *package;
        package = __Pyx_Import(PYIDENT("asyncio.coroutines"), NULL, 0);
        if (package) {
            patch_module = __Pyx_Coroutine_patch_module(
                PyObject_GetAttrString(package, "coroutines"), CSTRING("""\
try:
    coro_types = _module._COROUTINE_TYPES
except AttributeError: pass
else:
    if _cython_coroutine_type is not None and _cython_coroutine_type not in coro_types:
        coro_types = tuple(coro_types) + (_cython_coroutine_type,)
    if _cython_generator_type is not None and _cython_generator_type not in coro_types:
        coro_types = tuple(coro_types) + (_cython_generator_type,)
_module._COROUTINE_TYPES = coro_types
""")
            );
        } else {
            PyErr_Clear();
// Always enable fallback: even if we compile against 3.4.2, we might be running on 3.4.1 at some point.
//#if PY_VERSION_HEX < 0x03040200
            // Py3.4.1 used to have asyncio.tasks instead of asyncio.coroutines
            package = __Pyx_Import(PYIDENT("asyncio.tasks"), NULL, 0);
            if (unlikely(!package)) goto asyncio_done;
            patch_module = __Pyx_Coroutine_patch_module(
                PyObject_GetAttrString(package, "tasks"), CSTRING("""\
if hasattr(_module, 'iscoroutine'):
    old_types = getattr(_module.iscoroutine, '_cython_coroutine_types', None)
    if old_types is None or not isinstance(old_types, set):
        old_types = set()
        def cy_wrap(orig_func, type=type, cython_coroutine_types=old_types):
            def cy_iscoroutine(obj): return type(obj) in cython_coroutine_types or orig_func(obj)
            cy_iscoroutine._cython_coroutine_types = cython_coroutine_types
            return cy_iscoroutine
        _module.iscoroutine = cy_wrap(_module.iscoroutine)
    if _cython_coroutine_type is not None:
        old_types.add(_cython_coroutine_type)
    if _cython_generator_type is not None:
        old_types.add(_cython_generator_type)
""")
            );
//#endif
// Py < 0x03040200
        }
        Py_DECREF(package);
        if (unlikely(!patch_module)) goto ignore;
//#if PY_VERSION_HEX < 0x03040200
asyncio_done:
        PyErr_Clear();
//#endif
        asyncio_patched = 1;
#ifdef __Pyx_Generator_USED
        // now patch inspect.isgenerator() by looking up the imported module in the patched asyncio module
        {
            PyObject *inspect_module;
            if (patch_module) {
                inspect_module = PyObject_GetAttr(patch_module, PYIDENT("inspect"));
                Py_DECREF(patch_module);
            } else {
                inspect_module = __Pyx_Import(PYIDENT("inspect"), NULL, 0);
            }
            if (unlikely(!inspect_module)) goto ignore;
            inspect_module = __Pyx_patch_inspect(inspect_module);
            if (unlikely(!inspect_module)) {
                Py_DECREF(module);
                module = NULL;
            }
            Py_XDECREF(inspect_module);
        }
#else
        // avoid "unused" warning for __Pyx_patch_inspect()
        if ((0)) return __Pyx_patch_inspect(module);
#endif
    }
    return module;
ignore:
    PyErr_WriteUnraisable(module);
    if (unlikely(PyErr_WarnEx(PyExc_RuntimeWarning, "Cython module failed to patch asyncio package with custom generator type", 1) < 0)) {
        Py_DECREF(module);
        module = NULL;
    }
#else
    // avoid "unused" warning for __Pyx_Coroutine_patch_module()
    if ((0)) return __Pyx_patch_inspect(__Pyx_Coroutine_patch_module(module, NULL));
#endif
    return module;
}


//////////////////// PatchInspect.proto ////////////////////

// run after importing "inspect" to patch Cython generator support into it
static PyObject* __Pyx_patch_inspect(PyObject* module); /*proto*/

//////////////////// PatchInspect ////////////////////
//@requires: PatchModuleWithCoroutine

static PyObject* __Pyx_patch_inspect(PyObject* module) {
#if defined(__Pyx_Generator_USED) && (!defined(CYTHON_PATCH_INSPECT) || CYTHON_PATCH_INSPECT)
    static int inspect_patched = 0;
    if (unlikely((!inspect_patched) && module)) {
        module = __Pyx_Coroutine_patch_module(
            module, CSTRING("""\
old_types = getattr(_module.isgenerator, '_cython_generator_types', None)
if old_types is None or not isinstance(old_types, set):
    old_types = set()
    def cy_wrap(orig_func, type=type, cython_generator_types=old_types):
        def cy_isgenerator(obj): return type(obj) in cython_generator_types or orig_func(obj)
        cy_isgenerator._cython_generator_types = cython_generator_types
        return cy_isgenerator
    _module.isgenerator = cy_wrap(_module.isgenerator)
old_types.add(_cython_generator_type)
""")
        );
        inspect_patched = 1;
    }
#else
    // avoid "unused" warning for __Pyx_Coroutine_patch_module()
    if ((0)) return __Pyx_Coroutine_patch_module(module, NULL);
#endif
    return module;
}


//////////////////// StopAsyncIteration.proto ////////////////////

#define __Pyx_StopAsyncIteration_USED
static PyObject *__Pyx_PyExc_StopAsyncIteration;
static int __pyx_StopAsyncIteration_init(void); /*proto*/

//////////////////// StopAsyncIteration ////////////////////

#if PY_VERSION_HEX < 0x030500B1
static PyTypeObject __Pyx__PyExc_StopAsyncIteration_type = {
    PyVarObject_HEAD_INIT(0, 0)
    "StopAsyncIteration",               /*tp_name*/
    sizeof(PyBaseExceptionObject),      /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    0,                                  /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*tp_compare / reserved*/
    0,                                  /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    0,                                  /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,  /*tp_flags*/
    PyDoc_STR("Signal the end from iterator.__anext__()."),  /*tp_doc*/
    0,                                  /*tp_traverse*/
    0,                                  /*tp_clear*/
    0,                                  /*tp_richcompare*/
    0,                                  /*tp_weaklistoffset*/
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    0,                                  /*tp_methods*/
    0,                                  /*tp_members*/
    0,                                  /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    0,                                  /*tp_descr_get*/
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
#if CYTHON_COMPILING_IN_PYPY && PYPY_VERSION_NUM+0 >= 0x06000000
    0,                                          /*tp_pypy_flags*/
#endif
};
#endif

static int __pyx_StopAsyncIteration_init(void) {
#if PY_VERSION_HEX >= 0x030500B1
    __Pyx_PyExc_StopAsyncIteration = PyExc_StopAsyncIteration;
#else
    PyObject *builtins = PyEval_GetBuiltins();
    if (likely(builtins)) {
        PyObject *exc = PyMapping_GetItemString(builtins, (char*) "StopAsyncIteration");
        if (exc) {
            __Pyx_PyExc_StopAsyncIteration = exc;
            return 0;
        }
    }
    PyErr_Clear();

    __Pyx__PyExc_StopAsyncIteration_type.tp_traverse = ((PyTypeObject*)PyExc_BaseException)->tp_traverse;
    __Pyx__PyExc_StopAsyncIteration_type.tp_clear = ((PyTypeObject*)PyExc_BaseException)->tp_clear;
    __Pyx__PyExc_StopAsyncIteration_type.tp_dictoffset = ((PyTypeObject*)PyExc_BaseException)->tp_dictoffset;
    __Pyx__PyExc_StopAsyncIteration_type.tp_base = (PyTypeObject*)PyExc_Exception;

    __Pyx_PyExc_StopAsyncIteration = (PyObject*) __Pyx_FetchCommonType(&__Pyx__PyExc_StopAsyncIteration_type);
    if (unlikely(!__Pyx_PyExc_StopAsyncIteration))
        return -1;
    if (builtins && unlikely(PyMapping_SetItemString(builtins, (char*) "StopAsyncIteration", __Pyx_PyExc_StopAsyncIteration) < 0))
        return -1;
#endif
    return 0;
}
