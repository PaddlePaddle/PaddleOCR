/////////////// Profile.proto ///////////////
//@requires: Exceptions.c::PyErrFetchRestore
//@substitute: naming

// Note that cPython ignores PyTrace_EXCEPTION,
// but maybe some other profilers don't.

#ifndef CYTHON_PROFILE
#if CYTHON_COMPILING_IN_PYPY || CYTHON_COMPILING_IN_PYSTON
  #define CYTHON_PROFILE 0
#else
  #define CYTHON_PROFILE 1
#endif
#endif

#ifndef CYTHON_TRACE_NOGIL
  #define CYTHON_TRACE_NOGIL 0
#else
  #if CYTHON_TRACE_NOGIL && !defined(CYTHON_TRACE)
    #define CYTHON_TRACE 1
  #endif
#endif

#ifndef CYTHON_TRACE
  #define CYTHON_TRACE 0
#endif

#if CYTHON_TRACE
  #undef CYTHON_PROFILE_REUSE_FRAME
#endif

#ifndef CYTHON_PROFILE_REUSE_FRAME
  #define CYTHON_PROFILE_REUSE_FRAME 0
#endif

#if CYTHON_PROFILE || CYTHON_TRACE

  #include "compile.h"
  #include "frameobject.h"
  #include "traceback.h"

  #if CYTHON_PROFILE_REUSE_FRAME
    #define CYTHON_FRAME_MODIFIER static
    #define CYTHON_FRAME_DEL(frame)
  #else
    #define CYTHON_FRAME_MODIFIER
    #define CYTHON_FRAME_DEL(frame) Py_CLEAR(frame)
  #endif

  #define __Pyx_TraceDeclarations                                         \
      static PyCodeObject *$frame_code_cname = NULL;                      \
      CYTHON_FRAME_MODIFIER PyFrameObject *$frame_cname = NULL;           \
      int __Pyx_use_tracing = 0;

  #define __Pyx_TraceFrameInit(codeobj)                                   \
      if (codeobj) $frame_code_cname = (PyCodeObject*) codeobj;

#if PY_VERSION_HEX >= 0x030b00a2
  #define __Pyx_IsTracing(tstate, check_tracing, check_funcs) \
     (unlikely((tstate)->cframe->use_tracing) && \
         (!(check_tracing) || !(tstate)->tracing) && \
         (!(check_funcs) || (tstate)->c_profilefunc || (CYTHON_TRACE && (tstate)->c_tracefunc)))

  #define __Pyx_EnterTracing(tstate) PyThreadState_EnterTracing(tstate)

  #define __Pyx_LeaveTracing(tstate) PyThreadState_LeaveTracing(tstate)

#elif PY_VERSION_HEX >= 0x030a00b1
  #define __Pyx_IsTracing(tstate, check_tracing, check_funcs) \
     (unlikely((tstate)->cframe->use_tracing) && \
         (!(check_tracing) || !(tstate)->tracing) && \
         (!(check_funcs) || (tstate)->c_profilefunc || (CYTHON_TRACE && (tstate)->c_tracefunc)))

  #define __Pyx_EnterTracing(tstate) \
      do { tstate->tracing++; tstate->cframe->use_tracing = 0; } while (0)

  #define __Pyx_LeaveTracing(tstate) \
      do { \
          tstate->tracing--; \
          tstate->cframe->use_tracing = ((CYTHON_TRACE && tstate->c_tracefunc != NULL) \
                                 || tstate->c_profilefunc != NULL); \
      } while (0)

#else
  #define __Pyx_IsTracing(tstate, check_tracing, check_funcs) \
     (unlikely((tstate)->use_tracing) && \
         (!(check_tracing) || !(tstate)->tracing) && \
         (!(check_funcs) || (tstate)->c_profilefunc || (CYTHON_TRACE && (tstate)->c_tracefunc)))

  #define __Pyx_EnterTracing(tstate) \
      do { tstate->tracing++; tstate->use_tracing = 0; } while (0)

  #define __Pyx_LeaveTracing(tstate) \
      do { \
          tstate->tracing--; \
          tstate->use_tracing = ((CYTHON_TRACE && tstate->c_tracefunc != NULL) \
                                         || tstate->c_profilefunc != NULL); \
      } while (0)

#endif

  #ifdef WITH_THREAD
  #define __Pyx_TraceCall(funcname, srcfile, firstlineno, nogil, goto_error)             \
  if (nogil) {                                                                           \
      if (CYTHON_TRACE_NOGIL) {                                                          \
          PyThreadState *tstate;                                                         \
          PyGILState_STATE state = PyGILState_Ensure();                                  \
          tstate = __Pyx_PyThreadState_Current;                                          \
          if (__Pyx_IsTracing(tstate, 1, 1)) {                                           \
              __Pyx_use_tracing = __Pyx_TraceSetupAndCall(&$frame_code_cname, &$frame_cname, tstate, funcname, srcfile, firstlineno);  \
          }                                                                              \
          PyGILState_Release(state);                                                     \
          if (unlikely(__Pyx_use_tracing < 0)) goto_error;                               \
      }                                                                                  \
  } else {                                                                               \
      PyThreadState* tstate = PyThreadState_GET();                                       \
      if (__Pyx_IsTracing(tstate, 1, 1)) {                                               \
          __Pyx_use_tracing = __Pyx_TraceSetupAndCall(&$frame_code_cname, &$frame_cname, tstate, funcname, srcfile, firstlineno);  \
          if (unlikely(__Pyx_use_tracing < 0)) goto_error;                               \
      }                                                                                  \
  }
  #else
  #define __Pyx_TraceCall(funcname, srcfile, firstlineno, nogil, goto_error)             \
  {   PyThreadState* tstate = PyThreadState_GET();                                       \
      if (__Pyx_IsTracing(tstate, 1, 1)) {                                               \
          __Pyx_use_tracing = __Pyx_TraceSetupAndCall(&$frame_code_cname, &$frame_cname, tstate, funcname, srcfile, firstlineno);  \
          if (unlikely(__Pyx_use_tracing < 0)) goto_error;                               \
      }                                                                                  \
  }
  #endif

  #define __Pyx_TraceException()                                                           \
  if (likely(!__Pyx_use_tracing)); else {                                                  \
      PyThreadState* tstate = __Pyx_PyThreadState_Current;                                 \
      if (__Pyx_IsTracing(tstate, 0, 1)) {                                                 \
          __Pyx_EnterTracing(tstate);                                                      \
          PyObject *exc_info = __Pyx_GetExceptionTuple(tstate);                            \
          if (exc_info) {                                                                  \
              if (CYTHON_TRACE && tstate->c_tracefunc)                                     \
                  tstate->c_tracefunc(                                                     \
                      tstate->c_traceobj, $frame_cname, PyTrace_EXCEPTION, exc_info);      \
              tstate->c_profilefunc(                                                       \
                  tstate->c_profileobj, $frame_cname, PyTrace_EXCEPTION, exc_info);        \
              Py_DECREF(exc_info);                                                         \
          }                                                                                \
          __Pyx_LeaveTracing(tstate);                                                      \
      }                                                                                    \
  }

  static void __Pyx_call_return_trace_func(PyThreadState *tstate, PyFrameObject *frame, PyObject *result) {
      PyObject *type, *value, *traceback;
      __Pyx_ErrFetchInState(tstate, &type, &value, &traceback);
      __Pyx_EnterTracing(tstate);
      if (CYTHON_TRACE && tstate->c_tracefunc)
          tstate->c_tracefunc(tstate->c_traceobj, frame, PyTrace_RETURN, result);
      if (tstate->c_profilefunc)
          tstate->c_profilefunc(tstate->c_profileobj, frame, PyTrace_RETURN, result);
      CYTHON_FRAME_DEL(frame);
      __Pyx_LeaveTracing(tstate);
      __Pyx_ErrRestoreInState(tstate, type, value, traceback);
  }

  #ifdef WITH_THREAD
  #define __Pyx_TraceReturn(result, nogil)                                                \
  if (likely(!__Pyx_use_tracing)); else {                                                 \
      if (nogil) {                                                                        \
          if (CYTHON_TRACE_NOGIL) {                                                       \
              PyThreadState *tstate;                                                      \
              PyGILState_STATE state = PyGILState_Ensure();                               \
              tstate = __Pyx_PyThreadState_Current;                                       \
              if (__Pyx_IsTracing(tstate, 0, 0)) {                                        \
                  __Pyx_call_return_trace_func(tstate, $frame_cname, (PyObject*)result);  \
              }                                                                           \
              PyGILState_Release(state);                                                  \
          }                                                                               \
      } else {                                                                            \
          PyThreadState* tstate = __Pyx_PyThreadState_Current;                            \
          if (__Pyx_IsTracing(tstate, 0, 0)) {                                            \
              __Pyx_call_return_trace_func(tstate, $frame_cname, (PyObject*)result);      \
          }                                                                               \
      }                                                                                   \
  }
  #else
  #define __Pyx_TraceReturn(result, nogil)                                                \
  if (likely(!__Pyx_use_tracing)); else {                                                 \
      PyThreadState* tstate = __Pyx_PyThreadState_Current;                                \
      if (__Pyx_IsTracing(tstate, 0, 0)) {                                                \
          __Pyx_call_return_trace_func(tstate, $frame_cname, (PyObject*)result);          \
      }                                                                                   \
  }
  #endif

  static PyCodeObject *__Pyx_createFrameCodeObject(const char *funcname, const char *srcfile, int firstlineno); /*proto*/
  static int __Pyx_TraceSetupAndCall(PyCodeObject** code, PyFrameObject** frame, PyThreadState* tstate, const char *funcname, const char *srcfile, int firstlineno); /*proto*/

#else

  #define __Pyx_TraceDeclarations
  #define __Pyx_TraceFrameInit(codeobj)
  // mark error label as used to avoid compiler warnings
  #define __Pyx_TraceCall(funcname, srcfile, firstlineno, nogil, goto_error)   if ((1)); else goto_error;
  #define __Pyx_TraceException()
  #define __Pyx_TraceReturn(result, nogil)

#endif /* CYTHON_PROFILE */

#if CYTHON_TRACE
  // see call_trace_protected() in CPython's ceval.c
  static int __Pyx_call_line_trace_func(PyThreadState *tstate, PyFrameObject *frame, int lineno) {
      int ret;
      PyObject *type, *value, *traceback;
      __Pyx_ErrFetchInState(tstate, &type, &value, &traceback);
      __Pyx_PyFrame_SetLineNumber(frame, lineno);
      __Pyx_EnterTracing(tstate);

      ret = tstate->c_tracefunc(tstate->c_traceobj, frame, PyTrace_LINE, NULL);

      __Pyx_LeaveTracing(tstate);
      if (likely(!ret)) {
          __Pyx_ErrRestoreInState(tstate, type, value, traceback);
      } else {
          Py_XDECREF(type);
          Py_XDECREF(value);
          Py_XDECREF(traceback);
      }
      return ret;
  }

  #ifdef WITH_THREAD
  #define __Pyx_TraceLine(lineno, nogil, goto_error)                                       \
  if (likely(!__Pyx_use_tracing)); else {                                                  \
      if (nogil) {                                                                         \
          if (CYTHON_TRACE_NOGIL) {                                                        \
              int ret = 0;                                                                 \
              PyThreadState *tstate;                                                       \
              PyGILState_STATE state = PyGILState_Ensure();                                \
              tstate = __Pyx_PyThreadState_Current;                                        \
              if (__Pyx_IsTracing(tstate, 0, 0) && tstate->c_tracefunc && $frame_cname->f_trace) { \
                  ret = __Pyx_call_line_trace_func(tstate, $frame_cname, lineno);          \
              }                                                                            \
              PyGILState_Release(state);                                                   \
              if (unlikely(ret)) goto_error;                                               \
          }                                                                                \
      } else {                                                                             \
          PyThreadState* tstate = __Pyx_PyThreadState_Current;                             \
          if (__Pyx_IsTracing(tstate, 0, 0) && tstate->c_tracefunc && $frame_cname->f_trace) { \
              int ret = __Pyx_call_line_trace_func(tstate, $frame_cname, lineno);          \
              if (unlikely(ret)) goto_error;                                               \
          }                                                                                \
      }                                                                                    \
  }
  #else
  #define __Pyx_TraceLine(lineno, nogil, goto_error)                                       \
  if (likely(!__Pyx_use_tracing)); else {                                                  \
      PyThreadState* tstate = __Pyx_PyThreadState_Current;                                 \
      if (__Pyx_IsTracing(tstate, 0, 0) && tstate->c_tracefunc && $frame_cname->f_trace) { \
          int ret = __Pyx_call_line_trace_func(tstate, $frame_cname, lineno);              \
          if (unlikely(ret)) goto_error;                                                   \
      }                                                                                    \
  }
  #endif
#else
  // mark error label as used to avoid compiler warnings
  #define __Pyx_TraceLine(lineno, nogil, goto_error)   if ((1)); else goto_error;
#endif

/////////////// Profile ///////////////
//@substitute: naming

#if CYTHON_PROFILE

static int __Pyx_TraceSetupAndCall(PyCodeObject** code,
                                   PyFrameObject** frame,
                                   PyThreadState* tstate,
                                   const char *funcname,
                                   const char *srcfile,
                                   int firstlineno) {
    PyObject *type, *value, *traceback;
    int retval;
    if (*frame == NULL || !CYTHON_PROFILE_REUSE_FRAME) {
        if (*code == NULL) {
            *code = __Pyx_createFrameCodeObject(funcname, srcfile, firstlineno);
            if (*code == NULL) return 0;
        }
        *frame = PyFrame_New(
            tstate,                          /*PyThreadState *tstate*/
            *code,                           /*PyCodeObject *code*/
            $moddict_cname,                  /*PyObject *globals*/
            0                                /*PyObject *locals*/
        );
        if (*frame == NULL) return 0;
        if (CYTHON_TRACE && (*frame)->f_trace == NULL) {
            // this enables "f_lineno" lookup, at least in CPython ...
            Py_INCREF(Py_None);
            (*frame)->f_trace = Py_None;
        }
#if PY_VERSION_HEX < 0x030400B1
    } else {
        (*frame)->f_tstate = tstate;
#endif
    }
    __Pyx_PyFrame_SetLineNumber(*frame, firstlineno);

    retval = 1;
    __Pyx_EnterTracing(tstate);
    __Pyx_ErrFetchInState(tstate, &type, &value, &traceback);

    #if CYTHON_TRACE
    if (tstate->c_tracefunc)
        retval = tstate->c_tracefunc(tstate->c_traceobj, *frame, PyTrace_CALL, NULL) == 0;
    if (retval && tstate->c_profilefunc)
    #endif
        retval = tstate->c_profilefunc(tstate->c_profileobj, *frame, PyTrace_CALL, NULL) == 0;

    __Pyx_LeaveTracing(tstate);
    if (retval) {
        __Pyx_ErrRestoreInState(tstate, type, value, traceback);
        return __Pyx_IsTracing(tstate, 0, 0) && retval;
    } else {
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        return -1;
    }
}

static PyCodeObject *__Pyx_createFrameCodeObject(const char *funcname, const char *srcfile, int firstlineno) {
    PyCodeObject *py_code = 0;

#if PY_MAJOR_VERSION >= 3
    py_code = PyCode_NewEmpty(srcfile, funcname, firstlineno);
    // make CPython use a fresh dict for "f_locals" at need (see GH #1836)
    if (likely(py_code)) {
        py_code->co_flags |= CO_OPTIMIZED | CO_NEWLOCALS;
    }
#else
    PyObject *py_srcfile = 0;
    PyObject *py_funcname = 0;

    py_funcname = PyString_FromString(funcname);
    if (unlikely(!py_funcname)) goto bad;
    py_srcfile = PyString_FromString(srcfile);
    if (unlikely(!py_srcfile)) goto bad;

    py_code = PyCode_New(
        0,                /*int argcount,*/
        0,                /*int nlocals,*/
        0,                /*int stacksize,*/
        // make CPython use a fresh dict for "f_locals" at need (see GH #1836)
        CO_OPTIMIZED | CO_NEWLOCALS,  /*int flags,*/
        $empty_bytes,     /*PyObject *code,*/
        $empty_tuple,     /*PyObject *consts,*/
        $empty_tuple,     /*PyObject *names,*/
        $empty_tuple,     /*PyObject *varnames,*/
        $empty_tuple,     /*PyObject *freevars,*/
        $empty_tuple,     /*PyObject *cellvars,*/
        py_srcfile,       /*PyObject *filename,*/
        py_funcname,      /*PyObject *name,*/
        firstlineno,      /*int firstlineno,*/
        $empty_bytes      /*PyObject *lnotab*/
    );

bad:
    Py_XDECREF(py_srcfile);
    Py_XDECREF(py_funcname);
#endif

    return py_code;
}

#endif /* CYTHON_PROFILE */
