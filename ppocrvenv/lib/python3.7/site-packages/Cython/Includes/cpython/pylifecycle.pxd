# Interfaces to configure, query, create & destroy the Python runtime

from libc.stdio cimport FILE
from .pystate cimport PyThreadState


cdef extern from "Python.h":
    ctypedef int wchar_t

    void Py_SetProgramName(wchar_t *)
    wchar_t *Py_GetProgramName()

    void Py_SetPythonHome(wchar_t *)
    wchar_t *Py_GetPythonHome()

    # Only used by applications that embed the interpreter and need to
    # override the standard encoding determination mechanism
    int Py_SetStandardStreamEncoding(const char *encoding, const char *errors)

    void Py_Initialize()
    void Py_InitializeEx(int)
    void _Py_InitializeEx_Private(int, int)
    void Py_Finalize()
    int Py_FinalizeEx()
    int Py_IsInitialized()
    PyThreadState *Py_NewInterpreter()
    void Py_EndInterpreter(PyThreadState *)


    # _Py_PyAtExit is for the atexit module, Py_AtExit is for low-level
    # exit functions.
    void _Py_PyAtExit(void (*func)(object), object)
    int Py_AtExit(void (*func)())

    void Py_Exit(int)

    # Restore signals that the interpreter has called SIG_IGN on to SIG_DFL.
    void _Py_RestoreSignals()

    int Py_FdIsInteractive(FILE *, const char *)

    # Bootstrap __main__ (defined in Modules/main.c)
    int Py_Main(int argc, wchar_t **argv)

    # In getpath.c
    wchar_t *Py_GetProgramFullPath()
    wchar_t *Py_GetPrefix()
    wchar_t *Py_GetExecPrefix()
    wchar_t *Py_GetPath()
    void Py_SetPath(const wchar_t *)
    int _Py_CheckPython3()

    # In their own files
    const char *Py_GetVersion()
    const char *Py_GetPlatform()
    const char *Py_GetCopyright()
    const char *Py_GetCompiler()
    const char *Py_GetBuildInfo()
    const char *_Py_gitidentifier()
    const char *_Py_gitversion()

    ctypedef void (*PyOS_sighandler_t)(int)
    PyOS_sighandler_t PyOS_getsig(int)
    PyOS_sighandler_t PyOS_setsig(int, PyOS_sighandler_t)

    # Random
    int _PyOS_URandom(void *buffer, Py_ssize_t size)
    int _PyOS_URandomNonblock(void *buffer, Py_ssize_t size)
