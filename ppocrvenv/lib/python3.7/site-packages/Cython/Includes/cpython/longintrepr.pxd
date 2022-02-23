# Internals of the "long" type (Python 2) or "int" type (Python 3).

cdef extern from "Python.h":
    """
    #if PY_MAJOR_VERSION < 3
     #include "longintrepr.h"
    #endif
    """
    ctypedef unsigned int digit
    ctypedef int sdigit  # Python >= 2.7 only

    ctypedef class __builtin__.py_long [object PyLongObject]:
        cdef digit* ob_digit

    cdef py_long _PyLong_New(Py_ssize_t s)

    cdef long PyLong_SHIFT
    cdef digit PyLong_BASE
    cdef digit PyLong_MASK
