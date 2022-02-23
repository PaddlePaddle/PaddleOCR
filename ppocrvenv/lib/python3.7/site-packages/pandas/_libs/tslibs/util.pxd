
from cpython.object cimport PyTypeObject


cdef extern from *:
    """
    PyObject* char_to_string(const char* data) {
        return PyUnicode_FromString(data);
    }
    """
    object char_to_string(const char* data)


cdef extern from "Python.h":
    # Note: importing extern-style allows us to declare these as nogil
    # functions, whereas `from cpython cimport` does not.
    bint PyUnicode_Check(object obj) nogil
    bint PyBool_Check(object obj) nogil
    bint PyFloat_Check(object obj) nogil
    bint PyComplex_Check(object obj) nogil
    bint PyObject_TypeCheck(object obj, PyTypeObject* type) nogil

    # Note that following functions can potentially raise an exception,
    # thus they cannot be declared 'nogil'. Also PyUnicode_AsUTF8AndSize() can
    # potentially allocate memory inside in unlikely case of when underlying
    # unicode object was stored as non-utf8 and utf8 wasn't requested before.
    const char* PyUnicode_AsUTF8AndSize(object obj,
                                        Py_ssize_t* length) except NULL

from numpy cimport (
    float64_t,
    int64_t,
)


cdef extern from "numpy/arrayobject.h":
    PyTypeObject PyFloatingArrType_Type

cdef extern from "numpy/ndarrayobject.h":
    PyTypeObject PyTimedeltaArrType_Type
    PyTypeObject PyDatetimeArrType_Type
    PyTypeObject PyComplexFloatingArrType_Type
    PyTypeObject PyBoolArrType_Type

    bint PyArray_IsIntegerScalar(obj) nogil
    bint PyArray_Check(obj) nogil

cdef extern from "numpy/npy_common.h":
    int64_t NPY_MIN_INT64


cdef inline int64_t get_nat():
    return NPY_MIN_INT64


# --------------------------------------------------------------------
# Type Checking

cdef inline bint is_integer_object(object obj) nogil:
    """
    Cython equivalent of

    `isinstance(val, (int, long, np.integer)) and not isinstance(val, bool)`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_integer : bool

    Notes
    -----
    This counts np.timedelta64 objects as integers.
    """
    return (not PyBool_Check(obj) and PyArray_IsIntegerScalar(obj)
            and not is_timedelta64_object(obj))


cdef inline bint is_float_object(object obj) nogil:
    """
    Cython equivalent of `isinstance(val, (float, np.complex_))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_float : bool
    """
    return (PyFloat_Check(obj) or
            (PyObject_TypeCheck(obj, &PyFloatingArrType_Type)))


cdef inline bint is_complex_object(object obj) nogil:
    """
    Cython equivalent of `isinstance(val, (complex, np.complex_))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_complex : bool
    """
    return (PyComplex_Check(obj) or
            PyObject_TypeCheck(obj, &PyComplexFloatingArrType_Type))


cdef inline bint is_bool_object(object obj) nogil:
    """
    Cython equivalent of `isinstance(val, (bool, np.bool_))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_bool : bool
    """
    return (PyBool_Check(obj) or
            PyObject_TypeCheck(obj, &PyBoolArrType_Type))


cdef inline bint is_real_number_object(object obj) nogil:
    return is_bool_object(obj) or is_integer_object(obj) or is_float_object(obj)


cdef inline bint is_timedelta64_object(object obj) nogil:
    """
    Cython equivalent of `isinstance(val, np.timedelta64)`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_timedelta64 : bool
    """
    return PyObject_TypeCheck(obj, &PyTimedeltaArrType_Type)


cdef inline bint is_datetime64_object(object obj) nogil:
    """
    Cython equivalent of `isinstance(val, np.datetime64)`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_datetime64 : bool
    """
    return PyObject_TypeCheck(obj, &PyDatetimeArrType_Type)


cdef inline bint is_array(object val):
    """
    Cython equivalent of `isinstance(val, np.ndarray)`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_ndarray : bool
    """
    return PyArray_Check(val)


cdef inline bint is_nan(object val):
    """
    Check if val is a Not-A-Number float or complex, including
    float('NaN') and np.nan.

    Parameters
    ----------
    val : object

    Returns
    -------
    is_nan : bool
    """
    cdef float64_t fval
    if is_float_object(val):
        fval = val
        return fval != fval
    return is_complex_object(val) and val != val


cdef inline const char* get_c_string_buf_and_size(str py_string,
                                                  Py_ssize_t *length) except NULL:
    """
    Extract internal char* buffer of unicode or bytes object `py_string` with
    getting length of this internal buffer saved in `length`.

    Notes
    -----
    Python object owns memory, thus returned char* must not be freed.
    `length` can be NULL if getting buffer length is not needed.

    Parameters
    ----------
    py_string : str
    length : Py_ssize_t*

    Returns
    -------
    buf : const char*
    """
    return PyUnicode_AsUTF8AndSize(py_string, length)


cdef inline const char* get_c_string(str py_string) except NULL:
    return get_c_string_buf_and_size(py_string, NULL)
