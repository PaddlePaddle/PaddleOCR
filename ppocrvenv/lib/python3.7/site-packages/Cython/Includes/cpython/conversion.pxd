# From https://docs.python.org/3/c-api/conversion.html

from .object cimport PyObject

cdef extern from "Python.h":
    ctypedef struct va_list

    int PyOS_snprintf(char *str, size_t size, const char *format, ...)
    # Output not more than size bytes to str according to the format
    # string format and the extra arguments. See the Unix man page snprintf(2).

    int PyOS_vsnprintf(char *str, size_t size, const char *format, va_list va)
    # Output not more than size bytes to str according to the format
    # string format and the variable argument list va. Unix man page vsnprintf(2).

    double PyOS_string_to_double(const char *s, char **endptr, PyObject *overflow_exception) except? -1.0
    # Convert a string s to a double, raising a Python exception on failure. The set of
    # accepted strings corresponds to the set of strings accepted by Python’s float()
    # constructor, except that s must not have leading or trailing whitespace.
    # The conversion is independent of the current locale.

    enum:
        Py_DTSF_SIGN
        Py_DTSF_ADD_DOT_0
        Py_DTSF_ALT

    char* PyOS_double_to_string(double val, char format_code, int precision, int flags, int *ptype) except NULL
    # Convert a double val to a string using supplied format_code, precision, and flags.

    int PyOS_stricmp(const char *s1, const char *s2)
    # Case insensitive comparison of strings. The function works almost identically
    # to strcmp() except that it ignores the case.

    int PyOS_strnicmp(const char *s1, const char *s2, Py_ssize_t  size)
    # Case insensitive comparison of strings. The function works almost identically
    # to strncmp() except that it ignores the case.
