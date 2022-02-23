from .object cimport PyObject

cdef extern from "Python.h":
    bint PyByteArray_Check(object o)
    # Return true if the object o is a bytearray object or an instance of a subtype of the bytearray type.

    bint PyByteArray_CheckExact(object o)
    # Return true if the object o is a bytearray object, but not an instance of a subtype of the bytearray type.

    bytearray PyByteArray_FromObject(object o)
    # Return a new bytearray object from any object, o, that implements the buffer protocol.

    bytearray PyByteArray_FromStringAndSize(char *string, Py_ssize_t len)
    # Create a new bytearray object from string and its length, len. On failure, NULL is returned.

    bytearray PyByteArray_Concat(object a, object b)
    # Concat bytearrays a and b and return a new bytearray with the result.

    Py_ssize_t PyByteArray_Size(object bytearray)
    # Return the size of bytearray after checking for a NULL pointer.

    char* PyByteArray_AsString(object bytearray)
    # Return the contents of bytearray as a char array after checking for a NULL pointer.
    # The returned array always has an extra null byte appended.

    int PyByteArray_Resize(object bytearray, Py_ssize_t len)
    # Resize the internal buffer of bytearray to len.

    char* PyByteArray_AS_STRING(object bytearray)
    # Macro version of PyByteArray_AsString().

    Py_ssize_t PyByteArray_GET_SIZE(object bytearray)
    # Macro version of PyByteArray_Size().
