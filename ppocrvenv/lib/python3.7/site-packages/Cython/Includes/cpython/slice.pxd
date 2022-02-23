cdef extern from "Python.h":

    # PyTypeObject PySlice_Type
    #
    # The type object for slice objects. This is the same as slice and types.SliceType

    bint PySlice_Check(object ob)
    #
    # Return true if ob is a slice object; ob must not be NULL.

    slice PySlice_New(object start, object stop, object step)
    #
    # Return a new slice object with the given values. The start, stop, and step
    # parameters are used as the values of the slice object attributes of the same
    # names. Any of the values may be NULL, in which case the None will be used
    # for the corresponding attribute. Return NULL if the new object could not be
    # allocated.

    int PySlice_GetIndices(object slice, Py_ssize_t length,
                           Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step) except? -1
    #
    # Retrieve the start, stop and step indices from the slice object slice,
    # assuming a sequence of length length. Treats indices greater than length
    # as errors.
    #
    # Returns 0 on success and -1 on error with no exception set (unless one
    # of the indices was not None and failed to be converted to an integer,
    # in which case -1 is returned with an exception set).
    #
    # You probably do not want to use this function.
    #
    # Changed in version 3.2: The parameter type for the slice parameter was
    # PySliceObject* before.

    int PySlice_GetIndicesEx(object slice, Py_ssize_t length,
                             Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step,
                             Py_ssize_t *slicelength) except -1
    #
    # Usable replacement for PySlice_GetIndices(). Retrieve the start, stop, and step
    # indices from the slice object slice assuming a sequence of length length, and
    # store the length of the slice in slicelength. Out of bounds indices are clipped
    # in a manner consistent with the handling of normal slices.
    #
    # Returns 0 on success and -1 on error with exception set.
    #
    # Changed in version 3.2: The parameter type for the slice parameter was
    # PySliceObject* before.

    int PySlice_Unpack(object slice, Py_ssize_t *start, Py_ssize_t *stop,
                       Py_ssize_t *step) except -1
    # Extract the start, stop and step data members from a slice object as C
    # integers. Silently reduce values larger than PY_SSIZE_T_MAX to
    # PY_SSIZE_T_MAX, silently boost the start and stop values less than
    # PY_SSIZE_T_MIN to PY_SSIZE_T_MIN, and silently boost the step values
    # less than -PY_SSIZE_T_MAX to -PY_SSIZE_T_MAX.

    # Return -1 on error, 0 on success.

    # New in version 3.6.1.

    Py_ssize_t PySlice_AdjustIndices(Py_ssize_t length, Py_ssize_t *start,
                                     Py_ssize_t *stop, Py_ssize_t step)
    # Adjust start/end slice indices assuming a sequence of the specified
    # length. Out of bounds indices are clipped in a manner consistent with
    # the handling of normal slices.

    # Return the length of the slice. Always successful. Doesn’t call Python
    # code.

    # New in version 3.6.1.
