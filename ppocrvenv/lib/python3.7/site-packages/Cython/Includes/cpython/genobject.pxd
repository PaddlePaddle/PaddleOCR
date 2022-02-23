from .pystate cimport PyFrameObject

cdef extern from "Python.h":

    ###########################################################################
    # Generator Objects
    ###########################################################################

    bint PyGen_Check(object ob)
    # Return true if ob is a generator object; ob must not be NULL.

    bint PyGen_CheckExact(object ob)
    # Return true if ob's type is PyGen_Type; ob must not be NULL.

    object PyGen_New(PyFrameObject *frame)
    # Return value: New reference.
    # Create and return a new generator object based on the frame object. A
    # reference to frame is stolen by this function. The argument must not be
    # NULL.

    object PyGen_NewWithQualName(PyFrameObject *frame, object name, object qualname)
    # Return value: New reference.
    # Create and return a new generator object based on the frame object, with
    # __name__ and __qualname__ set to name and qualname. A reference to frame
    # is stolen by this function. The frame argument must not be NULL.
