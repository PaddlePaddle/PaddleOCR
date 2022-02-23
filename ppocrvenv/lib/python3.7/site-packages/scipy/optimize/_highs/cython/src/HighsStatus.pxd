# distutils: language=c++
# cython: language_level=3

from libcpp.string cimport string

cdef extern from "HighsStatus.h" nogil:
    ctypedef enum HighsStatus:
        HighsStatusOK "HighsStatus::OK"
        HighsStatusWarning "HighsStatus::Warning"
        HighsStatusError "HighsStatus::Error"

    string HighsStatusToString(HighsStatus status)
