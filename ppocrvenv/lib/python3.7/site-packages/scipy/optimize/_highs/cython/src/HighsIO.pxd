# distutils: language=c++
# cython: language_level=3

from libc.stdio cimport FILE

cdef extern from "HighsIO.h" nogil:
    void HighsPrintMessage(FILE* pass_output, const int level, const char* format, ...)

    cdef enum HighsPrintMessageLevel:
        ML_MIN = 0
        ML_NONE = ML_MIN
        ML_VERBOSE = 1
        ML_DETAILED = 2
        ML_MINIMAL = 4
        ML_ALWAYS = ML_VERBOSE | ML_DETAILED | ML_MINIMAL
        ML_MAX = ML_ALWAYS
