# distutils: language=c++
# cython: language_level=3

from libcpp.string cimport string

from .HConst cimport HighsModelStatus

cdef extern from "HighsModelUtils.h" nogil:
    # From HiGHS/src/lp_data/HighsModelUtils.h
    string utilHighsModelStatusToString(const HighsModelStatus model_status)
    string utilPrimalDualStatusToString(const int primal_dual_status)
