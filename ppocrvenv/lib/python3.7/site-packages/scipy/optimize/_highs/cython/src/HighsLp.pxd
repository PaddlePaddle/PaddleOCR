# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from .HConst cimport HighsBasisStatus

cdef extern from "HighsLp.h" nogil:
    # From HiGHS/src/lp_data/HighsLp.h
    cdef cppclass HighsLp:
        int numCol_
        int numRow_

        vector[int] Astart_
        vector[int] Aindex_
        vector[double] Avalue_
        vector[double] colCost_
        vector[double] colLower_
        vector[double] colUpper_
        vector[double] rowLower_
        vector[double] rowUpper_

        ObjSense sense_
        double offset_

        string model_name_
        string lp_name_

        vector[string] row_names_
        vector[string] col_names_

        vector[int] integrality_

    ctypedef enum ObjSense:
        ObjSenseMINIMIZE "ObjSense::MINIMIZE" = 1
        ObjSenseMAXIMIZE "ObjSense::MAXIMIZE" = -1

    cdef cppclass HighsSolution:
        vector[double] col_value
        vector[double] col_dual
        vector[double] row_value
        vector[double] row_dual

    cdef cppclass HighsBasis:
        bool valid_
        vector[HighsBasisStatus] col_status
        vector[HighsBasisStatus] row_status
