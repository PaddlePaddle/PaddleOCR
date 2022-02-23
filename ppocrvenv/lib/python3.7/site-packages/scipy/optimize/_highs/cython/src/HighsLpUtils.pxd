# distutils: language=c++
# cython: language_level=3

from .HighsStatus cimport HighsStatus
from .HighsLp cimport HighsLp
from .HighsOptions cimport HighsOptions

cdef extern from "HighsLpUtils.h" nogil:
    # From HiGHS/src/lp_data/HighsLpUtils.h
    HighsStatus assessLp(HighsLp& lp, const HighsOptions& options)
