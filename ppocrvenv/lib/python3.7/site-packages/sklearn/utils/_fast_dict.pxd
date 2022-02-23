# Author: Gael Varoquaux
# License: BSD
"""
Uses C++ map containers for fast dict-like behavior with keys being
integers, and values float.
"""

from libcpp.map cimport map as cpp_map

# Import the C-level symbols of numpy
cimport numpy as np

ctypedef np.float64_t DTYPE_t

ctypedef np.intp_t ITYPE_t

###############################################################################
# An object to be used in Python

cdef class IntFloatDict:
    cdef cpp_map[ITYPE_t, DTYPE_t] my_map
    cdef _to_arrays(self, ITYPE_t [:] keys, DTYPE_t [:] values)
