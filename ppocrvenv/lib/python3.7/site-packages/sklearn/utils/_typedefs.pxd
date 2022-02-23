#!python
cimport numpy as np

# Floating point/data type
ctypedef np.float64_t DTYPE_t  # WARNING: should match DTYPE in typedefs.pyx

cdef enum:
    DTYPECODE = np.NPY_FLOAT64
    ITYPECODE = np.NPY_INTP

# Index/integer type.
#  WARNING: ITYPE_t must be a signed integer type or you will have a bad time!
ctypedef np.intp_t ITYPE_t  # WARNING: should match ITYPE in typedefs.pyx

# Fused type for certain operations
ctypedef fused DITYPE_t:
    ITYPE_t
    DTYPE_t
