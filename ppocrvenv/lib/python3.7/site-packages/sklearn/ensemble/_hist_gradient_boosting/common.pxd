import numpy as np
cimport numpy as np

np.import_array()


ctypedef np.npy_float64 X_DTYPE_C
ctypedef np.npy_uint8 X_BINNED_DTYPE_C
ctypedef np.npy_float64 Y_DTYPE_C
ctypedef np.npy_float32 G_H_DTYPE_C
ctypedef np.npy_uint32 BITSET_INNER_DTYPE_C
ctypedef BITSET_INNER_DTYPE_C[8] BITSET_DTYPE_C

cdef packed struct hist_struct:
    # Same as histogram dtype but we need a struct to declare views. It needs
    # to be packed since by default numpy dtypes aren't aligned
    Y_DTYPE_C sum_gradients
    Y_DTYPE_C sum_hessians
    unsigned int count


cdef packed struct node_struct:
    # Equivalent struct to PREDICTOR_RECORD_DTYPE to use in memory views. It
    # needs to be packed since by default numpy dtypes aren't aligned
    Y_DTYPE_C value
    unsigned int count
    unsigned int feature_idx
    X_DTYPE_C num_threshold
    unsigned char missing_go_to_left
    unsigned int left
    unsigned int right
    Y_DTYPE_C gain
    unsigned int depth
    unsigned char is_leaf
    X_BINNED_DTYPE_C bin_threshold
    unsigned char is_categorical
    # The index of the corresponding bitsets in the Predictor's bitset arrays.
    # Only used if is_categorical is True
    unsigned int bitset_idx

cpdef enum MonotonicConstraint:
    NO_CST = 0
    POS = 1
    NEG = -1
