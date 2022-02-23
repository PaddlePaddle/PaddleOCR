
#------------------------------------------------------------------------------

"""
Dataset abstractions for sequential data access.
WARNING: Do not edit .pxd file directly, it is generated from .pxd.tp
"""

cimport numpy as np

# SequentialDataset and its two concrete subclasses are (optionally randomized)
# iterators over the rows of a matrix X and corresponding target values y.


cdef class SequentialDataset64:
    cdef int current_index
    cdef np.ndarray index
    cdef int *index_data_ptr
    cdef Py_ssize_t n_samples
    cdef np.uint32_t seed

    cdef void shuffle(self, np.uint32_t seed) nogil
    cdef int _get_next_index(self) nogil
    cdef int _get_random_index(self) nogil

    cdef void _sample(self, double **x_data_ptr, int **x_ind_ptr,
                      int *nnz, double *y, double *sample_weight,
                      int current_index) nogil
    cdef void next(self, double **x_data_ptr, int **x_ind_ptr,
                   int *nnz, double *y, double *sample_weight) nogil
    cdef int random(self, double **x_data_ptr, int **x_ind_ptr,
                    int *nnz, double *y, double *sample_weight) nogil


cdef class ArrayDataset64(SequentialDataset64):
    cdef np.ndarray X
    cdef np.ndarray Y
    cdef np.ndarray sample_weights
    cdef Py_ssize_t n_features
    cdef np.npy_intp X_stride
    cdef double *X_data_ptr
    cdef double *Y_data_ptr
    cdef np.ndarray feature_indices
    cdef int *feature_indices_ptr
    cdef double *sample_weight_data


cdef class CSRDataset64(SequentialDataset64):
    cdef np.ndarray X_data
    cdef np.ndarray X_indptr
    cdef np.ndarray X_indices
    cdef np.ndarray Y
    cdef np.ndarray sample_weights
    cdef double *X_data_ptr
    cdef int *X_indptr_ptr
    cdef int *X_indices_ptr
    cdef double *Y_data_ptr
    cdef double *sample_weight_data

#------------------------------------------------------------------------------

"""
Dataset abstractions for sequential data access.
WARNING: Do not edit .pxd file directly, it is generated from .pxd.tp
"""

cimport numpy as np

# SequentialDataset and its two concrete subclasses are (optionally randomized)
# iterators over the rows of a matrix X and corresponding target values y.


cdef class SequentialDataset32:
    cdef int current_index
    cdef np.ndarray index
    cdef int *index_data_ptr
    cdef Py_ssize_t n_samples
    cdef np.uint32_t seed

    cdef void shuffle(self, np.uint32_t seed) nogil
    cdef int _get_next_index(self) nogil
    cdef int _get_random_index(self) nogil

    cdef void _sample(self, float **x_data_ptr, int **x_ind_ptr,
                      int *nnz, float *y, float *sample_weight,
                      int current_index) nogil
    cdef void next(self, float **x_data_ptr, int **x_ind_ptr,
                   int *nnz, float *y, float *sample_weight) nogil
    cdef int random(self, float **x_data_ptr, int **x_ind_ptr,
                    int *nnz, float *y, float *sample_weight) nogil


cdef class ArrayDataset32(SequentialDataset32):
    cdef np.ndarray X
    cdef np.ndarray Y
    cdef np.ndarray sample_weights
    cdef Py_ssize_t n_features
    cdef np.npy_intp X_stride
    cdef float *X_data_ptr
    cdef float *Y_data_ptr
    cdef np.ndarray feature_indices
    cdef int *feature_indices_ptr
    cdef float *sample_weight_data


cdef class CSRDataset32(SequentialDataset32):
    cdef np.ndarray X_data
    cdef np.ndarray X_indptr
    cdef np.ndarray X_indices
    cdef np.ndarray Y
    cdef np.ndarray sample_weights
    cdef float *X_data_ptr
    cdef int *X_indptr_ptr
    cdef int *X_indices_ptr
    cdef float *Y_data_ptr
    cdef float *sample_weight_data
