from pandas._libs.util cimport numeric


cdef numeric kth_smallest_c(numeric* arr, Py_ssize_t k, Py_ssize_t n) nogil
