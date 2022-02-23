cdef extern from "../Zeros/zeros.h":
    ctypedef double (*callback_type)(double, void*)
    ctypedef struct scipy_zeros_info:
        int funcalls
        int iterations
        int error_num

cdef extern from "../Zeros/bisect.c" nogil:
    double bisect(callback_type f, double xa, double xb, double xtol,
                  double rtol, int iter, void *func_data,
                  scipy_zeros_info *solver_stats)

cdef extern from "../Zeros/ridder.c" nogil:
    double ridder(callback_type f, double xa, double xb, double xtol,
                  double rtol, int iter, void *func_data,
                  scipy_zeros_info *solver_stats)

cdef extern from "../Zeros/brenth.c" nogil:
    double brenth(callback_type f, double xa, double xb, double xtol,
                  double rtol, int iter, void *func_data,
                  scipy_zeros_info *solver_stats)

cdef extern from "../Zeros/brentq.c" nogil:
    double brentq(callback_type f, double xa, double xb, double xtol,
                  double rtol, int iter, void *func_data,
                  scipy_zeros_info *solver_stats)
