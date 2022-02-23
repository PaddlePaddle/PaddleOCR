# Legacy public Cython API declarations
#
# NOTE: due to the way Cython ABI compatibility works, **no changes
# should be made to this file** --- any API additions/changes should be
# done in `cython_optimize.pxd` (see gh-11793).

ctypedef double (*callback_type)(double, void*)

ctypedef struct zeros_parameters:
    callback_type function
    void* args

ctypedef struct zeros_full_output:
    int funcalls
    int iterations
    int error_num
    double root

cdef double bisect(callback_type f, double xa, double xb, void* args,
                   double xtol, double rtol, int iter,
                   zeros_full_output *full_output) nogil

cdef double ridder(callback_type f, double xa, double xb, void* args,
                   double xtol, double rtol, int iter,
                   zeros_full_output *full_output) nogil

cdef double brenth(callback_type f, double xa, double xb, void* args,
                   double xtol, double rtol, int iter,
                   zeros_full_output *full_output) nogil

cdef double brentq(callback_type f, double xa, double xb, void* args,
                   double xtol, double rtol, int iter,
                   zeros_full_output *full_output) nogil
