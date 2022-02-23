# distutils: language=c++
# cython: language_level=3

cdef extern from "highs_c_api.h" nogil:
    int Highs_passLp(void* highs, int numcol, int numrow, int numnz,
                     double* colcost, double* collower, double* colupper,
                     double* rowlower, double* rowupper,
                     int* astart, int* aindex,  double* avalue)
