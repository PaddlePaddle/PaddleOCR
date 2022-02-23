# Declare the class with cdef
cdef extern from "biasedurn/stocc.h" nogil:
    cdef cppclass CFishersNCHypergeometric:
        CFishersNCHypergeometric(int, int, int, double, double) except +
        int mode()
        double mean()
        double variance()
        double probability(int x)
        double moments(double * mean, double * var)

    cdef cppclass CWalleniusNCHypergeometric:
        CWalleniusNCHypergeometric() except +
        CWalleniusNCHypergeometric(int, int, int, double, double) except +
        int mode()
        double mean()
        double variance()
        double probability(int x)
        double moments(double * mean, double * var)

    cdef cppclass StochasticLib3:
        StochasticLib3(int seed) except +
        double Random() except +
        void SetAccuracy(double accur)
        int FishersNCHyp (int n, int m, int N, double odds) except +
        int WalleniusNCHyp (int n, int m, int N, double odds) except +
        double(*next_double)()
        double(*next_normal)(const double m, const double s)
