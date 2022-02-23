
cdef extern from "Python.h":

    void PyEval_InitThreads()
        # Initialize and acquire the global interpreter lock.

    int PyEval_ThreadsInitialized()
        # Returns a non-zero value if PyEval_InitThreads() has been called.
