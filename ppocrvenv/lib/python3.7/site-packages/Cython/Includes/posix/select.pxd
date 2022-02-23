from .types cimport sigset_t
from .time cimport timeval, timespec

cdef extern from "<sys/select.h>" nogil:
    ctypedef struct fd_set:
        pass

    int FD_SETSIZE
    void FD_SET(int, fd_set*)
    void FD_CLR(int, fd_set*)
    bint FD_ISSET(int, fd_set*)
    void FD_ZERO(fd_set*)

    int select(int nfds, fd_set *readfds, fd_set *writefds,
        fd_set *exceptfds, const timeval *timeout)

    int pselect(int nfds, fd_set *readfds, fd_set *writefds,
        fd_set *exceptfds, const timespec *timeout,
        const sigset_t *sigmask)
