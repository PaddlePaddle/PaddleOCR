# http://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/wait.h.html

from posix.types cimport pid_t, id_t
from posix.signal cimport siginfo_t
from posix.resource cimport rusage

cdef extern from "<sys/wait.h>" nogil:
    enum: WNOHANG
    enum: WUNTRACED
    enum: WCONTINUED
    enum: WEXITED
    enum: WSTOPPED
    enum: WNOWAIT

    int WEXITSTATUS(int status)
    int WIFCONTINUED(int status)
    int WIFEXITED(int status)
    int WIFSIGNALED(int status)
    int WIFSTOPPED(int status)
    int WSTOPSIG(int status)
    int WTERMSIG(int status)

    ctypedef int idtype_t
    enum: P_ALL             # idtype_t values
    enum: P_PID
    enum: P_PGID

    pid_t wait(int *stat_loc)
    pid_t waitpid(pid_t pid, int *status, int options)
    int waitid(idtype_t idtype, id_t id, siginfo_t *infop, int options)

# wait3 was in POSIX until 2008 while wait4 was never standardized.
# Even so, these calls are in almost every Unix, always in sys/wait.h.
# Hence, posix.wait is the least surprising place to declare them for Cython.
# libc may require _XXX_SOURCE to be defined at C-compile time to provide them.

    pid_t wait3(int *status, int options, rusage *rusage)
    pid_t wait4(pid_t pid, int *status, int options, rusage *rusage)
