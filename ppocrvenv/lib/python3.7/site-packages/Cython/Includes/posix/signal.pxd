# 7.14 Signal handling <signal.h>

from posix.types cimport pid_t, sigset_t, uid_t

cdef extern from "<signal.h>" nogil:

    cdef union sigval:
        int  sival_int
        void *sival_ptr

    cdef struct sigevent:
        int    sigev_notify
        int    sigev_signo
        sigval sigev_value
        void   sigev_notify_function(sigval)

    ctypedef struct siginfo_t:
        int    si_signo
        int    si_code
        int    si_errno
        pid_t  si_pid
        uid_t  si_uid
        void   *si_addr
        int    si_status
        long   si_band
        sigval si_value

    cdef struct sigaction_t "sigaction":
        void     sa_handler(int)
        void     sa_sigaction(int, siginfo_t *, void *)
        sigset_t sa_mask
        int      sa_flags

    ctypedef struct stack_t:
        void  *ss_sp
        int ss_flags
        size_t ss_size

    enum: SA_NOCLDSTOP
    enum: SIG_BLOCK
    enum: SIG_UNBLOCK
    enum: SIG_SETMASK
    enum: SA_ONSTACK
    enum: SA_RESETHAND
    enum: SA_RESTART
    enum: SA_SIGINFO
    enum: SA_NOCLDWAIT
    enum: SA_NODEFER
    enum: SS_ONSTACK
    enum: SS_DISABLE
    enum: MINSIGSTKSZ
    enum: SIGSTKSZ

    enum: SIGEV_NONE
    enum: SIGEV_SIGNAL
    enum: SIGEV_THREAD
    enum: SIGEV_THREAD_ID


    int          kill          (pid_t, int)
    int          killpg        (pid_t, int)
    int          sigaction     (int, const sigaction_t *, sigaction_t *)
    int          sigpending    (sigset_t *)
    int          sigprocmask   (int, const sigset_t *, sigset_t *)
    int          sigsuspend    (const sigset_t *)

    int          sigaddset     (sigset_t *, int)
    int          sigdelset     (sigset_t *, int)
    int          sigemptyset   (sigset_t *)
    int          sigfillset    (sigset_t *)
    int          sigismember   (const sigset_t *, int)

    int sigaltstack(const stack_t *, stack_t *)
