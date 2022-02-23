# http://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/time.h.html

from posix.types cimport suseconds_t, time_t, clockid_t, timer_t
from posix.signal cimport sigevent

cdef extern from "<sys/time.h>" nogil:
    enum: CLOCK_REALTIME
    enum: TIMER_ABSTIME
    enum: CLOCK_MONOTONIC

    # FreeBSD-specific clocks
    enum: CLOCK_UPTIME
    enum: CLOCK_UPTIME_PRECISE
    enum: CLOCK_UPTIME_FAST
    enum: CLOCK_REALTIME_PRECISE
    enum: CLOCK_REALTIME_FAST
    enum: CLOCK_MONOTONIC_PRECISE
    enum: CLOCK_MONOTONIC_FAST
    enum: CLOCK_SECOND

    # Linux-specific clocks
    enum: CLOCK_PROCESS_CPUTIME_ID
    enum: CLOCK_THREAD_CPUTIME_ID
    enum: CLOCK_MONOTONIC_RAW
    enum: CLOCK_REALTIME_COARSE
    enum: CLOCK_MONOTONIC_COARSE
    enum: CLOCK_BOOTTIME
    enum: CLOCK_REALTIME_ALARM
    enum: CLOCK_BOOTTIME_ALARM

    enum: ITIMER_REAL
    enum: ITIMER_VIRTUAL
    enum: ITIMER_PROF

    cdef struct timezone:
        int tz_minuteswest
        int dsttime

    cdef struct timeval:
        time_t      tv_sec
        suseconds_t tv_usec

    cdef struct timespec:
        time_t tv_sec
        long   tv_nsec

    cdef struct itimerval:
        timeval it_interval
        timeval it_value

    cdef struct itimerspec:
        timespec it_interval
        timespec it_value

    int nanosleep(const timespec *, timespec *)

    int getitimer(int, itimerval *)
    int gettimeofday(timeval *tp, timezone *tzp)
    int setitimer(int, const itimerval *, itimerval *)

    int clock_getcpuclockid(pid_t, clockid_t *)
    int clock_getres(clockid_t, timespec *)
    int clock_gettime(clockid_t, timespec *)
    int clock_nanosleep(clockid_t, int, const timespec *, timespec *)
    int clock_settime(clockid_t, const timespec *)

    int timer_create(clockid_t, sigevent *, timer_t *)
    int timer_delete(timer_t)
    int timer_gettime(timer_t, itimerspec *)
    int timer_getoverrun(timer_t)
    int timer_settime(timer_t, int, const itimerspec *, itimerspec *)
