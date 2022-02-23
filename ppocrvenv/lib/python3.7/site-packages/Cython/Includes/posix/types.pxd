# Note that the actual size of these types is system-dependent, and
# can't be detected at C compile time.  However, the generated C code
# will correctly use the actual size of these types *except* for
# determining promotion in binary arithmetic expressions involving
# mixed types.  In this case, operands are promoted to the declared
# larger type, with a bias towards typedef types.  Thus, with the
# declarations below, long + time_t will result in a time_t whereas
# long long + time_t will result in a long long which should be
# acceptable for either 32-bit or 64-bit signed time_t (though admittedly
# the POSIX standard doesn't even specify that time_t must be an integral
# type).

cdef extern from "<sys/types.h>":
    ctypedef long blkcnt_t
    ctypedef long blksize_t
    ctypedef long clockid_t
    ctypedef long dev_t
    ctypedef long gid_t
    ctypedef long id_t
    ctypedef unsigned long ino_t
    ctypedef long mode_t
    ctypedef long nlink_t
    ctypedef long off_t
    ctypedef long pid_t
    ctypedef struct sigset_t:
        pass
    ctypedef long suseconds_t
    ctypedef long time_t
    ctypedef long timer_t
    ctypedef long uid_t
