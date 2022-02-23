'''
Locking constants

Lock types:

- `EXCLUSIVE` exclusive lock
- `SHARED` shared lock

Lock flags:

- `NON_BLOCKING` non-blocking

Manually unlock, only needed internally

- `UNBLOCK` unlock
'''
import enum
import os

# The actual tests will execute the code anyhow so the following code can
# safely be ignored from the coverage tests
if os.name == 'nt':  # pragma: no cover
    import msvcrt

    LOCK_EX = 0x1  #: exclusive lock
    LOCK_SH = 0x2  #: shared lock
    LOCK_NB = 0x4  #: non-blocking
    LOCK_UN = msvcrt.LK_UNLCK  #: unlock

elif os.name == 'posix':  # pragma: no cover
    import fcntl

    LOCK_EX = fcntl.LOCK_EX  #: exclusive lock
    LOCK_SH = fcntl.LOCK_SH  #: shared lock
    LOCK_NB = fcntl.LOCK_NB  #: non-blocking
    LOCK_UN = fcntl.LOCK_UN  #: unlock

else:  # pragma: no cover
    raise RuntimeError('PortaLocker only defined for nt and posix platforms')


class LockFlags(enum.IntFlag):
    EXCLUSIVE = LOCK_EX  #: exclusive lock
    SHARED = LOCK_SH  #: shared lock
    NON_BLOCKING = LOCK_NB  #: non-blocking
    UNBLOCK = LOCK_UN  #: unlock
