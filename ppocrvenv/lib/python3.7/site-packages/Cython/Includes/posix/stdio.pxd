# POSIX additions to <stdio.h>.
# http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/stdio.h.html

from libc.stdio cimport FILE
from libc.stddef cimport wchar_t
from posix.types cimport off_t

cdef extern from "<stdio.h>" nogil:
    # File descriptors
    FILE *fdopen(int, const char *)
    int fileno(FILE *)

    # Pipes
    FILE *popen(const char *, const char *)
    int pclose(FILE *)

    # Memory streams (POSIX.2008)
    FILE *fmemopen(void *, size_t, const char *)
    FILE *open_memstream(char **, size_t *)
    FILE *open_wmemstream(wchar_t **, size_t *)

    # Seek and tell with off_t
    int fseeko(FILE *, off_t, int)
    off_t ftello(FILE *)

    # Locking (for multithreading)
    void flockfile(FILE *)
    int ftrylockfile(FILE *)
    void funlockfile(FILE *)
    int getc_unlocked(FILE *)
    int getchar_unlocked()
    int putc_unlocked(int, FILE *)
    int putchar_unlocked(int)

    # Reading lines and records (POSIX.2008)
    ssize_t getline(char **, size_t *, FILE *)
    ssize_t getdelim(char **, size_t *, int, FILE *)
