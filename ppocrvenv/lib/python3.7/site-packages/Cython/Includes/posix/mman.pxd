# http://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/mman.h.html

from posix.types cimport off_t, mode_t

cdef extern from "<sys/mman.h>" nogil:
    enum: PROT_EXEC                 # protection bits for mmap/mprotect
    enum: PROT_READ
    enum: PROT_WRITE
    enum: PROT_NONE

    enum: MAP_PRIVATE               # flag bits for mmap
    enum: MAP_SHARED
    enum: MAP_FIXED
    enum: MAP_ANON                  # These three are not in POSIX, but are
    enum: MAP_ANONYMOUS             # fairly common in spelling/semantics
    enum: MAP_STACK

    enum: MAP_LOCKED                # Typically available only on Linux
    enum: MAP_HUGETLB
    enum: MAP_POPULATE
    enum: MAP_NORESERVE
    enum: MAP_GROWSDOWN

    enum: MAP_NOCORE                #  Typically available only on BSD
    enum: MAP_NOSYNC

    void *MAP_FAILED

    void *mmap(void *addr, size_t Len, int prot, int flags, int fd, off_t off)
    int   munmap(void *addr, size_t Len)
    int   mprotect(void *addr, size_t Len, int prot)

    enum: MS_ASYNC
    enum: MS_SYNC
    enum: MS_INVALIDATE
    int   msync(void *addr, size_t Len, int flags)

    enum: POSIX_MADV_NORMAL         # POSIX advice flags
    enum: POSIX_MADV_SEQUENTIAL
    enum: POSIX_MADV_RANDOM
    enum: POSIX_MADV_WILLNEED
    enum: POSIX_MADV_DONTNEED
    int   posix_madvise(void *addr, size_t Len, int advice)

    enum: MCL_CURRENT
    enum: MCL_FUTURE
    int   mlock(const void *addr, size_t Len)
    int   munlock(const void *addr, size_t Len)
    int   mlockall(int flags)
    int   munlockall()
    # Linux-specific
    enum: MLOCK_ONFAULT
    enum: MCL_ONFAULT
    int   mlock2(const void *addr, size_t len, int flags)

    int shm_open(const char *name, int oflag, mode_t mode)
    int shm_unlink(const char *name)

    # often available
    enum: MADV_NORMAL               # pre-POSIX advice flags; should translate 1-1 to POSIX_*
    enum: MADV_RANDOM               # but in practice it is not always the same.
    enum: MADV_SEQUENTIAL
    enum: MADV_WILLNEED
    enum: MADV_DONTNEED
    enum: MADV_REMOVE               # other pre-POSIX advice flags; often available
    enum: MADV_DONTFORK
    enum: MADV_DOFORK
    enum: MADV_HWPOISON
    enum: MADV_MERGEABLE,
    enum: MADV_UNMERGEABLE
    enum: MADV_SOFT_OFFLINE
    enum: MADV_HUGEPAGE
    enum: MADV_NOHUGEPAGE
    enum: MADV_DONTDUMP
    enum: MADV_DODUMP
    enum: MADV_FREE
    enum: MADV_WIPEONFORK
    enum: MADV_KEEPONFORK
    int   madvise(void *addr, size_t Len, int advice)

    # sometimes available
    int   mincore(void *addr, size_t Len, unsigned char *vec)

    # These two are Linux specific but sometimes very efficient
    void *mremap(void *old_addr, size_t old_len, size_t new_len, int flags, ...)
    int   remap_file_pages(void *addr, size_t Len, int prot,
                           size_t pgoff, int flags)

    # The rare but standardized typed memory option
    enum: POSIX_TYPED_MEM_ALLOCATE
    enum: POSIX_TYPED_MEM_ALLOCATE_CONTIG
    enum: POSIX_TYPED_MEM_MAP_ALLOCATABLE
    int posix_typed_mem_open(const char *name, int oflag, int tflag)
    int posix_mem_offset(const void *addr, size_t Len, off_t *off,
                         size_t *contig_len, int *fildes)

    cdef struct posix_typed_mem_info:
        size_t posix_tmi_length
    int posix_typed_mem_get_info(int fildes, posix_typed_mem_info *info)
