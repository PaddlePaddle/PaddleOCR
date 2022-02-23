from posix.types cimport (blkcnt_t, blksize_t, dev_t, gid_t, ino_t, mode_t,
                          nlink_t, off_t, time_t, uid_t)


cdef extern from "<sys/stat.h>" nogil:
    cdef struct struct_stat "stat":
        dev_t   st_dev
        ino_t   st_ino
        mode_t  st_mode
        nlink_t st_nlink
        uid_t   st_uid
        gid_t   st_gid
        dev_t   st_rdev
        off_t   st_size
        blksize_t st_blksize
        blkcnt_t st_blocks
        time_t  st_atime
        time_t  st_mtime
        time_t  st_ctime

        # st_birthtime exists on *BSD and OS X.
        # Under Linux, defining it here does not hurt. Compilation under Linux
        # will only (and rightfully) fail when attempting to use the field.
        time_t  st_birthtime

# POSIX prescribes including both <sys/stat.h> and <unistd.h> for these
cdef extern from "<unistd.h>" nogil:
    int fchmod(int, mode_t)
    int chmod(const char *, mode_t)

    int fstat(int, struct_stat *)
    int lstat(const char *, struct_stat *)
    int stat(const char *, struct_stat *)

    # Macros for st_mode
    mode_t S_ISREG(mode_t)
    mode_t S_ISDIR(mode_t)
    mode_t S_ISCHR(mode_t)
    mode_t S_ISBLK(mode_t)
    mode_t S_ISFIFO(mode_t)
    mode_t S_ISLNK(mode_t)
    mode_t S_ISSOCK(mode_t)

    mode_t S_IFMT
    mode_t S_IFREG
    mode_t S_IFDIR
    mode_t S_IFCHR
    mode_t S_IFBLK
    mode_t S_IFIFO
    mode_t S_IFLNK
    mode_t S_IFSOCK

    # Permissions
    mode_t S_ISUID
    mode_t S_ISGID
    mode_t S_ISVTX

    mode_t S_IRWXU
    mode_t S_IRUSR
    mode_t S_IWUSR
    mode_t S_IXUSR

    mode_t S_IRWXG
    mode_t S_IRGRP
    mode_t S_IWGRP
    mode_t S_IXGRP

    mode_t S_IRWXO
    mode_t S_IROTH
    mode_t S_IWOTH
    mode_t S_IXOTH
