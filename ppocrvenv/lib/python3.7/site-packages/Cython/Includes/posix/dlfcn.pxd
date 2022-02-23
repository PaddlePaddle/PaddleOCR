# POSIX dynamic linking/loading interface.
# http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/dlfcn.h.html

cdef extern from "<dlfcn.h>" nogil:
    void *dlopen(const char *, int)
    char *dlerror()
    void *dlsym(void *, const char *)
    int dlclose(void *)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL
