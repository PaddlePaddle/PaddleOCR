

cdef extern from "pythread.h":

    ctypedef void *PyThread_type_lock
    ctypedef void *PyThread_type_sema

    void PyThread_init_thread()
    long PyThread_start_new_thread(void (*)(void *), void *)  # FIXME: legacy
    #unsigned long PyThread_start_new_thread(void (*)(void *), void *)  # returned 'long' before Py3.7
    void PyThread_exit_thread()
    long PyThread_get_thread_ident()  # FIXME: legacy
    #unsigned long PyThread_get_thread_ident()  # returned 'long' before Py3.7

    PyThread_type_lock PyThread_allocate_lock()
    void PyThread_free_lock(PyThread_type_lock)
    int PyThread_acquire_lock(PyThread_type_lock, int mode) nogil
    void PyThread_release_lock(PyThread_type_lock) nogil

    enum:
        # 'mode' in PyThread_acquire_lock()
        WAIT_LOCK    #   1
        NOWAIT_LOCK  #   0

    ctypedef enum PyLockStatus:
        # return values of PyThread_acquire_lock() in CPython 3.2+
        PY_LOCK_FAILURE = 0
        PY_LOCK_ACQUIRED = 1
        PY_LOCK_INTR

    size_t PyThread_get_stacksize()
    int PyThread_set_stacksize(size_t)

    # Thread Local Storage (TLS) API deprecated in CPython 3.7+
    int PyThread_create_key()
    void PyThread_delete_key(int)
    int PyThread_set_key_value(int, void *)
    void * PyThread_get_key_value(int)
    void PyThread_delete_key_value(int key)

    # Cleanup after a fork
    void PyThread_ReInitTLS()

    # Thread Specific Storage (TSS) API in CPython 3.7+ (also backported)
    #ctypedef struct Py_tss_t: pass   # Cython built-in type
    Py_tss_t Py_tss_NEEDS_INIT        # Not normally useful: Cython auto-initialises declared "Py_tss_t" variables.
    Py_tss_t * PyThread_tss_alloc()
    void PyThread_tss_free(Py_tss_t *key)
    int PyThread_tss_is_created(Py_tss_t *key)
    int PyThread_tss_create(Py_tss_t *key)
    void PyThread_tss_delete(Py_tss_t *key)
    int PyThread_tss_set(Py_tss_t *key, void *value)
    void * PyThread_tss_get(Py_tss_t *key)
