cdef extern from "<setjmp.h>" nogil:
    ctypedef struct jmp_buf:
        pass
    int setjmp(jmp_buf state)
    void longjmp(jmp_buf state, int value)

    ctypedef struct sigjmp_buf:
        pass
    int sigsetjmp(sigjmp_buf state, int savesigs)
    void siglongjmp(sigjmp_buf state, int value)
