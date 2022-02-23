cdef extern from "<functional>" namespace "std" nogil:
    cdef cppclass function[T]:
        function() except +
        function(T*) except +
        function(function&) except +
        function(void*) except +

        function operator=(T*)
        function operator=(function&)
        function operator=(void*)
        function operator=[U](U)

        bint operator bool()
