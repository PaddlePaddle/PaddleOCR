cdef extern from "<utility>" namespace "std" nogil:
    cdef cppclass pair[T, U]:
        ctypedef T first_type
        ctypedef U second_type
        T first
        U second
        pair() except +
        pair(pair&) except +
        pair(T&, U&) except +
        bint operator==(pair&, pair&)
        bint operator!=(pair&, pair&)
        bint operator<(pair&, pair&)
        bint operator>(pair&, pair&)
        bint operator<=(pair&, pair&)
        bint operator>=(pair&, pair&)

cdef extern from * namespace "cython_std" nogil:
    """
    #if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1600)
    // move should be defined for these versions of MSVC, but __cplusplus isn't set usefully
    #include <type_traits>

    namespace cython_std {
    template <typename T> typename std::remove_reference<T>::type&& move(T& t) noexcept { return std::move(t); }
    template <typename T> typename std::remove_reference<T>::type&& move(T&& t) noexcept { return std::move(t); }
    }

    #endif
    """
    cdef T move[T](T)
