from libcpp cimport bool, nullptr_t, nullptr

cdef extern from "<memory>" namespace "std" nogil:
    cdef cppclass default_delete[T]:
        default_delete()

    cdef cppclass allocator[T]:
        allocator()
        allocator(const allocator &)
        #allocator(const allocator[U] &) #unique_ptr unit tests fail w/this
        T * address(T &)
        const T * address(const T &) const
        T * allocate( size_t n ) # Not to standard.  should be a second default argument
        void deallocate(T * , size_t)
        size_t max_size() const
        void construct( T *, const T &) #C++98.  The C++11 version is variadic AND perfect-forwarding
        void destroy(T *) #C++98
        void destroy[U](U *) #unique_ptr unit tests fail w/this


    cdef cppclass unique_ptr[T,DELETER=*]:
        unique_ptr()
        unique_ptr(nullptr_t)
        unique_ptr(T*)
        unique_ptr(unique_ptr[T]&)

        # Modifiers
        T* release()
        void reset()
        void reset(nullptr_t)
        void reset(T*)
        void swap(unique_ptr&)

        # Observers
        T* get()
        T& operator*()
        #T* operator->() # Not Supported
        bool operator bool()
        bool operator!()

        bool operator==(const unique_ptr&)
        bool operator!=(const unique_ptr&)
        bool operator<(const unique_ptr&)
        bool operator>(const unique_ptr&)
        bool operator<=(const unique_ptr&)
        bool operator>=(const unique_ptr&)

        bool operator==(nullptr_t)
        bool operator!=(nullptr_t)

    # Forward Declaration not working ("Compiler crash in AnalyseDeclarationsTransform")
    #cdef cppclass weak_ptr[T]

    cdef cppclass shared_ptr[T]:
        shared_ptr()
        shared_ptr(nullptr_t)
        shared_ptr(T*)
        shared_ptr(shared_ptr[T]&)
        shared_ptr(shared_ptr[T]&, T*)
        shared_ptr(unique_ptr[T]&)
        #shared_ptr(weak_ptr[T]&) # Not Supported

        # Modifiers
        void reset()
        void reset(T*)
        void swap(shared_ptr&)

        # Observers
        T* get()
        T& operator*()
        #T* operator->() # Not Supported
        long use_count()
        bool unique()
        bool operator bool()
        bool operator!()
        #bool owner_before[Y](const weak_ptr[Y]&) # Not Supported
        bool owner_before[Y](const shared_ptr[Y]&)

        bool operator==(const shared_ptr&)
        bool operator!=(const shared_ptr&)
        bool operator<(const shared_ptr&)
        bool operator>(const shared_ptr&)
        bool operator<=(const shared_ptr&)
        bool operator>=(const shared_ptr&)

        bool operator==(nullptr_t)
        bool operator!=(nullptr_t)

    cdef cppclass weak_ptr[T]:
        weak_ptr()
        weak_ptr(weak_ptr[T]&)
        weak_ptr(shared_ptr[T]&)

        # Modifiers
        void reset()
        void swap(weak_ptr&)

        # Observers
        long use_count()
        bool expired()
        shared_ptr[T] lock()
        bool owner_before[Y](const weak_ptr[Y]&)
        bool owner_before[Y](const shared_ptr[Y]&)

    # Smart pointer non-member operations
    shared_ptr[T] make_shared[T](...) except +

    # Temporaries used for exception handling break generated code
    unique_ptr[T] make_unique[T](...) # except +

    # No checking on the compatibility of T and U.
    cdef shared_ptr[T] static_pointer_cast[T, U](const shared_ptr[U]&)
    cdef shared_ptr[T] dynamic_pointer_cast[T, U](const shared_ptr[U]&)
    cdef shared_ptr[T] const_pointer_cast[T, U](const shared_ptr[U]&)
    cdef shared_ptr[T] reinterpret_pointer_cast[T, U](const shared_ptr[U]&)
