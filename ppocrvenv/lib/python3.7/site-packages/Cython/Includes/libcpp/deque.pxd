cdef extern from "<deque>" namespace "std" nogil:
    cdef cppclass deque[T,ALLOCATOR=*]:
        ctypedef T value_type
        ctypedef ALLOCATOR allocator_type

        # these should really be allocator_type.size_type and
        # allocator_type.difference_type to be true to the C++ definition
        # but cython doesn't support deferred access on template arguments
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        cppclass iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_type)
            iterator operator-(size_type)
            difference_type operator-(iterator)
            bint operator==(iterator)
            bint operator!=(iterator)
            bint operator<(iterator)
            bint operator>(iterator)
            bint operator<=(iterator)
            bint operator>=(iterator)
        cppclass reverse_iterator:
            T& operator*()
            reverse_iterator operator++()
            reverse_iterator operator--()
            reverse_iterator operator+(size_type)
            reverse_iterator operator-(size_type)
            difference_type operator-(reverse_iterator)
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator>=(reverse_iterator)
        cppclass const_iterator(iterator):
            pass
        cppclass const_reverse_iterator(reverse_iterator):
            pass
        deque() except +
        deque(deque&) except +
        deque(size_t) except +
        deque(size_t, T&) except +
        #deque[input_iterator](input_iterator, input_iterator)
        T& operator[](size_t)
        #deque& operator=(deque&)
        bint operator==(deque&, deque&)
        bint operator!=(deque&, deque&)
        bint operator<(deque&, deque&)
        bint operator>(deque&, deque&)
        bint operator<=(deque&, deque&)
        bint operator>=(deque&, deque&)
        void assign(size_t, T&)
        void assign(input_iterator, input_iterator)
        T& at(size_t)
        T& back()
        iterator begin()
        const_iterator const_begin "begin"()
        void clear()
        bint empty()
        iterator end()
        const_iterator const_end "end"()
        iterator erase(iterator)
        iterator erase(iterator, iterator)
        T& front()
        iterator insert(iterator, T&)
        void insert(iterator, size_t, T&)
        void insert(iterator, input_iterator, input_iterator)
        size_t max_size()
        void pop_back()
        void pop_front()
        void push_back(T&)
        void push_front(T&)
        reverse_iterator rbegin()
        #const_reverse_iterator rbegin()
        reverse_iterator rend()
        #const_reverse_iterator rend()
        void resize(size_t)
        void resize(size_t, T&)
        size_t size()
        void swap(deque&)

        # C++11 methods
        void shrink_to_fit()
