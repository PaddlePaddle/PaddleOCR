cdef extern from "<list>" namespace "std" nogil:
    cdef cppclass list[T,ALLOCATOR=*]:
        ctypedef T value_type
        ctypedef ALLOCATOR allocator_type

        # these should really be allocator_type.size_type and
        # allocator_type.difference_type to be true to the C++ definition
        # but cython doesn't support deferred access on template arguments
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        cppclass iterator:
            iterator()
            iterator(iterator &)
            T& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass reverse_iterator:
            reverse_iterator()
            reverse_iterator(iterator &)
            T& operator*()
            reverse_iterator operator++()
            reverse_iterator operator--()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
        cppclass const_iterator(iterator):
            pass
        cppclass const_reverse_iterator(reverse_iterator):
            pass
        list() except +
        list(list&) except +
        list(size_t, T&) except +
        #list operator=(list&)
        bint operator==(list&, list&)
        bint operator!=(list&, list&)
        bint operator<(list&, list&)
        bint operator>(list&, list&)
        bint operator<=(list&, list&)
        bint operator>=(list&, list&)
        void assign(size_t, T&)
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
        size_t max_size()
        void merge(list&)
        #void merge(list&, BinPred)
        void pop_back()
        void pop_front()
        void push_back(T&)
        void push_front(T&)
        reverse_iterator rbegin()
        const_reverse_iterator const_rbegin "rbegin"()
        void remove(T&)
        #void remove_if(UnPred)
        reverse_iterator rend()
        const_reverse_iterator const_rend "rend"()
        void resize(size_t, T&)
        void reverse()
        size_t size()
        void sort()
        #void sort(BinPred)
        void splice(iterator, list&)
        void splice(iterator, list&, iterator)
        void splice(iterator, list&, iterator, iterator)
        void swap(list&)
        void unique()
        #void unique(BinPred)
