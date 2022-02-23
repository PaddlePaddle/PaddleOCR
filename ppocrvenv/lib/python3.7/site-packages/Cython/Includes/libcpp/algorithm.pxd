from libcpp cimport bool


cdef extern from "<algorithm>" namespace "std" nogil:
    # Sorting and searching
    bool binary_search[Iter, T](Iter first, Iter last, const T& value)
    bool binary_search[Iter, T, Compare](Iter first, Iter last, const T& value,
                                         Compare comp)

    Iter lower_bound[Iter, T](Iter first, Iter last, const T& value)
    Iter lower_bound[Iter, T, Compare](Iter first, Iter last, const T& value,
                                       Compare comp)

    Iter upper_bound[Iter, T](Iter first, Iter last, const T& value)
    Iter upper_bound[Iter, T, Compare](Iter first, Iter last, const T& value,
                                       Compare comp)

    void partial_sort[Iter](Iter first, Iter middle, Iter last)
    void partial_sort[Iter, Compare](Iter first, Iter middle, Iter last,
                                     Compare comp)

    void sort[Iter](Iter first, Iter last)
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)

    # Removing duplicates
    Iter unique[Iter](Iter first, Iter last)
    Iter unique[Iter, BinaryPredicate](Iter first, Iter last, BinaryPredicate p)

    # Binary heaps (priority queues)
    void make_heap[Iter](Iter first, Iter last)
    void make_heap[Iter, Compare](Iter first, Iter last, Compare comp)

    void pop_heap[Iter](Iter first, Iter last)
    void pop_heap[Iter, Compare](Iter first, Iter last, Compare comp)

    void push_heap[Iter](Iter first, Iter last)
    void push_heap[Iter, Compare](Iter first, Iter last, Compare comp)

    void sort_heap[Iter](Iter first, Iter last)
    void sort_heap[Iter, Compare](Iter first, Iter last, Compare comp)

    # Copy
    OutputIter copy[InputIter,OutputIter](InputIter,InputIter,OutputIter)
