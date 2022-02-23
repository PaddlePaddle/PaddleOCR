#Basic reference: http://www.cplusplus.com/reference/iterator/
#Most of these classes are in fact empty structs

cdef extern from "<iterator>" namespace "std" nogil:
    cdef cppclass iterator[Category,T,Distance,Pointer,Reference]:
        pass
    cdef cppclass output_iterator_tag:
        pass
    cdef cppclass input_iterator_tag:
        pass
    cdef cppclass forward_iterator_tag(input_iterator_tag):
        pass
    cdef cppclass bidirectional_iterator_tag(forward_iterator_tag):
        pass
    cdef cppclass random_access_iterator_tag(bidirectional_iterator_tag):
        pass

    cdef cppclass back_insert_iterator[T](iterator[output_iterator_tag,void,void,void,void]):
        pass
    cdef cppclass front_insert_iterator[T](iterator[output_iterator_tag,void,void,void,void]):
        pass
    cdef cppclass insert_iterator[T](iterator[output_iterator_tag,void,void,void,void]):
        pass
    back_insert_iterator[CONTAINER] back_inserter[CONTAINER](CONTAINER &)
    front_insert_iterator[CONTAINER] front_inserter[CONTAINER](CONTAINER &)
    ##Note: this is the C++98 version of inserter.
    ##The C++11 versions's prototype relies on typedef members of classes, which Cython doesn't currently support:
    ##template <class Container>
    ##insert_iterator<Container> inserter (Container& x, typename Container::iterator it)
    insert_iterator[CONTAINER] inserter[CONTAINER,ITERATOR](CONTAINER &, ITERATOR)


