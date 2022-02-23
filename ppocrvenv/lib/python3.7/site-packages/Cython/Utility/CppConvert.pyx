# TODO: Figure out how many of the pass-by-value copies the compiler can eliminate.


#################### string.from_py ####################

cdef extern from *:
    cdef cppclass string "{{type}}":
        string()
        string(char* c_str, size_t size)
    cdef const char* __Pyx_PyObject_AsStringAndSize(object, Py_ssize_t*) except NULL

@cname("{{cname}}")
cdef string {{cname}}(object o) except *:
    cdef Py_ssize_t length = 0
    cdef const char* data = __Pyx_PyObject_AsStringAndSize(o, &length)
    return string(data, length)


#################### string.to_py ####################

#cimport cython
#from libcpp.string cimport string
cdef extern from *:
    cdef cppclass string "{{type}}":
        char* data()
        size_t size()

{{for py_type in ['PyObject', 'PyUnicode', 'PyStr', 'PyBytes', 'PyByteArray']}}
cdef extern from *:
    cdef object __Pyx_{{py_type}}_FromStringAndSize(const char*, size_t)

@cname("{{cname.replace("PyObject", py_type, 1)}}")
cdef inline object {{cname.replace("PyObject", py_type, 1)}}(const string& s):
    return __Pyx_{{py_type}}_FromStringAndSize(s.data(), s.size())
{{endfor}}


#################### vector.from_py ####################

cdef extern from *:
    cdef cppclass vector "std::vector" [T]:
        void push_back(T&)

@cname("{{cname}}")
cdef vector[X] {{cname}}(object o) except *:
    cdef vector[X] v
    for item in o:
        v.push_back(<X>item)
    return v


#################### vector.to_py ####################

cdef extern from *:
    cdef cppclass vector "const std::vector" [T]:
        size_t size()
        T& operator[](size_t)

@cname("{{cname}}")
cdef object {{cname}}(vector[X]& v):
    return [v[i] for i in range(v.size())]


#################### list.from_py ####################

cdef extern from *:
    cdef cppclass cpp_list "std::list" [T]:
        void push_back(T&)

@cname("{{cname}}")
cdef cpp_list[X] {{cname}}(object o) except *:
    cdef cpp_list[X] l
    for item in o:
        l.push_back(<X>item)
    return l


#################### list.to_py ####################

cimport cython

cdef extern from *:
    cdef cppclass cpp_list "std::list" [T]:
        cppclass const_iterator:
            T& operator*()
            const_iterator operator++()
            bint operator!=(const_iterator)
        const_iterator begin()
        const_iterator end()

@cname("{{cname}}")
cdef object {{cname}}(const cpp_list[X]& v):
    o = []
    cdef cpp_list[X].const_iterator iter = v.begin()
    while iter != v.end():
        o.append(cython.operator.dereference(iter))
        cython.operator.preincrement(iter)
    return o


#################### set.from_py ####################

cdef extern from *:
    cdef cppclass set "std::{{maybe_unordered}}set" [T]:
        void insert(T&)

@cname("{{cname}}")
cdef set[X] {{cname}}(object o) except *:
    cdef set[X] s
    for item in o:
        s.insert(<X>item)
    return s


#################### set.to_py ####################

cimport cython

cdef extern from *:
    cdef cppclass cpp_set "std::{{maybe_unordered}}set" [T]:
        cppclass const_iterator:
            T& operator*()
            const_iterator operator++()
            bint operator!=(const_iterator)
        const_iterator begin()
        const_iterator end()

@cname("{{cname}}")
cdef object {{cname}}(const cpp_set[X]& s):
    o = set()
    cdef cpp_set[X].const_iterator iter = s.begin()
    while iter != s.end():
        o.add(cython.operator.dereference(iter))
        cython.operator.preincrement(iter)
    return o

#################### pair.from_py ####################

cdef extern from *:
    cdef cppclass pair "std::pair" [T, U]:
        pair()
        pair(T&, U&)

@cname("{{cname}}")
cdef pair[X,Y] {{cname}}(object o) except *:
    x, y = o
    return pair[X,Y](<X>x, <Y>y)


#################### pair.to_py ####################

cdef extern from *:
    cdef cppclass pair "std::pair" [T, U]:
        T first
        U second

@cname("{{cname}}")
cdef object {{cname}}(const pair[X,Y]& p):
    return p.first, p.second


#################### map.from_py ####################

cdef extern from *:
    cdef cppclass pair "std::pair" [T, U]:
        pair(T&, U&)
    cdef cppclass map "std::{{maybe_unordered}}map" [T, U]:
        void insert(pair[T, U]&)
    cdef cppclass vector "std::vector" [T]:
        pass


@cname("{{cname}}")
cdef map[X,Y] {{cname}}(object o) except *:
    cdef dict d = o
    cdef map[X,Y] m
    for key, value in d.iteritems():
        m.insert(pair[X,Y](<X>key, <Y>value))
    return m


#################### map.to_py ####################
# TODO: Work out const so that this can take a const
# reference rather than pass by value.

cimport cython

cdef extern from *:
    cdef cppclass map "std::{{maybe_unordered}}map" [T, U]:
        cppclass value_type:
            T first
            U second
        cppclass const_iterator:
            value_type& operator*()
            const_iterator operator++()
            bint operator!=(const_iterator)
        const_iterator begin()
        const_iterator end()

@cname("{{cname}}")
cdef object {{cname}}(const map[X,Y]& s):
    o = {}
    cdef const map[X,Y].value_type *key_value
    cdef map[X,Y].const_iterator iter = s.begin()
    while iter != s.end():
        key_value = &cython.operator.dereference(iter)
        o[key_value.first] = key_value.second
        cython.operator.preincrement(iter)
    return o


#################### complex.from_py ####################

cdef extern from *:
    cdef cppclass std_complex "std::complex" [T]:
        std_complex()
        std_complex(T, T) except +

@cname("{{cname}}")
cdef std_complex[X] {{cname}}(object o) except *:
    cdef double complex z = o
    return std_complex[X](<X>z.real, <X>z.imag)


#################### complex.to_py ####################

cdef extern from *:
    cdef cppclass std_complex "std::complex" [T]:
        X real()
        X imag()

@cname("{{cname}}")
cdef object {{cname}}(const std_complex[X]& z):
    cdef double complex tmp
    tmp.real = <double>z.real()
    tmp.imag = <double>z.imag()
    return tmp
