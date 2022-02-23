# Note: add integer versions of the functions?

cdef extern from "<complex>" namespace "std" nogil:
    cdef cppclass complex[T]:
        complex() except +
        complex(T, T) except +
        complex(complex[T]&) except +
        # How to make the converting constructor, i.e. convert complex[double]
        # to complex[float]?

        complex[T] operator+(complex[T]&)
        complex[T] operator-(complex[T]&)
        complex[T] operator+(complex[T]&, complex[T]&)
        complex[T] operator+(complex[T]&, T&)
        complex[T] operator+(T&, complex[T]&)
        complex[T] operator-(complex[T]&, complex[T]&)
        complex[T] operator-(complex[T]&, T&)
        complex[T] operator-(T&, complex[T]&)
        complex[T] operator*(complex[T]&, complex[T]&)
        complex[T] operator*(complex[T]&, T&)
        complex[T] operator*(T&, complex[T]&)
        complex[T] operator/(complex[T]&, complex[T]&)
        complex[T] operator/(complex[T]&, T&)
        complex[T] operator/(T&, complex[T]&)

        bint operator==(complex[T]&, complex[T]&)
        bint operator==(complex[T]&, T&)
        bint operator==(T&, complex[T]&)
        bint operator!=(complex[T]&, complex[T]&)
        bint operator!=(complex[T]&, T&)
        bint operator!=(T&, complex[T]&)

        # Access real part
        T real()
        void real(T)

        # Access imaginary part
        T imag()
        void imag(T)

    # Return real part
    T real[T](complex[T]&)
    long double real(long double)
    double real(double)
    float real(float)

    # Return imaginary part
    T imag[T](complex[T]&)
    long double imag(long double)
    double imag(double)
    float imag(float)

    T abs[T](complex[T]&)
    T arg[T](complex[T]&)
    long double arg(long double)
    double arg(double)
    float arg(float)

    T norm[T](complex[T])
    long double norm(long double)
    double norm(double)
    float norm(float)

    complex[T] conj[T](complex[T]&)
    complex[long double] conj(long double)
    complex[double] conj(double)
    complex[float] conj(float)

    complex[T] proj[T](complex[T])
    complex[long double] proj(long double)
    complex[double] proj(double)
    complex[float] proj(float)

    complex[T] polar[T](T&, T&)
    complex[T] ploar[T](T&)

    complex[T] exp[T](complex[T]&)
    complex[T] log[T](complex[T]&)
    complex[T] log10[T](complex[T]&)

    complex[T] pow[T](complex[T]&, complex[T]&)
    complex[T] pow[T](complex[T]&, T&)
    complex[T] pow[T](T&, complex[T]&)
    # There are some promotion versions too

    complex[T] sqrt[T](complex[T]&)

    complex[T] sin[T](complex[T]&)
    complex[T] cos[T](complex[T]&)
    complex[T] tan[T](complex[T]&)
    complex[T] asin[T](complex[T]&)
    complex[T] acos[T](complex[T]&)
    complex[T] atan[T](complex[T]&)

    complex[T] sinh[T](complex[T]&)
    complex[T] cosh[T](complex[T]&)
    complex[T] tanh[T](complex[T]&)

    complex[T] asinh[T](complex[T]&)
    complex[T] acosh[T](complex[T]&)
    complex[T] atanh[T](complex[T]&)
