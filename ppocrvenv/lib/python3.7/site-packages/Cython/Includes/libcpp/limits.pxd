cdef extern from "<limits>" namespace "std" nogil:
   enum float_round_style:
        round_indeterminate       = -1
        round_toward_zero         = 0
        round_to_nearest          = 1
        round_toward_infinity     = 2
        round_toward_neg_infinity = 3

   enum float_denorm_style:
        denorm_indeterminate  = -1
        denorm_absent         = 0
        denorm_present        = 1

   #The static methods can be called as, e.g. numeric_limits[int].round_error(), etc.
   #The const data members should be declared as static.  Cython currently doesn't allow that
   #and/or I can't figure it out, so you must instantiate an object to access, e.g.
   #cdef numeric_limits[double] lm
   #print lm.round_style
   cdef cppclass numeric_limits[T]:
    const bint is_specialized
    @staticmethod
    T min()
    @staticmethod
    T max()
    const int digits
    const int  digits10
    const bint is_signed
    const bint is_integer
    const bint is_exact
    const int radix
    @staticmethod
    T epsilon()
    @staticmethod
    T round_error()

    const int  min_exponent
    const int  min_exponent10
    const int  max_exponent
    const int  max_exponent10

    const bint has_infinity
    const bint has_quiet_NaN
    const bint has_signaling_NaN
    const float_denorm_style has_denorm
    const bint has_denorm_loss
    @staticmethod
    T infinity()
    @staticmethod
    T quiet_NaN()
    @staticmethod
    T signaling_NaN()
    @staticmethod
    T denorm_min()

    const bint is_iec559
    const bint is_bounded
    const bint is_modulo

    const bint traps
    const bint tinyness_before
    const float_round_style round_style
