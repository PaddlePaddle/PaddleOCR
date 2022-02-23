# 5.2.4.2.2 Characteristics of floating types <float.h>

cdef extern from "<float.h>":

    const float FLT_RADIX

    const float FLT_MANT_DIG
    const double DBL_MANT_DIG
    const long double LDBL_MANT_DIG

    const double DECIMAL_DIG

    const float FLT_DIG
    const double DBL_DIG
    const long double LDBL_DIG

    const float FLT_MIN_EXP
    const double DBL_MIN_EXP
    const long double LDBL_MIN_EXP

    const float FLT_MIN_10_EXP
    const double DBL_MIN_10_EXP
    const long double LDBL_MIN_10_EXP

    const float FLT_MAX_EXP
    const double DBL_MAX_EXP
    const long double LDBL_MAX_EXP

    const float FLT_MAX_10_EXP
    const double DBL_MAX_10_EXP
    const long double LDBL_MAX_10_EXP

    const float FLT_MAX
    const double DBL_MAX
    const long double LDBL_MAX

    const float FLT_EPSILON
    const double DBL_EPSILON
    const long double LDBL_EPSILON

    const float FLT_MIN
    const double DBL_MIN
    const long double LDBL_MIN
