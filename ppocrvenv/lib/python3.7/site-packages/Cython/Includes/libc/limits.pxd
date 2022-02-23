# 5.2.4.2.1 Sizes of integer types <limits.h>

cdef extern from "<limits.h>":
    const int CHAR_BIT
    const int MB_LEN_MAX

    const char CHAR_MIN
    const char CHAR_MAX

    const signed char SCHAR_MIN
    const signed char SCHAR_MAX
    const unsigned char UCHAR_MAX

    const short SHRT_MIN
    const short SHRT_MAX
    const unsigned short USHRT_MAX

    const int INT_MIN
    const int INT_MAX
    const unsigned int UINT_MAX

    const long LONG_MIN
    const long LONG_MAX
    const unsigned long ULONG_MAX

    const long long LLONG_MIN
    const long long LLONG_MAX
    const unsigned long long ULLONG_MAX
