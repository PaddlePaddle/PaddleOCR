from cpython.datetime cimport (
    datetime,
    timedelta,
    tzinfo,
)


cdef tzinfo utc_pytz

cpdef bint is_utc(tzinfo tz)
cdef bint is_tzlocal(tzinfo tz)

cdef bint treat_tz_as_pytz(tzinfo tz)

cpdef bint tz_compare(tzinfo start, tzinfo end)
cpdef object get_timezone(tzinfo tz)
cpdef tzinfo maybe_get_tz(object tz)

cdef timedelta get_utcoffset(tzinfo tz, datetime obj)
cdef bint is_fixed_offset(tzinfo tz)

cdef object get_dst_info(tzinfo tz)
