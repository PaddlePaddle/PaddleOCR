import cython

from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDateTime_IMPORT,
    datetime,
    tzinfo,
)

# import datetime C API
PyDateTime_IMPORT


cimport numpy as cnp
from numpy cimport (
    float64_t,
    int64_t,
    ndarray,
)

import numpy as np

cnp.import_array()

import pytz

from pandas._libs.tslibs.np_datetime cimport (
    _string_to_dts,
    check_dts_bounds,
    dt64_to_dtstruct,
    dtstruct_to_dt64,
    get_datetime64_value,
    npy_datetimestruct,
    pydate_to_dt64,
    pydatetime_to_dt64,
)
from pandas._libs.util cimport (
    is_datetime64_object,
    is_float_object,
    is_integer_object,
)

from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import parse_datetime_string

from pandas._libs.tslibs.conversion cimport (
    _TSObject,
    cast_from_unit,
    convert_datetime_to_tsobject,
    get_datetime64_nanos,
    precision_from_unit,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    c_nat_strings as nat_strings,
)
from pandas._libs.tslibs.timestamps cimport _Timestamp

from pandas._libs.tslibs.timestamps import Timestamp

# Note: this is the only non-tslibs intra-pandas dependency here

from pandas._libs.missing cimport checknull_with_nat_and_na
from pandas._libs.tslibs.tzconversion cimport tz_localize_to_utc_single


def _test_parse_iso8601(ts: str):
    """
    TESTING ONLY: Parse string into Timestamp using iso8601 parser. Used
    only for testing, actual construction uses `convert_str_to_tsobject`
    """
    cdef:
        _TSObject obj
        int out_local = 0, out_tzoffset = 0

    obj = _TSObject()

    if ts == 'now':
        return Timestamp.utcnow()
    elif ts == 'today':
        return Timestamp.now().normalize()

    _string_to_dts(ts, &obj.dts, &out_local, &out_tzoffset, True)
    obj.value = dtstruct_to_dt64(&obj.dts)
    check_dts_bounds(&obj.dts)
    if out_local == 1:
        obj.tzinfo = pytz.FixedOffset(out_tzoffset)
        obj.value = tz_localize_to_utc_single(obj.value, obj.tzinfo)
        return Timestamp(obj.value, tz=obj.tzinfo)
    else:
        return Timestamp(obj.value)


@cython.wraparound(False)
@cython.boundscheck(False)
def format_array_from_datetime(
    ndarray[int64_t] values,
    tzinfo tz=None,
    str format=None,
    object na_rep=None
) -> np.ndarray:
    """
    return a np object array of the string formatted values

    Parameters
    ----------
    values : a 1-d i8 array
    tz : tzinfo or None, default None
    format : str or None, default None
          a strftime capable string
    na_rep : optional, default is None
          a nat format

    Returns
    -------
    np.ndarray[object]
    """
    cdef:
        int64_t val, ns, N = len(values)
        ndarray[int64_t] consider_values
        bint show_ms = False, show_us = False, show_ns = False
        bint basic_format = False
        ndarray[object] result = np.empty(N, dtype=object)
        object ts, res
        npy_datetimestruct dts

    if na_rep is None:
        na_rep = 'NaT'

    # if we don't have a format nor tz, then choose
    # a format based on precision
    basic_format = format is None and tz is None
    if basic_format:
        consider_values = values[values != NPY_NAT]
        show_ns = (consider_values % 1000).any()

        if not show_ns:
            consider_values //= 1000
            show_us = (consider_values % 1000).any()

            if not show_ms:
                consider_values //= 1000
                show_ms = (consider_values % 1000).any()

    for i in range(N):
        val = values[i]

        if val == NPY_NAT:
            result[i] = na_rep
        elif basic_format:

            dt64_to_dtstruct(val, &dts)
            res = (f'{dts.year}-{dts.month:02d}-{dts.day:02d} '
                   f'{dts.hour:02d}:{dts.min:02d}:{dts.sec:02d}')

            if show_ns:
                ns = dts.ps // 1000
                res += f'.{ns + dts.us * 1000:09d}'
            elif show_us:
                res += f'.{dts.us:06d}'
            elif show_ms:
                res += f'.{dts.us // 1000:03d}'

            result[i] = res

        else:

            ts = Timestamp(val, tz=tz)
            if format is None:
                result[i] = str(ts)
            else:

                # invalid format string
                # requires dates > 1900
                try:
                    result[i] = ts.strftime(format)
                except ValueError:
                    result[i] = str(ts)

    return result


def array_with_unit_to_datetime(
    ndarray values,
    str unit,
    str errors="coerce"
):
    """
    Convert the ndarray to datetime according to the time unit.

    This function converts an array of objects into a numpy array of
    datetime64[ns]. It returns the converted array
    and also returns the timezone offset

    if errors:
      - raise: return converted values or raise OutOfBoundsDatetime
          if out of range on the conversion or
          ValueError for other conversions (e.g. a string)
      - ignore: return non-convertible values as the same unit
      - coerce: NaT for non-convertibles

    Parameters
    ----------
    values : ndarray
         Date-like objects to convert.
    unit : str
         Time unit to use during conversion.
    errors : str, default 'raise'
         Error behavior when parsing.

    Returns
    -------
    result : ndarray of m8 values
    tz : parsed timezone offset or None
    """
    cdef:
        Py_ssize_t i, j, n=len(values)
        int64_t m
        int prec = 0
        ndarray[float64_t] fvalues
        bint is_ignore = errors=='ignore'
        bint is_coerce = errors=='coerce'
        bint is_raise = errors=='raise'
        bint need_to_iterate = True
        ndarray[int64_t] iresult
        ndarray[object] oresult
        ndarray mask
        object tz = None

    assert is_ignore or is_coerce or is_raise

    if unit == "ns":
        if issubclass(values.dtype.type, (np.integer, np.float_)):
            result = values.astype("M8[ns]", copy=False)
        else:
            result, tz = array_to_datetime(
                values.astype(object, copy=False),
                errors=errors,
            )
        return result, tz

    m, p = precision_from_unit(unit)

    if is_raise:
        # try a quick conversion to i8/f8
        # if we have nulls that are not type-compat
        # then need to iterate

        if values.dtype.kind == "i" or values.dtype.kind == "f":
            iresult = values.astype("i8", copy=False)
            # fill missing values by comparing to NPY_NAT
            mask = iresult == NPY_NAT
            iresult[mask] = 0
            fvalues = iresult.astype("f8") * m
            need_to_iterate = False

        if not need_to_iterate:
            # check the bounds
            if (fvalues < Timestamp.min.value).any() or (
                (fvalues > Timestamp.max.value).any()
            ):
                raise OutOfBoundsDatetime(f"cannot convert input with unit '{unit}'")

            if values.dtype.kind == "i":
                result = (iresult * m).astype("M8[ns]")

            elif values.dtype.kind == "f":
                fresult = (values * m).astype("f8")
                fresult[mask] = 0
                if prec:
                    fresult = round(fresult, prec)
                result = fresult.astype("M8[ns]", copy=False)

            iresult = result.view("i8")
            iresult[mask] = NPY_NAT

            return result, tz

    result = np.empty(n, dtype='M8[ns]')
    iresult = result.view('i8')

    try:
        for i in range(n):
            val = values[i]

            if checknull_with_nat_and_na(val):
                iresult[i] = NPY_NAT

            elif is_integer_object(val) or is_float_object(val):

                if val != val or val == NPY_NAT:
                    iresult[i] = NPY_NAT
                else:
                    try:
                        iresult[i] = cast_from_unit(val, unit)
                    except OverflowError:
                        if is_raise:
                            raise OutOfBoundsDatetime(
                                f"cannot convert input {val} with the unit '{unit}'"
                            )
                        elif is_ignore:
                            raise AssertionError
                        iresult[i] = NPY_NAT

            elif isinstance(val, str):
                if len(val) == 0 or val in nat_strings:
                    iresult[i] = NPY_NAT

                else:
                    try:
                        iresult[i] = cast_from_unit(float(val), unit)
                    except ValueError:
                        if is_raise:
                            raise ValueError(
                                f"non convertible value {val} with the unit '{unit}'"
                            )
                        elif is_ignore:
                            raise AssertionError
                        iresult[i] = NPY_NAT
                    except OverflowError:
                        if is_raise:
                            raise OutOfBoundsDatetime(
                                f"cannot convert input {val} with the unit '{unit}'"
                            )
                        elif is_ignore:
                            raise AssertionError
                        iresult[i] = NPY_NAT

            else:

                if is_raise:
                    raise ValueError(
                        f"unit='{unit}' not valid with non-numerical val='{val}'"
                    )
                if is_ignore:
                    raise AssertionError

                iresult[i] = NPY_NAT

        return result, tz

    except AssertionError:
        pass

    # we have hit an exception
    # and are in ignore mode
    # redo as object

    oresult = np.empty(n, dtype=object)
    for i in range(n):
        val = values[i]

        if checknull_with_nat_and_na(val):
            oresult[i] = <object>NaT
        elif is_integer_object(val) or is_float_object(val):

            if val != val or val == NPY_NAT:
                oresult[i] = <object>NaT
            else:
                try:
                    oresult[i] = Timestamp(cast_from_unit(val, unit))
                except OverflowError:
                    oresult[i] = val

        elif isinstance(val, str):
            if len(val) == 0 or val in nat_strings:
                oresult[i] = <object>NaT

            else:
                oresult[i] = val

    return oresult, tz


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef array_to_datetime(
    ndarray[object] values,
    str errors='raise',
    bint dayfirst=False,
    bint yearfirst=False,
    bint utc=False,
    bint require_iso8601=False,
    bint allow_mixed=False,
):
    """
    Converts a 1D array of date-like values to a numpy array of either:
        1) datetime64[ns] data
        2) datetime.datetime objects, if OutOfBoundsDatetime or TypeError
           is encountered

    Also returns a pytz.FixedOffset if an array of strings with the same
    timezone offset is passed and utc=True is not passed. Otherwise, None
    is returned

    Handles datetime.date, datetime.datetime, np.datetime64 objects, numeric,
    strings

    Parameters
    ----------
    values : ndarray of object
         date-like objects to convert
    errors : str, default 'raise'
         error behavior when parsing
    dayfirst : bool, default False
         dayfirst parsing behavior when encountering datetime strings
    yearfirst : bool, default False
         yearfirst parsing behavior when encountering datetime strings
    utc : bool, default False
         indicator whether the dates should be UTC
    require_iso8601 : bool, default False
         indicator whether the datetime string should be iso8601
    allow_mixed : bool, default False
        Whether to allow mixed datetimes and integers.

    Returns
    -------
    np.ndarray
        May be datetime64[ns] or object dtype
    tzinfo or None
    """
    cdef:
        Py_ssize_t i, n = len(values)
        object val, py_dt, tz, tz_out = None
        ndarray[int64_t] iresult
        ndarray[object] oresult
        npy_datetimestruct dts
        bint utc_convert = bool(utc)
        bint seen_integer = False
        bint seen_string = False
        bint seen_datetime = False
        bint seen_datetime_offset = False
        bint is_raise = errors=='raise'
        bint is_ignore = errors=='ignore'
        bint is_coerce = errors=='coerce'
        bint is_same_offsets
        _TSObject _ts
        int64_t value
        int out_local = 0, out_tzoffset = 0
        float offset_seconds, tz_offset
        set out_tzoffset_vals = set()
        bint string_to_dts_failed

    # specify error conditions
    assert is_raise or is_ignore or is_coerce

    result = np.empty(n, dtype='M8[ns]')
    iresult = result.view('i8')

    try:
        for i in range(n):
            val = values[i]

            try:
                if checknull_with_nat_and_na(val):
                    iresult[i] = NPY_NAT

                elif PyDateTime_Check(val):
                    seen_datetime = True
                    if val.tzinfo is not None:
                        if utc_convert:
                            _ts = convert_datetime_to_tsobject(val, None)
                            iresult[i] = _ts.value
                        else:
                            raise ValueError('Tz-aware datetime.datetime '
                                             'cannot be converted to '
                                             'datetime64 unless utc=True')
                    elif isinstance(val, _Timestamp):
                        iresult[i] = val.value
                    else:
                        iresult[i] = pydatetime_to_dt64(val, &dts)
                        check_dts_bounds(&dts)

                elif PyDate_Check(val):
                    seen_datetime = True
                    iresult[i] = pydate_to_dt64(val, &dts)
                    check_dts_bounds(&dts)

                elif is_datetime64_object(val):
                    seen_datetime = True
                    iresult[i] = get_datetime64_nanos(val)

                elif is_integer_object(val) or is_float_object(val):
                    # these must be ns unit by-definition
                    seen_integer = True

                    if val != val or val == NPY_NAT:
                        iresult[i] = NPY_NAT
                    elif is_raise or is_ignore:
                        iresult[i] = val
                    else:
                        # coerce
                        # we now need to parse this as if unit='ns'
                        # we can ONLY accept integers at this point
                        # if we have previously (or in future accept
                        # datetimes/strings, then we must coerce)
                        try:
                            iresult[i] = cast_from_unit(val, 'ns')
                        except OverflowError:
                            iresult[i] = NPY_NAT

                elif isinstance(val, str):
                    # string
                    seen_string = True

                    if len(val) == 0 or val in nat_strings:
                        iresult[i] = NPY_NAT
                        continue

                    string_to_dts_failed = _string_to_dts(
                        val, &dts, &out_local,
                        &out_tzoffset, False
                    )
                    if string_to_dts_failed:
                        # An error at this point is a _parsing_ error
                        # specifically _not_ OutOfBoundsDatetime
                        if _parse_today_now(val, &iresult[i]):
                            continue
                        elif require_iso8601:
                            # if requiring iso8601 strings, skip trying
                            # other formats
                            if is_coerce:
                                iresult[i] = NPY_NAT
                                continue
                            elif is_raise:
                                raise ValueError(
                                    f"time data {val} doesn't match format specified"
                                )
                            return values, tz_out

                        try:
                            py_dt = parse_datetime_string(val,
                                                          dayfirst=dayfirst,
                                                          yearfirst=yearfirst)
                            # If the dateutil parser returned tzinfo, capture it
                            # to check if all arguments have the same tzinfo
                            tz = py_dt.utcoffset()

                        except (ValueError, OverflowError):
                            if is_coerce:
                                iresult[i] = NPY_NAT
                                continue
                            raise TypeError("invalid string coercion to datetime")

                        if tz is not None:
                            seen_datetime_offset = True
                            # dateutil timezone objects cannot be hashed, so
                            # store the UTC offsets in seconds instead
                            out_tzoffset_vals.add(tz.total_seconds())
                        else:
                            # Add a marker for naive string, to track if we are
                            # parsing mixed naive and aware strings
                            out_tzoffset_vals.add('naive')

                        _ts = convert_datetime_to_tsobject(py_dt, None)
                        iresult[i] = _ts.value
                    if not string_to_dts_failed:
                        # No error reported by string_to_dts, pick back up
                        # where we left off
                        value = dtstruct_to_dt64(&dts)
                        if out_local == 1:
                            seen_datetime_offset = True
                            # Store the out_tzoffset in seconds
                            # since we store the total_seconds of
                            # dateutil.tz.tzoffset objects
                            out_tzoffset_vals.add(out_tzoffset * 60.)
                            tz = pytz.FixedOffset(out_tzoffset)
                            value = tz_localize_to_utc_single(value, tz)
                            out_local = 0
                            out_tzoffset = 0
                        else:
                            # Add a marker for naive string, to track if we are
                            # parsing mixed naive and aware strings
                            out_tzoffset_vals.add('naive')
                        iresult[i] = value
                        check_dts_bounds(&dts)

                else:
                    if is_coerce:
                        iresult[i] = NPY_NAT
                    else:
                        raise TypeError(f"{type(val)} is not convertible to datetime")

            except OutOfBoundsDatetime:
                if is_coerce:
                    iresult[i] = NPY_NAT
                    continue
                elif require_iso8601 and isinstance(val, str):
                    # GH#19382 for just-barely-OutOfBounds falling back to
                    # dateutil parser will return incorrect result because
                    # it will ignore nanoseconds
                    if is_raise:

                        # Still raise OutOfBoundsDatetime,
                        # as error message is informative.
                        raise

                    assert is_ignore
                    return values, tz_out
                raise

    except OutOfBoundsDatetime:
        if is_raise:
            raise

        return ignore_errors_out_of_bounds_fallback(values), tz_out

    except TypeError:
        return _array_to_datetime_object(values, errors, dayfirst, yearfirst)

    if seen_datetime and seen_integer:
        # we have mixed datetimes & integers

        if is_coerce:
            # coerce all of the integers/floats to NaT, preserve
            # the datetimes and other convertibles
            for i in range(n):
                val = values[i]
                if is_integer_object(val) or is_float_object(val):
                    result[i] = NPY_NAT
        elif allow_mixed:
            pass
        elif is_raise:
            raise ValueError("mixed datetimes and integers in passed array")
        else:
            return _array_to_datetime_object(values, errors, dayfirst, yearfirst)

    if seen_datetime_offset and not utc_convert:
        # GH#17697
        # 1) If all the offsets are equal, return one offset for
        #    the parsed dates to (maybe) pass to DatetimeIndex
        # 2) If the offsets are different, then force the parsing down the
        #    object path where an array of datetimes
        #    (with individual dateutil.tzoffsets) are returned
        is_same_offsets = len(out_tzoffset_vals) == 1
        if not is_same_offsets:
            return _array_to_datetime_object(values, errors, dayfirst, yearfirst)
        else:
            tz_offset = out_tzoffset_vals.pop()
            tz_out = pytz.FixedOffset(tz_offset / 60.)
    return result, tz_out


cdef ndarray[object] ignore_errors_out_of_bounds_fallback(ndarray[object] values):
    """
    Fallback for array_to_datetime if an OutOfBoundsDatetime is raised
    and errors == "ignore"

    Parameters
    ----------
    values : ndarray[object]

    Returns
    -------
    ndarray[object]
    """
    cdef:
        Py_ssize_t i, n = len(values)
        object val

    oresult = np.empty(n, dtype=object)

    for i in range(n):
        val = values[i]

        # set as nan except if its a NaT
        if checknull_with_nat_and_na(val):
            if isinstance(val, float):
                oresult[i] = np.nan
            else:
                oresult[i] = NaT
        elif is_datetime64_object(val):
            if get_datetime64_value(val) == NPY_NAT:
                oresult[i] = NaT
            else:
                oresult[i] = val.item()
        else:
            oresult[i] = val
    return oresult


@cython.wraparound(False)
@cython.boundscheck(False)
cdef _array_to_datetime_object(
    ndarray[object] values,
    str errors,
    bint dayfirst=False,
    bint yearfirst=False,
):
    """
    Fall back function for array_to_datetime

    Attempts to parse datetime strings with dateutil to return an array
    of datetime objects

    Parameters
    ----------
    values : ndarray[object]
         date-like objects to convert
    errors : str
         error behavior when parsing
    dayfirst : bool, default False
         dayfirst parsing behavior when encountering datetime strings
    yearfirst : bool, default False
         yearfirst parsing behavior when encountering datetime strings

    Returns
    -------
    np.ndarray[object]
    Literal[None]
    """
    cdef:
        Py_ssize_t i, n = len(values)
        object val
        bint is_ignore = errors == 'ignore'
        bint is_coerce = errors == 'coerce'
        bint is_raise = errors == 'raise'
        ndarray[object] oresult
        npy_datetimestruct dts

    assert is_raise or is_ignore or is_coerce

    oresult = np.empty(n, dtype=object)

    # We return an object array and only attempt to parse:
    # 1) NaT or NaT-like values
    # 2) datetime strings, which we return as datetime.datetime
    for i in range(n):
        val = values[i]
        if checknull_with_nat_and_na(val) or PyDateTime_Check(val):
            # GH 25978. No need to parse NaT-like or datetime-like vals
            oresult[i] = val
        elif isinstance(val, str):
            if len(val) == 0 or val in nat_strings:
                oresult[i] = 'NaT'
                continue
            try:
                oresult[i] = parse_datetime_string(val, dayfirst=dayfirst,
                                                   yearfirst=yearfirst)
                pydatetime_to_dt64(oresult[i], &dts)
                check_dts_bounds(&dts)
            except (ValueError, OverflowError):
                if is_coerce:
                    oresult[i] = <object>NaT
                    continue
                if is_raise:
                    raise
                return values, None
        else:
            if is_raise:
                raise
            return values, None
    return oresult, None


cdef inline bint _parse_today_now(str val, int64_t* iresult):
    # We delay this check for as long as possible
    # because it catches relatively rare cases
    if val == 'now':
        # Note: this is *not* the same as Timestamp('now')
        iresult[0] = Timestamp.utcnow().value
        return True
    elif val == 'today':
        iresult[0] = Timestamp.today().value
        return True
    return False
