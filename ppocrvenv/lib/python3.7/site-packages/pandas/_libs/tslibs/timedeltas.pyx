import collections
import warnings

import cython

from cpython.object cimport (
    Py_EQ,
    Py_NE,
    PyObject_RichCompare,
)

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
)

cnp.import_array()

from cpython.datetime cimport (
    PyDateTime_Check,
    PyDateTime_IMPORT,
    PyDelta_Check,
    timedelta,
)

PyDateTime_IMPORT


cimport pandas._libs.tslibs.util as util
from pandas._libs.tslibs.base cimport ABCTimestamp
from pandas._libs.tslibs.conversion cimport (
    cast_from_unit,
    precision_from_unit,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    c_nat_strings as nat_strings,
    checknull_with_nat,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    cmp_scalar,
    get_datetime64_unit,
    get_timedelta64_value,
    pandas_timedeltastruct,
    td64_to_tdstruct,
)
from pandas._libs.tslibs.offsets cimport is_tick_object
from pandas._libs.tslibs.util cimport (
    is_array,
    is_datetime64_object,
    is_float_object,
    is_integer_object,
    is_timedelta64_object,
)

from pandas._libs.tslibs.fields import (
    RoundTo,
    round_nsint64,
)

# ----------------------------------------------------------------------
# Constants

# components named tuple
Components = collections.namedtuple(
    "Components",
    [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ],
)

cdef dict timedelta_abbrevs = {
    "Y": "Y",
    "y": "Y",
    "M": "M",
    "W": "W",
    "w": "W",
    "D": "D",
    "d": "D",
    "days": "D",
    "day": "D",
    "hours": "h",
    "hour": "h",
    "hr": "h",
    "h": "h",
    "m": "m",
    "minute": "m",
    "min": "m",
    "minutes": "m",
    "t": "m",
    "s": "s",
    "seconds": "s",
    "sec": "s",
    "second": "s",
    "ms": "ms",
    "milliseconds": "ms",
    "millisecond": "ms",
    "milli": "ms",
    "millis": "ms",
    "l": "ms",
    "us": "us",
    "microseconds": "us",
    "microsecond": "us",
    "µs": "us",
    "micro": "us",
    "micros": "us",
    "u": "us",
    "ns": "ns",
    "nanoseconds": "ns",
    "nano": "ns",
    "nanos": "ns",
    "nanosecond": "ns",
    "n": "ns",
}

_no_input = object()


# ----------------------------------------------------------------------
# API

@cython.boundscheck(False)
@cython.wraparound(False)
def ints_to_pytimedelta(const int64_t[:] arr, box=False):
    """
    convert an i8 repr to an ndarray of timedelta or Timedelta (if box ==
    True)

    Parameters
    ----------
    arr : ndarray[int64_t]
    box : bool, default False

    Returns
    -------
    result : ndarray[object]
        array of Timedelta or timedeltas objects
    """
    cdef:
        Py_ssize_t i, n = len(arr)
        int64_t value
        object[:] result = np.empty(n, dtype=object)

    for i in range(n):

        value = arr[i]
        if value == NPY_NAT:
            result[i] = <object>NaT
        else:
            if box:
                result[i] = Timedelta(value)
            else:
                result[i] = timedelta(microseconds=int(value) / 1000)

    return result.base  # .base to access underlying np.ndarray


# ----------------------------------------------------------------------

cpdef int64_t delta_to_nanoseconds(delta) except? -1:
    if is_tick_object(delta):
        return delta.nanos
    if isinstance(delta, _Timedelta):
        delta = delta.value
    if is_timedelta64_object(delta):
        return get_timedelta64_value(ensure_td64ns(delta))
    if is_integer_object(delta):
        return delta
    if PyDelta_Check(delta):
        try:
            return (
                delta.days * 24 * 60 * 60 * 1_000_000
                + delta.seconds * 1_000_000
                + delta.microseconds
            ) * 1000
        except OverflowError as err:
            from pandas._libs.tslibs.conversion import OutOfBoundsTimedelta
            raise OutOfBoundsTimedelta(*err.args) from err

    raise TypeError(type(delta))


cdef str npy_unit_to_abbrev(NPY_DATETIMEUNIT unit):
    if unit == NPY_DATETIMEUNIT.NPY_FR_ns or unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        # generic -> default to nanoseconds
        return "ns"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_us:
        return "us"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_ms:
        return "ms"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_s:
        return "s"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_m:
        return "m"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_h:
        return "h"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_D:
        return "D"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_W:
        return "W"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_M:
        return "M"
    elif unit == NPY_DATETIMEUNIT.NPY_FR_Y:
        return "Y"
    else:
        raise NotImplementedError(unit)


@cython.overflowcheck(True)
cdef object ensure_td64ns(object ts):
    """
    Overflow-safe implementation of td64.astype("m8[ns]")

    Parameters
    ----------
    ts : np.timedelta64

    Returns
    -------
    np.timedelta64[ns]
    """
    cdef:
        NPY_DATETIMEUNIT td64_unit
        int64_t td64_value, mult
        str unitstr

    td64_unit = get_datetime64_unit(ts)
    if (
        td64_unit != NPY_DATETIMEUNIT.NPY_FR_ns
        and td64_unit != NPY_DATETIMEUNIT.NPY_FR_GENERIC
    ):
        unitstr = npy_unit_to_abbrev(td64_unit)

        td64_value = get_timedelta64_value(ts)

        mult = precision_from_unit(unitstr)[0]
        try:
            # NB: cython#1381 this cannot be *=
            td64_value = td64_value * mult
        except OverflowError as err:
            from pandas._libs.tslibs.conversion import OutOfBoundsTimedelta
            raise OutOfBoundsTimedelta(ts) from err

        return np.timedelta64(td64_value, "ns")

    return ts


cdef convert_to_timedelta64(object ts, str unit):
    """
    Convert an incoming object to a timedelta64 if possible.
    Before calling, unit must be standardized to avoid repeated unit conversion

    Handle these types of objects:
        - timedelta/Timedelta
        - timedelta64
        - an offset
        - np.int64 (with unit providing a possible modifier)
        - None/NaT

    Return an ns based int64
    """
    if checknull_with_nat(ts):
        return np.timedelta64(NPY_NAT, "ns")
    elif isinstance(ts, _Timedelta):
        # already in the proper format
        ts = np.timedelta64(ts.value, "ns")
    elif is_timedelta64_object(ts):
        ts = ensure_td64ns(ts)
    elif is_integer_object(ts):
        if ts == NPY_NAT:
            return np.timedelta64(NPY_NAT, "ns")
        else:
            if unit in ["Y", "M", "W"]:
                ts = np.timedelta64(ts, unit)
            else:
                ts = cast_from_unit(ts, unit)
                ts = np.timedelta64(ts, "ns")
    elif is_float_object(ts):
        if unit in ["Y", "M", "W"]:
            ts = np.timedelta64(int(ts), unit)
        else:
            ts = cast_from_unit(ts, unit)
            ts = np.timedelta64(ts, "ns")
    elif isinstance(ts, str):
        if (len(ts) > 0 and ts[0] == "P") or (len(ts) > 1 and ts[:2] == "-P"):
            ts = parse_iso_format_string(ts)
        else:
            ts = parse_timedelta_string(ts)
        ts = np.timedelta64(ts, "ns")
    elif is_tick_object(ts):
        ts = np.timedelta64(ts.nanos, "ns")

    if PyDelta_Check(ts):
        ts = np.timedelta64(delta_to_nanoseconds(ts), "ns")
    elif not is_timedelta64_object(ts):
        raise ValueError(f"Invalid type for timedelta scalar: {type(ts)}")
    return ts.astype("timedelta64[ns]")


@cython.boundscheck(False)
@cython.wraparound(False)
def array_to_timedelta64(
    ndarray[object] values, str unit=None, str errors="raise"
) -> ndarray:
    """
    Convert an ndarray to an array of timedeltas. If errors == 'coerce',
    coerce non-convertible objects to NaT. Otherwise, raise.

    Returns
    -------
    np.ndarray[timedelta64ns]
    """

    cdef:
        Py_ssize_t i, n
        int64_t[:] iresult

    if errors not in {'ignore', 'raise', 'coerce'}:
        raise ValueError("errors must be one of {'ignore', 'raise', or 'coerce'}")

    n = values.shape[0]
    result = np.empty(n, dtype='m8[ns]')
    iresult = result.view('i8')

    if unit is not None:
        for i in range(n):
            if isinstance(values[i], str) and errors != "coerce":
                raise ValueError(
                    "unit must not be specified if the input contains a str"
                )

    # Usually, we have all strings. If so, we hit the fast path.
    # If this path fails, we try conversion a different way, and
    # this is where all of the error handling will take place.
    try:
        for i in range(n):
            if values[i] is NaT:
                # we allow this check in the fast-path because NaT is a C-object
                #  so this is an inexpensive check
                iresult[i] = NPY_NAT
            else:
                result[i] = parse_timedelta_string(values[i])
    except (TypeError, ValueError):
        parsed_unit = parse_timedelta_unit(unit or 'ns')
        for i in range(n):
            try:
                result[i] = convert_to_timedelta64(values[i], parsed_unit)
            except ValueError as err:
                if errors == 'coerce':
                    result[i] = NPY_NAT
                elif "unit abbreviation w/o a number" in str(err):
                    # re-raise with more pertinent message
                    msg = f"Could not convert '{values[i]}' to NumPy timedelta"
                    raise ValueError(msg) from err
                else:
                    raise

    return iresult.base  # .base to access underlying np.ndarray


cdef inline int64_t parse_timedelta_string(str ts) except? -1:
    """
    Parse a regular format timedelta string. Return an int64_t (in ns)
    or raise a ValueError on an invalid parse.
    """

    cdef:
        unicode c
        bint neg = 0, have_dot = 0, have_value = 0, have_hhmmss = 0
        object current_unit = None
        int64_t result = 0, m = 0, r
        list number = [], frac = [], unit = []

    # neg : tracks if we have a leading negative for the value
    # have_dot : tracks if we are processing a dot (either post hhmmss or
    #            inside an expression)
    # have_value : track if we have at least 1 leading unit
    # have_hhmmss : tracks if we have a regular format hh:mm:ss

    if len(ts) == 0 or ts in nat_strings:
        return NPY_NAT

    for c in ts:

        # skip whitespace / commas
        if c == ' ' or c == ',':
            pass

        # positive signs are ignored
        elif c == '+':
            pass

        # neg
        elif c == '-':

            if neg or have_value or have_hhmmss:
                raise ValueError("only leading negative signs are allowed")

            neg = 1

        # number (ascii codes)
        elif ord(c) >= 48 and ord(c) <= 57:

            if have_dot:

                # we found a dot, but now its just a fraction
                if len(unit):
                    number.append(c)
                    have_dot = 0
                else:
                    frac.append(c)

            elif not len(unit):
                number.append(c)

            else:
                r = timedelta_from_spec(number, frac, unit)
                unit, number, frac = [], [c], []

                result += timedelta_as_neg(r, neg)

        # hh:mm:ss.
        elif c == ':':

            # we flip this off if we have a leading value
            if have_value:
                neg = 0

            # we are in the pattern hh:mm:ss pattern
            if len(number):
                if current_unit is None:
                    current_unit = 'h'
                    m = 1000000000 * 3600
                elif current_unit == 'h':
                    current_unit = 'm'
                    m = 1000000000 * 60
                elif current_unit == 'm':
                    current_unit = 's'
                    m = 1000000000
                r = <int64_t>int(''.join(number)) * m
                result += timedelta_as_neg(r, neg)
                have_hhmmss = 1
            else:
                raise ValueError(f"expecting hh:mm:ss format, received: {ts}")

            unit, number = [], []

        # after the decimal point
        elif c == '.':

            if len(number) and current_unit is not None:

                # by definition we had something like
                # so we need to evaluate the final field from a
                # hh:mm:ss (so current_unit is 'm')
                if current_unit != 'm':
                    raise ValueError("expected hh:mm:ss format before .")
                m = 1000000000
                r = <int64_t>int(''.join(number)) * m
                result += timedelta_as_neg(r, neg)
                have_value = 1
                unit, number, frac = [], [], []

            have_dot = 1

        # unit
        else:
            unit.append(c)
            have_value = 1
            have_dot = 0

    # we had a dot, but we have a fractional
    # value since we have an unit
    if have_dot and len(unit):
        r = timedelta_from_spec(number, frac, unit)
        result += timedelta_as_neg(r, neg)

    # we have a dot as part of a regular format
    # e.g. hh:mm:ss.fffffff
    elif have_dot:

        if ((len(number) or len(frac)) and not len(unit)
                and current_unit is None):
            raise ValueError("no units specified")

        if len(frac) > 0 and len(frac) <= 3:
            m = 10**(3 -len(frac)) * 1000 * 1000
        elif len(frac) > 3 and len(frac) <= 6:
            m = 10**(6 -len(frac)) * 1000
        elif len(frac) > 6 and len(frac) <= 9:
            m = 10**(9 -len(frac))
        else:
            m = 1
            frac = frac[:9]
        r = <int64_t>int(''.join(frac)) * m
        result += timedelta_as_neg(r, neg)

    # we have a regular format
    # we must have seconds at this point (hence the unit is still 'm')
    elif current_unit is not None:
        if current_unit != 'm':
            raise ValueError("expected hh:mm:ss format")
        m = 1000000000
        r = <int64_t>int(''.join(number)) * m
        result += timedelta_as_neg(r, neg)

    # we have a last abbreviation
    elif len(unit):
        if len(number):
            r = timedelta_from_spec(number, frac, unit)
            result += timedelta_as_neg(r, neg)
        else:
            raise ValueError("unit abbreviation w/o a number")

    # we only have symbols and no numbers
    elif len(number) == 0:
        raise ValueError("symbols w/o a number")

    # treat as nanoseconds
    # but only if we don't have anything else
    else:
        if have_value:
            raise ValueError("have leftover units")
        if len(number):
            r = timedelta_from_spec(number, frac, 'ns')
            result += timedelta_as_neg(r, neg)

    return result


cdef inline int64_t timedelta_as_neg(int64_t value, bint neg):
    """

    Parameters
    ----------
    value : int64_t of the timedelta value
    neg : bool if the a negative value
    """
    if neg:
        return -value
    return value


cdef inline timedelta_from_spec(object number, object frac, object unit):
    """

    Parameters
    ----------
    number : a list of number digits
    frac : a list of frac digits
    unit : a list of unit characters
    """
    cdef:
        str n

    try:
        unit = ''.join(unit)

        if unit in ["M", "Y", "y"]:
            warnings.warn(
                "Units 'M', 'Y' and 'y' do not represent unambiguous "
                "timedelta values and will be removed in a future version",
                FutureWarning,
                stacklevel=2,
            )

        if unit == 'M':
            # To parse ISO 8601 string, 'M' should be treated as minute,
            # not month
            unit = 'm'
        unit = parse_timedelta_unit(unit)
    except KeyError:
        raise ValueError(f"invalid abbreviation: {unit}")

    n = ''.join(number) + '.' + ''.join(frac)
    return cast_from_unit(float(n), unit)


cpdef inline str parse_timedelta_unit(str unit):
    """
    Parameters
    ----------
    unit : str or None

    Returns
    -------
    str
        Canonical unit string.

    Raises
    ------
    ValueError : on non-parseable input
    """
    if unit is None:
        return "ns"
    elif unit == "M":
        return unit
    try:
        return timedelta_abbrevs[unit.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"invalid unit abbreviation: {unit}")

# ----------------------------------------------------------------------
# Timedelta ops utilities

cdef bint _validate_ops_compat(other):
    # return True if we are compat with operating
    if checknull_with_nat(other):
        return True
    elif is_any_td_scalar(other):
        return True
    elif isinstance(other, str):
        return True
    return False


def _op_unary_method(func, name):
    def f(self):
        return Timedelta(func(self.value), unit='ns')
    f.__name__ = name
    return f


def _binary_op_method_timedeltalike(op, name):
    # define a binary operation that only works if the other argument is
    # timedelta like or an array of timedeltalike
    def f(self, other):
        if other is NaT:
            return NaT

        elif is_datetime64_object(other) or (
           PyDateTime_Check(other) and not isinstance(other, ABCTimestamp)):
            # this case is for a datetime object that is specifically
            # *not* a Timestamp, as the Timestamp case will be
            # handled after `_validate_ops_compat` returns False below
            from pandas._libs.tslibs.timestamps import Timestamp
            return op(self, Timestamp(other))
            # We are implicitly requiring the canonical behavior to be
            # defined by Timestamp methods.

        elif is_array(other):
            # nd-array like
            if other.dtype.kind in ['m', 'M']:
                return op(self.to_timedelta64(), other)
            elif other.dtype.kind == 'O':
                return np.array([op(self, x) for x in other])
            else:
                return NotImplemented

        elif not _validate_ops_compat(other):
            # Includes any of our non-cython classes
            return NotImplemented

        try:
            other = Timedelta(other)
        except ValueError:
            # failed to parse as timedelta
            return NotImplemented

        if other is NaT:
            # e.g. if original other was timedelta64('NaT')
            return NaT
        return Timedelta(op(self.value, other.value), unit='ns')

    f.__name__ = name
    return f


# ----------------------------------------------------------------------
# Timedelta Construction

cdef inline int64_t parse_iso_format_string(str ts) except? -1:
    """
    Extracts and cleanses the appropriate values from a match object with
    groups for each component of an ISO 8601 duration

    Parameters
    ----------
    ts: str
        ISO 8601 Duration formatted string

    Returns
    -------
    ns: int64_t
        Precision in nanoseconds of matched ISO 8601 duration

    Raises
    ------
    ValueError
        If ``ts`` cannot be parsed
    """

    cdef:
        unicode c
        int64_t result = 0, r
        int p = 0, sign = 1
        object dec_unit = 'ms', err_msg
        bint have_dot = 0, have_value = 0, neg = 0
        list number = [], unit = []

    err_msg = f"Invalid ISO 8601 Duration format - {ts}"

    if ts[0] == "-":
        sign = -1
        ts = ts[1:]

    for c in ts:
        # number (ascii codes)
        if 48 <= ord(c) <= 57:

            have_value = 1
            if have_dot:
                if p == 3 and dec_unit != 'ns':
                    unit.append(dec_unit)
                    if dec_unit == 'ms':
                        dec_unit = 'us'
                    elif dec_unit == 'us':
                        dec_unit = 'ns'
                    p = 0
                p += 1

            if not len(unit):
                number.append(c)
            else:
                r = timedelta_from_spec(number, '0', unit)
                result += timedelta_as_neg(r, neg)

                neg = 0
                unit, number = [], [c]
        else:
            if c == 'P' or c == 'T':
                pass  # ignore marking characters P and T
            elif c == '-':
                if neg or have_value:
                    raise ValueError(err_msg)
                else:
                    neg = 1
            elif c == "+":
                pass
            elif c in ['W', 'D', 'H', 'M']:
                if c in ['H', 'M'] and len(number) > 2:
                    raise ValueError(err_msg)
                if c == 'M':
                    c = 'min'
                unit.append(c)
                r = timedelta_from_spec(number, '0', unit)
                result += timedelta_as_neg(r, neg)

                neg = 0
                unit, number = [], []
            elif c == '.':
                # append any seconds
                if len(number):
                    r = timedelta_from_spec(number, '0', 'S')
                    result += timedelta_as_neg(r, neg)
                    unit, number = [], []
                have_dot = 1
            elif c == 'S':
                if have_dot:  # ms, us, or ns
                    if not len(number) or p > 3:
                        raise ValueError(err_msg)
                    # pad to 3 digits as required
                    pad = 3 - p
                    while pad > 0:
                        number.append('0')
                        pad -= 1

                    r = timedelta_from_spec(number, '0', dec_unit)
                    result += timedelta_as_neg(r, neg)
                else:  # seconds
                    r = timedelta_from_spec(number, '0', 'S')
                    result += timedelta_as_neg(r, neg)
            else:
                raise ValueError(err_msg)

    if not have_value:
        # Received string only - never parsed any values
        raise ValueError(err_msg)

    return sign*result


cdef _to_py_int_float(v):
    # Note: This used to be defined inside Timedelta.__new__
    # but cython will not allow `cdef` functions to be defined dynamically.
    if is_integer_object(v):
        return int(v)
    elif is_float_object(v):
        return float(v)
    raise TypeError(f"Invalid type {type(v)}. Must be int or float.")


# Similar to Timestamp/datetime, this is a construction requirement for
# timedeltas that we need to do object instantiation in python. This will
# serve as a C extension type that shadows the Python class, where we do any
# heavy lifting.
cdef class _Timedelta(timedelta):
    # cdef readonly:
    #    int64_t value      # nanoseconds
    #    object freq        # frequency reference
    #    bint is_populated  # are my components populated
    #    int64_t _d, _h, _m, _s, _ms, _us, _ns

    # higher than np.ndarray and np.matrix
    __array_priority__ = 100

    def __hash__(_Timedelta self):
        if self._has_ns():
            return hash(self.value)
        else:
            return timedelta.__hash__(self)

    def __richcmp__(_Timedelta self, object other, int op):
        cdef:
            _Timedelta ots
            int ndim

        if isinstance(other, _Timedelta):
            ots = other
        elif is_any_td_scalar(other):
            ots = Timedelta(other)
            # TODO: watch out for overflows

        elif other is NaT:
            return op == Py_NE

        elif util.is_array(other):
            # TODO: watch out for zero-dim
            if other.dtype.kind == "m":
                return PyObject_RichCompare(self.asm8, other, op)
            elif other.dtype.kind == "O":
                # operate element-wise
                return np.array(
                    [PyObject_RichCompare(self, x, op) for x in other],
                    dtype=bool,
                )
            if op == Py_EQ:
                return np.zeros(other.shape, dtype=bool)
            elif op == Py_NE:
                return np.ones(other.shape, dtype=bool)
            return NotImplemented  # let other raise TypeError

        else:
            return NotImplemented

        return cmp_scalar(self.value, ots.value, op)

    cpdef bint _has_ns(self):
        return self.value % 1000 != 0

    def _ensure_components(_Timedelta self):
        """
        compute the components
        """
        if self.is_populated:
            return

        cdef:
            pandas_timedeltastruct tds

        td64_to_tdstruct(self.value, &tds)
        self._d = tds.days
        self._h = tds.hrs
        self._m = tds.min
        self._s = tds.sec
        self._ms = tds.ms
        self._us = tds.us
        self._ns = tds.ns
        self._seconds = tds.seconds
        self._microseconds = tds.microseconds

        self.is_populated = 1

    cpdef timedelta to_pytimedelta(_Timedelta self):
        """
        Convert a pandas Timedelta object into a python timedelta object.

        Timedelta objects are internally saved as numpy datetime64[ns] dtype.
        Use to_pytimedelta() to convert to object dtype.

        Returns
        -------
        datetime.timedelta or numpy.array of datetime.timedelta

        See Also
        --------
        to_timedelta : Convert argument to Timedelta type.

        Notes
        -----
        Any nanosecond resolution will be lost.
        """
        return timedelta(microseconds=int(self.value) / 1000)

    def to_timedelta64(self) -> np.timedelta64:
        """
        Return a numpy.timedelta64 object with 'ns' precision.
        """
        return np.timedelta64(self.value, 'ns')

    def to_numpy(self, dtype=None, copy=False) -> np.timedelta64:
        """
        Convert the Timedelta to a NumPy timedelta64.

        .. versionadded:: 0.25.0

        This is an alias method for `Timedelta.to_timedelta64()`. The dtype and
        copy parameters are available here only for compatibility. Their values
        will not affect the return value.

        Returns
        -------
        numpy.timedelta64

        See Also
        --------
        Series.to_numpy : Similar method for Series.
        """
        return self.to_timedelta64()

    def view(self, dtype):
        """
        Array view compatibility.
        """
        return np.timedelta64(self.value).view(dtype)

    @property
    def components(self):
        """
        Return a components namedtuple-like.
        """
        self._ensure_components()
        # return the named tuple
        return Components(self._d, self._h, self._m, self._s,
                          self._ms, self._us, self._ns)

    @property
    def delta(self):
        """
        Return the timedelta in nanoseconds (ns), for internal compatibility.

        Returns
        -------
        int
            Timedelta in nanoseconds.

        Examples
        --------
        >>> td = pd.Timedelta('1 days 42 ns')
        >>> td.delta
        86400000000042

        >>> td = pd.Timedelta('3 s')
        >>> td.delta
        3000000000

        >>> td = pd.Timedelta('3 ms 5 us')
        >>> td.delta
        3005000

        >>> td = pd.Timedelta(42, unit='ns')
        >>> td.delta
        42
        """
        return self.value

    @property
    def asm8(self) -> np.timedelta64:
        """
        Return a numpy timedelta64 array scalar view.

        Provides access to the array scalar view (i.e. a combination of the
        value and the units) associated with the numpy.timedelta64().view(),
        including a 64-bit integer representation of the timedelta in
        nanoseconds (Python int compatible).

        Returns
        -------
        numpy timedelta64 array scalar view
            Array scalar view of the timedelta in nanoseconds.

        Examples
        --------
        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.asm8
        numpy.timedelta64(86520000003042,'ns')

        >>> td = pd.Timedelta('2 min 3 s')
        >>> td.asm8
        numpy.timedelta64(123000000000,'ns')

        >>> td = pd.Timedelta('3 ms 5 us')
        >>> td.asm8
        numpy.timedelta64(3005000,'ns')

        >>> td = pd.Timedelta(42, unit='ns')
        >>> td.asm8
        numpy.timedelta64(42,'ns')
        """
        return np.int64(self.value).view('m8[ns]')

    @property
    def resolution_string(self) -> str:
        """
        Return a string representing the lowest timedelta resolution.

        Each timedelta has a defined resolution that represents the lowest OR
        most granular level of precision. Each level of resolution is
        represented by a short string as defined below:

        Resolution:     Return value

        * Days:         'D'
        * Hours:        'H'
        * Minutes:      'T'
        * Seconds:      'S'
        * Milliseconds: 'L'
        * Microseconds: 'U'
        * Nanoseconds:  'N'

        Returns
        -------
        str
            Timedelta resolution.

        Examples
        --------
        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')
        >>> td.resolution_string
        'N'

        >>> td = pd.Timedelta('1 days 2 min 3 us')
        >>> td.resolution_string
        'U'

        >>> td = pd.Timedelta('2 min 3 s')
        >>> td.resolution_string
        'S'

        >>> td = pd.Timedelta(36, unit='us')
        >>> td.resolution_string
        'U'
        """
        self._ensure_components()
        if self._ns:
            return "N"
        elif self._us:
            return "U"
        elif self._ms:
            return "L"
        elif self._s:
            return "S"
        elif self._m:
            return "T"
        elif self._h:
            return "H"
        else:
            return "D"

    @property
    def nanoseconds(self):
        """
        Return the number of nanoseconds (n), where 0 <= n < 1 microsecond.

        Returns
        -------
        int
            Number of nanoseconds.

        See Also
        --------
        Timedelta.components : Return all attributes with assigned values
            (i.e. days, hours, minutes, seconds, milliseconds, microseconds,
            nanoseconds).

        Examples
        --------
        **Using string input**

        >>> td = pd.Timedelta('1 days 2 min 3 us 42 ns')

        >>> td.nanoseconds
        42

        **Using integer input**

        >>> td = pd.Timedelta(42, unit='ns')
        >>> td.nanoseconds
        42
        """
        self._ensure_components()
        return self._ns

    def _repr_base(self, format=None) -> str:
        """

        Parameters
        ----------
        format : None|all|sub_day|long

        Returns
        -------
        converted : string of a Timedelta

        """
        cdef object sign, seconds_pretty, subs, fmt, comp_dict

        self._ensure_components()

        if self._d < 0:
            sign = " +"
        else:
            sign = " "

        if format == 'all':
            fmt = ("{days} days{sign}{hours:02}:{minutes:02}:{seconds:02}."
                   "{milliseconds:03}{microseconds:03}{nanoseconds:03}")
        else:
            # if we have a partial day
            subs = (self._h or self._m or self._s or
                    self._ms or self._us or self._ns)

            if self._ms or self._us or self._ns:
                seconds_fmt = "{seconds:02}.{milliseconds:03}{microseconds:03}"
                if self._ns:
                    # GH#9309
                    seconds_fmt += "{nanoseconds:03}"
            else:
                seconds_fmt = "{seconds:02}"

            if format == 'sub_day' and not self._d:
                fmt = "{hours:02}:{minutes:02}:" + seconds_fmt
            elif subs or format == 'long':
                fmt = "{days} days{sign}{hours:02}:{minutes:02}:" + seconds_fmt
            else:
                fmt = "{days} days"

        comp_dict = self.components._asdict()
        comp_dict['sign'] = sign

        return fmt.format(**comp_dict)

    def __repr__(self) -> str:
        repr_based = self._repr_base(format='long')
        return f"Timedelta('{repr_based}')"

    def __str__(self) -> str:
        return self._repr_base(format='long')

    def __bool__(self) -> bool:
        return self.value != 0

    def isoformat(self) -> str:
        """
        Format Timedelta as ISO 8601 Duration like
        ``P[n]Y[n]M[n]DT[n]H[n]M[n]S``, where the ``[n]`` s are replaced by the
        values. See https://en.wikipedia.org/wiki/ISO_8601#Durations.

        Returns
        -------
        str

        See Also
        --------
        Timestamp.isoformat : Function is used to convert the given
            Timestamp object into the ISO format.

        Notes
        -----
        The longest component is days, whose value may be larger than
        365.
        Every component is always included, even if its value is 0.
        Pandas uses nanosecond precision, so up to 9 decimal places may
        be included in the seconds component.
        Trailing 0's are removed from the seconds component after the decimal.
        We do not 0 pad components, so it's `...T5H...`, not `...T05H...`

        Examples
        --------
        >>> td = pd.Timedelta(days=6, minutes=50, seconds=3,
        ...                   milliseconds=10, microseconds=10, nanoseconds=12)

        >>> td.isoformat()
        'P6DT0H50M3.010010012S'
        >>> pd.Timedelta(hours=1, seconds=10).isoformat()
        'P0DT1H0M10S'
        >>> pd.Timedelta(days=500.5).isoformat()
        'P500DT12H0M0S'
        """
        components = self.components
        seconds = (f'{components.seconds}.'
                   f'{components.milliseconds:0>3}'
                   f'{components.microseconds:0>3}'
                   f'{components.nanoseconds:0>3}')
        # Trim unnecessary 0s, 1.000000000 -> 1
        seconds = seconds.rstrip('0').rstrip('.')
        tpl = (f'P{components.days}DT{components.hours}'
               f'H{components.minutes}M{seconds}S')
        return tpl


# Python front end to C extension type _Timedelta
# This serves as the box for timedelta64

class Timedelta(_Timedelta):
    """
    Represents a duration, the difference between two dates or times.

    Timedelta is the pandas equivalent of python's ``datetime.timedelta``
    and is interchangeable with it in most cases.

    Parameters
    ----------
    value : Timedelta, timedelta, np.timedelta64, str, or int
    unit : str, default 'ns'
        Denote the unit of the input, if input is an integer.

        Possible values:

        * 'W', 'D', 'T', 'S', 'L', 'U', or 'N'
        * 'days' or 'day'
        * 'hours', 'hour', 'hr', or 'h'
        * 'minutes', 'minute', 'min', or 'm'
        * 'seconds', 'second', or 'sec'
        * 'milliseconds', 'millisecond', 'millis', or 'milli'
        * 'microseconds', 'microsecond', 'micros', or 'micro'
        * 'nanoseconds', 'nanosecond', 'nanos', 'nano', or 'ns'.

    **kwargs
        Available kwargs: {days, seconds, microseconds,
        milliseconds, minutes, hours, weeks}.
        Values for construction in compat with datetime.timedelta.
        Numpy ints and floats will be coerced to python ints and floats.

    Notes
    -----
    The ``.value`` attribute is always in ns.

    If the precision is higher than nanoseconds, the precision of the duration is
    truncated to nanoseconds.
    """

    def __new__(cls, object value=_no_input, unit=None, **kwargs):
        cdef _Timedelta td_base

        if value is _no_input:
            if not len(kwargs):
                raise ValueError("cannot construct a Timedelta without a "
                                 "value/unit or descriptive keywords "
                                 "(days,seconds....)")

            kwargs = {key: _to_py_int_float(kwargs[key]) for key in kwargs}

            nano = convert_to_timedelta64(kwargs.pop('nanoseconds', 0), 'ns')
            try:
                value = nano + convert_to_timedelta64(timedelta(**kwargs),
                                                      'ns')
            except TypeError as e:
                raise ValueError(
                    "cannot construct a Timedelta from the passed arguments, "
                    "allowed keywords are "
                    "[weeks, days, hours, minutes, seconds, "
                    "milliseconds, microseconds, nanoseconds]"
                )

        if unit in {'Y', 'y', 'M'}:
            raise ValueError(
                "Units 'M', 'Y', and 'y' are no longer supported, as they do not "
                "represent unambiguous timedelta values durations."
            )

        # GH 30543 if pd.Timedelta already passed, return it
        # check that only value is passed
        if isinstance(value, _Timedelta) and unit is None and len(kwargs) == 0:
            return value
        elif isinstance(value, _Timedelta):
            value = value.value
        elif isinstance(value, str):
            if unit is not None:
                raise ValueError("unit must not be specified if the value is a str")
            if (len(value) > 0 and value[0] == 'P') or (
                len(value) > 1 and value[:2] == '-P'
            ):
                value = parse_iso_format_string(value)
            else:
                value = parse_timedelta_string(value)
            value = np.timedelta64(value)
        elif PyDelta_Check(value):
            value = convert_to_timedelta64(value, 'ns')
        elif is_timedelta64_object(value):
            if unit is not None:
                value = value.astype(f'timedelta64[{unit}]')
            value = ensure_td64ns(value)
        elif is_tick_object(value):
            value = np.timedelta64(value.nanos, 'ns')
        elif is_integer_object(value) or is_float_object(value):
            # unit=None is de-facto 'ns'
            unit = parse_timedelta_unit(unit)
            value = convert_to_timedelta64(value, unit)
        elif checknull_with_nat(value):
            return NaT
        else:
            raise ValueError(
                "Value must be Timedelta, string, integer, "
                f"float, timedelta or convertible, not {type(value).__name__}"
            )

        if is_timedelta64_object(value):
            value = value.view('i8')

        # nat
        if value == NPY_NAT:
            return NaT

        # make timedelta happy
        td_base = _Timedelta.__new__(cls, microseconds=int(value) // 1000)
        td_base.value = value
        td_base.is_populated = 0
        return td_base

    def __setstate__(self, state):
        (value) = state
        self.value = value

    def __reduce__(self):
        object_state = self.value,
        return (Timedelta, object_state)

    @cython.cdivision(True)
    def _round(self, freq, mode):
        cdef:
            int64_t result, unit, remainder
            ndarray[int64_t] arr

        from pandas._libs.tslibs.offsets import to_offset
        unit = to_offset(freq).nanos

        arr = np.array([self.value], dtype="i8")
        result = round_nsint64(arr, mode, unit)[0]
        return Timedelta(result, unit="ns")

    def round(self, freq):
        """
        Round the Timedelta to the specified resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.

        Returns
        -------
        a new Timedelta rounded to the given resolution of `freq`

        Raises
        ------
        ValueError if the freq cannot be converted
        """
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN)

    def floor(self, freq):
        """
        Return a new Timedelta floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.
        """
        return self._round(freq, RoundTo.MINUS_INFTY)

    def ceil(self, freq):
        """
        Return a new Timedelta ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution.
        """
        return self._round(freq, RoundTo.PLUS_INFTY)

    # ----------------------------------------------------------------
    # Arithmetic Methods
    # TODO: Can some of these be defined in the cython class?

    __inv__ = _op_unary_method(lambda x: -x, '__inv__')
    __neg__ = _op_unary_method(lambda x: -x, '__neg__')
    __pos__ = _op_unary_method(lambda x: x, '__pos__')
    __abs__ = _op_unary_method(lambda x: abs(x), '__abs__')

    __add__ = _binary_op_method_timedeltalike(lambda x, y: x + y, '__add__')
    __radd__ = _binary_op_method_timedeltalike(lambda x, y: x + y, '__radd__')
    __sub__ = _binary_op_method_timedeltalike(lambda x, y: x - y, '__sub__')
    __rsub__ = _binary_op_method_timedeltalike(lambda x, y: y - x, '__rsub__')

    def __mul__(self, other):
        if is_integer_object(other) or is_float_object(other):
            return Timedelta(other * self.value, unit='ns')

        elif is_array(other):
            # ndarray-like
            return other * self.to_timedelta64()

        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = Timedelta(other)
            if other is NaT:
                return np.nan
            return self.value / float(other.value)

        elif is_integer_object(other) or is_float_object(other):
            # integers or floats
            return Timedelta(self.value / other, unit='ns')

        elif is_array(other):
            return self.to_timedelta64() / other

        return NotImplemented

    def __rtruediv__(self, other):
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = Timedelta(other)
            if other is NaT:
                return np.nan
            return float(other.value) / self.value

        elif is_array(other):
            if other.dtype.kind == "O":
                # GH#31869
                return np.array([x / self for x in other])
            return other / self.to_timedelta64()

        return NotImplemented

    def __floordiv__(self, other):
        # numpy does not implement floordiv for timedelta64 dtype, so we cannot
        # just defer
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = Timedelta(other)
            if other is NaT:
                return np.nan
            return self.value // other.value

        elif is_integer_object(other) or is_float_object(other):
            return Timedelta(self.value // other, unit='ns')

        elif is_array(other):
            if other.dtype.kind == 'm':
                # also timedelta-like
                return _broadcast_floordiv_td64(self.value, other, _floordiv)
            elif other.dtype.kind in ['i', 'u', 'f']:
                if other.ndim == 0:
                    return Timedelta(self.value // other)
                else:
                    return self.to_timedelta64() // other

            raise TypeError(f'Invalid dtype {other.dtype} for __floordiv__')

        return NotImplemented

    def __rfloordiv__(self, other):
        # numpy does not implement floordiv for timedelta64 dtype, so we cannot
        # just defer
        if _should_cast_to_timedelta(other):
            # We interpret NaT as timedelta64("NaT")
            other = Timedelta(other)
            if other is NaT:
                return np.nan
            return other.value // self.value

        elif is_array(other):
            if other.dtype.kind == 'm':
                # also timedelta-like
                return _broadcast_floordiv_td64(self.value, other, _rfloordiv)

            # Includes integer array // Timedelta, disallowed in GH#19761
            raise TypeError(f'Invalid dtype {other.dtype} for __floordiv__')

        return NotImplemented

    def __mod__(self, other):
        # Naive implementation, room for optimization
        return self.__divmod__(other)[1]

    def __rmod__(self, other):
        # Naive implementation, room for optimization
        return self.__rdivmod__(other)[1]

    def __divmod__(self, other):
        # Naive implementation, room for optimization
        div = self // other
        return div, self - div * other

    def __rdivmod__(self, other):
        # Naive implementation, room for optimization
        div = other // self
        return div, other - div * self


cdef bint is_any_td_scalar(object obj):
    """
    Cython equivalent for `isinstance(obj, (timedelta, np.timedelta64, Tick))`

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return (
        PyDelta_Check(obj) or is_timedelta64_object(obj) or is_tick_object(obj)
    )


cdef bint _should_cast_to_timedelta(object obj):
    """
    Should we treat this object as a Timedelta for the purpose of a binary op
    """
    return (
        is_any_td_scalar(obj) or obj is None or obj is NaT or isinstance(obj, str)
    )


cdef _floordiv(int64_t value, right):
    return value // right


cdef _rfloordiv(int64_t value, right):
    # analogous to referencing operator.div, but there is no operator.rfloordiv
    return right // value


cdef _broadcast_floordiv_td64(
    int64_t value,
    ndarray other,
    object (*operation)(int64_t value, object right)
):
    """
    Boilerplate code shared by Timedelta.__floordiv__ and
    Timedelta.__rfloordiv__ because np.timedelta64 does not implement these.

    Parameters
    ----------
    value : int64_t; `self.value` from a Timedelta object
    other : object
    operation : function, either _floordiv or _rfloordiv

    Returns
    -------
    result : varies based on `other`
    """
    # assumes other.dtype.kind == 'm', i.e. other is timedelta-like

    # We need to watch out for np.timedelta64('NaT').
    mask = other.view('i8') == NPY_NAT

    if other.ndim == 0:
        if mask:
            return np.nan

        return operation(value, other.astype('m8[ns]').astype('i8'))

    else:
        res = operation(value, other.astype('m8[ns]').astype('i8'))

        if mask.any():
            res = res.astype('f8')
            res[mask] = np.nan
        return res


# resolution in ns
Timedelta.min = Timedelta(np.iinfo(np.int64).min + 1)
Timedelta.max = Timedelta(np.iinfo(np.int64).max)
Timedelta.resolution = Timedelta(nanoseconds=1)
