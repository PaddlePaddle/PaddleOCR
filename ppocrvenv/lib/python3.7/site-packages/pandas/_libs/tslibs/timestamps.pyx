"""
_Timestamp is a c-defined subclass of datetime.datetime

_Timestamp is PITA. Because we inherit from datetime, which has very specific
construction requirements, we need to do object instantiation in python
(see Timestamp class below). This will serve as a C extension type that
shadows the python class, where we do any heavy lifting.
"""
import warnings

cimport cython

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int8_t,
    int64_t,
    ndarray,
    uint8_t,
)

cnp.import_array()

from cpython.datetime cimport (  # alias bc `tzinfo` is a kwarg below
    PyDate_Check,
    PyDateTime_Check,
    PyDateTime_IMPORT,
    PyDelta_Check,
    PyTZInfo_Check,
    datetime,
    time,
    tzinfo as tzinfo_type,
)
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject_RichCompare,
    PyObject_RichCompareBool,
)

PyDateTime_IMPORT

from pandas._libs.tslibs cimport ccalendar
from pandas._libs.tslibs.base cimport ABCTimestamp
from pandas._libs.tslibs.conversion cimport (
    _TSObject,
    convert_datetime_to_tsobject,
    convert_to_tsobject,
    normalize_i8_stamp,
)
from pandas._libs.tslibs.util cimport (
    is_array,
    is_datetime64_object,
    is_float_object,
    is_integer_object,
    is_timedelta64_object,
)

from pandas._libs.tslibs.fields import (
    RoundTo,
    get_date_name_field,
    get_start_end_field,
    round_nsint64,
)

from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)
from pandas._libs.tslibs.np_datetime cimport (
    check_dts_bounds,
    cmp_scalar,
    dt64_to_dtstruct,
    npy_datetimestruct,
    pydatetime_to_dt64,
)

from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

from pandas._libs.tslibs.offsets cimport (
    is_offset_object,
    to_offset,
)
from pandas._libs.tslibs.timedeltas cimport (
    delta_to_nanoseconds,
    is_any_td_scalar,
)

from pandas._libs.tslibs.timedeltas import Timedelta

from pandas._libs.tslibs.timezones cimport (
    get_timezone,
    is_utc,
    maybe_get_tz,
    treat_tz_as_pytz,
    tz_compare,
    utc_pytz as UTC,
)
from pandas._libs.tslibs.tzconversion cimport (
    tz_convert_from_utc_single,
    tz_localize_to_utc_single,
)

# ----------------------------------------------------------------------
# Constants
_zero_time = time(0, 0)
_no_input = object()

# ----------------------------------------------------------------------


cdef inline object create_timestamp_from_ts(int64_t value,
                                            npy_datetimestruct dts,
                                            tzinfo tz, object freq, bint fold):
    """ convenience routine to construct a Timestamp from its parts """
    cdef _Timestamp ts_base
    ts_base = _Timestamp.__new__(Timestamp, dts.year, dts.month,
                                 dts.day, dts.hour, dts.min,
                                 dts.sec, dts.us, tz, fold=fold)
    ts_base.value = value
    ts_base._freq = freq
    ts_base.nanosecond = dts.ps // 1000

    return ts_base


def _unpickle_timestamp(value, freq, tz):
    # GH#41949 dont warn on unpickle if we have a freq
    ts = Timestamp(value, tz=tz)
    ts._set_freq(freq)
    return ts


# ----------------------------------------------------------------------

def integer_op_not_supported(obj):
    # GH#22535 add/sub of integers and int-arrays is no longer allowed
    # Note we return rather than raise the exception so we can raise in
    #  the caller; mypy finds this more palatable.
    cls = type(obj).__name__

    # GH#30886 using an fstring raises SystemError
    int_addsub_msg = (
        f"Addition/subtraction of integers and integer-arrays with {cls} is "
        "no longer supported.  Instead of adding/subtracting `n`, "
        "use `n * obj.freq`"
    )
    return TypeError(int_addsub_msg)


# ----------------------------------------------------------------------

cdef class _Timestamp(ABCTimestamp):

    # higher than np.ndarray and np.matrix
    __array_priority__ = 100
    dayofweek = _Timestamp.day_of_week
    dayofyear = _Timestamp.day_of_year

    cpdef void _set_freq(self, freq):
        # set the ._freq attribute without going through the constructor,
        #  which would issue a warning
        # Caller is responsible for validation
        self._freq = freq

    @property
    def freq(self):
        warnings.warn(
            "Timestamp.freq is deprecated and will be removed in a future version",
            FutureWarning,
            stacklevel=1,
        )
        return self._freq

    def __hash__(_Timestamp self):
        if self.nanosecond:
            return hash(self.value)
        return datetime.__hash__(self)

    def __richcmp__(_Timestamp self, object other, int op):
        cdef:
            _Timestamp ots
            int ndim

        if isinstance(other, _Timestamp):
            ots = other
        elif other is NaT:
            return op == Py_NE
        elif PyDateTime_Check(other) or is_datetime64_object(other):
            if self.nanosecond == 0 and PyDateTime_Check(other):
                val = self.to_pydatetime()
                return PyObject_RichCompareBool(val, other, op)

            try:
                ots = type(self)(other)
            except ValueError:
                if is_datetime64_object(other):
                    # cast non-nano dt64 to pydatetime
                    other = other.astype(object)
                return self._compare_outside_nanorange(other, op)

        elif is_array(other):
            # avoid recursion error GH#15183
            if other.dtype.kind == "M":
                if self.tz is None:
                    return PyObject_RichCompare(self.asm8, other, op)
                elif op == Py_NE:
                    return np.ones(other.shape, dtype=np.bool_)
                elif op == Py_EQ:
                    return np.zeros(other.shape, dtype=np.bool_)
                raise TypeError(
                    "Cannot compare tz-naive and tz-aware timestamps"
                )
            elif other.dtype.kind == "O":
                # Operate element-wise
                return np.array(
                    [PyObject_RichCompare(self, x, op) for x in other],
                    dtype=bool,
                )
            elif op == Py_NE:
                return np.ones(other.shape, dtype=np.bool_)
            elif op == Py_EQ:
                return np.zeros(other.shape, dtype=np.bool_)
            return NotImplemented

        elif PyDate_Check(other):
            # returning NotImplemented defers to the `date` implementation
            #  which incorrectly drops tz and normalizes to midnight
            #  before comparing
            # We follow the stdlib datetime behavior of never being equal
            warnings.warn(
                "Comparison of Timestamp with datetime.date is deprecated in "
                "order to match the standard library behavior.  "
                "In a future version these will be considered non-comparable."
                "Use 'ts == pd.Timestamp(date)' or 'ts.date() == date' instead.",
                FutureWarning,
                stacklevel=1,
            )
            return NotImplemented
        else:
            return NotImplemented

        if not self._can_compare(ots):
            if op == Py_NE or op == Py_EQ:
                return NotImplemented
            raise TypeError(
                "Cannot compare tz-naive and tz-aware timestamps"
            )
        return cmp_scalar(self.value, ots.value, op)

    cdef bint _compare_outside_nanorange(_Timestamp self, datetime other,
                                         int op) except -1:
        cdef:
            datetime dtval = self.to_pydatetime(warn=False)

        if not self._can_compare(other):
            return NotImplemented

        if self.nanosecond == 0:
            return PyObject_RichCompareBool(dtval, other, op)

        # otherwise we have dtval < self
        if op == Py_NE:
            return True
        if op == Py_EQ:
            return False
        if op == Py_LE or op == Py_LT:
            return self.year <= other.year
        if op == Py_GE or op == Py_GT:
            return self.year >= other.year

    cdef bint _can_compare(self, datetime other):
        if self.tzinfo is not None:
            return other.tzinfo is not None
        return other.tzinfo is None

    def __add__(self, other):
        cdef:
            int64_t nanos = 0

        if is_any_td_scalar(other):
            nanos = delta_to_nanoseconds(other)
            result = type(self)(self.value + nanos, tz=self.tzinfo)
            if result is not NaT:
                result._set_freq(self._freq)  # avoid warning in constructor
            return result

        elif is_integer_object(other):
            raise integer_op_not_supported(self)

        elif is_array(other):
            if other.dtype.kind in ['i', 'u']:
                raise integer_op_not_supported(self)
            if other.dtype.kind == "m":
                if self.tz is None:
                    return self.asm8 + other
                return np.asarray(
                    [self + other[n] for n in range(len(other))],
                    dtype=object,
                )

        elif not isinstance(self, _Timestamp):
            # cython semantics, args have been switched and this is __radd__
            return other.__add__(self)

        return NotImplemented

    def __sub__(self, other):

        if is_any_td_scalar(other) or is_integer_object(other):
            neg_other = -other
            return self + neg_other

        elif is_array(other):
            if other.dtype.kind in ['i', 'u']:
                raise integer_op_not_supported(self)
            if other.dtype.kind == "m":
                if self.tz is None:
                    return self.asm8 - other
                return np.asarray(
                    [self - other[n] for n in range(len(other))],
                    dtype=object,
                )
            return NotImplemented

        if other is NaT:
            return NaT

        # coerce if necessary if we are a Timestamp-like
        if (PyDateTime_Check(self)
                and (PyDateTime_Check(other) or is_datetime64_object(other))):
            # both_timestamps is to determine whether Timedelta(self - other)
            # should raise the OOB error, or fall back returning a timedelta.
            both_timestamps = (isinstance(other, _Timestamp) and
                               isinstance(self, _Timestamp))
            if isinstance(self, _Timestamp):
                other = type(self)(other)
            else:
                self = type(other)(self)

            # validate tz's
            if not tz_compare(self.tzinfo, other.tzinfo):
                raise TypeError("Timestamp subtraction must have the "
                                "same timezones or no timezones")

            # scalar Timestamp/datetime - Timestamp/datetime -> yields a
            # Timedelta
            try:
                return Timedelta(self.value - other.value)
            except (OverflowError, OutOfBoundsDatetime) as err:
                if isinstance(other, _Timestamp):
                    if both_timestamps:
                        raise OutOfBoundsDatetime(
                            "Result is too large for pandas.Timedelta. Convert inputs "
                            "to datetime.datetime with 'Timestamp.to_pydatetime()' "
                            "before subtracting."
                        ) from err
                # We get here in stata tests, fall back to stdlib datetime
                #  method and return stdlib timedelta object
                pass
        elif is_datetime64_object(self):
            # GH#28286 cython semantics for __rsub__, `other` is actually
            #  the Timestamp
            return type(other)(self) - other

        return NotImplemented

    # -----------------------------------------------------------------

    cdef int64_t _maybe_convert_value_to_local(self):
        """Convert UTC i8 value to local i8 value if tz exists"""
        cdef:
            int64_t val
            tzinfo own_tz = self.tzinfo
            npy_datetimestruct dts

        if own_tz is not None and not is_utc(own_tz):
            val = pydatetime_to_dt64(self, &dts) + self.nanosecond
        else:
            val = self.value
        return val

    cdef bint _get_start_end_field(self, str field, freq):
        cdef:
            int64_t val
            dict kwds
            ndarray[uint8_t, cast=True] out
            int month_kw

        if freq:
            kwds = freq.kwds
            month_kw = kwds.get('startingMonth', kwds.get('month', 12))
            freqstr = self._freqstr
        else:
            month_kw = 12
            freqstr = None

        val = self._maybe_convert_value_to_local()
        out = get_start_end_field(np.array([val], dtype=np.int64),
                                  field, freqstr, month_kw)
        return out[0]

    cdef _warn_on_field_deprecation(self, freq, str field):
        """
        Warn if the removal of .freq change the value of start/end properties.
        """
        cdef:
            bint needs = False

        if freq is not None:
            kwds = freq.kwds
            month_kw = kwds.get("startingMonth", kwds.get("month", 12))
            freqstr = self._freqstr
            if month_kw != 12:
                needs = True
            if freqstr.startswith("B"):
                needs = True

            if needs:
                warnings.warn(
                    "Timestamp.freq is deprecated and will be removed in a future "
                    "version. When you have a freq, use "
                    f"freq.{field}(timestamp) instead",
                    FutureWarning,
                    stacklevel=1,
                )

    @property
    def is_month_start(self) -> bool:
        """
        Return True if date is first day of month.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_month_start
        False

        >>> ts = pd.Timestamp(2020, 1, 1)
        >>> ts.is_month_start
        True
        """
        if self._freq is None:
            # fast-path for non-business frequencies
            return self.day == 1
        self._warn_on_field_deprecation(self._freq, "is_month_start")
        return self._get_start_end_field("is_month_start", self._freq)

    @property
    def is_month_end(self) -> bool:
        """
        Return True if date is last day of month.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_month_end
        False

        >>> ts = pd.Timestamp(2020, 12, 31)
        >>> ts.is_month_end
        True
        """
        if self._freq is None:
            # fast-path for non-business frequencies
            return self.day == self.days_in_month
        self._warn_on_field_deprecation(self._freq, "is_month_end")
        return self._get_start_end_field("is_month_end", self._freq)

    @property
    def is_quarter_start(self) -> bool:
        """
        Return True if date is first day of the quarter.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_quarter_start
        False

        >>> ts = pd.Timestamp(2020, 4, 1)
        >>> ts.is_quarter_start
        True
        """
        if self._freq is None:
            # fast-path for non-business frequencies
            return self.day == 1 and self.month % 3 == 1
        self._warn_on_field_deprecation(self._freq, "is_quarter_start")
        return self._get_start_end_field("is_quarter_start", self._freq)

    @property
    def is_quarter_end(self) -> bool:
        """
        Return True if date is last day of the quarter.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_quarter_end
        False

        >>> ts = pd.Timestamp(2020, 3, 31)
        >>> ts.is_quarter_end
        True
        """
        if self._freq is None:
            # fast-path for non-business frequencies
            return (self.month % 3) == 0 and self.day == self.days_in_month
        self._warn_on_field_deprecation(self._freq, "is_quarter_end")
        return self._get_start_end_field("is_quarter_end", self._freq)

    @property
    def is_year_start(self) -> bool:
        """
        Return True if date is first day of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_year_start
        False

        >>> ts = pd.Timestamp(2020, 1, 1)
        >>> ts.is_year_start
        True
        """
        if self._freq is None:
            # fast-path for non-business frequencies
            return self.day == self.month == 1
        self._warn_on_field_deprecation(self._freq, "is_year_start")
        return self._get_start_end_field("is_year_start", self._freq)

    @property
    def is_year_end(self) -> bool:
        """
        Return True if date is last day of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_year_end
        False

        >>> ts = pd.Timestamp(2020, 12, 31)
        >>> ts.is_year_end
        True
        """
        if self._freq is None:
            # fast-path for non-business frequencies
            return self.month == 12 and self.day == 31
        self._warn_on_field_deprecation(self._freq, "is_year_end")
        return self._get_start_end_field("is_year_end", self._freq)

    cdef _get_date_name_field(self, str field, object locale):
        cdef:
            int64_t val
            object[:] out

        val = self._maybe_convert_value_to_local()
        out = get_date_name_field(np.array([val], dtype=np.int64),
                                  field, locale=locale)
        return out[0]

    def day_name(self, locale=None) -> str:
        """
        Return the day name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the day name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.day_name()
        'Saturday'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.day_name()
        nan
        """
        return self._get_date_name_field("day_name", locale)

    def month_name(self, locale=None) -> str:
        """
        Return the month name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the month name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.month_name()
        'March'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.month_name()
        nan
        """
        return self._get_date_name_field("month_name", locale)

    @property
    def is_leap_year(self) -> bool:
        """
        Return True if year is a leap year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.is_leap_year
        True
        """
        return bool(ccalendar.is_leapyear(self.year))

    @property
    def day_of_week(self) -> int:
        """
        Return day of the week.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.day_of_week
        5
        """
        return self.weekday()

    @property
    def day_of_year(self) -> int:
        """
        Return the day of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.day_of_year
        74
        """
        return ccalendar.get_day_of_year(self.year, self.month, self.day)

    @property
    def quarter(self) -> int:
        """
        Return the quarter of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.quarter
        1
        """
        return ((self.month - 1) // 3) + 1

    @property
    def week(self) -> int:
        """
        Return the week number of the year.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.week
        11
        """
        return ccalendar.get_week_of_year(self.year, self.month, self.day)

    @property
    def days_in_month(self) -> int:
        """
        Return the number of days in the month.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14)
        >>> ts.days_in_month
        31
        """
        return ccalendar.get_days_in_month(self.year, self.month)

    # -----------------------------------------------------------------
    # Transformation Methods

    def normalize(self) -> "Timestamp":
        """
        Normalize Timestamp to midnight, preserving tz information.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14, 15, 30)
        >>> ts.normalize()
        Timestamp('2020-03-14 00:00:00')
        """
        cdef:
            local_val = self._maybe_convert_value_to_local()
            int64_t normalized

        normalized = normalize_i8_stamp(local_val)
        return Timestamp(normalized).tz_localize(self.tzinfo)

    # -----------------------------------------------------------------
    # Pickle Methods

    def __reduce_ex__(self, protocol):
        # python 3.6 compat
        # https://bugs.python.org/issue28730
        # now __reduce_ex__ is defined and higher priority than __reduce__
        return self.__reduce__()

    def __setstate__(self, state):
        self.value = state[0]
        self._freq = state[1]
        self.tzinfo = state[2]

    def __reduce__(self):
        object_state = self.value, self._freq, self.tzinfo
        return (_unpickle_timestamp, object_state)

    # -----------------------------------------------------------------
    # Rendering Methods

    def isoformat(self, sep: str = "T") -> str:
        base = super(_Timestamp, self).isoformat(sep=sep)
        if self.nanosecond == 0:
            return base

        if self.tzinfo is not None:
            base1, base2 = base[:-6], base[-6:]
        else:
            base1, base2 = base, ""

        if self.microsecond != 0:
            base1 += f"{self.nanosecond:03d}"
        else:
            base1 += f".{self.nanosecond:09d}"

        return base1 + base2

    def __repr__(self) -> str:
        stamp = self._repr_base
        zone = None

        try:
            stamp += self.strftime('%z')
        except ValueError:
            year2000 = self.replace(year=2000)
            stamp += year2000.strftime('%z')

        if self.tzinfo:
            zone = get_timezone(self.tzinfo)
        try:
            stamp += zone.strftime(' %%Z')
        except AttributeError:
            # e.g. tzlocal has no `strftime`
            pass

        tz = f", tz='{zone}'" if zone is not None else ""
        freq = "" if self._freq is None else f", freq='{self._freqstr}'"

        return f"Timestamp('{stamp}'{tz}{freq})"

    @property
    def _repr_base(self) -> str:
        return f"{self._date_repr} {self._time_repr}"

    @property
    def _date_repr(self) -> str:
        # Ideal here would be self.strftime("%Y-%m-%d"), but
        # the datetime strftime() methods require year >= 1900
        return f'{self.year}-{self.month:02d}-{self.day:02d}'

    @property
    def _time_repr(self) -> str:
        result = f'{self.hour:02d}:{self.minute:02d}:{self.second:02d}'

        if self.nanosecond != 0:
            result += f'.{self.nanosecond + 1000 * self.microsecond:09d}'
        elif self.microsecond != 0:
            result += f'.{self.microsecond:06d}'

        return result

    @property
    def _short_repr(self) -> str:
        # format a Timestamp with only _date_repr if possible
        # otherwise _repr_base
        if (self.hour == 0 and
                self.minute == 0 and
                self.second == 0 and
                self.microsecond == 0 and
                self.nanosecond == 0):
            return self._date_repr
        return self._repr_base

    # -----------------------------------------------------------------
    # Conversion Methods

    @property
    def asm8(self) -> np.datetime64:
        """
        Return numpy datetime64 format in nanoseconds.

        Examples
        --------
        >>> ts = pd.Timestamp(2020, 3, 14, 15)
        >>> ts.asm8
        numpy.datetime64('2020-03-14T15:00:00.000000000')
        """
        return np.datetime64(self.value, 'ns')

    def timestamp(self):
        """
        Return POSIX timestamp as float.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.timestamp()
        1584199972.192548
        """
        # GH 17329
        # Note: Naive timestamps will not match datetime.stdlib
        return round(self.value / 1e9, 6)

    cpdef datetime to_pydatetime(_Timestamp self, bint warn=True):
        """
        Convert a Timestamp object to a native Python datetime object.

        If warn=True, issue a warning if nanoseconds is nonzero.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.to_pydatetime()
        datetime.datetime(2020, 3, 14, 15, 32, 52, 192548)

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_pydatetime()
        NaT
        """
        if self.nanosecond != 0 and warn:
            warnings.warn("Discarding nonzero nanoseconds in conversion",
                          UserWarning, stacklevel=2)

        return datetime(self.year, self.month, self.day,
                        self.hour, self.minute, self.second,
                        self.microsecond, self.tzinfo)

    cpdef to_datetime64(self):
        """
        Return a numpy.datetime64 object with 'ns' precision.
        """
        return np.datetime64(self.value, "ns")

    def to_numpy(self, dtype=None, copy=False) -> np.datetime64:
        """
        Convert the Timestamp to a NumPy datetime64.

        .. versionadded:: 0.25.0

        This is an alias method for `Timestamp.to_datetime64()`. The dtype and
        copy parameters are available here only for compatibility. Their values
        will not affect the return value.

        Returns
        -------
        numpy.datetime64

        See Also
        --------
        DatetimeIndex.to_numpy : Similar method for DatetimeIndex.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.to_numpy()
        numpy.datetime64('2020-03-14T15:32:52.192548651')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_numpy()
        numpy.datetime64('NaT')
        """
        return self.to_datetime64()

    def to_period(self, freq=None):
        """
        Return an period of which this timestamp is an observation.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.to_period(freq='Y) # Year end frequency
        numpy.datetime64('2020-03-14T15:32:52.192548651')

        >>> ts.to_period(freq='M') # Month end frequency
        Period('2020-03', 'M')

        >>> ts.to_period(freq='W') # Weekly frequency
        Period('2020-03-09/2020-03-15', 'W-SUN')

        >>> ts.to_period(freq='Q') # Quarter end frequency
        Period('2020Q1', 'Q-DEC')
        """
        from pandas import Period

        if self.tz is not None:
            # GH#21333
            warnings.warn(
                "Converting to Period representation will drop timezone information.",
                UserWarning,
            )

        if freq is None:
            freq = self._freq
            warnings.warn(
                "In a future version, calling 'Timestamp.to_period()' without "
                "passing a 'freq' will raise an exception.",
                FutureWarning,
                stacklevel=2,
            )

        return Period(self, freq=freq)


# ----------------------------------------------------------------------

# Python front end to C extension type _Timestamp
# This serves as the box for datetime64


class Timestamp(_Timestamp):
    """
    Pandas replacement for python datetime.datetime object.

    Timestamp is the pandas equivalent of python's Datetime
    and is interchangeable with it in most cases. It's the type used
    for the entries that make up a DatetimeIndex, and other timeseries
    oriented data structures in pandas.

    Parameters
    ----------
    ts_input : datetime-like, str, int, float
        Value to be converted to Timestamp.
    freq : str, DateOffset
        Offset which Timestamp will have.
    tz : str, pytz.timezone, dateutil.tz.tzfile or None
        Time zone for time which Timestamp will have.
    unit : str
        Unit used for conversion if ts_input is of type int or float. The
        valid values are 'D', 'h', 'm', 's', 'ms', 'us', and 'ns'. For
        example, 's' means seconds and 'ms' means milliseconds.
    year, month, day : int
    hour, minute, second, microsecond : int, optional, default 0
    nanosecond : int, optional, default 0
    tzinfo : datetime.tzinfo, optional, default None
    fold : {0, 1}, default None, keyword-only
        Due to daylight saving time, one wall clock time can occur twice
        when shifting from summer to winter time; fold describes whether the
        datetime-like corresponds  to the first (0) or the second time (1)
        the wall clock hits the ambiguous time

        .. versionadded:: 1.1.0

    Notes
    -----
    There are essentially three calling conventions for the constructor. The
    primary form accepts four parameters. They can be passed by position or
    keyword.

    The other two forms mimic the parameters from ``datetime.datetime``. They
    can be passed by either position or keyword, but not both mixed together.

    Examples
    --------
    Using the primary calling convention:

    This converts a datetime-like string

    >>> pd.Timestamp('2017-01-01T12')
    Timestamp('2017-01-01 12:00:00')

    This converts a float representing a Unix epoch in units of seconds

    >>> pd.Timestamp(1513393355.5, unit='s')
    Timestamp('2017-12-16 03:02:35.500000')

    This converts an int representing a Unix-epoch in units of seconds
    and for a particular timezone

    >>> pd.Timestamp(1513393355, unit='s', tz='US/Pacific')
    Timestamp('2017-12-15 19:02:35-0800', tz='US/Pacific')

    Using the other two forms that mimic the API for ``datetime.datetime``:

    >>> pd.Timestamp(2017, 1, 1, 12)
    Timestamp('2017-01-01 12:00:00')

    >>> pd.Timestamp(year=2017, month=1, day=1, hour=12)
    Timestamp('2017-01-01 12:00:00')
    """

    @classmethod
    def fromordinal(cls, ordinal, freq=None, tz=None):
        """
        Timestamp.fromordinal(ordinal, freq=None, tz=None)

        Passed an ordinal, translate and convert to a ts.
        Note: by definition there cannot be any tz info on the ordinal itself.

        Parameters
        ----------
        ordinal : int
            Date corresponding to a proleptic Gregorian ordinal.
        freq : str, DateOffset
            Offset to apply to the Timestamp.
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for the Timestamp.

        Examples
        --------
        >>> pd.Timestamp.fromordinal(737425)
        Timestamp('2020-01-01 00:00:00')
        """
        return cls(datetime.fromordinal(ordinal),
                   freq=freq, tz=tz)

    @classmethod
    def now(cls, tz=None):
        """
        Timestamp.now(tz=None)

        Return new Timestamp object representing current time local to
        tz.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.now()
        Timestamp('2020-11-16 22:06:16.378782')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.now()
        NaT
        """
        if isinstance(tz, str):
            tz = maybe_get_tz(tz)
        return cls(datetime.now(tz))

    @classmethod
    def today(cls, tz=None):
        """
        Timestamp.today(cls, tz=None)

        Return the current time in the local timezone.  This differs
        from datetime.today() in that it can be localized to a
        passed timezone.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.today()
        Timestamp('2020-11-16 22:37:39.969883')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.today()
        NaT
        """
        return cls.now(tz)

    @classmethod
    def utcnow(cls):
        """
        Timestamp.utcnow()

        Return a new Timestamp representing UTC day and time.

        Examples
        --------
        >>> pd.Timestamp.utcnow()
        Timestamp('2020-11-16 22:50:18.092888+0000', tz='UTC')
        """
        return cls.now(UTC)

    @classmethod
    def utcfromtimestamp(cls, ts):
        """
        Timestamp.utcfromtimestamp(ts)

        Construct a naive UTC datetime from a POSIX timestamp.

        Examples
        --------
        >>> pd.Timestamp.fromtimestamp(1584199972)
        Timestamp('2020-03-14 15:32:52')
        """
        return cls(datetime.utcfromtimestamp(ts))

    @classmethod
    def fromtimestamp(cls, ts):
        """
        Timestamp.fromtimestamp(ts)

        Transform timestamp[, tz] to tz's local time from POSIX timestamp.

        Examples
        --------
        >>> pd.Timestamp.utcfromtimestamp(1584199972)
        Timestamp('2020-03-14 15:32:52')

        Note that the output may change depending on your local time.
        """
        return cls(datetime.fromtimestamp(ts))

    def strftime(self, format):
        """
        Timestamp.strftime(format)

        Return a string representing the given POSIX timestamp
        controlled by an explicit format string.

        Parameters
        ----------
        format : str
            Format string to convert Timestamp to string.
            See strftime documentation for more information on the format string:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.strftime('%Y-%m-%d %X')
        '2020-03-14 15:32:52'
        """
        return datetime.strftime(self, format)

    # Issue 25016.
    @classmethod
    def strptime(cls, date_string, format):
        """
        Timestamp.strptime(string, format)

        Function is not implemented. Use pd.to_datetime().
        """
        raise NotImplementedError(
            "Timestamp.strptime() is not implemented. "
            "Use to_datetime() to parse date strings."
        )

    @classmethod
    def combine(cls, date, time):
        """
        Timestamp.combine(date, time)

        Combine date, time into datetime with same date and time fields.

        Examples
        --------
        >>> from datetime import date, time
        >>> pd.Timestamp.combine(date(2020, 3, 14), time(15, 30, 15))
        Timestamp('2020-03-14 15:30:15')
        """
        return cls(datetime.combine(date, time))

    def __new__(
        cls,
        object ts_input=_no_input,
        object freq=None,
        tz=None,
        unit=None,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        nanosecond=None,
        tzinfo_type tzinfo=None,
        *,
        fold=None,
    ):
        # The parameter list folds together legacy parameter names (the first
        # four) and positional and keyword parameter names from pydatetime.
        #
        # There are three calling forms:
        #
        # - In the legacy form, the first parameter, ts_input, is required
        #   and may be datetime-like, str, int, or float. The second
        #   parameter, offset, is optional and may be str or DateOffset.
        #
        # - ints in the first, second, and third arguments indicate
        #   pydatetime positional arguments. Only the first 8 arguments
        #   (standing in for year, month, day, hour, minute, second,
        #   microsecond, tzinfo) may be non-None. As a shortcut, we just
        #   check that the second argument is an int.
        #
        # - Nones for the first four (legacy) arguments indicate pydatetime
        #   keyword arguments. year, month, and day are required. As a
        #   shortcut, we just check that the first argument was not passed.
        #
        # Mixing pydatetime positional and keyword arguments is forbidden!

        cdef:
            _TSObject ts
            tzinfo_type tzobj

        _date_attributes = [year, month, day, hour, minute, second,
                            microsecond, nanosecond]

        if tzinfo is not None:
            # GH#17690 tzinfo must be a datetime.tzinfo object, ensured
            #  by the cython annotation.
            if tz is not None:
                raise ValueError('Can provide at most one of tz, tzinfo')

            # User passed tzinfo instead of tz; avoid silently ignoring
            tz, tzinfo = tzinfo, None

        # Allow fold only for unambiguous input
        if fold is not None:
            if fold not in [0, 1]:
                raise ValueError(
                    "Valid values for the fold argument are None, 0, or 1."
                )

            if (ts_input is not _no_input and not (
                    PyDateTime_Check(ts_input) and
                    getattr(ts_input, 'tzinfo', None) is None)):
                raise ValueError(
                    "Cannot pass fold with possibly unambiguous input: int, "
                    "float, numpy.datetime64, str, or timezone-aware "
                    "datetime-like. Pass naive datetime-like or build "
                    "Timestamp from components."
                )

            if tz is not None and PyTZInfo_Check(tz) and treat_tz_as_pytz(tz):
                raise ValueError(
                    "pytz timezones do not support fold. Please use dateutil "
                    "timezones."
                )

            if hasattr(ts_input, 'fold'):
                ts_input = ts_input.replace(fold=fold)

        # GH 30543 if pd.Timestamp already passed, return it
        # check that only ts_input is passed
        # checking verbosely, because cython doesn't optimize
        # list comprehensions (as of cython 0.29.x)
        if (isinstance(ts_input, _Timestamp) and freq is None and
                tz is None and unit is None and year is None and
                month is None and day is None and hour is None and
                minute is None and second is None and
                microsecond is None and nanosecond is None and
                tzinfo is None):
            return ts_input
        elif isinstance(ts_input, str):
            # User passed a date string to parse.
            # Check that the user didn't also pass a date attribute kwarg.
            if any(arg is not None for arg in _date_attributes):
                raise ValueError(
                    "Cannot pass a date attribute keyword "
                    "argument when passing a date string"
                )

        elif ts_input is _no_input:
            # GH 31200
            # When year, month or day is not given, we call the datetime
            # constructor to make sure we get the same error message
            # since Timestamp inherits datetime
            datetime_kwargs = {
                "hour": hour or 0,
                "minute": minute or 0,
                "second": second or 0,
                "microsecond": microsecond or 0,
                "fold": fold or 0
            }
            if year is not None:
                datetime_kwargs["year"] = year
            if month is not None:
                datetime_kwargs["month"] = month
            if day is not None:
                datetime_kwargs["day"] = day

            ts_input = datetime(**datetime_kwargs)

        elif is_integer_object(freq):
            # User passed positional arguments:
            # Timestamp(year, month, day[, hour[, minute[, second[,
            # microsecond[, nanosecond[, tzinfo]]]]]])
            ts_input = datetime(ts_input, freq, tz, unit or 0,
                                year or 0, month or 0, day or 0, fold=fold or 0)
            nanosecond = hour
            tz = minute
            freq = None
            unit = None

        if getattr(ts_input, 'tzinfo', None) is not None and tz is not None:
            raise ValueError("Cannot pass a datetime or Timestamp with tzinfo with "
                             "the tz parameter. Use tz_convert instead.")

        tzobj = maybe_get_tz(tz)
        ts = convert_to_tsobject(ts_input, tzobj, unit, 0, 0, nanosecond or 0)

        if ts.value == NPY_NAT:
            return NaT

        if freq is None:
            # GH 22311: Try to extract the frequency of a given Timestamp input
            freq = getattr(ts_input, '_freq', None)
        else:
            warnings.warn(
                "The 'freq' argument in Timestamp is deprecated and will be "
                "removed in a future version.",
                FutureWarning,
                stacklevel=1,
            )
            if not is_offset_object(freq):
                freq = to_offset(freq)

        return create_timestamp_from_ts(ts.value, ts.dts, ts.tzinfo, freq, ts.fold)

    def _round(self, freq, mode, ambiguous='raise', nonexistent='raise'):
        cdef:
            int64_t nanos = to_offset(freq).nanos

        if self.tz is not None:
            value = self.tz_localize(None).value
        else:
            value = self.value

        value = np.array([value], dtype=np.int64)

        # Will only ever contain 1 element for timestamp
        r = round_nsint64(value, mode, nanos)[0]
        result = Timestamp(r, unit='ns')
        if self.tz is not None:
            result = result.tz_localize(
                self.tz, ambiguous=ambiguous, nonexistent=nonexistent
            )
        return result

    def round(self, freq, ambiguous='raise', nonexistent='raise'):
        """
        Round the Timestamp to the specified resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        a new Timestamp rounded to the given resolution of `freq`

        Raises
        ------
        ValueError if the freq cannot be converted

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be rounded using multiple frequency units:

        >>> ts.round(freq='H') # hour
        Timestamp('2020-03-14 16:00:00')

        >>> ts.round(freq='T') # minute
        Timestamp('2020-03-14 15:33:00')

        >>> ts.round(freq='S') # seconds
        Timestamp('2020-03-14 15:32:52')

        >>> ts.round(freq='L') # milliseconds
        Timestamp('2020-03-14 15:32:52.193000')

        ``freq`` can also be a multiple of a single unit, like '5T' (i.e.  5 minutes):

        >>> ts.round(freq='5T')
        Timestamp('2020-03-14 15:35:00')

        or a combination of multiple units, like '1H30T' (i.e. 1 hour and 30 minutes):

        >>> ts.round(freq='1H30T')
        Timestamp('2020-03-14 15:00:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.round()
        NaT
        """
        return self._round(
            freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent
        )

    def floor(self, freq, ambiguous='raise', nonexistent='raise'):
        """
        Return a new Timestamp floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be floored using multiple frequency units:

        >>> ts.floor(freq='H') # hour
        Timestamp('2020-03-14 15:00:00')

        >>> ts.floor(freq='T') # minute
        Timestamp('2020-03-14 15:32:00')

        >>> ts.floor(freq='S') # seconds
        Timestamp('2020-03-14 15:32:52')

        >>> ts.floor(freq='N') # nanoseconds
        Timestamp('2020-03-14 15:32:52.192548651')

        ``freq`` can also be a multiple of a single unit, like '5T' (i.e.  5 minutes):

        >>> ts.floor(freq='5T')
        Timestamp('2020-03-14 15:30:00')

        or a combination of multiple units, like '1H30T' (i.e. 1 hour and 30 minutes):

        >>> ts.floor(freq='1H30T')
        Timestamp('2020-03-14 15:00:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.floor()
        NaT
        """
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    def ceil(self, freq, ambiguous='raise', nonexistent='raise'):
        """
        Return a new Timestamp ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be ceiled using multiple frequency units:

        >>> ts.ceil(freq='H') # hour
        Timestamp('2020-03-14 16:00:00')

        >>> ts.ceil(freq='T') # minute
        Timestamp('2020-03-14 15:33:00')

        >>> ts.ceil(freq='S') # seconds
        Timestamp('2020-03-14 15:32:53')

        >>> ts.ceil(freq='U') # microseconds
        Timestamp('2020-03-14 15:32:52.192549')

        ``freq`` can also be a multiple of a single unit, like '5T' (i.e.  5 minutes):

        >>> ts.ceil(freq='5T')
        Timestamp('2020-03-14 15:35:00')

        or a combination of multiple units, like '1H30T' (i.e. 1 hour and 30 minutes):

        >>> ts.ceil(freq='1H30T')
        Timestamp('2020-03-14 16:30:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.ceil()
        NaT
        """
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    @property
    def tz(self):
        """
        Alias for tzinfo.

        Examples
        --------
        >>> ts = pd.Timestamp(1584226800, unit='s', tz='Europe/Stockholm')
        >>> ts.tz
        <DstTzInfo 'Europe/Stockholm' CET+1:00:00 STD>
        """
        return self.tzinfo

    @tz.setter
    def tz(self, value):
        # GH 3746: Prevent localizing or converting the index by setting tz
        raise AttributeError(
            "Cannot directly set timezone. "
            "Use tz_localize() or tz_convert() as appropriate"
        )

    @property
    def _freqstr(self):
        return getattr(self._freq, "freqstr", self._freq)

    @property
    def freqstr(self):
        """
        Return the total number of days in the month.
        """
        warnings.warn(
            "Timestamp.freqstr is deprecated and will be removed in a future version.",
            FutureWarning,
            stacklevel=1,
        )
        return self._freqstr

    def tz_localize(self, tz, ambiguous='raise', nonexistent='raise'):
        """
        Convert naive Timestamp to local time zone, or remove
        timezone from tz-aware Timestamp.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding local time.

        ambiguous : bool, 'NaT', default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, \
default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            The behavior is as follows:

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        localized : Timestamp

        Raises
        ------
        TypeError
            If the Timestamp is tz-aware and tz is not None.

        Examples
        --------
        Create a naive timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651')

        Add 'Europe/Stockholm' as timezone:

        >>> ts.tz_localize(tz='Europe/Stockholm')
        Timestamp('2020-03-14 15:32:52.192548651+0100', tz='Europe/Stockholm')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_localize()
        NaT
        """
        if ambiguous == 'infer':
            raise ValueError('Cannot infer offset with only one time.')

        nonexistent_options = ('raise', 'NaT', 'shift_forward', 'shift_backward')
        if nonexistent not in nonexistent_options and not PyDelta_Check(nonexistent):
            raise ValueError(
                "The nonexistent argument must be one of 'raise', "
                "'NaT', 'shift_forward', 'shift_backward' or a timedelta object"
            )

        if self.tzinfo is None:
            # tz naive, localize
            tz = maybe_get_tz(tz)
            if not isinstance(ambiguous, str):
                ambiguous = [ambiguous]
            value = tz_localize_to_utc_single(self.value, tz,
                                              ambiguous=ambiguous,
                                              nonexistent=nonexistent)
            out = Timestamp(value, tz=tz)
            if out is not NaT:
                out._set_freq(self._freq)  # avoid warning in constructor
            return out
        else:
            if tz is None:
                # reset tz
                value = tz_convert_from_utc_single(self.value, self.tz)
                out = Timestamp(value, tz=tz)
                if out is not NaT:
                    out._set_freq(self._freq)  # avoid warning in constructor
                return out
            else:
                raise TypeError(
                    "Cannot localize tz-aware Timestamp, use tz_convert for conversions"
                )

    def tz_convert(self, tz):
        """
        Convert tz-aware Timestamp to another time zone.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding UTC time.

        Returns
        -------
        converted : Timestamp

        Raises
        ------
        TypeError
            If Timestamp is tz-naive.

        Examples
        --------
        Create a timestamp object with UTC timezone:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Change to Tokyo timezone:

        >>> ts.tz_convert(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Can also use ``astimezone``:

        >>> ts.astimezone(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_convert(tz='Asia/Tokyo')
        NaT
        """
        if self.tzinfo is None:
            # tz naive, use tz_localize
            raise TypeError(
                "Cannot convert tz-naive Timestamp, use tz_localize to localize"
            )
        else:
            # Same UTC timestamp, different time zone
            out = Timestamp(self.value, tz=tz)
            if out is not NaT:
                out._set_freq(self._freq)  # avoid warning in constructor
            return out

    astimezone = tz_convert

    def replace(
        self,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        nanosecond=None,
        tzinfo=object,
        fold=None,
    ):
        """
        Implements datetime.replace, handles nanoseconds.

        Parameters
        ----------
        year : int, optional
        month : int, optional
        day : int, optional
        hour : int, optional
        minute : int, optional
        second : int, optional
        microsecond : int, optional
        nanosecond : int, optional
        tzinfo : tz-convertible, optional
        fold : int, optional

        Returns
        -------
        Timestamp with fields replaced

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Replace year and the hour:

        >>> ts.replace(year=1999, hour=10)
        Timestamp('1999-03-14 10:32:52.192548651+0000', tz='UTC')

        Replace timezone (not a conversion):

        >>> import pytz
        >>> ts.replace(tzinfo=pytz.timezone('US/Pacific'))
        Timestamp('2020-03-14 15:32:52.192548651-0700', tz='US/Pacific')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.replace(tzinfo=pytz.timezone('US/Pacific'))
        NaT
        """

        cdef:
            npy_datetimestruct dts
            int64_t value
            object k, v
            datetime ts_input
            tzinfo_type tzobj

        # set to naive if needed
        tzobj = self.tzinfo
        value = self.value

        # GH 37610. Preserve fold when replacing.
        if fold is None:
            fold = self.fold

        if tzobj is not None:
            value = tz_convert_from_utc_single(value, tzobj)

        # setup components
        dt64_to_dtstruct(value, &dts)
        dts.ps = self.nanosecond * 1000

        # replace
        def validate(k, v):
            """ validate integers """
            if not is_integer_object(v):
                raise ValueError(
                    f"value must be an integer, received {type(v)} for {k}"
                )
            return v

        if year is not None:
            dts.year = validate('year', year)
        if month is not None:
            dts.month = validate('month', month)
        if day is not None:
            dts.day = validate('day', day)
        if hour is not None:
            dts.hour = validate('hour', hour)
        if minute is not None:
            dts.min = validate('minute', minute)
        if second is not None:
            dts.sec = validate('second', second)
        if microsecond is not None:
            dts.us = validate('microsecond', microsecond)
        if nanosecond is not None:
            dts.ps = validate('nanosecond', nanosecond) * 1000
        if tzinfo is not object:
            tzobj = tzinfo

        # reconstruct & check bounds
        if tzobj is not None and treat_tz_as_pytz(tzobj):
            # replacing across a DST boundary may induce a new tzinfo object
            # see GH#18319
            ts_input = tzobj.localize(datetime(dts.year, dts.month, dts.day,
                                               dts.hour, dts.min, dts.sec,
                                               dts.us),
                                      is_dst=not bool(fold))
            tzobj = ts_input.tzinfo
        else:
            kwargs = {'year': dts.year, 'month': dts.month, 'day': dts.day,
                      'hour': dts.hour, 'minute': dts.min, 'second': dts.sec,
                      'microsecond': dts.us, 'tzinfo': tzobj,
                      'fold': fold}
            ts_input = datetime(**kwargs)

        ts = convert_datetime_to_tsobject(ts_input, tzobj)
        value = ts.value + (dts.ps // 1000)
        if value != NPY_NAT:
            check_dts_bounds(&dts)

        return create_timestamp_from_ts(value, dts, tzobj, self._freq, fold)

    def to_julian_date(self) -> np.float64:
        """
        Convert TimeStamp to a Julian Date.
        0 Julian date is noon January 1, 4713 BC.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52')
        >>> ts.to_julian_date()
        2458923.147824074
        """
        year = self.year
        month = self.month
        day = self.day
        if month <= 2:
            year -= 1
            month += 12
        return (day +
                np.fix((153 * month - 457) / 5) +
                365 * year +
                np.floor(year / 4) -
                np.floor(year / 100) +
                np.floor(year / 400) +
                1721118.5 +
                (self.hour +
                 self.minute / 60.0 +
                 self.second / 3600.0 +
                 self.microsecond / 3600.0 / 1e+6 +
                 self.nanosecond / 3600.0 / 1e+9
                ) / 24.0)


# Aliases
Timestamp.weekofyear = Timestamp.week
Timestamp.daysinmonth = Timestamp.days_in_month

# Add the min and max fields at the class level
cdef int64_t _NS_UPPER_BOUND = np.iinfo(np.int64).max
cdef int64_t _NS_LOWER_BOUND = NPY_NAT + 1

# Resolution is in nanoseconds
Timestamp.min = Timestamp(_NS_LOWER_BOUND)
Timestamp.max = Timestamp(_NS_UPPER_BOUND)
Timestamp.resolution = Timedelta(nanoseconds=1)  # GH#21336, GH#21365
