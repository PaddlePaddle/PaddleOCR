import operator
import re
import time
import warnings

import cython

from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDateTime_IMPORT,
    PyDelta_Check,
    date,
    datetime,
    time as dt_time,
    timedelta,
)

PyDateTime_IMPORT

from dateutil.easter import easter
from dateutil.relativedelta import relativedelta
import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
)

cnp.import_array()

# TODO: formalize having _libs.properties "above" tslibs in the dependency structure

from pandas._libs.properties import cache_readonly

from pandas._libs.tslibs cimport util
from pandas._libs.tslibs.util cimport (
    is_datetime64_object,
    is_float_object,
    is_integer_object,
)

from pandas._libs.tslibs.ccalendar import (
    MONTH_ALIASES,
    MONTH_TO_CAL_NUM,
    int_to_weekday,
    weekday_to_int,
)

from pandas._libs.tslibs.ccalendar cimport (
    DAY_NANOS,
    dayofweek,
    get_days_in_month,
    get_firstbday,
    get_lastbday,
)
from pandas._libs.tslibs.conversion cimport (
    convert_datetime_to_tsobject,
    localize_pydatetime,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)
from pandas._libs.tslibs.np_datetime cimport (
    dt64_to_dtstruct,
    dtstruct_to_dt64,
    npy_datetimestruct,
    pydate_to_dtstruct,
)
from pandas._libs.tslibs.tzconversion cimport tz_convert_from_utc_single

from .dtypes cimport PeriodDtypeCode
from .timedeltas cimport delta_to_nanoseconds

from .timedeltas import Timedelta

from .timestamps cimport _Timestamp

from .timestamps import Timestamp

# ---------------------------------------------------------------------
# Misc Helpers

cdef bint is_offset_object(object obj):
    return isinstance(obj, BaseOffset)


cdef bint is_tick_object(object obj):
    return isinstance(obj, Tick)


cdef datetime _as_datetime(datetime obj):
    if isinstance(obj, _Timestamp):
        return obj.to_pydatetime()
    return obj


cdef bint _is_normalized(datetime dt):
    if dt.hour != 0 or dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
        # Regardless of whether dt is datetime vs Timestamp
        return False
    if isinstance(dt, _Timestamp):
        return dt.nanosecond == 0
    return True


def apply_wrapper_core(func, self, other) -> ndarray:
    result = func(self, other)
    result = np.asarray(result)

    if self.normalize:
        # TODO: Avoid circular/runtime import
        from .vectorized import normalize_i8_timestamps
        result = normalize_i8_timestamps(result.view("i8"), None)

    return result


def apply_index_wraps(func):
    # Note: normally we would use `@functools.wraps(func)`, but this does
    # not play nicely with cython class methods
    def wrapper(self, other):
        # other is a DatetimeArray
        result = apply_wrapper_core(func, self, other)
        result = type(other)(result)
        warnings.warn("'Offset.apply_index(other)' is deprecated. "
                      "Use 'offset + other' instead.", FutureWarning)
        return result

    return wrapper


def apply_array_wraps(func):
    # Note: normally we would use `@functools.wraps(func)`, but this does
    # not play nicely with cython class methods
    def wrapper(self, other) -> np.ndarray:
        # other is a DatetimeArray
        result = apply_wrapper_core(func, self, other)
        return result

    # do @functools.wraps(func) manually since it doesn't work on cdef funcs
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def apply_wraps(func):
    # Note: normally we would use `@functools.wraps(func)`, but this does
    # not play nicely with cython class methods

    def wrapper(self, other):

        if other is NaT:
            return NaT
        elif isinstance(other, BaseOffset) or PyDelta_Check(other):
            # timedelta path
            return func(self, other)
        elif is_datetime64_object(other) or PyDate_Check(other):
            # PyDate_Check includes date, datetime
            other = Timestamp(other)
        else:
            # This will end up returning NotImplemented back in __add__
            raise ApplyTypeError

        tz = other.tzinfo
        nano = other.nanosecond

        if self._adjust_dst:
            other = other.tz_localize(None)

        result = func(self, other)

        result = Timestamp(result)
        if self._adjust_dst:
            result = result.tz_localize(tz)

        if self.normalize:
            result = result.normalize()

        # nanosecond may be deleted depending on offset process
        if not self.normalize and nano != 0:
            if result.nanosecond != nano:
                if result.tz is not None:
                    # convert to UTC
                    value = result.tz_localize(None).value
                else:
                    value = result.value
                result = Timestamp(value + nano)

        if tz is not None and result.tzinfo is None:
            result = result.tz_localize(tz)

        return result

    # do @functools.wraps(func) manually since it doesn't work on cdef funcs
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


cdef _wrap_timedelta_result(result):
    """
    Tick operations dispatch to their Timedelta counterparts.  Wrap the result
    of these operations in a Tick if possible.

    Parameters
    ----------
    result : object

    Returns
    -------
    object
    """
    if PyDelta_Check(result):
        # convert Timedelta back to a Tick
        return delta_to_tick(result)

    return result

# ---------------------------------------------------------------------
# Business Helpers


cdef _get_calendar(weekmask, holidays, calendar):
    """
    Generate busdaycalendar
    """
    if isinstance(calendar, np.busdaycalendar):
        if not holidays:
            holidays = tuple(calendar.holidays)
        elif not isinstance(holidays, tuple):
            holidays = tuple(holidays)
        else:
            # trust that calendar.holidays and holidays are
            # consistent
            pass
        return calendar, holidays

    if holidays is None:
        holidays = []
    try:
        holidays = holidays + calendar.holidays().tolist()
    except AttributeError:
        pass
    holidays = [_to_dt64D(dt) for dt in holidays]
    holidays = tuple(sorted(holidays))

    kwargs = {'weekmask': weekmask}
    if holidays:
        kwargs['holidays'] = holidays

    busdaycalendar = np.busdaycalendar(**kwargs)
    return busdaycalendar, holidays


cdef _to_dt64D(dt):
    # Currently
    # > np.datetime64(dt.datetime(2013,5,1),dtype='datetime64[D]')
    # numpy.datetime64('2013-05-01T02:00:00.000000+0200')
    # Thus astype is needed to cast datetime to datetime64[D]
    if getattr(dt, 'tzinfo', None) is not None:
        # Get the nanosecond timestamp,
        #  equiv `Timestamp(dt).value` or `dt.timestamp() * 10**9`
        nanos = getattr(dt, "nanosecond", 0)
        i8 = convert_datetime_to_tsobject(dt, tz=None, nanos=nanos).value
        dt = tz_convert_from_utc_single(i8, dt.tzinfo)
        dt = np.int64(dt).astype('datetime64[ns]')
    else:
        dt = np.datetime64(dt)
    if dt.dtype.name != "datetime64[D]":
        dt = dt.astype("datetime64[D]")
    return dt


# ---------------------------------------------------------------------
# Validation


cdef _validate_business_time(t_input):
    if isinstance(t_input, str):
        try:
            t = time.strptime(t_input, '%H:%M')
            return dt_time(hour=t.tm_hour, minute=t.tm_min)
        except ValueError:
            raise ValueError("time data must match '%H:%M' format")
    elif isinstance(t_input, dt_time):
        if t_input.second != 0 or t_input.microsecond != 0:
            raise ValueError(
                "time data must be specified only with hour and minute")
        return t_input
    else:
        raise ValueError("time data must be string or datetime.time")


# ---------------------------------------------------------------------
# Constructor Helpers

_relativedelta_kwds = {"years", "months", "weeks", "days", "year", "month",
                       "day", "weekday", "hour", "minute", "second",
                       "microsecond", "nanosecond", "nanoseconds", "hours",
                       "minutes", "seconds", "microseconds"}


cdef _determine_offset(kwds):
    # timedelta is used for sub-daily plural offsets and all singular
    # offsets relativedelta is used for plural offsets of daily length or
    # more nanosecond(s) are handled by apply_wraps
    kwds_no_nanos = dict(
        (k, v) for k, v in kwds.items()
        if k not in ('nanosecond', 'nanoseconds')
    )
    # TODO: Are nanosecond and nanoseconds allowed somewhere?

    _kwds_use_relativedelta = ('years', 'months', 'weeks', 'days',
                               'year', 'month', 'week', 'day', 'weekday',
                               'hour', 'minute', 'second', 'microsecond')

    use_relativedelta = False
    if len(kwds_no_nanos) > 0:
        if any(k in _kwds_use_relativedelta for k in kwds_no_nanos):
            offset = relativedelta(**kwds_no_nanos)
            use_relativedelta = True
        else:
            # sub-daily offset - use timedelta (tz-aware)
            offset = timedelta(**kwds_no_nanos)
    else:
        offset = timedelta(1)
    return offset, use_relativedelta


# ---------------------------------------------------------------------
# Mixins & Singletons


class ApplyTypeError(TypeError):
    # sentinel class for catching the apply error to return NotImplemented
    pass


# ---------------------------------------------------------------------
# Base Classes

cdef class BaseOffset:
    """
    Base class for DateOffset methods that are not overridden by subclasses.
    """
    _day_opt = None
    _attributes = tuple(["n", "normalize"])
    _use_relativedelta = False
    _adjust_dst = True
    _deprecations = frozenset(["isAnchored", "onOffset"])

    # cdef readonly:
    #    int64_t n
    #    bint normalize
    #    dict _cache

    def __init__(self, n=1, normalize=False):
        n = self._validate_n(n)
        self.n = n
        self.normalize = normalize
        self._cache = {}

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            try:
                # GH#23524 if to_offset fails, we are dealing with an
                #  incomparable type so == is False and != is True
                other = to_offset(other)
            except ValueError:
                # e.g. "infer"
                return False
        try:
            return self._params == other._params
        except AttributeError:
            # other is not a DateOffset object
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self) -> int:
        return hash(self._params)

    @cache_readonly
    def _params(self):
        """
        Returns a tuple containing all of the attributes needed to evaluate
        equality between two DateOffset objects.
        """
        d = getattr(self, "__dict__", {})
        all_paras = d.copy()
        all_paras["n"] = self.n
        all_paras["normalize"] = self.normalize
        for attr in self._attributes:
            if hasattr(self, attr) and attr not in d:
                # cython attributes are not in __dict__
                all_paras[attr] = getattr(self, attr)

        if 'holidays' in all_paras and not all_paras['holidays']:
            all_paras.pop('holidays')
        exclude = ['kwds', 'name', 'calendar']
        attrs = [(k, v) for k, v in all_paras.items()
                 if (k not in exclude) and (k[0] != '_')]
        attrs = sorted(set(attrs))
        params = tuple([str(type(self))] + attrs)
        return params

    @property
    def kwds(self) -> dict:
        # for backwards-compatibility
        kwds = {name: getattr(self, name, None) for name in self._attributes
                if name not in ["n", "normalize"]}
        return {name: kwds[name] for name in kwds if kwds[name] is not None}

    @property
    def base(self):
        """
        Returns a copy of the calling offset object with n=1 and all other
        attributes equal.
        """
        return type(self)(n=1, normalize=self.normalize, **self.kwds)

    def __add__(self, other):
        if not isinstance(self, BaseOffset):
            # cython semantics; this is __radd__
            return other.__add__(self)
        try:
            return self.apply(other)
        except ApplyTypeError:
            return NotImplemented

    def __sub__(self, other):
        if PyDateTime_Check(other):
            raise TypeError('Cannot subtract datetime from offset.')
        elif type(other) == type(self):
            return type(self)(self.n - other.n, normalize=self.normalize,
                              **self.kwds)
        elif not isinstance(self, BaseOffset):
            # cython semantics, this is __rsub__
            return (-other).__add__(self)
        else:  # pragma: no cover
            return NotImplemented

    def __call__(self, other):
        warnings.warn(
            "DateOffset.__call__ is deprecated and will be removed in a future "
            "version.  Use `offset + other` instead.",
            FutureWarning,
            stacklevel=1,
        )
        return self.apply(other)

    def __mul__(self, other):
        if util.is_array(other):
            return np.array([self * x for x in other])
        elif is_integer_object(other):
            return type(self)(n=other * self.n, normalize=self.normalize,
                              **self.kwds)
        elif not isinstance(self, BaseOffset):
            # cython semantics, this is __rmul__
            return other.__mul__(self)
        return NotImplemented

    def __neg__(self):
        # Note: we are deferring directly to __mul__ instead of __rmul__, as
        # that allows us to use methods that can go in a `cdef class`
        return self * -1

    def copy(self):
        # Note: we are deferring directly to __mul__ instead of __rmul__, as
        # that allows us to use methods that can go in a `cdef class`
        return self * 1

    # ------------------------------------------------------------------
    # Name and Rendering Methods

    def __repr__(self) -> str:
        # _output_name used by B(Year|Quarter)(End|Begin) to
        #  expand "B" -> "Business"
        class_name = getattr(self, "_output_name", type(self).__name__)

        if abs(self.n) != 1:
            plural = "s"
        else:
            plural = ""

        n_str = ""
        if self.n != 1:
            n_str = f"{self.n} * "

        out = f"<{n_str}{class_name}{plural}{self._repr_attrs()}>"
        return out

    def _repr_attrs(self) -> str:
        exclude = {"n", "inc", "normalize"}
        attrs = []
        for attr in sorted(self._attributes):
            # _attributes instead of __dict__ because cython attrs are not in __dict__
            if attr.startswith("_") or attr == "kwds" or not hasattr(self, attr):
                # DateOffset may not have some of these attributes
                continue
            elif attr not in exclude:
                value = getattr(self, attr)
                attrs.append(f"{attr}={value}")

        out = ""
        if attrs:
            out += ": " + ", ".join(attrs)
        return out

    @property
    def name(self) -> str:
        return self.rule_code

    @property
    def _prefix(self) -> str:
        raise NotImplementedError("Prefix not defined")

    @property
    def rule_code(self) -> str:
        return self._prefix

    @cache_readonly
    def freqstr(self) -> str:
        try:
            code = self.rule_code
        except NotImplementedError:
            return str(repr(self))

        if self.n != 1:
            fstr = f"{self.n}{code}"
        else:
            fstr = code

        try:
            if self._offset:
                fstr += self._offset_str()
        except AttributeError:
            # TODO: standardize `_offset` vs `offset` naming convention
            pass

        return fstr

    def _offset_str(self) -> str:
        return ""

    # ------------------------------------------------------------------

    @apply_index_wraps
    def apply_index(self, dtindex):
        """
        Vectorized apply of DateOffset to DatetimeIndex,
        raises NotImplementedError for offsets without a
        vectorized implementation.

        .. deprecated:: 1.1.0

           Use ``offset + dtindex`` instead.

        Parameters
        ----------
        index : DatetimeIndex

        Returns
        -------
        DatetimeIndex

        Raises
        ------
        NotImplementedError
            When the specific offset subclass does not have a vectorized
            implementation.
        """
        raise NotImplementedError(
            f"DateOffset subclass {type(self).__name__} "
            "does not have a vectorized implementation"
        )

    @apply_array_wraps
    def _apply_array(self, dtarr):
        raise NotImplementedError(
            f"DateOffset subclass {type(self).__name__} "
            "does not have a vectorized implementation"
        )

    def rollback(self, dt) -> datetime:
        """
        Roll provided date backward to next offset only if not on offset.

        Returns
        -------
        TimeStamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.
        """
        dt = Timestamp(dt)
        if not self.is_on_offset(dt):
            dt = dt - type(self)(1, normalize=self.normalize, **self.kwds)
        return dt

    def rollforward(self, dt) -> datetime:
        """
        Roll provided date forward to next offset only if not on offset.

        Returns
        -------
        TimeStamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.
        """
        dt = Timestamp(dt)
        if not self.is_on_offset(dt):
            dt = dt + type(self)(1, normalize=self.normalize, **self.kwds)
        return dt

    def _get_offset_day(self, other: datetime) -> int:
        # subclass must implement `_day_opt`; calling from the base class
        # will implicitly assume day_opt = "business_end", see get_day_of_month.
        cdef:
            npy_datetimestruct dts
        pydate_to_dtstruct(other, &dts)
        return get_day_of_month(&dts, self._day_opt)

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False

        # Default (slow) method for determining if some date is a member of the
        # date range generated by this offset. Subclasses may have this
        # re-implemented in a nicer way.
        a = dt
        b = (dt + self) - self
        return a == b

    # ------------------------------------------------------------------

    # Staticmethod so we can call from Tick.__init__, will be unnecessary
    #  once BaseOffset is a cdef class and is inherited by Tick
    @staticmethod
    def _validate_n(n) -> int:
        """
        Require that `n` be an integer.

        Parameters
        ----------
        n : int

        Returns
        -------
        nint : int

        Raises
        ------
        TypeError if `int(n)` raises
        ValueError if n != int(n)
        """
        if util.is_timedelta64_object(n):
            raise TypeError(f'`n` argument must be an integer, got {type(n)}')
        try:
            nint = int(n)
        except (ValueError, TypeError):
            raise TypeError(f'`n` argument must be an integer, got {type(n)}')
        if n != nint:
            raise ValueError(f'`n` argument must be an integer, got {n}')
        return nint

    def __setstate__(self, state):
        """
        Reconstruct an instance from a pickled state
        """
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self._cache = state.pop("_cache", {})
        # At this point we expect state to be empty

    def __getstate__(self):
        """
        Return a pickleable state
        """
        state = {}
        state["n"] = self.n
        state["normalize"] = self.normalize

        # we don't want to actually pickle the calendar object
        # as its a np.busyday; we recreate on deserialization
        state.pop("calendar", None)
        if "kwds" in state:
            state["kwds"].pop("calendar", None)

        return state

    @property
    def nanos(self):
        raise ValueError(f"{self} is a non-fixed frequency")

    def onOffset(self, dt) -> bool:
        warnings.warn(
            "onOffset is a deprecated, use is_on_offset instead",
            FutureWarning,
            stacklevel=1,
        )
        return self.is_on_offset(dt)

    def isAnchored(self) -> bool:
        warnings.warn(
            "isAnchored is a deprecated, use is_anchored instead",
            FutureWarning,
            stacklevel=1,
        )
        return self.is_anchored()

    def is_anchored(self) -> bool:
        # TODO: Does this make sense for the general case?  It would help
        # if there were a canonical docstring for what is_anchored means.
        return self.n == 1

    # ------------------------------------------------------------------

    def is_month_start(self, _Timestamp ts):
        return ts._get_start_end_field("is_month_start", self)

    def is_month_end(self, _Timestamp ts):
        return ts._get_start_end_field("is_month_end", self)

    def is_quarter_start(self, _Timestamp ts):
        return ts._get_start_end_field("is_quarter_start", self)

    def is_quarter_end(self, _Timestamp ts):
        return ts._get_start_end_field("is_quarter_end", self)

    def is_year_start(self, _Timestamp ts):
        return ts._get_start_end_field("is_year_start", self)

    def is_year_end(self, _Timestamp ts):
        return ts._get_start_end_field("is_year_end", self)


cdef class SingleConstructorOffset(BaseOffset):
    @classmethod
    def _from_name(cls, suffix=None):
        # default _from_name calls cls with no args
        if suffix:
            raise ValueError(f"Bad freq suffix {suffix}")
        return cls()

    def __reduce__(self):
        # This __reduce__ implementation is for all BaseOffset subclasses
        #  except for RelativeDeltaOffset
        # np.busdaycalendar objects do not pickle nicely, but we can reconstruct
        #  from attributes that do get pickled.
        tup = tuple(
            getattr(self, attr) if attr != "calendar" else None
            for attr in self._attributes
        )
        return type(self), tup


# ---------------------------------------------------------------------
# Tick Offsets

cdef class Tick(SingleConstructorOffset):
    # ensure that reversed-ops with numpy scalars return NotImplemented
    __array_priority__ = 1000
    _adjust_dst = False
    _prefix = "undefined"
    _attributes = tuple(["n", "normalize"])

    def __init__(self, n=1, normalize=False):
        n = self._validate_n(n)
        self.n = n
        self.normalize = False
        self._cache = {}
        if normalize:
            # GH#21427
            raise ValueError(
                "Tick offset with `normalize=True` are not allowed."
            )

    # FIXME: Without making this cpdef, we get AttributeError when calling
    #  from __mul__
    cpdef Tick _next_higher_resolution(Tick self):
        if type(self) is Day:
            return Hour(self.n * 24)
        if type(self) is Hour:
            return Minute(self.n * 60)
        if type(self) is Minute:
            return Second(self.n * 60)
        if type(self) is Second:
            return Milli(self.n * 1000)
        if type(self) is Milli:
            return Micro(self.n * 1000)
        if type(self) is Micro:
            return Nano(self.n * 1000)
        raise ValueError("Could not convert to integer offset at any resolution")

    # --------------------------------------------------------------------

    def _repr_attrs(self) -> str:
        # Since cdef classes have no __dict__, we need to override
        return ""

    @property
    def delta(self):
        return self.n * Timedelta(self._nanos_inc)

    @property
    def nanos(self) -> int64_t:
        return self.n * self._nanos_inc

    # FIXME: This should be typed as datetime, but we DatetimeLikeIndex.insert
    #  checks self.freq.is_on_offset with a Timedelta sometimes.
    def is_on_offset(self, dt) -> bool:
        return True

    def is_anchored(self) -> bool:
        return False

    # This is identical to BaseOffset.__hash__, but has to be redefined here
    # for Python 3, because we've redefined __eq__.
    def __hash__(self) -> int:
        return hash(self._params)

    # --------------------------------------------------------------------
    # Comparison and Arithmetic Methods

    def __eq__(self, other):
        if isinstance(other, str):
            try:
                # GH#23524 if to_offset fails, we are dealing with an
                #  incomparable type so == is False and != is True
                other = to_offset(other)
            except ValueError:
                # e.g. "infer"
                return False
        return self.delta == other

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        return self.delta.__le__(other)

    def __lt__(self, other):
        return self.delta.__lt__(other)

    def __ge__(self, other):
        return self.delta.__ge__(other)

    def __gt__(self, other):
        return self.delta.__gt__(other)

    def __mul__(self, other):
        if not isinstance(self, Tick):
            # cython semantics, this is __rmul__
            return other.__mul__(self)
        if is_float_object(other):
            n = other * self.n
            # If the new `n` is an integer, we can represent it using the
            #  same Tick subclass as self, otherwise we need to move up
            #  to a higher-resolution subclass
            if np.isclose(n % 1, 0):
                return type(self)(int(n))
            new_self = self._next_higher_resolution()
            return new_self * other
        return BaseOffset.__mul__(self, other)

    def __truediv__(self, other):
        if not isinstance(self, Tick):
            # cython semantics mean the args are sometimes swapped
            result = other.delta.__rtruediv__(self)
        else:
            result = self.delta.__truediv__(other)
        return _wrap_timedelta_result(result)

    def __add__(self, other):
        if not isinstance(self, Tick):
            # cython semantics; this is __radd__
            return other.__add__(self)

        if isinstance(other, Tick):
            if type(self) == type(other):
                return type(self)(self.n + other.n)
            else:
                return delta_to_tick(self.delta + other.delta)
        try:
            return self.apply(other)
        except ApplyTypeError:
            # Includes pd.Period
            return NotImplemented
        except OverflowError as err:
            raise OverflowError(
                f"the add operation between {self} and {other} will overflow"
            ) from err

    def apply(self, other):
        # Timestamp can handle tz and nano sec, thus no need to use apply_wraps
        if isinstance(other, _Timestamp):
            # GH#15126
            return other + self.delta
        elif other is NaT:
            return NaT
        elif is_datetime64_object(other) or PyDate_Check(other):
            # PyDate_Check includes date, datetime
            return Timestamp(other) + self

        if PyDelta_Check(other):
            return other + self.delta
        elif isinstance(other, type(self)):
            # TODO: this is reached in tests that specifically call apply,
            #  but should not be reached "naturally" because __add__ should
            #  catch this case first.
            return type(self)(self.n + other.n)

        raise ApplyTypeError(f"Unhandled type: {type(other).__name__}")

    # --------------------------------------------------------------------
    # Pickle Methods

    def __setstate__(self, state):
        self.n = state["n"]
        self.normalize = False


cdef class Day(Tick):
    _nanos_inc = 24 * 3600 * 1_000_000_000
    _prefix = "D"
    _period_dtype_code = PeriodDtypeCode.D


cdef class Hour(Tick):
    _nanos_inc = 3600 * 1_000_000_000
    _prefix = "H"
    _period_dtype_code = PeriodDtypeCode.H


cdef class Minute(Tick):
    _nanos_inc = 60 * 1_000_000_000
    _prefix = "T"
    _period_dtype_code = PeriodDtypeCode.T


cdef class Second(Tick):
    _nanos_inc = 1_000_000_000
    _prefix = "S"
    _period_dtype_code = PeriodDtypeCode.S


cdef class Milli(Tick):
    _nanos_inc = 1_000_000
    _prefix = "L"
    _period_dtype_code = PeriodDtypeCode.L


cdef class Micro(Tick):
    _nanos_inc = 1000
    _prefix = "U"
    _period_dtype_code = PeriodDtypeCode.U


cdef class Nano(Tick):
    _nanos_inc = 1
    _prefix = "N"
    _period_dtype_code = PeriodDtypeCode.N


def delta_to_tick(delta: timedelta) -> Tick:
    if delta.microseconds == 0 and getattr(delta, "nanoseconds", 0) == 0:
        # nanoseconds only for pd.Timedelta
        if delta.seconds == 0:
            return Day(delta.days)
        else:
            seconds = delta.days * 86400 + delta.seconds
            if seconds % 3600 == 0:
                return Hour(seconds / 3600)
            elif seconds % 60 == 0:
                return Minute(seconds / 60)
            else:
                return Second(seconds)
    else:
        nanos = delta_to_nanoseconds(delta)
        if nanos % 1_000_000 == 0:
            return Milli(nanos // 1_000_000)
        elif nanos % 1000 == 0:
            return Micro(nanos // 1000)
        else:  # pragma: no cover
            return Nano(nanos)


# --------------------------------------------------------------------

cdef class RelativeDeltaOffset(BaseOffset):
    """
    DateOffset subclass backed by a dateutil relativedelta object.
    """
    _attributes = tuple(["n", "normalize"] + list(_relativedelta_kwds))
    _adjust_dst = False

    def __init__(self, n=1, normalize=False, **kwds):
        BaseOffset.__init__(self, n, normalize)

        off, use_rd = _determine_offset(kwds)
        object.__setattr__(self, "_offset", off)
        object.__setattr__(self, "_use_relativedelta", use_rd)
        for key in kwds:
            val = kwds[key]
            object.__setattr__(self, key, val)

    def __getstate__(self):
        """
        Return a pickleable state
        """
        # RelativeDeltaOffset (technically DateOffset) is the only non-cdef
        #  class, so the only one with __dict__
        state = self.__dict__.copy()
        state["n"] = self.n
        state["normalize"] = self.normalize
        return state

    def __setstate__(self, state):
        """
        Reconstruct an instance from a pickled state
        """

        if "offset" in state:
            # Older (<0.22.0) versions have offset attribute instead of _offset
            if "_offset" in state:  # pragma: no cover
                raise AssertionError("Unexpected key `_offset`")
            state["_offset"] = state.pop("offset")
            state["kwds"]["offset"] = state["_offset"]

        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self._cache = state.pop("_cache", {})

        self.__dict__.update(state)

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        if self._use_relativedelta:
            other = _as_datetime(other)

        if len(self.kwds) > 0:
            tzinfo = getattr(other, "tzinfo", None)
            if tzinfo is not None and self._use_relativedelta:
                # perform calculation in UTC
                other = other.replace(tzinfo=None)

            if self.n > 0:
                for i in range(self.n):
                    other = other + self._offset
            else:
                for i in range(-self.n):
                    other = other - self._offset

            if tzinfo is not None and self._use_relativedelta:
                # bring tz back from UTC calculation
                other = localize_pydatetime(other, tzinfo)

            return Timestamp(other)
        else:
            return other + timedelta(self.n)

    @apply_index_wraps
    def apply_index(self, dtindex):
        """
        Vectorized apply of DateOffset to DatetimeIndex,
        raises NotImplementedError for offsets without a
        vectorized implementation.

        Parameters
        ----------
        index : DatetimeIndex

        Returns
        -------
        ndarray[datetime64[ns]]
        """
        return self._apply_array(dtindex)

    @apply_array_wraps
    def _apply_array(self, dtarr):
        dt64other = np.asarray(dtarr)
        kwds = self.kwds
        relativedelta_fast = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "microseconds",
        }
        # relativedelta/_offset path only valid for base DateOffset
        if self._use_relativedelta and set(kwds).issubset(relativedelta_fast):

            months = (kwds.get("years", 0) * 12 + kwds.get("months", 0)) * self.n
            if months:
                shifted = shift_months(dt64other.view("i8"), months)
                dt64other = shifted.view("datetime64[ns]")

            weeks = kwds.get("weeks", 0) * self.n
            if weeks:
                dt64other = dt64other + Timedelta(days=7 * weeks)

            timedelta_kwds = {
                k: v
                for k, v in kwds.items()
                if k in ["days", "hours", "minutes", "seconds", "microseconds"]
            }
            if timedelta_kwds:
                delta = Timedelta(**timedelta_kwds)
                dt64other = dt64other + (self.n * delta)
            return dt64other
        elif not self._use_relativedelta and hasattr(self, "_offset"):
            # timedelta
            return dt64other + Timedelta(self._offset * self.n)
        else:
            # relativedelta with other keywords
            kwd = set(kwds) - relativedelta_fast
            raise NotImplementedError(
                "DateOffset with relativedelta "
                f"keyword(s) {kwd} not able to be "
                "applied vectorized"
            )

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        # TODO: see GH#1395
        return True


class OffsetMeta(type):
    """
    Metaclass that allows us to pretend that all BaseOffset subclasses
    inherit from DateOffset (which is needed for backward-compatibility).
    """

    @classmethod
    def __instancecheck__(cls, obj) -> bool:
        return isinstance(obj, BaseOffset)

    @classmethod
    def __subclasscheck__(cls, obj) -> bool:
        return issubclass(obj, BaseOffset)


# TODO: figure out a way to use a metaclass with a cdef class
class DateOffset(RelativeDeltaOffset, metaclass=OffsetMeta):
    """
    Standard kind of date increment used for a date range.

    Works exactly like relativedelta in terms of the keyword args you
    pass in, use of the keyword n is discouraged-- you would be better
    off specifying n in the keywords you use, but regardless it is
    there for you. n is needed for DateOffset subclasses.

    DateOffset work as follows.  Each offset specify a set of dates
    that conform to the DateOffset.  For example, Bday defines this
    set to be the set of dates that are weekdays (M-F).  To test if a
    date is in the set of a DateOffset dateOffset we can use the
    is_on_offset method: dateOffset.is_on_offset(date).

    If a date is not on a valid date, the rollback and rollforward
    methods can be used to roll the date to the nearest valid date
    before/after the date.

    DateOffsets can be created to move dates forward a given number of
    valid dates.  For example, Bday(2) can be added to a date to move
    it two business days forward.  If the date does not start on a
    valid date, first it is moved to a valid date.  Thus pseudo code
    is:

    def __add__(date):
      date = rollback(date) # does nothing if date is valid
      return date + <n number of periods>

    When a date offset is created for a negative number of periods,
    the date is first rolled forward.  The pseudo code is:

    def __add__(date):
      date = rollforward(date) # does nothing is date is valid
      return date + <n number of periods>

    Zero presents a problem.  Should it roll forward or back?  We
    arbitrarily have it rollforward:

    date + BDay(0) == BDay.rollforward(date)

    Since 0 is a bit weird, we suggest avoiding its use.

    Parameters
    ----------
    n : int, default 1
        The number of time periods the offset represents.
    normalize : bool, default False
        Whether to round the result of a DateOffset addition down to the
        previous midnight.
    **kwds
        Temporal parameter that add to or replace the offset value.

        Parameters that **add** to the offset (like Timedelta):

        - years
        - months
        - weeks
        - days
        - hours
        - minutes
        - seconds
        - microseconds
        - nanoseconds

        Parameters that **replace** the offset value:

        - year
        - month
        - day
        - weekday
        - hour
        - minute
        - second
        - microsecond
        - nanosecond.

    See Also
    --------
    dateutil.relativedelta.relativedelta : The relativedelta type is designed
        to be applied to an existing datetime an can replace specific components of
        that datetime, or represents an interval of time.

    Examples
    --------
    >>> from pandas.tseries.offsets import DateOffset
    >>> ts = pd.Timestamp('2017-01-01 09:10:11')
    >>> ts + DateOffset(months=3)
    Timestamp('2017-04-01 09:10:11')

    >>> ts = pd.Timestamp('2017-01-01 09:10:11')
    >>> ts + DateOffset(months=2)
    Timestamp('2017-03-01 09:10:11')
    """
    def __setattr__(self, name, value):
        raise AttributeError("DateOffset objects are immutable.")

# --------------------------------------------------------------------


cdef class BusinessMixin(SingleConstructorOffset):
    """
    Mixin to business types to provide related functions.
    """

    cdef readonly:
        timedelta _offset
        # Only Custom subclasses use weekmask, holiday, calendar
        object weekmask, holidays, calendar

    def __init__(self, n=1, normalize=False, offset=timedelta(0)):
        BaseOffset.__init__(self, n, normalize)
        self._offset = offset

    cpdef _init_custom(self, weekmask, holidays, calendar):
        """
        Additional __init__ for Custom subclasses.
        """
        calendar, holidays = _get_calendar(
            weekmask=weekmask, holidays=holidays, calendar=calendar
        )
        # Custom offset instances are identified by the
        # following two attributes. See DateOffset._params()
        # holidays, weekmask
        self.weekmask = weekmask
        self.holidays = holidays
        self.calendar = calendar

    @property
    def offset(self):
        """
        Alias for self._offset.
        """
        # Alias for backward compat
        return self._offset

    def _repr_attrs(self) -> str:
        if self.offset:
            attrs = [f"offset={repr(self.offset)}"]
        else:
            attrs = []
        out = ""
        if attrs:
            out += ": " + ", ".join(attrs)
        return out

    cpdef __setstate__(self, state):
        # We need to use a cdef/cpdef method to set the readonly _offset attribute
        if "_offset" in state:
            self._offset = state.pop("_offset")
        elif "offset" in state:
            # Older (<0.22.0) versions have offset attribute instead of _offset
            self._offset = state.pop("offset")

        if self._prefix.startswith("C"):
            # i.e. this is a Custom class
            weekmask = state.pop("weekmask")
            holidays = state.pop("holidays")
            calendar, holidays = _get_calendar(weekmask=weekmask,
                                               holidays=holidays,
                                               calendar=None)
            self.weekmask = weekmask
            self.calendar = calendar
            self.holidays = holidays

        BaseOffset.__setstate__(self, state)


cdef class BusinessDay(BusinessMixin):
    """
    DateOffset subclass representing possibly n business days.
    """
    _period_dtype_code = PeriodDtypeCode.B
    _prefix = "B"
    _attributes = tuple(["n", "normalize", "offset"])

    cpdef __setstate__(self, state):
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        if "_offset" in state:
            self._offset = state.pop("_offset")
        elif "offset" in state:
            self._offset = state.pop("offset")
        self._cache = state.pop("_cache", {})

    def _offset_str(self) -> str:
        def get_str(td):
            off_str = ""
            if td.days > 0:
                off_str += str(td.days) + "D"
            if td.seconds > 0:
                s = td.seconds
                hrs = int(s / 3600)
                if hrs != 0:
                    off_str += str(hrs) + "H"
                    s -= hrs * 3600
                mts = int(s / 60)
                if mts != 0:
                    off_str += str(mts) + "Min"
                    s -= mts * 60
                if s != 0:
                    off_str += str(s) + "s"
            if td.microseconds > 0:
                off_str += str(td.microseconds) + "us"
            return off_str

        if PyDelta_Check(self.offset):
            zero = timedelta(0, 0, 0)
            if self.offset >= zero:
                off_str = "+" + get_str(self.offset)
            else:
                off_str = "-" + get_str(-self.offset)
            return off_str
        else:
            return "+" + repr(self.offset)

    @apply_wraps
    def apply(self, other):
        if PyDateTime_Check(other):
            n = self.n
            wday = other.weekday()

            # avoid slowness below by operating on weeks first
            weeks = n // 5
            if n <= 0 and wday > 4:
                # roll forward
                n += 1

            n -= 5 * weeks

            # n is always >= 0 at this point
            if n == 0 and wday > 4:
                # roll back
                days = 4 - wday
            elif wday > 4:
                # roll forward
                days = (7 - wday) + (n - 1)
            elif wday + n <= 4:
                # shift by n days without leaving the current week
                days = n
            else:
                # shift by n days plus 2 to get past the weekend
                days = n + 2

            result = other + timedelta(days=7 * weeks + days)
            if self.offset:
                result = result + self.offset
            return result

        elif PyDelta_Check(other) or isinstance(other, Tick):
            return BusinessDay(
                self.n, offset=self.offset + other, normalize=self.normalize
            )
        else:
            raise ApplyTypeError(
                "Only know how to combine business day with datetime or timedelta."
            )

    @apply_index_wraps
    def apply_index(self, dtindex):
        return self._apply_array(dtindex)

    @apply_array_wraps
    def _apply_array(self, dtarr):
        i8other = dtarr.view("i8")
        res = _shift_bdays(i8other, self.n)
        if self.offset:
            res = res.view("M8[ns]") + Timedelta(self.offset)
            res = res.view("i8")
        return res

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.weekday() < 5


cdef class BusinessHour(BusinessMixin):
    """
    DateOffset subclass representing possibly n business hours.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    start : str, default "09:00"
        Start time of your custom business hour in 24h format.
    end : str, default: "17:00"
        End time of your custom business hour in 24h format.
    """

    _prefix = "BH"
    _anchor = 0
    _attributes = tuple(["n", "normalize", "start", "end", "offset"])
    _adjust_dst = False

    cdef readonly:
        tuple start, end

    def __init__(
            self, n=1, normalize=False, start="09:00", end="17:00", offset=timedelta(0)
        ):
        BusinessMixin.__init__(self, n, normalize, offset)

        # must be validated here to equality check
        if np.ndim(start) == 0:
            # i.e. not is_list_like
            start = [start]
        if not len(start):
            raise ValueError("Must include at least 1 start time")

        if np.ndim(end) == 0:
            # i.e. not is_list_like
            end = [end]
        if not len(end):
            raise ValueError("Must include at least 1 end time")

        start = np.array([_validate_business_time(x) for x in start])
        end = np.array([_validate_business_time(x) for x in end])

        # Validation of input
        if len(start) != len(end):
            raise ValueError("number of starting time and ending time must be the same")
        num_openings = len(start)

        # sort starting and ending time by starting time
        index = np.argsort(start)

        # convert to tuple so that start and end are hashable
        start = tuple(start[index])
        end = tuple(end[index])

        total_secs = 0
        for i in range(num_openings):
            total_secs += self._get_business_hours_by_sec(start[i], end[i])
            total_secs += self._get_business_hours_by_sec(
                end[i], start[(i + 1) % num_openings]
            )
        if total_secs != 24 * 60 * 60:
            raise ValueError(
                "invalid starting and ending time(s): "
                "opening hours should not touch or overlap with "
                "one another"
            )

        self.start = start
        self.end = end

    cpdef __setstate__(self, state):
        start = state.pop("start")
        start = (start,) if np.ndim(start) == 0 else tuple(start)
        end = state.pop("end")
        end = (end,) if np.ndim(end) == 0 else tuple(end)
        self.start = start
        self.end = end

        state.pop("kwds", {})
        state.pop("next_bday", None)
        BusinessMixin.__setstate__(self, state)

    def _repr_attrs(self) -> str:
        out = super()._repr_attrs()
        hours = ",".join(
            f'{st.strftime("%H:%M")}-{en.strftime("%H:%M")}'
            for st, en in zip(self.start, self.end)
        )
        attrs = [f"{self._prefix}={hours}"]
        out += ": " + ", ".join(attrs)
        return out

    def _get_business_hours_by_sec(self, start, end):
        """
        Return business hours in a day by seconds.
        """
        # create dummy datetime to calculate business hours in a day
        dtstart = datetime(2014, 4, 1, start.hour, start.minute)
        day = 1 if start < end else 2
        until = datetime(2014, 4, day, end.hour, end.minute)
        return int((until - dtstart).total_seconds())

    def _get_closing_time(self, dt: datetime) -> datetime:
        """
        Get the closing time of a business hour interval by its opening time.

        Parameters
        ----------
        dt : datetime
            Opening time of a business hour interval.

        Returns
        -------
        result : datetime
            Corresponding closing time.
        """
        for i, st in enumerate(self.start):
            if st.hour == dt.hour and st.minute == dt.minute:
                return dt + timedelta(
                    seconds=self._get_business_hours_by_sec(st, self.end[i])
                )
        assert False

    @cache_readonly
    def next_bday(self):
        """
        Used for moving to next business day.
        """
        if self.n >= 0:
            nb_offset = 1
        else:
            nb_offset = -1
        if self._prefix.startswith("C"):
            # CustomBusinessHour
            return CustomBusinessDay(
                n=nb_offset,
                weekmask=self.weekmask,
                holidays=self.holidays,
                calendar=self.calendar,
            )
        else:
            return BusinessDay(n=nb_offset)

    def _next_opening_time(self, other, sign=1):
        """
        If self.n and sign have the same sign, return the earliest opening time
        later than or equal to current time.
        Otherwise the latest opening time earlier than or equal to current
        time.

        Opening time always locates on BusinessDay.
        However, closing time may not if business hour extends over midnight.

        Parameters
        ----------
        other : datetime
            Current time.
        sign : int, default 1.
            Either 1 or -1. Going forward in time if it has the same sign as
            self.n. Going backward in time otherwise.

        Returns
        -------
        result : datetime
            Next opening time.
        """
        earliest_start = self.start[0]
        latest_start = self.start[-1]

        if not self.next_bday.is_on_offset(other):
            # today is not business day
            other = other + sign * self.next_bday
            if self.n * sign >= 0:
                hour, minute = earliest_start.hour, earliest_start.minute
            else:
                hour, minute = latest_start.hour, latest_start.minute
        else:
            if self.n * sign >= 0:
                if latest_start < other.time():
                    # current time is after latest starting time in today
                    other = other + sign * self.next_bday
                    hour, minute = earliest_start.hour, earliest_start.minute
                else:
                    # find earliest starting time no earlier than current time
                    for st in self.start:
                        if other.time() <= st:
                            hour, minute = st.hour, st.minute
                            break
            else:
                if other.time() < earliest_start:
                    # current time is before earliest starting time in today
                    other = other + sign * self.next_bday
                    hour, minute = latest_start.hour, latest_start.minute
                else:
                    # find latest starting time no later than current time
                    for st in reversed(self.start):
                        if other.time() >= st:
                            hour, minute = st.hour, st.minute
                            break

        return datetime(other.year, other.month, other.day, hour, minute)

    def _prev_opening_time(self, other: datetime) -> datetime:
        """
        If n is positive, return the latest opening time earlier than or equal
        to current time.
        Otherwise the earliest opening time later than or equal to current
        time.

        Parameters
        ----------
        other : datetime
            Current time.

        Returns
        -------
        result : datetime
            Previous opening time.
        """
        return self._next_opening_time(other, sign=-1)

    @apply_wraps
    def rollback(self, dt: datetime) -> datetime:
        """
        Roll provided date backward to next offset only if not on offset.
        """
        if not self.is_on_offset(dt):
            if self.n >= 0:
                dt = self._prev_opening_time(dt)
            else:
                dt = self._next_opening_time(dt)
            return self._get_closing_time(dt)
        return dt

    @apply_wraps
    def rollforward(self, dt: datetime) -> datetime:
        """
        Roll provided date forward to next offset only if not on offset.
        """
        if not self.is_on_offset(dt):
            if self.n >= 0:
                return self._next_opening_time(dt)
            else:
                return self._prev_opening_time(dt)
        return dt

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        # used for detecting edge condition
        nanosecond = getattr(other, "nanosecond", 0)
        # reset timezone and nanosecond
        # other may be a Timestamp, thus not use replace
        other = datetime(
            other.year,
            other.month,
            other.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        n = self.n

        # adjust other to reduce number of cases to handle
        if n >= 0:
            if other.time() in self.end or not self._is_on_offset(other):
                other = self._next_opening_time(other)
        else:
            if other.time() in self.start:
                # adjustment to move to previous business day
                other = other - timedelta(seconds=1)
            if not self._is_on_offset(other):
                other = self._next_opening_time(other)
                other = self._get_closing_time(other)

        # get total business hours by sec in one business day
        businesshours = sum(
            self._get_business_hours_by_sec(st, en)
            for st, en in zip(self.start, self.end)
        )

        bd, r = divmod(abs(n * 60), businesshours // 60)
        if n < 0:
            bd, r = -bd, -r

        # adjust by business days first
        if bd != 0:
            if self._prefix.startswith("C"):
                # GH#30593 this is a Custom offset
                skip_bd = CustomBusinessDay(
                    n=bd,
                    weekmask=self.weekmask,
                    holidays=self.holidays,
                    calendar=self.calendar,
                )
            else:
                skip_bd = BusinessDay(n=bd)
            # midnight business hour may not on BusinessDay
            if not self.next_bday.is_on_offset(other):
                prev_open = self._prev_opening_time(other)
                remain = other - prev_open
                other = prev_open + skip_bd + remain
            else:
                other = other + skip_bd

        # remaining business hours to adjust
        bhour_remain = timedelta(minutes=r)

        if n >= 0:
            while bhour_remain != timedelta(0):
                # business hour left in this business time interval
                bhour = (
                    self._get_closing_time(self._prev_opening_time(other)) - other
                )
                if bhour_remain < bhour:
                    # finish adjusting if possible
                    other += bhour_remain
                    bhour_remain = timedelta(0)
                else:
                    # go to next business time interval
                    bhour_remain -= bhour
                    other = self._next_opening_time(other + bhour)
        else:
            while bhour_remain != timedelta(0):
                # business hour left in this business time interval
                bhour = self._next_opening_time(other) - other
                if (
                    bhour_remain > bhour
                    or bhour_remain == bhour
                    and nanosecond != 0
                ):
                    # finish adjusting if possible
                    other += bhour_remain
                    bhour_remain = timedelta(0)
                else:
                    # go to next business time interval
                    bhour_remain -= bhour
                    other = self._get_closing_time(
                        self._next_opening_time(
                            other + bhour - timedelta(seconds=1)
                        )
                    )

        return other

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False

        if dt.tzinfo is not None:
            dt = datetime(
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond
            )
        # Valid BH can be on the different BusinessDay during midnight
        # Distinguish by the time spent from previous opening time
        return self._is_on_offset(dt)

    def _is_on_offset(self, dt: datetime) -> bool:
        """
        Slight speedups using calculated values.
        """
        # if self.normalize and not _is_normalized(dt):
        #     return False
        # Valid BH can be on the different BusinessDay during midnight
        # Distinguish by the time spent from previous opening time
        if self.n >= 0:
            op = self._prev_opening_time(dt)
        else:
            op = self._next_opening_time(dt)
        span = (dt - op).total_seconds()
        businesshours = 0
        for i, st in enumerate(self.start):
            if op.hour == st.hour and op.minute == st.minute:
                businesshours = self._get_business_hours_by_sec(st, self.end[i])
        if span <= businesshours:
            return True
        else:
            return False


cdef class WeekOfMonthMixin(SingleConstructorOffset):
    """
    Mixin for methods common to WeekOfMonth and LastWeekOfMonth.
    """

    cdef readonly:
        int weekday, week

    def __init__(self, n=1, normalize=False, weekday=0):
        BaseOffset.__init__(self, n, normalize)
        self.weekday = weekday

        if weekday < 0 or weekday > 6:
            raise ValueError(f"Day must be 0<=day<=6, got {weekday}")

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        compare_day = self._get_offset_day(other)

        months = self.n
        months = roll_convention(other.day, months, compare_day)

        shifted = shift_month(other, months, "start")
        to_day = self._get_offset_day(shifted)
        return shift_day(shifted, to_day - shifted.day)

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.day == self._get_offset_day(dt)

    @property
    def rule_code(self) -> str:
        weekday = int_to_weekday.get(self.weekday, "")
        if self.week == -1:
            # LastWeekOfMonth
            return f"{self._prefix}-{weekday}"
        return f"{self._prefix}-{self.week + 1}{weekday}"


# ----------------------------------------------------------------------
# Year-Based Offset Classes

cdef class YearOffset(SingleConstructorOffset):
    """
    DateOffset that just needs a month.
    """
    _attributes = tuple(["n", "normalize", "month"])

    # _default_month: int  # FIXME: python annotation here breaks things

    cdef readonly:
        int month

    def __init__(self, n=1, normalize=False, month=None):
        BaseOffset.__init__(self, n, normalize)

        month = month if month is not None else self._default_month
        self.month = month

        if month < 1 or month > 12:
            raise ValueError("Month must go from 1 to 12")

    cpdef __setstate__(self, state):
        self.month = state.pop("month")
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self._cache = {}

    @classmethod
    def _from_name(cls, suffix=None):
        kwargs = {}
        if suffix:
            kwargs["month"] = MONTH_TO_CAL_NUM[suffix]
        return cls(**kwargs)

    @property
    def rule_code(self) -> str:
        month = MONTH_ALIASES[self.month]
        return f"{self._prefix}-{month}"

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.month == self.month and dt.day == self._get_offset_day(dt)

    def _get_offset_day(self, other: datetime) -> int:
        # override BaseOffset method to use self.month instead of other.month
        cdef:
            npy_datetimestruct dts
        pydate_to_dtstruct(other, &dts)
        dts.month = self.month
        return get_day_of_month(&dts, self._day_opt)

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        years = roll_qtrday(other, self.n, self.month, self._day_opt, modby=12)
        months = years * 12 + (self.month - other.month)
        return shift_month(other, months, self._day_opt)

    @apply_index_wraps
    def apply_index(self, dtindex):
        return self._apply_array(dtindex)

    @apply_array_wraps
    def _apply_array(self, dtarr):
        shifted = shift_quarters(
            dtarr.view("i8"), self.n, self.month, self._day_opt, modby=12
        )
        return shifted


cdef class BYearEnd(YearOffset):
    """
    DateOffset increments between the last business day of the year.

    Examples
    --------
    >>> from pandas.tseries.offset import BYearEnd
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts - BYearEnd()
    Timestamp('2019-12-31 05:01:15')
    >>> ts + BYearEnd()
    Timestamp('2020-12-31 05:01:15')
    >>> ts + BYearEnd(3)
    Timestamp('2022-12-30 05:01:15')
    >>> ts + BYearEnd(-3)
    Timestamp('2017-12-29 05:01:15')
    >>> ts + BYearEnd(month=11)
    Timestamp('2020-11-30 05:01:15')
    """

    _outputName = "BusinessYearEnd"
    _default_month = 12
    _prefix = "BA"
    _day_opt = "business_end"


cdef class BYearBegin(YearOffset):
    """
    DateOffset increments between the first business day of the year.

    Examples
    --------
    >>> from pandas.tseries.offset import BYearBegin
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BYearBegin()
    Timestamp('2021-01-01 05:01:15')
    >>> ts - BYearBegin()
    Timestamp('2020-01-01 05:01:15')
    >>> ts + BYearBegin(-1)
    Timestamp('2020-01-01 05:01:15')
    >>> ts + BYearBegin(2)
    Timestamp('2022-01-03 05:01:15')
    """

    _outputName = "BusinessYearBegin"
    _default_month = 1
    _prefix = "BAS"
    _day_opt = "business_start"


cdef class YearEnd(YearOffset):
    """
    DateOffset increments between calendar year ends.
    """

    _default_month = 12
    _prefix = "A"
    _day_opt = "end"

    cdef readonly:
        int _period_dtype_code

    def __init__(self, n=1, normalize=False, month=None):
        # Because YearEnd can be the freq for a Period, define its
        #  _period_dtype_code at construction for performance
        YearOffset.__init__(self, n, normalize, month)
        self._period_dtype_code = PeriodDtypeCode.A + self.month % 12


cdef class YearBegin(YearOffset):
    """
    DateOffset increments between calendar year begin dates.
    """

    _default_month = 1
    _prefix = "AS"
    _day_opt = "start"


# ----------------------------------------------------------------------
# Quarter-Based Offset Classes

cdef class QuarterOffset(SingleConstructorOffset):
    _attributes = tuple(["n", "normalize", "startingMonth"])
    # TODO: Consider combining QuarterOffset and YearOffset __init__ at some
    #       point.  Also apply_index, is_on_offset, rule_code if
    #       startingMonth vs month attr names are resolved

    # FIXME: python annotations here breaks things
    # _default_starting_month: int
    # _from_name_starting_month: int

    cdef readonly:
        int startingMonth

    def __init__(self, n=1, normalize=False, startingMonth=None):
        BaseOffset.__init__(self, n, normalize)

        if startingMonth is None:
            startingMonth = self._default_starting_month
        self.startingMonth = startingMonth

    cpdef __setstate__(self, state):
        self.startingMonth = state.pop("startingMonth")
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")

    @classmethod
    def _from_name(cls, suffix=None):
        kwargs = {}
        if suffix:
            kwargs["startingMonth"] = MONTH_TO_CAL_NUM[suffix]
        else:
            if cls._from_name_starting_month is not None:
                kwargs["startingMonth"] = cls._from_name_starting_month
        return cls(**kwargs)

    @property
    def rule_code(self) -> str:
        month = MONTH_ALIASES[self.startingMonth]
        return f"{self._prefix}-{month}"

    def is_anchored(self) -> bool:
        return self.n == 1 and self.startingMonth is not None

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        mod_month = (dt.month - self.startingMonth) % 3
        return mod_month == 0 and dt.day == self._get_offset_day(dt)

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        # months_since: find the calendar quarter containing other.month,
        # e.g. if other.month == 8, the calendar quarter is [Jul, Aug, Sep].
        # Then find the month in that quarter containing an is_on_offset date for
        # self.  `months_since` is the number of months to shift other.month
        # to get to this on-offset month.
        months_since = other.month % 3 - self.startingMonth % 3
        qtrs = roll_qtrday(
            other, self.n, self.startingMonth, day_opt=self._day_opt, modby=3
        )
        months = qtrs * 3 - months_since
        return shift_month(other, months, self._day_opt)

    @apply_index_wraps
    def apply_index(self, dtindex):
        return self._apply_array(dtindex)

    @apply_array_wraps
    def _apply_array(self, dtarr):
        shifted = shift_quarters(
            dtarr.view("i8"), self.n, self.startingMonth, self._day_opt
        )
        return shifted


cdef class BQuarterEnd(QuarterOffset):
    """
    DateOffset increments between the last business day of each Quarter.

    startingMonth = 1 corresponds to dates like 1/31/2007, 4/30/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 5/31/2007, ...
    startingMonth = 3 corresponds to dates like 3/30/2007, 6/29/2007, ...

    Examples
    --------
    >>> from pandas.tseries.offset import BQuarterEnd
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BQuarterEnd()
    Timestamp('2020-06-30 05:01:15')
    >>> ts + BQuarterEnd(2)
    Timestamp('2020-09-30 05:01:15')
    >>> ts + BQuarterEnd(1, startingMonth=2)
    Timestamp('2020-05-29 05:01:15')
    >>> ts + BQuarterEnd(startingMonth=2)
    Timestamp('2020-05-29 05:01:15')
    """
    _output_name = "BusinessQuarterEnd"
    _default_starting_month = 3
    _from_name_starting_month = 12
    _prefix = "BQ"
    _day_opt = "business_end"


cdef class BQuarterBegin(QuarterOffset):
    """
    DateOffset increments between the first business day of each Quarter.

    startingMonth = 1 corresponds to dates like 1/01/2007, 4/01/2007, ...
    startingMonth = 2 corresponds to dates like 2/01/2007, 5/01/2007, ...
    startingMonth = 3 corresponds to dates like 3/01/2007, 6/01/2007, ...

    Examples
    --------
    >>> from pandas.tseries.offset import BQuarterBegin
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BQuarterBegin()
    Timestamp('2020-06-01 05:01:15')
    >>> ts + BQuarterBegin(2)
    Timestamp('2020-09-01 05:01:15')
    >>> ts + BQuarterBegin(startingMonth=2)
    Timestamp('2020-08-03 05:01:15')
    >>> ts + BQuarterBegin(-1)
    Timestamp('2020-03-02 05:01:15')
    """
    _output_name = "BusinessQuarterBegin"
    _default_starting_month = 3
    _from_name_starting_month = 1
    _prefix = "BQS"
    _day_opt = "business_start"


cdef class QuarterEnd(QuarterOffset):
    """
    DateOffset increments between Quarter end dates.

    startingMonth = 1 corresponds to dates like 1/31/2007, 4/30/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 5/31/2007, ...
    startingMonth = 3 corresponds to dates like 3/31/2007, 6/30/2007, ...
    """
    _default_starting_month = 3
    _prefix = "Q"
    _day_opt = "end"

    cdef readonly:
        int _period_dtype_code

    def __init__(self, n=1, normalize=False, startingMonth=None):
        # Because QuarterEnd can be the freq for a Period, define its
        #  _period_dtype_code at construction for performance
        QuarterOffset.__init__(self, n, normalize, startingMonth)
        self._period_dtype_code = PeriodDtypeCode.Q_DEC + self.startingMonth % 12


cdef class QuarterBegin(QuarterOffset):
    """
    DateOffset increments between Quarter start dates.

    startingMonth = 1 corresponds to dates like 1/01/2007, 4/01/2007, ...
    startingMonth = 2 corresponds to dates like 2/01/2007, 5/01/2007, ...
    startingMonth = 3 corresponds to dates like 3/01/2007, 6/01/2007, ...
    """
    _default_starting_month = 3
    _from_name_starting_month = 1
    _prefix = "QS"
    _day_opt = "start"


# ----------------------------------------------------------------------
# Month-Based Offset Classes

cdef class MonthOffset(SingleConstructorOffset):
    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.day == self._get_offset_day(dt)

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        compare_day = self._get_offset_day(other)
        n = roll_convention(other.day, self.n, compare_day)
        return shift_month(other, n, self._day_opt)

    @apply_index_wraps
    def apply_index(self, dtindex):
        return self._apply_array(dtindex)

    @apply_array_wraps
    def _apply_array(self, dtarr):
        shifted = shift_months(dtarr.view("i8"), self.n, self._day_opt)
        return shifted

    cpdef __setstate__(self, state):
        state.pop("_use_relativedelta", False)
        state.pop("offset", None)
        state.pop("_offset", None)
        state.pop("kwds", {})

        BaseOffset.__setstate__(self, state)


cdef class MonthEnd(MonthOffset):
    """
    DateOffset of one month end.
    """
    _period_dtype_code = PeriodDtypeCode.M
    _prefix = "M"
    _day_opt = "end"


cdef class MonthBegin(MonthOffset):
    """
    DateOffset of one month at beginning.
    """
    _prefix = "MS"
    _day_opt = "start"


cdef class BusinessMonthEnd(MonthOffset):
    """
    DateOffset increments between the last business day of the month

    Examples
    --------
    >>> from pandas.tseries.offset import BMonthEnd
    >>> ts = pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BMonthEnd()
    Timestamp('2020-05-29 05:01:15')
    >>> ts + BMonthEnd(2)
    Timestamp('2020-06-30 05:01:15')
    >>> ts + BMonthEnd(-2)
    Timestamp('2020-03-31 05:01:15')
    """
    _prefix = "BM"
    _day_opt = "business_end"


cdef class BusinessMonthBegin(MonthOffset):
    """
    DateOffset of one month at the first business day.

    Examples
    --------
    >>> from pandas.tseries.offset import BMonthBegin
    >>> ts=pd.Timestamp('2020-05-24 05:01:15')
    >>> ts + BMonthBegin()
    Timestamp('2020-06-01 05:01:15')
    >>> ts + BMonthBegin(2)
    Timestamp('2020-07-01 05:01:15')
    >>> ts + BMonthBegin(-3)
    Timestamp('2020-03-02 05:01:15')
    """
    _prefix = "BMS"
    _day_opt = "business_start"


# ---------------------------------------------------------------------
# Semi-Month Based Offsets

cdef class SemiMonthOffset(SingleConstructorOffset):
    _default_day_of_month = 15
    _min_day_of_month = 2
    _attributes = tuple(["n", "normalize", "day_of_month"])

    cdef readonly:
        int day_of_month

    def __init__(self, n=1, normalize=False, day_of_month=None):
        BaseOffset.__init__(self, n, normalize)

        if day_of_month is None:
            day_of_month = self._default_day_of_month

        self.day_of_month = int(day_of_month)
        if not self._min_day_of_month <= self.day_of_month <= 27:
            raise ValueError(
                "day_of_month must be "
                f"{self._min_day_of_month}<=day_of_month<=27, "
                f"got {self.day_of_month}"
            )

    cpdef __setstate__(self, state):
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.day_of_month = state.pop("day_of_month")

    @classmethod
    def _from_name(cls, suffix=None):
        return cls(day_of_month=suffix)

    @property
    def rule_code(self) -> str:
        suffix = f"-{self.day_of_month}"
        return self._prefix + suffix

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        is_start = isinstance(self, SemiMonthBegin)

        # shift `other` to self.day_of_month, incrementing `n` if necessary
        n = roll_convention(other.day, self.n, self.day_of_month)

        days_in_month = get_days_in_month(other.year, other.month)
        # For SemiMonthBegin on other.day == 1 and
        # SemiMonthEnd on other.day == days_in_month,
        # shifting `other` to `self.day_of_month` _always_ requires
        # incrementing/decrementing `n`, regardless of whether it is
        # initially positive.
        if is_start and (self.n <= 0 and other.day == 1):
            n -= 1
        elif (not is_start) and (self.n > 0 and other.day == days_in_month):
            n += 1

        if is_start:
            months = n // 2 + n % 2
            to_day = 1 if n % 2 else self.day_of_month
        else:
            months = n // 2
            to_day = 31 if n % 2 else self.day_of_month

        return shift_month(other, months, to_day)

    @apply_index_wraps
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def apply_index(self, dtindex):
        return self._apply_array(dtindex)

    @apply_array_wraps
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def _apply_array(self, dtarr):
        cdef:
            int64_t[:] i8other = dtarr.view("i8")
            Py_ssize_t i, count = len(i8other)
            int64_t val
            int64_t[:] out = np.empty(count, dtype="i8")
            npy_datetimestruct dts
            int months, to_day, nadj, n = self.n
            int days_in_month, day, anchor_dom = self.day_of_month
            bint is_start = isinstance(self, SemiMonthBegin)

        with nogil:
            for i in range(count):
                val = i8other[i]
                if val == NPY_NAT:
                    out[i] = NPY_NAT
                    continue

                dt64_to_dtstruct(val, &dts)
                day = dts.day

                # Adjust so that we are always looking at self.day_of_month,
                #  incrementing/decrementing n if necessary.
                nadj = roll_convention(day, n, anchor_dom)

                days_in_month = get_days_in_month(dts.year, dts.month)
                # For SemiMonthBegin on other.day == 1 and
                #  SemiMonthEnd on other.day == days_in_month,
                #  shifting `other` to `self.day_of_month` _always_ requires
                #  incrementing/decrementing `n`, regardless of whether it is
                #  initially positive.
                if is_start and (n <= 0 and day == 1):
                    nadj -= 1
                elif (not is_start) and (n > 0 and day == days_in_month):
                    nadj += 1

                if is_start:
                    # See also: SemiMonthBegin._apply
                    months = nadj // 2 + nadj % 2
                    to_day = 1 if nadj % 2 else anchor_dom

                else:
                    # See also: SemiMonthEnd._apply
                    months = nadj // 2
                    to_day = 31 if nadj % 2 else anchor_dom

                dts.year = year_add_months(dts, months)
                dts.month = month_add_months(dts, months)
                days_in_month = get_days_in_month(dts.year, dts.month)
                dts.day = min(to_day, days_in_month)

                out[i] = dtstruct_to_dt64(&dts)

        return out.base


cdef class SemiMonthEnd(SemiMonthOffset):
    """
    Two DateOffset's per month repeating on the last
    day of the month and day_of_month.

    Parameters
    ----------
    n : int
    normalize : bool, default False
    day_of_month : int, {1, 3,...,27}, default 15
    """

    _prefix = "SM"
    _min_day_of_month = 1

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        days_in_month = get_days_in_month(dt.year, dt.month)
        return dt.day in (self.day_of_month, days_in_month)


cdef class SemiMonthBegin(SemiMonthOffset):
    """
    Two DateOffset's per month repeating on the first
    day of the month and day_of_month.

    Parameters
    ----------
    n : int
    normalize : bool, default False
    day_of_month : int, {2, 3,...,27}, default 15
    """

    _prefix = "SMS"

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.day in (1, self.day_of_month)


# ---------------------------------------------------------------------
# Week-Based Offset Classes


cdef class Week(SingleConstructorOffset):
    """
    Weekly offset.

    Parameters
    ----------
    weekday : int or None, default None
        Always generate specific day of week. 0 for Monday.
    """

    _inc = timedelta(weeks=1)
    _prefix = "W"
    _attributes = tuple(["n", "normalize", "weekday"])

    cdef readonly:
        object weekday  # int or None
        int _period_dtype_code

    def __init__(self, n=1, normalize=False, weekday=None):
        BaseOffset.__init__(self, n, normalize)
        self.weekday = weekday

        if self.weekday is not None:
            if self.weekday < 0 or self.weekday > 6:
                raise ValueError(f"Day must be 0<=day<=6, got {self.weekday}")

            self._period_dtype_code = PeriodDtypeCode.W_SUN + (weekday + 1) % 7

    cpdef __setstate__(self, state):
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        self._cache = state.pop("_cache", {})

    def is_anchored(self) -> bool:
        return self.n == 1 and self.weekday is not None

    @apply_wraps
    def apply(self, other):
        if self.weekday is None:
            return other + self.n * self._inc

        if not PyDateTime_Check(other):
            raise TypeError(
                f"Cannot add {type(other).__name__} to {type(self).__name__}"
            )

        k = self.n
        otherDay = other.weekday()
        if otherDay != self.weekday:
            other = other + timedelta((self.weekday - otherDay) % 7)
            if k > 0:
                k -= 1

        return other + timedelta(weeks=k)

    @apply_index_wraps
    def apply_index(self, dtindex):
        return self._apply_array(dtindex)

    @apply_array_wraps
    def _apply_array(self, dtarr):
        if self.weekday is None:
            td = timedelta(days=7 * self.n)
            td64 = np.timedelta64(td, "ns")
            return dtarr + td64
        else:
            i8other = dtarr.view("i8")
            return self._end_apply_index(i8other)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef _end_apply_index(self, const int64_t[:] i8other):
        """
        Add self to the given DatetimeIndex, specialized for case where
        self.weekday is non-null.

        Parameters
        ----------
        i8other : const int64_t[:]

        Returns
        -------
        ndarray[int64_t]
        """
        cdef:
            Py_ssize_t i, count = len(i8other)
            int64_t val
            int64_t[:] out = np.empty(count, dtype="i8")
            npy_datetimestruct dts
            int wday, days, weeks, n = self.n
            int anchor_weekday = self.weekday

        with nogil:
            for i in range(count):
                val = i8other[i]
                if val == NPY_NAT:
                    out[i] = NPY_NAT
                    continue

                dt64_to_dtstruct(val, &dts)
                wday = dayofweek(dts.year, dts.month, dts.day)

                days = 0
                weeks = n
                if wday != anchor_weekday:
                    days = (anchor_weekday - wday) % 7
                    if weeks > 0:
                        weeks -= 1

                out[i] = val + (7 * weeks + days) * DAY_NANOS

        return out.base

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        elif self.weekday is None:
            return True
        return dt.weekday() == self.weekday

    @property
    def rule_code(self) -> str:
        suffix = ""
        if self.weekday is not None:
            weekday = int_to_weekday[self.weekday]
            suffix = f"-{weekday}"
        return self._prefix + suffix

    @classmethod
    def _from_name(cls, suffix=None):
        if not suffix:
            weekday = None
        else:
            weekday = weekday_to_int[suffix]
        return cls(weekday=weekday)


cdef class WeekOfMonth(WeekOfMonthMixin):
    """
    Describes monthly dates like "the Tuesday of the 2nd week of each month".

    Parameters
    ----------
    n : int
    week : int {0, 1, 2, 3, ...}, default 0
        A specific integer for the week of the month.
        e.g. 0 is 1st week of month, 1 is the 2nd week, etc.
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.
    """

    _prefix = "WOM"
    _attributes = tuple(["n", "normalize", "week", "weekday"])

    def __init__(self, n=1, normalize=False, week=0, weekday=0):
        WeekOfMonthMixin.__init__(self, n, normalize, weekday)
        self.week = week

        if self.week < 0 or self.week > 3:
            raise ValueError(f"Week must be 0<=week<=3, got {self.week}")

    cpdef __setstate__(self, state):
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        self.week = state.pop("week")

    def _get_offset_day(self, other: datetime) -> int:
        """
        Find the day in the same month as other that has the same
        weekday as self.weekday and is the self.week'th such day in the month.

        Parameters
        ----------
        other : datetime

        Returns
        -------
        day : int
        """
        mstart = datetime(other.year, other.month, 1)
        wday = mstart.weekday()
        shift_days = (self.weekday - wday) % 7
        return 1 + shift_days + self.week * 7

    @classmethod
    def _from_name(cls, suffix=None):
        if not suffix:
            raise ValueError(f"Prefix {repr(cls._prefix)} requires a suffix.")
        # TODO: handle n here...
        # only one digit weeks (1 --> week 0, 2 --> week 1, etc.)
        week = int(suffix[0]) - 1
        weekday = weekday_to_int[suffix[1:]]
        return cls(week=week, weekday=weekday)


cdef class LastWeekOfMonth(WeekOfMonthMixin):
    """
    Describes monthly dates in last week of month like "the last Tuesday of
    each month".

    Parameters
    ----------
    n : int, default 1
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.
    """

    _prefix = "LWOM"
    _attributes = tuple(["n", "normalize", "weekday"])

    def __init__(self, n=1, normalize=False, weekday=0):
        WeekOfMonthMixin.__init__(self, n, normalize, weekday)
        self.week = -1

        if self.n == 0:
            raise ValueError("N cannot be 0")

    cpdef __setstate__(self, state):
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        self.week = -1

    def _get_offset_day(self, other: datetime) -> int:
        """
        Find the day in the same month as other that has the same
        weekday as self.weekday and is the last such day in the month.

        Parameters
        ----------
        other: datetime

        Returns
        -------
        day: int
        """
        dim = get_days_in_month(other.year, other.month)
        mend = datetime(other.year, other.month, dim)
        wday = mend.weekday()
        shift_days = (wday - self.weekday) % 7
        return dim - shift_days

    @classmethod
    def _from_name(cls, suffix=None):
        if not suffix:
            raise ValueError(f"Prefix {repr(cls._prefix)} requires a suffix.")
        # TODO: handle n here...
        weekday = weekday_to_int[suffix]
        return cls(weekday=weekday)


# ---------------------------------------------------------------------
# Special Offset Classes

cdef class FY5253Mixin(SingleConstructorOffset):
    cdef readonly:
        int startingMonth
        int weekday
        str variation

    def __init__(
        self, n=1, normalize=False, weekday=0, startingMonth=1, variation="nearest"
    ):
        BaseOffset.__init__(self, n, normalize)
        self.startingMonth = startingMonth
        self.weekday = weekday
        self.variation = variation

        if self.n == 0:
            raise ValueError("N cannot be 0")

        if self.variation not in ["nearest", "last"]:
            raise ValueError(f"{self.variation} is not a valid variation")

    cpdef __setstate__(self, state):
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self.weekday = state.pop("weekday")
        self.variation = state.pop("variation")

    def is_anchored(self) -> bool:
        return (
            self.n == 1 and self.startingMonth is not None and self.weekday is not None
        )

    # --------------------------------------------------------------------
    # Name-related methods

    @property
    def rule_code(self) -> str:
        prefix = self._prefix
        suffix = self.get_rule_code_suffix()
        return f"{prefix}-{suffix}"

    def _get_suffix_prefix(self) -> str:
        if self.variation == "nearest":
            return "N"
        else:
            return "L"

    def get_rule_code_suffix(self) -> str:
        prefix = self._get_suffix_prefix()
        month = MONTH_ALIASES[self.startingMonth]
        weekday = int_to_weekday[self.weekday]
        return f"{prefix}-{month}-{weekday}"


cdef class FY5253(FY5253Mixin):
    """
    Describes 52-53 week fiscal year. This is also known as a 4-4-5 calendar.

    It is used by companies that desire that their
    fiscal year always end on the same day of the week.

    It is a method of managing accounting periods.
    It is a common calendar structure for some industries,
    such as retail, manufacturing and parking industry.

    For more information see:
    https://en.wikipedia.org/wiki/4-4-5_calendar

    The year may either:

    - end on the last X day of the Y month.
    - end on the last X day closest to the last day of the Y month.

    X is a specific day of the week.
    Y is a certain month of the year

    Parameters
    ----------
    n : int
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    startingMonth : int {1, 2, ... 12}, default 1
        The month in which the fiscal year ends.

    variation : str, default "nearest"
        Method of employing 4-4-5 calendar.

        There are two options:

        - "nearest" means year end is **weekday** closest to last day of month in year.
        - "last" means year end is final **weekday** of the final month in fiscal year.
    """

    _prefix = "RE"
    _attributes = tuple(["n", "normalize", "weekday", "startingMonth", "variation"])

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        dt = datetime(dt.year, dt.month, dt.day)
        year_end = self.get_year_end(dt)

        if self.variation == "nearest":
            # We have to check the year end of "this" cal year AND the previous
            return year_end == dt or self.get_year_end(shift_month(dt, -1, None)) == dt
        else:
            return year_end == dt

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        norm = Timestamp(other).normalize()

        n = self.n
        prev_year = self.get_year_end(datetime(other.year - 1, self.startingMonth, 1))
        cur_year = self.get_year_end(datetime(other.year, self.startingMonth, 1))
        next_year = self.get_year_end(datetime(other.year + 1, self.startingMonth, 1))

        prev_year = localize_pydatetime(prev_year, other.tzinfo)
        cur_year = localize_pydatetime(cur_year, other.tzinfo)
        next_year = localize_pydatetime(next_year, other.tzinfo)

        # Note: next_year.year == other.year + 1, so we will always
        # have other < next_year
        if norm == prev_year:
            n -= 1
        elif norm == cur_year:
            pass
        elif n > 0:
            if norm < prev_year:
                n -= 2
            elif prev_year < norm < cur_year:
                n -= 1
            elif cur_year < norm < next_year:
                pass
        else:
            if cur_year < norm < next_year:
                n += 1
            elif prev_year < norm < cur_year:
                pass
            elif (
                norm.year == prev_year.year
                and norm < prev_year
                and prev_year - norm <= timedelta(6)
            ):
                # GH#14774, error when next_year.year == cur_year.year
                # e.g. prev_year == datetime(2004, 1, 3),
                # other == datetime(2004, 1, 1)
                n -= 1
            else:
                assert False

        shifted = datetime(other.year + n, self.startingMonth, 1)
        result = self.get_year_end(shifted)
        result = datetime(
            result.year,
            result.month,
            result.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        return result

    def get_year_end(self, dt: datetime) -> datetime:
        assert dt.tzinfo is None

        dim = get_days_in_month(dt.year, self.startingMonth)
        target_date = datetime(dt.year, self.startingMonth, dim)
        wkday_diff = self.weekday - target_date.weekday()
        if wkday_diff == 0:
            # year_end is the same for "last" and "nearest" cases
            return target_date

        if self.variation == "last":
            days_forward = (wkday_diff % 7) - 7

            # days_forward is always negative, so we always end up
            # in the same year as dt
            return target_date + timedelta(days=days_forward)
        else:
            # variation == "nearest":
            days_forward = wkday_diff % 7
            if days_forward <= 3:
                # The upcoming self.weekday is closer than the previous one
                return target_date + timedelta(days_forward)
            else:
                # The previous self.weekday is closer than the upcoming one
                return target_date + timedelta(days_forward - 7)

    @classmethod
    def _parse_suffix(cls, varion_code, startingMonth_code, weekday_code):
        if varion_code == "N":
            variation = "nearest"
        elif varion_code == "L":
            variation = "last"
        else:
            raise ValueError(f"Unable to parse varion_code: {varion_code}")

        startingMonth = MONTH_TO_CAL_NUM[startingMonth_code]
        weekday = weekday_to_int[weekday_code]

        return {
            "weekday": weekday,
            "startingMonth": startingMonth,
            "variation": variation,
        }

    @classmethod
    def _from_name(cls, *args):
        return cls(**cls._parse_suffix(*args))


cdef class FY5253Quarter(FY5253Mixin):
    """
    DateOffset increments between business quarter dates
    for 52-53 week fiscal year (also known as a 4-4-5 calendar).

    It is used by companies that desire that their
    fiscal year always end on the same day of the week.

    It is a method of managing accounting periods.
    It is a common calendar structure for some industries,
    such as retail, manufacturing and parking industry.

    For more information see:
    https://en.wikipedia.org/wiki/4-4-5_calendar

    The year may either:

    - end on the last X day of the Y month.
    - end on the last X day closest to the last day of the Y month.

    X is a specific day of the week.
    Y is a certain month of the year

    startingMonth = 1 corresponds to dates like 1/31/2007, 4/30/2007, ...
    startingMonth = 2 corresponds to dates like 2/28/2007, 5/31/2007, ...
    startingMonth = 3 corresponds to dates like 3/30/2007, 6/29/2007, ...

    Parameters
    ----------
    n : int
    weekday : int {0, 1, ..., 6}, default 0
        A specific integer for the day of the week.

        - 0 is Monday
        - 1 is Tuesday
        - 2 is Wednesday
        - 3 is Thursday
        - 4 is Friday
        - 5 is Saturday
        - 6 is Sunday.

    startingMonth : int {1, 2, ..., 12}, default 1
        The month in which fiscal years end.

    qtr_with_extra_week : int {1, 2, 3, 4}, default 1
        The quarter number that has the leap or 14 week when needed.

    variation : str, default "nearest"
        Method of employing 4-4-5 calendar.

        There are two options:

        - "nearest" means year end is **weekday** closest to last day of month in year.
        - "last" means year end is final **weekday** of the final month in fiscal year.
    """

    _prefix = "REQ"
    _attributes = tuple(
        [
            "n",
            "normalize",
            "weekday",
            "startingMonth",
            "qtr_with_extra_week",
            "variation",
        ]
    )

    cdef readonly:
        int qtr_with_extra_week

    def __init__(
        self,
        n=1,
        normalize=False,
        weekday=0,
        startingMonth=1,
        qtr_with_extra_week=1,
        variation="nearest",
    ):
        FY5253Mixin.__init__(
            self, n, normalize, weekday, startingMonth, variation
        )
        self.qtr_with_extra_week = qtr_with_extra_week

    cpdef __setstate__(self, state):
        FY5253Mixin.__setstate__(self, state)
        self.qtr_with_extra_week = state.pop("qtr_with_extra_week")

    @cache_readonly
    def _offset(self):
        return FY5253(
            startingMonth=self.startingMonth,
            weekday=self.weekday,
            variation=self.variation,
        )

    def _rollback_to_year(self, other: datetime):
        """
        Roll `other` back to the most recent date that was on a fiscal year
        end.

        Return the date of that year-end, the number of full quarters
        elapsed between that year-end and other, and the remaining Timedelta
        since the most recent quarter-end.

        Parameters
        ----------
        other : datetime or Timestamp

        Returns
        -------
        tuple of
        prev_year_end : Timestamp giving most recent fiscal year end
        num_qtrs : int
        tdelta : Timedelta
        """
        num_qtrs = 0

        norm = Timestamp(other).tz_localize(None)
        start = self._offset.rollback(norm)
        # Note: start <= norm and self._offset.is_on_offset(start)

        if start < norm:
            # roll adjustment
            qtr_lens = self.get_weeks(norm)

            # check that qtr_lens is consistent with self._offset addition
            end = shift_day(start, days=7 * sum(qtr_lens))
            assert self._offset.is_on_offset(end), (start, end, qtr_lens)

            tdelta = norm - start
            for qlen in qtr_lens:
                if qlen * 7 <= tdelta.days:
                    num_qtrs += 1
                    tdelta -= Timedelta(days=qlen * 7)
                else:
                    break
        else:
            tdelta = Timedelta(0)

        # Note: we always have tdelta.value >= 0
        return start, num_qtrs, tdelta

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        # Note: self.n == 0 is not allowed.

        n = self.n

        prev_year_end, num_qtrs, tdelta = self._rollback_to_year(other)
        res = prev_year_end
        n += num_qtrs
        if self.n <= 0 and tdelta.value > 0:
            n += 1

        # Possible speedup by handling years first.
        years = n // 4
        if years:
            res += self._offset * years
            n -= years * 4

        # Add an extra day to make *sure* we are getting the quarter lengths
        # for the upcoming year, not the previous year
        qtr_lens = self.get_weeks(res + Timedelta(days=1))

        # Note: we always have 0 <= n < 4
        weeks = sum(qtr_lens[:n])
        if weeks:
            res = shift_day(res, days=weeks * 7)

        return res

    def get_weeks(self, dt: datetime):
        ret = [13] * 4

        year_has_extra_week = self.year_has_extra_week(dt)

        if year_has_extra_week:
            ret[self.qtr_with_extra_week - 1] = 14

        return ret

    def year_has_extra_week(self, dt: datetime) -> bool:
        # Avoid round-down errors --> normalize to get
        # e.g. '370D' instead of '360D23H'
        norm = Timestamp(dt).normalize().tz_localize(None)

        next_year_end = self._offset.rollforward(norm)
        prev_year_end = norm - self._offset
        weeks_in_year = (next_year_end - prev_year_end).days / 7
        assert weeks_in_year in [52, 53], weeks_in_year
        return weeks_in_year == 53

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        if self._offset.is_on_offset(dt):
            return True

        next_year_end = dt - self._offset

        qtr_lens = self.get_weeks(dt)

        current = next_year_end
        for qtr_len in qtr_lens:
            current = shift_day(current, days=qtr_len * 7)
            if dt == current:
                return True
        return False

    @property
    def rule_code(self) -> str:
        suffix = FY5253Mixin.rule_code.__get__(self)
        qtr = self.qtr_with_extra_week
        return f"{suffix}-{qtr}"

    @classmethod
    def _from_name(cls, *args):
        return cls(
            **dict(FY5253._parse_suffix(*args[:-1]), qtr_with_extra_week=int(args[-1]))
        )


cdef class Easter(SingleConstructorOffset):
    """
    DateOffset for the Easter holiday using logic defined in dateutil.

    Right now uses the revised method which is valid in years 1583-4099.
    """

    cpdef __setstate__(self, state):
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        current_easter = easter(other.year)
        current_easter = datetime(
            current_easter.year, current_easter.month, current_easter.day
        )
        current_easter = localize_pydatetime(current_easter, other.tzinfo)

        n = self.n
        if n >= 0 and other < current_easter:
            n -= 1
        elif n < 0 and other > current_easter:
            n += 1
        # TODO: Why does this handle the 0 case the opposite of others?

        # NOTE: easter returns a datetime.date so we have to convert to type of
        # other
        new = easter(other.year + n)
        new = datetime(
            new.year,
            new.month,
            new.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        return new

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        return date(dt.year, dt.month, dt.day) == easter(dt.year)


# ----------------------------------------------------------------------
# Custom Offset classes


cdef class CustomBusinessDay(BusinessDay):
    """
    DateOffset subclass representing custom business days excluding holidays.

    Parameters
    ----------
    n : int, default 1
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : pd.HolidayCalendar or np.busdaycalendar
    offset : timedelta, default timedelta(0)
    """

    _prefix = "C"
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "offset"]
    )

    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        offset=timedelta(0),
    ):
        BusinessDay.__init__(self, n, normalize, offset)
        self._init_custom(weekmask, holidays, calendar)

    cpdef __setstate__(self, state):
        self.holidays = state.pop("holidays")
        self.weekmask = state.pop("weekmask")
        BusinessDay.__setstate__(self, state)

    @apply_wraps
    def apply(self, other):
        if self.n <= 0:
            roll = "forward"
        else:
            roll = "backward"

        if PyDateTime_Check(other):
            date_in = other
            np_dt = np.datetime64(date_in.date())

            np_incr_dt = np.busday_offset(
                np_dt, self.n, roll=roll, busdaycal=self.calendar
            )

            dt_date = np_incr_dt.astype(datetime)
            result = datetime.combine(dt_date, date_in.time())

            if self.offset:
                result = result + self.offset
            return result

        elif PyDelta_Check(other) or isinstance(other, Tick):
            return BDay(self.n, offset=self.offset + other, normalize=self.normalize)
        else:
            raise ApplyTypeError(
                "Only know how to combine trading day with "
                "datetime, datetime64 or timedelta."
            )

    def apply_index(self, dtindex):
        raise NotImplementedError

    def _apply_array(self, dtarr):
        raise NotImplementedError

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        day64 = _to_dt64D(dt)
        return np.is_busday(day64, busdaycal=self.calendar)


cdef class CustomBusinessHour(BusinessHour):
    """
    DateOffset subclass representing possibly n custom business days.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    start : str, default "09:00"
        Start time of your custom business hour in 24h format.
    end : str, default: "17:00"
        End time of your custom business hour in 24h format.
    """

    _prefix = "CBH"
    _anchor = 0
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "start", "end", "offset"]
    )

    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        start="09:00",
        end="17:00",
        offset=timedelta(0),
    ):
        BusinessHour.__init__(self, n, normalize, start=start, end=end, offset=offset)
        self._init_custom(weekmask, holidays, calendar)


cdef class _CustomBusinessMonth(BusinessMixin):
    """
    DateOffset subclass representing custom business month(s).

    Increments between beginning/end of month dates.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : pd.HolidayCalendar or np.busdaycalendar
        Calendar to integrate.
    offset : timedelta, default timedelta(0)
        Time offset to apply.
    """

    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "offset"]
    )

    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        offset=timedelta(0),
    ):
        BusinessMixin.__init__(self, n, normalize, offset)
        self._init_custom(weekmask, holidays, calendar)

    @cache_readonly
    def cbday_roll(self):
        """
        Define default roll function to be called in apply method.
        """
        cbday = CustomBusinessDay(n=self.n, normalize=False, **self.kwds)

        if self._prefix.endswith("S"):
            # MonthBegin
            roll_func = cbday.rollforward
        else:
            # MonthEnd
            roll_func = cbday.rollback
        return roll_func

    @cache_readonly
    def m_offset(self):
        if self._prefix.endswith("S"):
            # MonthBegin
            moff = MonthBegin(n=1, normalize=False)
        else:
            # MonthEnd
            moff = MonthEnd(n=1, normalize=False)
        return moff

    @cache_readonly
    def month_roll(self):
        """
        Define default roll function to be called in apply method.
        """
        if self._prefix.endswith("S"):
            # MonthBegin
            roll_func = self.m_offset.rollback
        else:
            # MonthEnd
            roll_func = self.m_offset.rollforward
        return roll_func

    @apply_wraps
    def apply(self, other: datetime) -> datetime:
        # First move to month offset
        cur_month_offset_date = self.month_roll(other)

        # Find this custom month offset
        compare_date = self.cbday_roll(cur_month_offset_date)
        n = roll_convention(other.day, self.n, compare_date.day)

        new = cur_month_offset_date + n * self.m_offset
        result = self.cbday_roll(new)
        return result


cdef class CustomBusinessMonthEnd(_CustomBusinessMonth):
    _prefix = "CBM"


cdef class CustomBusinessMonthBegin(_CustomBusinessMonth):
    _prefix = "CBMS"


BDay = BusinessDay
BMonthEnd = BusinessMonthEnd
BMonthBegin = BusinessMonthBegin
CBMonthEnd = CustomBusinessMonthEnd
CBMonthBegin = CustomBusinessMonthBegin
CDay = CustomBusinessDay

# ----------------------------------------------------------------------
# to_offset helpers

prefix_mapping = {
    offset._prefix: offset
    for offset in [
        YearBegin,  # 'AS'
        YearEnd,  # 'A'
        BYearBegin,  # 'BAS'
        BYearEnd,  # 'BA'
        BusinessDay,  # 'B'
        BusinessMonthBegin,  # 'BMS'
        BusinessMonthEnd,  # 'BM'
        BQuarterEnd,  # 'BQ'
        BQuarterBegin,  # 'BQS'
        BusinessHour,  # 'BH'
        CustomBusinessDay,  # 'C'
        CustomBusinessMonthEnd,  # 'CBM'
        CustomBusinessMonthBegin,  # 'CBMS'
        CustomBusinessHour,  # 'CBH'
        MonthEnd,  # 'M'
        MonthBegin,  # 'MS'
        Nano,  # 'N'
        SemiMonthEnd,  # 'SM'
        SemiMonthBegin,  # 'SMS'
        Week,  # 'W'
        Second,  # 'S'
        Minute,  # 'T'
        Micro,  # 'U'
        QuarterEnd,  # 'Q'
        QuarterBegin,  # 'QS'
        Milli,  # 'L'
        Hour,  # 'H'
        Day,  # 'D'
        WeekOfMonth,  # 'WOM'
        FY5253,
        FY5253Quarter,
    ]
}

# hack to handle WOM-1MON
opattern = re.compile(
    r"([+\-]?\d*|[+\-]?\d*\.\d*)\s*([A-Za-z]+([\-][\dA-Za-z\-]+)?)"
)

_lite_rule_alias = {
    "W": "W-SUN",
    "Q": "Q-DEC",

    "A": "A-DEC",      # YearEnd(month=12),
    "Y": "A-DEC",
    "AS": "AS-JAN",    # YearBegin(month=1),
    "YS": "AS-JAN",
    "BA": "BA-DEC",    # BYearEnd(month=12),
    "BY": "BA-DEC",
    "BAS": "BAS-JAN",  # BYearBegin(month=1),
    "BYS": "BAS-JAN",

    "Min": "T",
    "min": "T",
    "ms": "L",
    "us": "U",
    "ns": "N",
}

_dont_uppercase = {"MS", "ms"}

INVALID_FREQ_ERR_MSG = "Invalid frequency: {0}"

# TODO: still needed?
# cache of previously seen offsets
_offset_map = {}


# TODO: better name?
def _get_offset(name: str) -> BaseOffset:
    """
    Return DateOffset object associated with rule name.

    Examples
    --------
    _get_offset('EOM') --> BMonthEnd(1)
    """
    if name not in _dont_uppercase:
        name = name.upper()
        name = _lite_rule_alias.get(name, name)
        name = _lite_rule_alias.get(name.lower(), name)
    else:
        name = _lite_rule_alias.get(name, name)

    if name not in _offset_map:
        try:
            split = name.split("-")
            klass = prefix_mapping[split[0]]
            # handles case where there's no suffix (and will TypeError if too
            # many '-')
            offset = klass._from_name(*split[1:])
        except (ValueError, TypeError, KeyError) as err:
            # bad prefix or suffix
            raise ValueError(INVALID_FREQ_ERR_MSG.format(name)) from err
        # cache
        _offset_map[name] = offset

    return _offset_map[name]


cpdef to_offset(freq):
    """
    Return DateOffset object from string or tuple representation
    or datetime.timedelta object.

    Parameters
    ----------
    freq : str, tuple, datetime.timedelta, DateOffset or None

    Returns
    -------
    DateOffset or None

    Raises
    ------
    ValueError
        If freq is an invalid frequency

    See Also
    --------
    DateOffset : Standard kind of date increment used for a date range.

    Examples
    --------
    >>> to_offset("5min")
    <5 * Minutes>

    >>> to_offset("1D1H")
    <25 * Hours>

    >>> to_offset("2W")
    <2 * Weeks: weekday=6>

    >>> to_offset("2B")
    <2 * BusinessDays>

    >>> to_offset(pd.Timedelta(days=1))
    <Day>

    >>> to_offset(Hour())
    <Hour>
    """
    if freq is None:
        return None

    if isinstance(freq, BaseOffset):
        return freq

    if isinstance(freq, tuple):
        raise TypeError(
            f"to_offset does not support tuples {freq}, pass as a string instead"
        )

    elif PyDelta_Check(freq):
        return delta_to_tick(freq)

    elif isinstance(freq, str):
        delta = None
        stride_sign = None

        try:
            split = opattern.split(freq)
            if split[-1] != "" and not split[-1].isspace():
                # the last element must be blank
                raise ValueError("last element must be blank")

            tups = zip(split[0::4], split[1::4], split[2::4])
            for n, (sep, stride, name) in enumerate(tups):
                if sep != "" and not sep.isspace():
                    raise ValueError("separator must be spaces")
                prefix = _lite_rule_alias.get(name) or name
                if stride_sign is None:
                    stride_sign = -1 if stride.startswith("-") else 1
                if not stride:
                    stride = 1

                if prefix in {"D", "H", "T", "S", "L", "U", "N"}:
                    # For these prefixes, we have something like "3H" or
                    #  "2.5T", so we can construct a Timedelta with the
                    #  matching unit and get our offset from delta_to_tick
                    td = Timedelta(1, unit=prefix)
                    off = delta_to_tick(td)
                    offset = off * float(stride)
                    if n != 0:
                        # If n==0, then stride_sign is already incorporated
                        #  into the offset
                        offset *= stride_sign
                else:
                    stride = int(stride)
                    offset = _get_offset(name)
                    offset = offset * int(np.fabs(stride) * stride_sign)

                if delta is None:
                    delta = offset
                else:
                    delta = delta + offset
        except (ValueError, TypeError) as err:
            raise ValueError(INVALID_FREQ_ERR_MSG.format(freq)) from err
    else:
        delta = None

    if delta is None:
        raise ValueError(INVALID_FREQ_ERR_MSG.format(freq))

    return delta


# ----------------------------------------------------------------------
# RelativeDelta Arithmetic

def shift_day(other: datetime, days: int) -> datetime:
    """
    Increment the datetime `other` by the given number of days, retaining
    the time-portion of the datetime.  For tz-naive datetimes this is
    equivalent to adding a timedelta.  For tz-aware datetimes it is similar to
    dateutil's relativedelta.__add__, but handles pytz tzinfo objects.

    Parameters
    ----------
    other : datetime or Timestamp
    days : int

    Returns
    -------
    shifted: datetime or Timestamp
    """
    if other.tzinfo is None:
        return other + timedelta(days=days)

    tz = other.tzinfo
    naive = other.replace(tzinfo=None)
    shifted = naive + timedelta(days=days)
    return localize_pydatetime(shifted, tz)


cdef inline int year_add_months(npy_datetimestruct dts, int months) nogil:
    """
    New year number after shifting npy_datetimestruct number of months.
    """
    return dts.year + (dts.month + months - 1) // 12


cdef inline int month_add_months(npy_datetimestruct dts, int months) nogil:
    """
    New month number after shifting npy_datetimestruct
    number of months.
    """
    cdef:
        int new_month = (dts.month + months) % 12
    return 12 if new_month == 0 else new_month


@cython.wraparound(False)
@cython.boundscheck(False)
cdef shift_quarters(
    const int64_t[:] dtindex,
    int quarters,
    int q1start_month,
    object day_opt,
    int modby=3,
):
    """
    Given an int64 array representing nanosecond timestamps, shift all elements
    by the specified number of quarters using DateOffset semantics.

    Parameters
    ----------
    dtindex : int64_t[:] timestamps for input dates
    quarters : int number of quarters to shift
    q1start_month : int month in which Q1 begins by convention
    day_opt : {'start', 'end', 'business_start', 'business_end'}
    modby : int (3 for quarters, 12 for years)

    Returns
    -------
    out : ndarray[int64_t]
    """
    cdef:
        Py_ssize_t count = len(dtindex)
        int64_t[:] out = np.empty(count, dtype="int64")

    if day_opt not in ["start", "end", "business_start", "business_end"]:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")

    _shift_quarters(dtindex, out, count, quarters, q1start_month, day_opt, modby)
    return np.asarray(out)


@cython.wraparound(False)
@cython.boundscheck(False)
def shift_months(const int64_t[:] dtindex, int months, object day_opt=None):
    """
    Given an int64-based datetime index, shift all elements
    specified number of months using DateOffset semantics

    day_opt: {None, 'start', 'end', 'business_start', 'business_end'}
       * None: day of month
       * 'start' 1st day of month
       * 'end' last day of month
    """
    cdef:
        Py_ssize_t i
        npy_datetimestruct dts
        int count = len(dtindex)
        int64_t[:] out = np.empty(count, dtype="int64")

    if day_opt is None:
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = NPY_NAT
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                dts.year = year_add_months(dts, months)
                dts.month = month_add_months(dts, months)

                dts.day = min(dts.day, get_days_in_month(dts.year, dts.month))
                out[i] = dtstruct_to_dt64(&dts)
    elif day_opt in ["start", "end", "business_start", "business_end"]:
        _shift_months(dtindex, out, count, months, day_opt)
    else:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")

    return np.asarray(out)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _shift_months(const int64_t[:] dtindex,
                               int64_t[:] out,
                               Py_ssize_t count,
                               int months,
                               str day_opt) nogil:
    """
    See shift_months.__doc__
    """
    cdef:
        Py_ssize_t i
        int months_to_roll
        npy_datetimestruct dts

    for i in range(count):
        if dtindex[i] == NPY_NAT:
            out[i] = NPY_NAT
            continue

        dt64_to_dtstruct(dtindex[i], &dts)
        months_to_roll = months

        months_to_roll = _roll_qtrday(&dts, months_to_roll, 0, day_opt)

        dts.year = year_add_months(dts, months_to_roll)
        dts.month = month_add_months(dts, months_to_roll)
        dts.day = get_day_of_month(&dts, day_opt)

        out[i] = dtstruct_to_dt64(&dts)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _shift_quarters(const int64_t[:] dtindex,
                                 int64_t[:] out,
                                 Py_ssize_t count,
                                 int quarters,
                                 int q1start_month,
                                 str day_opt,
                                 int modby) nogil:
    """
    See shift_quarters.__doc__
    """
    cdef:
        Py_ssize_t i
        int months_since, n
        npy_datetimestruct dts

    for i in range(count):
        if dtindex[i] == NPY_NAT:
            out[i] = NPY_NAT
            continue

        dt64_to_dtstruct(dtindex[i], &dts)
        n = quarters

        months_since = (dts.month - q1start_month) % modby
        n = _roll_qtrday(&dts, n, months_since, day_opt)

        dts.year = year_add_months(dts, modby * n - months_since)
        dts.month = month_add_months(dts, modby * n - months_since)
        dts.day = get_day_of_month(&dts, day_opt)

        out[i] = dtstruct_to_dt64(&dts)


cdef ndarray[int64_t] _shift_bdays(const int64_t[:] i8other, int periods):
    """
    Implementation of BusinessDay.apply_offset.

    Parameters
    ----------
    i8other : const int64_t[:]
    periods : int

    Returns
    -------
    ndarray[int64_t]
    """
    cdef:
        Py_ssize_t i, n = len(i8other)
        int64_t[:] result = np.empty(n, dtype="i8")
        int64_t val, res
        int wday, nadj, days
        npy_datetimestruct dts

    for i in range(n):
        val = i8other[i]
        if val == NPY_NAT:
            result[i] = NPY_NAT
        else:
            # The rest of this is effectively a copy of BusinessDay.apply
            nadj = periods
            weeks = nadj // 5
            dt64_to_dtstruct(val, &dts)
            wday = dayofweek(dts.year, dts.month, dts.day)

            if nadj <= 0 and wday > 4:
                # roll forward
                nadj += 1

            nadj -= 5 * weeks

            # nadj is always >= 0 at this point
            if nadj == 0 and wday > 4:
                # roll back
                days = 4 - wday
            elif wday > 4:
                # roll forward
                days = (7 - wday) + (nadj - 1)
            elif wday + nadj <= 4:
                # shift by n days without leaving the current week
                days = nadj
            else:
                # shift by nadj days plus 2 to get past the weekend
                days = nadj + 2

            res = val + (7 * weeks + days) * DAY_NANOS
            result[i] = res

    return result.base


def shift_month(stamp: datetime, months: int, day_opt: object=None) -> datetime:
    """
    Given a datetime (or Timestamp) `stamp`, an integer `months` and an
    option `day_opt`, return a new datetimelike that many months later,
    with day determined by `day_opt` using relativedelta semantics.

    Scalar analogue of shift_months

    Parameters
    ----------
    stamp : datetime or Timestamp
    months : int
    day_opt : None, 'start', 'end', 'business_start', 'business_end', or int
        None: returned datetimelike has the same day as the input, or the
              last day of the month if the new month is too short
        'start': returned datetimelike has day=1
        'end': returned datetimelike has day on the last day of the month
        'business_start': returned datetimelike has day on the first
            business day of the month
        'business_end': returned datetimelike has day on the last
            business day of the month
        int: returned datetimelike has day equal to day_opt

    Returns
    -------
    shifted : datetime or Timestamp (same as input `stamp`)
    """
    cdef:
        int year, month, day
        int days_in_month, dy

    dy = (stamp.month + months) // 12
    month = (stamp.month + months) % 12

    if month == 0:
        month = 12
        dy -= 1
    year = stamp.year + dy

    if day_opt is None:
        days_in_month = get_days_in_month(year, month)
        day = min(stamp.day, days_in_month)
    elif day_opt == "start":
        day = 1
    elif day_opt == "end":
        day = get_days_in_month(year, month)
    elif day_opt == "business_start":
        # first business day of month
        day = get_firstbday(year, month)
    elif day_opt == "business_end":
        # last business day of month
        day = get_lastbday(year, month)
    elif is_integer_object(day_opt):
        days_in_month = get_days_in_month(year, month)
        day = min(day_opt, days_in_month)
    else:
        raise ValueError(day_opt)
    return stamp.replace(year=year, month=month, day=day)


cdef inline int get_day_of_month(npy_datetimestruct* dts, str day_opt) nogil:
    """
    Find the day in `other`'s month that satisfies a DateOffset's is_on_offset
    policy, as described by the `day_opt` argument.

    Parameters
    ----------
    dts : npy_datetimestruct*
    day_opt : {'start', 'end', 'business_start', 'business_end'}
        'start': returns 1
        'end': returns last day of the month
        'business_start': returns the first business day of the month
        'business_end': returns the last business day of the month

    Returns
    -------
    day_of_month : int

    Examples
    -------
    >>> other = datetime(2017, 11, 14)
    >>> get_day_of_month(other, 'start')
    1
    >>> get_day_of_month(other, 'end')
    30

    Notes
    -----
    Caller is responsible for ensuring one of the four accepted day_opt values
    is passed.
    """

    if day_opt == "start":
        return 1
    elif day_opt == "end":
        return get_days_in_month(dts.year, dts.month)
    elif day_opt == "business_start":
        # first business day of month
        return get_firstbday(dts.year, dts.month)
    else:
        # i.e. day_opt == "business_end":
        # last business day of month
        return get_lastbday(dts.year, dts.month)


cpdef int roll_convention(int other, int n, int compare) nogil:
    """
    Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : int, generally the day component of a datetime
    n : number of periods to increment, before adjusting for rolling
    compare : int, generally the day component of a datetime, in the same
              month as the datetime form which `other` was taken.

    Returns
    -------
    n : int number of periods to increment
    """
    if n > 0 and other < compare:
        n -= 1
    elif n <= 0 and other > compare:
        # as if rolled forward already
        n += 1
    return n


def roll_qtrday(other: datetime, n: int, month: int,
                day_opt: str, modby: int) -> int:
    """
    Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : datetime or Timestamp
    n : number of periods to increment, before adjusting for rolling
    month : int reference month giving the first month of the year
    day_opt : {'start', 'end', 'business_start', 'business_end'}
        The convention to use in finding the day in a given month against
        which to compare for rollforward/rollbackward decisions.
    modby : int 3 for quarters, 12 for years

    Returns
    -------
    n : int number of periods to increment

    See Also
    --------
    get_day_of_month : Find the day in a month provided an offset.
    """
    cdef:
        int months_since
        npy_datetimestruct dts

    if day_opt not in ["start", "end", "business_start", "business_end"]:
        raise ValueError(day_opt)

    pydate_to_dtstruct(other, &dts)

    if modby == 12:
        # We care about the month-of-year, not month-of-quarter, so skip mod
        months_since = other.month - month
    else:
        months_since = other.month % modby - month % modby

    return _roll_qtrday(&dts, n, months_since, day_opt)


cdef inline int _roll_qtrday(npy_datetimestruct* dts,
                             int n,
                             int months_since,
                             str day_opt) nogil except? -1:
    """
    See roll_qtrday.__doc__
    """

    if n > 0:
        if months_since < 0 or (months_since == 0 and
                                dts.day < get_day_of_month(dts, day_opt)):
            # pretend to roll back if on same month but
            # before compare_day
            n -= 1
    else:
        if months_since > 0 or (months_since == 0 and
                                dts.day > get_day_of_month(dts, day_opt)):
            # make sure to roll forward, so negate
            n += 1
    return n
