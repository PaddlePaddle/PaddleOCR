from __future__ import annotations

from datetime import (
    datetime,
    time,
    timedelta,
    tzinfo,
)
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    tslib,
)
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Resolution,
    Timestamp,
    conversion,
    fields,
    get_resolution,
    iNaT,
    ints_to_pydatetime,
    is_date_array_normalized,
    normalize_i8_timestamps,
    timezones,
    to_offset,
    tzconversion,
)
from pandas.errors import PerformanceWarning

from pandas.core.dtypes.cast import astype_dt64_to_dt64tz
from pandas.core.dtypes.common import (
    DT64NS_DTYPE,
    INT64_DTYPE,
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_dtype_equal,
    is_extension_array_dtype,
    is_float_dtype,
    is_object_dtype,
    is_period_dtype,
    is_sparse,
    is_string_dtype,
    is_timedelta64_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.core.dtypes.missing import isna

from pandas.core.algorithms import checked_add_with_arr
from pandas.core.arrays import (
    ExtensionArray,
    datetimelike as dtl,
)
from pandas.core.arrays._ranges import generate_regular_range
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import extract_array

from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (
    BDay,
    Day,
    Tick,
)

if TYPE_CHECKING:
    from typing import Literal

    from pandas import DataFrame
    from pandas.core.arrays import (
        PeriodArray,
        TimedeltaArray,
    )

_midnight = time(0, 0)


def tz_to_dtype(tz):
    """
    Return a datetime64[ns] dtype appropriate for the given timezone.

    Parameters
    ----------
    tz : tzinfo or None

    Returns
    -------
    np.dtype or Datetime64TZDType
    """
    if tz is None:
        return DT64NS_DTYPE
    else:
        return DatetimeTZDtype(tz=tz)


def _field_accessor(name, field, docstring=None):
    def f(self):
        values = self._local_timestamps()

        if field in self._bool_ops:
            if field.endswith(("start", "end")):
                freq = self.freq
                month_kw = 12
                if freq:
                    kwds = freq.kwds
                    month_kw = kwds.get("startingMonth", kwds.get("month", 12))

                result = fields.get_start_end_field(
                    values, field, self.freqstr, month_kw
                )
            else:
                result = fields.get_date_field(values, field)

            # these return a boolean by-definition
            return result

        if field in self._object_ops:
            result = fields.get_date_name_field(values, field)
            result = self._maybe_mask_results(result, fill_value=None)

        else:
            result = fields.get_date_field(values, field)
            result = self._maybe_mask_results(
                result, fill_value=None, convert="float64"
            )

        return result

    f.__name__ = name
    f.__doc__ = docstring
    return property(f)


class DatetimeArray(dtl.TimelikeOps, dtl.DatelikeOps):
    """
    Pandas ExtensionArray for tz-naive or tz-aware datetime data.

    .. warning::

       DatetimeArray is currently experimental, and its API may change
       without warning. In particular, :attr:`DatetimeArray.dtype` is
       expected to change to always be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    values : Series, Index, DatetimeArray, ndarray
        The datetime data.

        For DatetimeArray `values` (or a Series or Index boxing one),
        `dtype` and `freq` will be extracted from `values`.

    dtype : numpy.dtype or DatetimeTZDtype
        Note that the only NumPy dtype allowed is 'datetime64[ns]'.
    freq : str or Offset, optional
        The frequency.
    copy : bool, default False
        Whether to copy the underlying array of values.

    Attributes
    ----------
    None

    Methods
    -------
    None
    """

    _typ = "datetimearray"
    _scalar_type = Timestamp
    _recognized_scalars = (datetime, np.datetime64)
    _is_recognized_dtype = is_datetime64_any_dtype
    _infer_matches = ("datetime", "datetime64", "date")

    # define my properties & methods for delegation
    _bool_ops: list[str] = [
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_leap_year",
    ]
    _object_ops: list[str] = ["freq", "tz"]
    _field_ops: list[str] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "weekofyear",
        "week",
        "weekday",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "days_in_month",
        "daysinmonth",
        "microsecond",
        "nanosecond",
    ]
    _other_ops: list[str] = ["date", "time", "timetz"]
    _datetimelike_ops: list[str] = _field_ops + _object_ops + _bool_ops + _other_ops
    _datetimelike_methods: list[str] = [
        "to_period",
        "tz_localize",
        "tz_convert",
        "normalize",
        "strftime",
        "round",
        "floor",
        "ceil",
        "month_name",
        "day_name",
    ]

    # ndim is inherited from ExtensionArray, must exist to ensure
    #  Timestamp.__richcmp__(DateTimeArray) operates pointwise

    # ensure that operations with numpy arrays defer to our implementation
    __array_priority__ = 1000

    # -----------------------------------------------------------------
    # Constructors

    _dtype: np.dtype | DatetimeTZDtype
    _freq = None

    def __init__(self, values, dtype=DT64NS_DTYPE, freq=None, copy: bool = False):
        values = extract_array(values, extract_numpy=True)
        if isinstance(values, IntegerArray):
            values = values.to_numpy("int64", na_value=iNaT)

        inferred_freq = getattr(values, "_freq", None)

        if isinstance(values, type(self)):
            # validation
            dtz = getattr(dtype, "tz", None)
            if dtz and values.tz is None:
                dtype = DatetimeTZDtype(tz=dtype.tz)
            elif dtz and values.tz:
                if not timezones.tz_compare(dtz, values.tz):
                    msg = (
                        "Timezone of the array and 'dtype' do not match. "
                        f"'{dtz}' != '{values.tz}'"
                    )
                    raise TypeError(msg)
            elif values.tz:
                dtype = values.dtype

            if freq is None:
                freq = values.freq
            values = values._ndarray

        if not isinstance(values, np.ndarray):
            raise ValueError(
                f"Unexpected type '{type(values).__name__}'. 'values' must be "
                "a DatetimeArray, ndarray, or Series or Index containing one of those."
            )
        if values.ndim not in [1, 2]:
            raise ValueError("Only 1-dimensional input arrays are supported.")

        if values.dtype == "i8":
            # for compat with datetime/timedelta/period shared methods,
            #  we can sometimes get here with int64 values.  These represent
            #  nanosecond UTC (or tz-naive) unix timestamps
            values = values.view(DT64NS_DTYPE)

        if values.dtype != DT64NS_DTYPE:
            raise ValueError(
                "The dtype of 'values' is incorrect. Must be 'datetime64[ns]'. "
                f"Got {values.dtype} instead."
            )

        dtype = _validate_dt64_dtype(dtype)

        if freq == "infer":
            raise ValueError(
                "Frequency inference not allowed in DatetimeArray.__init__. "
                "Use 'pd.array()' instead."
            )

        if copy:
            values = values.copy()
        if freq:
            freq = to_offset(freq)
        if getattr(dtype, "tz", None):
            # https://github.com/pandas-dev/pandas/issues/18595
            # Ensure that we have a standard timezone for pytz objects.
            # Without this, things like adding an array of timedeltas and
            # a  tz-aware Timestamp (with a tz specific to its datetime) will
            # be incorrect(ish?) for the array as a whole
            dtype = DatetimeTZDtype(tz=timezones.tz_standardize(dtype.tz))

        NDArrayBacked.__init__(self, values=values, dtype=dtype)
        self._freq = freq

        if inferred_freq is None and freq is not None:
            type(self)._validate_frequency(self, freq)

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls, values: np.ndarray, freq: BaseOffset | None = None, dtype=DT64NS_DTYPE
    ) -> DatetimeArray:
        assert isinstance(values, np.ndarray)
        assert values.dtype == DT64NS_DTYPE

        result = super()._simple_new(values, dtype)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False):
        return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_not_strict(
        cls,
        data,
        dtype=None,
        copy: bool = False,
        tz=None,
        freq=lib.no_default,
        dayfirst: bool = False,
        yearfirst: bool = False,
        ambiguous="raise",
    ):
        explicit_none = freq is None
        freq = freq if freq is not lib.no_default else None

        freq, freq_infer = dtl.maybe_infer_freq(freq)

        subarr, tz, inferred_freq = sequence_to_dt64ns(
            data,
            dtype=dtype,
            copy=copy,
            tz=tz,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            ambiguous=ambiguous,
        )

        freq, freq_infer = dtl.validate_inferred_freq(freq, inferred_freq, freq_infer)
        if explicit_none:
            freq = None

        dtype = tz_to_dtype(tz)
        result = cls._simple_new(subarr, freq=freq, dtype=dtype)

        if inferred_freq is None and freq is not None:
            # this condition precludes `freq_infer`
            cls._validate_frequency(result, freq, ambiguous=ambiguous)

        elif freq_infer:
            # Set _freq directly to bypass duplicative _validate_frequency
            # check.
            result._freq = to_offset(result.inferred_freq)

        return result

    @classmethod
    def _generate_range(
        cls,
        start,
        end,
        periods,
        freq,
        tz=None,
        normalize=False,
        ambiguous="raise",
        nonexistent="raise",
        closed=None,
    ):

        periods = dtl.validate_periods(periods)
        if freq is None and any(x is None for x in [periods, start, end]):
            raise ValueError("Must provide freq argument if no data is supplied")

        if com.count_not_none(start, end, periods, freq) != 3:
            raise ValueError(
                "Of the four parameters: start, end, periods, "
                "and freq, exactly three must be specified"
            )
        freq = to_offset(freq)

        if start is not None:
            start = Timestamp(start)

        if end is not None:
            end = Timestamp(end)

        if start is NaT or end is NaT:
            raise ValueError("Neither `start` nor `end` can be NaT")

        left_closed, right_closed = dtl.validate_endpoints(closed)
        start, end, _normalized = _maybe_normalize_endpoints(start, end, normalize)
        tz = _infer_tz_from_endpoints(start, end, tz)

        if tz is not None:
            # Localize the start and end arguments
            start_tz = None if start is None else start.tz
            end_tz = None if end is None else end.tz
            start = _maybe_localize_point(
                start, start_tz, start, freq, tz, ambiguous, nonexistent
            )
            end = _maybe_localize_point(
                end, end_tz, end, freq, tz, ambiguous, nonexistent
            )
        if freq is not None:
            # We break Day arithmetic (fixed 24 hour) here and opt for
            # Day to mean calendar day (23/24/25 hour). Therefore, strip
            # tz info from start and day to avoid DST arithmetic
            if isinstance(freq, Day):
                if start is not None:
                    start = start.tz_localize(None)
                if end is not None:
                    end = end.tz_localize(None)

            if isinstance(freq, Tick):
                values = generate_regular_range(start, end, periods, freq)
            else:
                xdr = generate_range(start=start, end=end, periods=periods, offset=freq)
                values = np.array([x.value for x in xdr], dtype=np.int64)

            _tz = start.tz if start is not None else end.tz
            values = values.view("M8[ns]")
            index = cls._simple_new(values, freq=freq, dtype=tz_to_dtype(_tz))

            if tz is not None and index.tz is None:
                arr = tzconversion.tz_localize_to_utc(
                    index.asi8, tz, ambiguous=ambiguous, nonexistent=nonexistent
                )

                index = cls(arr)

                # index is localized datetime64 array -> have to convert
                # start/end as well to compare
                if start is not None:
                    start = start.tz_localize(tz, ambiguous, nonexistent).asm8
                if end is not None:
                    end = end.tz_localize(tz, ambiguous, nonexistent).asm8
        else:
            # Create a linearly spaced date_range in local time
            # Nanosecond-granularity timestamps aren't always correctly
            # representable with doubles, so we limit the range that we
            # pass to np.linspace as much as possible
            arr = (
                np.linspace(0, end.value - start.value, periods, dtype="int64")
                + start.value
            )
            dtype = tz_to_dtype(tz)
            arr = arr.astype("M8[ns]", copy=False)
            index = cls._simple_new(arr, freq=None, dtype=dtype)

        if not left_closed and len(index) and index[0] == start:
            # TODO: overload DatetimeLikeArrayMixin.__getitem__
            index = cast(DatetimeArray, index[1:])
        if not right_closed and len(index) and index[-1] == end:
            # TODO: overload DatetimeLikeArrayMixin.__getitem__
            index = cast(DatetimeArray, index[:-1])

        dtype = tz_to_dtype(tz)
        return cls._simple_new(index._ndarray, freq=freq, dtype=dtype)

    # -----------------------------------------------------------------
    # DatetimeLike Interface

    def _unbox_scalar(self, value, setitem: bool = False) -> np.datetime64:
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timestamp.")
        self._check_compatible_with(value, setitem=setitem)
        return value.asm8

    def _scalar_from_string(self, value) -> Timestamp | NaTType:
        return Timestamp(value, tz=self.tz)

    def _check_compatible_with(self, other, setitem: bool = False):
        if other is NaT:
            return
        self._assert_tzawareness_compat(other)
        if setitem:
            # Stricter check for setitem vs comparison methods
            if not timezones.tz_compare(self.tz, other.tz):
                raise ValueError(f"Timezones don't match. '{self.tz}' != '{other.tz}'")

    # -----------------------------------------------------------------
    # Descriptive Properties

    def _box_func(self, x) -> Timestamp | NaTType:
        ts = Timestamp(x, tz=self.tz)
        # Non-overlapping identity check (left operand type: "Timestamp",
        # right operand type: "NaTType")
        if ts is not NaT:  # type: ignore[comparison-overlap]
            # GH#41586
            # do this instead of passing to the constructor to avoid FutureWarning
            ts._set_freq(self.freq)
        return ts

    @property
    # error: Return type "Union[dtype, DatetimeTZDtype]" of "dtype"
    # incompatible with return type "ExtensionDtype" in supertype
    # "ExtensionArray"
    def dtype(self) -> np.dtype | DatetimeTZDtype:  # type: ignore[override]
        """
        The dtype for the DatetimeArray.

        .. warning::

           A future version of pandas will change dtype to never be a
           ``numpy.dtype``. Instead, :attr:`DatetimeArray.dtype` will
           always be an instance of an ``ExtensionDtype`` subclass.

        Returns
        -------
        numpy.dtype or DatetimeTZDtype
            If the values are tz-naive, then ``np.dtype('datetime64[ns]')``
            is returned.

            If the values are tz-aware, then the ``DatetimeTZDtype``
            is returned.
        """
        return self._dtype

    @property
    def tz(self) -> tzinfo | None:
        """
        Return timezone, if any.

        Returns
        -------
        datetime.tzinfo, pytz.tzinfo.BaseTZInfo, dateutil.tz.tz.tzfile, or None
            Returns None when the array is tz-naive.
        """
        # GH 18595
        return getattr(self.dtype, "tz", None)

    @tz.setter
    def tz(self, value):
        # GH 3746: Prevent localizing or converting the index by setting tz
        raise AttributeError(
            "Cannot directly set timezone. Use tz_localize() "
            "or tz_convert() as appropriate"
        )

    @property
    def tzinfo(self) -> tzinfo | None:
        """
        Alias for tz attribute
        """
        return self.tz

    @property  # NB: override with cache_readonly in immutable subclasses
    def is_normalized(self) -> bool:
        """
        Returns True if all of the dates are at midnight ("no time")
        """
        return is_date_array_normalized(self.asi8, self.tz)

    @property  # NB: override with cache_readonly in immutable subclasses
    def _resolution_obj(self) -> Resolution:
        return get_resolution(self.asi8, self.tz)

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def __array__(self, dtype=None) -> np.ndarray:
        if dtype is None and self.tz:
            # The default for tz-aware is object, to preserve tz info
            dtype = object

        return super().__array__(dtype=dtype)

    def __iter__(self):
        """
        Return an iterator over the boxed values

        Yields
        ------
        tstamp : Timestamp
        """
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            # convert in chunks of 10k for efficiency
            data = self.asi8
            length = len(self)
            chunksize = 10000
            chunks = (length // chunksize) + 1

            with warnings.catch_warnings():
                # filter out warnings about Timestamp.freq
                warnings.filterwarnings("ignore", category=FutureWarning)

                for i in range(chunks):
                    start_i = i * chunksize
                    end_i = min((i + 1) * chunksize, length)
                    converted = ints_to_pydatetime(
                        data[start_i:end_i], tz=self.tz, freq=self.freq, box="timestamp"
                    )
                    yield from converted

    def astype(self, dtype, copy: bool = True):
        # We handle
        #   --> datetime
        #   --> period
        # DatetimeLikeArrayMixin Super handles the rest.
        dtype = pandas_dtype(dtype)

        if is_dtype_equal(dtype, self.dtype):
            if copy:
                return self.copy()
            return self

        elif is_datetime64_ns_dtype(dtype):
            return astype_dt64_to_dt64tz(self, dtype, copy, via_utc=False)

        elif self.tz is None and is_datetime64_dtype(dtype) and dtype != self.dtype:
            # unit conversion e.g. datetime64[s]
            return self._ndarray.astype(dtype)

        elif is_period_dtype(dtype):
            return self.to_period(freq=dtype.freq)
        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy)

    # -----------------------------------------------------------------
    # Rendering Methods

    @dtl.ravel_compat
    def _format_native_types(
        self, na_rep="NaT", date_format=None, **kwargs
    ) -> np.ndarray:
        from pandas.io.formats.format import get_format_datetime64_from_values

        fmt = get_format_datetime64_from_values(self, date_format)

        return tslib.format_array_from_datetime(
            self.asi8, tz=self.tz, format=fmt, na_rep=na_rep
        )

    # -----------------------------------------------------------------
    # Comparison Methods

    def _has_same_tz(self, other) -> bool:

        # vzone shouldn't be None if value is non-datetime like
        if isinstance(other, np.datetime64):
            # convert to Timestamp as np.datetime64 doesn't have tz attr
            other = Timestamp(other)

        if not hasattr(other, "tzinfo"):
            return False
        other_tz = other.tzinfo
        return timezones.tz_compare(self.tzinfo, other_tz)

    def _assert_tzawareness_compat(self, other) -> None:
        # adapted from _Timestamp._assert_tzawareness_compat
        other_tz = getattr(other, "tzinfo", None)
        other_dtype = getattr(other, "dtype", None)

        if is_datetime64tz_dtype(other_dtype):
            # Get tzinfo from Series dtype
            other_tz = other.dtype.tz
        if other is NaT:
            # pd.NaT quacks both aware and naive
            pass
        elif self.tz is None:
            if other_tz is not None:
                raise TypeError(
                    "Cannot compare tz-naive and tz-aware datetime-like objects."
                )
        elif other_tz is None:
            raise TypeError(
                "Cannot compare tz-naive and tz-aware datetime-like objects"
            )

    # -----------------------------------------------------------------
    # Arithmetic Methods

    def _sub_datetime_arraylike(self, other):
        """subtract DatetimeArray/Index or ndarray[datetime64]"""
        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")

        if isinstance(other, np.ndarray):
            assert is_datetime64_dtype(other)
            other = type(self)(other)

        if not self._has_same_tz(other):
            # require tz compat
            raise TypeError(
                f"{type(self).__name__} subtraction must have the same "
                "timezones or no timezones"
            )

        self_i8 = self.asi8
        other_i8 = other.asi8
        arr_mask = self._isnan | other._isnan
        new_values = checked_add_with_arr(self_i8, -other_i8, arr_mask=arr_mask)
        if self._hasnans or other._hasnans:
            np.putmask(new_values, arr_mask, iNaT)
        return new_values.view("timedelta64[ns]")

    def _add_offset(self, offset) -> DatetimeArray:
        if self.ndim == 2:
            return self.ravel()._add_offset(offset).reshape(self.shape)

        assert not isinstance(offset, Tick)
        try:
            if self.tz is not None:
                values = self.tz_localize(None)
            else:
                values = self
            result = offset._apply_array(values).view("M8[ns]")
            result = DatetimeArray._simple_new(result)
            result = result.tz_localize(self.tz)

        except NotImplementedError:
            warnings.warn(
                "Non-vectorized DateOffset being applied to Series or DatetimeIndex",
                PerformanceWarning,
            )
            result = self.astype("O") + offset
            if not len(self):
                # GH#30336 _from_sequence won't be able to infer self.tz
                return type(self)._from_sequence(result).tz_localize(self.tz)

        return type(self)._from_sequence(result)

    def _sub_datetimelike_scalar(self, other):
        # subtract a datetime from myself, yielding a ndarray[timedelta64[ns]]
        assert isinstance(other, (datetime, np.datetime64))
        assert other is not NaT
        other = Timestamp(other)
        # error: Non-overlapping identity check (left operand type: "Timestamp",
        # right operand type: "NaTType")
        if other is NaT:  # type: ignore[comparison-overlap]
            return self - NaT

        if not self._has_same_tz(other):
            # require tz compat
            raise TypeError(
                "Timestamp subtraction must have the same timezones or no timezones"
            )

        i8 = self.asi8
        result = checked_add_with_arr(i8, -other.value, arr_mask=self._isnan)
        result = self._maybe_mask_results(result)
        return result.view("timedelta64[ns]")

    # -----------------------------------------------------------------
    # Timezone Conversion and Localization Methods

    def _local_timestamps(self) -> np.ndarray:
        """
        Convert to an i8 (unix-like nanosecond timestamp) representation
        while keeping the local timezone and not using UTC.
        This is used to calculate time-of-day information as if the timestamps
        were timezone-naive.
        """
        if self.tz is None or timezones.is_utc(self.tz):
            return self.asi8
        return tzconversion.tz_convert_from_utc(self.asi8, self.tz)

    def tz_convert(self, tz) -> DatetimeArray:
        """
        Convert tz-aware Datetime Array/Index from one time zone to another.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time. Corresponding timestamps would be converted
            to this time zone of the Datetime Array/Index. A `tz` of None will
            convert to UTC and remove the timezone information.

        Returns
        -------
        Array or Index

        Raises
        ------
        TypeError
            If Datetime Array/Index is tz-naive.

        See Also
        --------
        DatetimeIndex.tz : A timezone that has a variable offset from UTC.
        DatetimeIndex.tz_localize : Localize tz-naive DatetimeIndex to a
            given time zone, or remove timezone from a tz-aware DatetimeIndex.

        Examples
        --------
        With the `tz` parameter, we can change the DatetimeIndex
        to other time zones:

        >>> dti = pd.date_range(start='2014-08-01 09:00',
        ...                     freq='H', periods=3, tz='Europe/Berlin')

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                      dtype='datetime64[ns, Europe/Berlin]', freq='H')

        >>> dti.tz_convert('US/Central')
        DatetimeIndex(['2014-08-01 02:00:00-05:00',
                       '2014-08-01 03:00:00-05:00',
                       '2014-08-01 04:00:00-05:00'],
                      dtype='datetime64[ns, US/Central]', freq='H')

        With the ``tz=None``, we can remove the timezone (after converting
        to UTC if necessary):

        >>> dti = pd.date_range(start='2014-08-01 09:00', freq='H',
        ...                     periods=3, tz='Europe/Berlin')

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                        dtype='datetime64[ns, Europe/Berlin]', freq='H')

        >>> dti.tz_convert(None)
        DatetimeIndex(['2014-08-01 07:00:00',
                       '2014-08-01 08:00:00',
                       '2014-08-01 09:00:00'],
                        dtype='datetime64[ns]', freq='H')
        """
        tz = timezones.maybe_get_tz(tz)

        if self.tz is None:
            # tz naive, use tz_localize
            raise TypeError(
                "Cannot convert tz-naive timestamps, use tz_localize to localize"
            )

        # No conversion since timestamps are all UTC to begin with
        dtype = tz_to_dtype(tz)
        return self._simple_new(self._ndarray, dtype=dtype, freq=self.freq)

    @dtl.ravel_compat
    def tz_localize(self, tz, ambiguous="raise", nonexistent="raise") -> DatetimeArray:
        """
        Localize tz-naive Datetime Array/Index to tz-aware
        Datetime Array/Index.

        This method takes a time zone (tz) naive Datetime Array/Index object
        and makes this time zone aware. It does not move the time to another
        time zone.

        This method can also be used to do the inverse -- to create a time
        zone unaware object from an aware object. To that end, pass `tz=None`.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone to convert timestamps to. Passing ``None`` will
            remove the time zone information preserving local time.
        ambiguous : 'infer', 'NaT', bool array, default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False signifies a
              non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise an AmbiguousTimeError if there are ambiguous
              times.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, \
default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - 'shift_forward' will shift the nonexistent time forward to the
              closest existing time
            - 'shift_backward' will shift the nonexistent time backward to the
              closest existing time
            - 'NaT' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        Same type as self
            Array/Index converted to the specified time zone.

        Raises
        ------
        TypeError
            If the Datetime Array/Index is tz-aware and tz is not None.

        See Also
        --------
        DatetimeIndex.tz_convert : Convert tz-aware DatetimeIndex from
            one time zone to another.

        Examples
        --------
        >>> tz_naive = pd.date_range('2018-03-01 09:00', periods=3)
        >>> tz_naive
        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',
                       '2018-03-03 09:00:00'],
                      dtype='datetime64[ns]', freq='D')

        Localize DatetimeIndex in US/Eastern time zone:

        >>> tz_aware = tz_naive.tz_localize(tz='US/Eastern')
        >>> tz_aware
        DatetimeIndex(['2018-03-01 09:00:00-05:00',
                       '2018-03-02 09:00:00-05:00',
                       '2018-03-03 09:00:00-05:00'],
                      dtype='datetime64[ns, US/Eastern]', freq=None)

        With the ``tz=None``, we can remove the time zone information
        while keeping the local time (not converted to UTC):

        >>> tz_aware.tz_localize(None)
        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',
                       '2018-03-03 09:00:00'],
                      dtype='datetime64[ns]', freq=None)

        Be careful with DST changes. When there is sequential data, pandas can
        infer the DST time:

        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:30:00',
        ...                               '2018-10-28 02:00:00',
        ...                               '2018-10-28 02:30:00',
        ...                               '2018-10-28 02:00:00',
        ...                               '2018-10-28 02:30:00',
        ...                               '2018-10-28 03:00:00',
        ...                               '2018-10-28 03:30:00']))
        >>> s.dt.tz_localize('CET', ambiguous='infer')
        0   2018-10-28 01:30:00+02:00
        1   2018-10-28 02:00:00+02:00
        2   2018-10-28 02:30:00+02:00
        3   2018-10-28 02:00:00+01:00
        4   2018-10-28 02:30:00+01:00
        5   2018-10-28 03:00:00+01:00
        6   2018-10-28 03:30:00+01:00
        dtype: datetime64[ns, CET]

        In some cases, inferring the DST is impossible. In such cases, you can
        pass an ndarray to the ambiguous parameter to set the DST explicitly

        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:20:00',
        ...                               '2018-10-28 02:36:00',
        ...                               '2018-10-28 03:46:00']))
        >>> s.dt.tz_localize('CET', ambiguous=np.array([True, True, False]))
        0   2018-10-28 01:20:00+02:00
        1   2018-10-28 02:36:00+02:00
        2   2018-10-28 03:46:00+01:00
        dtype: datetime64[ns, CET]

        If the DST transition causes nonexistent times, you can shift these
        dates forward or backwards with a timedelta object or `'shift_forward'`
        or `'shift_backwards'`.

        >>> s = pd.to_datetime(pd.Series(['2015-03-29 02:30:00',
        ...                               '2015-03-29 03:30:00']))
        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_forward')
        0   2015-03-29 03:00:00+02:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]

        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_backward')
        0   2015-03-29 01:59:59.999999999+01:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]

        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent=pd.Timedelta('1H'))
        0   2015-03-29 03:30:00+02:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]
        """
        nonexistent_options = ("raise", "NaT", "shift_forward", "shift_backward")
        if nonexistent not in nonexistent_options and not isinstance(
            nonexistent, timedelta
        ):
            raise ValueError(
                "The nonexistent argument must be one of 'raise', "
                "'NaT', 'shift_forward', 'shift_backward' or "
                "a timedelta object"
            )

        if self.tz is not None:
            if tz is None:
                new_dates = tzconversion.tz_convert_from_utc(self.asi8, self.tz)
            else:
                raise TypeError("Already tz-aware, use tz_convert to convert.")
        else:
            tz = timezones.maybe_get_tz(tz)
            # Convert to UTC

            new_dates = tzconversion.tz_localize_to_utc(
                self.asi8, tz, ambiguous=ambiguous, nonexistent=nonexistent
            )
        new_dates = new_dates.view(DT64NS_DTYPE)
        dtype = tz_to_dtype(tz)

        freq = None
        if timezones.is_utc(tz) or (len(self) == 1 and not isna(new_dates[0])):
            # we can preserve freq
            # TODO: Also for fixed-offsets
            freq = self.freq
        elif tz is None and self.tz is None:
            # no-op
            freq = self.freq
        return self._simple_new(new_dates, dtype=dtype, freq=freq)

    # ----------------------------------------------------------------
    # Conversion Methods - Vectorized analogues of Timestamp methods

    def to_pydatetime(self) -> np.ndarray:
        """
        Return Datetime Array/Index as object ndarray of datetime.datetime
        objects.

        Returns
        -------
        datetimes : ndarray[object]
        """
        return ints_to_pydatetime(self.asi8, tz=self.tz)

    def normalize(self) -> DatetimeArray:
        """
        Convert times to midnight.

        The time component of the date-time is converted to midnight i.e.
        00:00:00. This is useful in cases, when the time does not matter.
        Length is unaltered. The timezones are unaffected.

        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on Datetime Array/Index.

        Returns
        -------
        DatetimeArray, DatetimeIndex or Series
            The same type as the original data. Series will have the same
            name and index. DatetimeIndex will have the same name.

        See Also
        --------
        floor : Floor the datetimes to the specified freq.
        ceil : Ceil the datetimes to the specified freq.
        round : Round the datetimes to the specified freq.

        Examples
        --------
        >>> idx = pd.date_range(start='2014-08-01 10:00', freq='H',
        ...                     periods=3, tz='Asia/Calcutta')
        >>> idx
        DatetimeIndex(['2014-08-01 10:00:00+05:30',
                       '2014-08-01 11:00:00+05:30',
                       '2014-08-01 12:00:00+05:30'],
                        dtype='datetime64[ns, Asia/Calcutta]', freq='H')
        >>> idx.normalize()
        DatetimeIndex(['2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30'],
                       dtype='datetime64[ns, Asia/Calcutta]', freq=None)
        """
        new_values = normalize_i8_timestamps(self.asi8, self.tz)
        return type(self)(new_values)._with_freq("infer").tz_localize(self.tz)

    @dtl.ravel_compat
    def to_period(self, freq=None) -> PeriodArray:
        """
        Cast to PeriodArray/Index at a particular frequency.

        Converts DatetimeArray/Index to PeriodArray/Index.

        Parameters
        ----------
        freq : str or Offset, optional
            One of pandas' :ref:`offset strings <timeseries.offset_aliases>`
            or an Offset object. Will be inferred by default.

        Returns
        -------
        PeriodArray/Index

        Raises
        ------
        ValueError
            When converting a DatetimeArray/Index with non-regular values,
            so that a frequency cannot be inferred.

        See Also
        --------
        PeriodIndex: Immutable ndarray holding ordinal values.
        DatetimeIndex.to_pydatetime: Return DatetimeIndex as object.

        Examples
        --------
        >>> df = pd.DataFrame({"y": [1, 2, 3]},
        ...                   index=pd.to_datetime(["2000-03-31 00:00:00",
        ...                                         "2000-05-31 00:00:00",
        ...                                         "2000-08-31 00:00:00"]))
        >>> df.index.to_period("M")
        PeriodIndex(['2000-03', '2000-05', '2000-08'],
                    dtype='period[M]')

        Infer the daily frequency

        >>> idx = pd.date_range("2017-01-01", periods=2)
        >>> idx.to_period()
        PeriodIndex(['2017-01-01', '2017-01-02'],
                    dtype='period[D]')
        """
        from pandas.core.arrays import PeriodArray

        if self.tz is not None:
            warnings.warn(
                "Converting to PeriodArray/Index representation "
                "will drop timezone information.",
                UserWarning,
            )

        if freq is None:
            freq = self.freqstr or self.inferred_freq

            if freq is None:
                raise ValueError(
                    "You must pass a freq argument as current index has none."
                )

            res = get_period_alias(freq)

            #  https://github.com/pandas-dev/pandas/issues/33358
            if res is None:
                res = freq

            freq = res

        return PeriodArray._from_datetime64(self._ndarray, freq, tz=self.tz)

    def to_perioddelta(self, freq) -> TimedeltaArray:
        """
        Calculate TimedeltaArray of difference between index
        values and index converted to PeriodArray at specified
        freq. Used for vectorized offsets.

        Parameters
        ----------
        freq : Period frequency

        Returns
        -------
        TimedeltaArray/Index
        """
        # Deprecaation GH#34853
        warnings.warn(
            "to_perioddelta is deprecated and will be removed in a "
            "future version.  "
            "Use `dtindex - dtindex.to_period(freq).to_timestamp()` instead",
            FutureWarning,
            # stacklevel chosen to be correct for when called from DatetimeIndex
            stacklevel=3,
        )
        from pandas.core.arrays.timedeltas import TimedeltaArray

        i8delta = self.asi8 - self.to_period(freq).to_timestamp().asi8
        m8delta = i8delta.view("m8[ns]")
        return TimedeltaArray(m8delta)

    # -----------------------------------------------------------------
    # Properties - Vectorized Timestamp Properties/Methods

    def month_name(self, locale=None):
        """
        Return the month names of the DateTimeIndex with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the month name.
            Default is English locale.

        Returns
        -------
        Index
            Index of month names.

        Examples
        --------
        >>> idx = pd.date_range(start='2018-01', freq='M', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31'],
                      dtype='datetime64[ns]', freq='M')
        >>> idx.month_name()
        Index(['January', 'February', 'March'], dtype='object')
        """
        values = self._local_timestamps()

        result = fields.get_date_name_field(values, "month_name", locale=locale)
        result = self._maybe_mask_results(result, fill_value=None)
        return result

    def day_name(self, locale=None):
        """
        Return the day names of the DateTimeIndex with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the day name.
            Default is English locale.

        Returns
        -------
        Index
            Index of day names.

        Examples
        --------
        >>> idx = pd.date_range(start='2018-01-01', freq='D', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.day_name()
        Index(['Monday', 'Tuesday', 'Wednesday'], dtype='object')
        """
        values = self._local_timestamps()

        result = fields.get_date_name_field(values, "day_name", locale=locale)
        result = self._maybe_mask_results(result, fill_value=None)
        return result

    @property
    def time(self) -> np.ndarray:
        """
        Returns numpy array of datetime.time. The time part of the Timestamps.
        """
        # If the Timestamps have a timezone that is not UTC,
        # convert them into their i8 representation while
        # keeping their timezone and not using UTC
        timestamps = self._local_timestamps()

        return ints_to_pydatetime(timestamps, box="time")

    @property
    def timetz(self) -> np.ndarray:
        """
        Returns numpy array of datetime.time also containing timezone
        information. The time part of the Timestamps.
        """
        return ints_to_pydatetime(self.asi8, self.tz, box="time")

    @property
    def date(self) -> np.ndarray:
        """
        Returns numpy array of python datetime.date objects (namely, the date
        part of Timestamps without timezone information).
        """
        # If the Timestamps have a timezone that is not UTC,
        # convert them into their i8 representation while
        # keeping their timezone and not using UTC
        timestamps = self._local_timestamps()

        return ints_to_pydatetime(timestamps, box="date")

    def isocalendar(self) -> DataFrame:
        """
        Returns a DataFrame with the year, week, and day calculated according to
        the ISO 8601 standard.

        .. versionadded:: 1.1.0

        Returns
        -------
        DataFrame
            with columns year, week and day

        See Also
        --------
        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,
            week number, and weekday for the given Timestamp object.
        datetime.date.isocalendar : Return a named tuple object with
            three components: year, week and weekday.

        Examples
        --------
        >>> idx = pd.date_range(start='2019-12-29', freq='D', periods=4)
        >>> idx.isocalendar()
                    year  week  day
        2019-12-29  2019    52    7
        2019-12-30  2020     1    1
        2019-12-31  2020     1    2
        2020-01-01  2020     1    3
        >>> idx.isocalendar().week
        2019-12-29    52
        2019-12-30     1
        2019-12-31     1
        2020-01-01     1
        Freq: D, Name: week, dtype: UInt32
        """
        from pandas import DataFrame

        values = self._local_timestamps()
        sarray = fields.build_isocalendar_sarray(values)
        iso_calendar_df = DataFrame(
            sarray, columns=["year", "week", "day"], dtype="UInt32"
        )
        if self._hasnans:
            iso_calendar_df.iloc[self._isnan] = None
        return iso_calendar_df

    @property
    def weekofyear(self):
        """
        The week ordinal of the year.

        .. deprecated:: 1.1.0

        weekofyear and week have been deprecated.
        Please use DatetimeIndex.isocalendar().week instead.
        """
        warnings.warn(
            "weekofyear and week have been deprecated, please use "
            "DatetimeIndex.isocalendar().week instead, which returns "
            "a Series.  To exactly reproduce the behavior of week and "
            "weekofyear and return an Index, you may call "
            "pd.Int64Index(idx.isocalendar().week)",
            FutureWarning,
            stacklevel=3,
        )
        week_series = self.isocalendar().week
        if week_series.hasnans:
            return week_series.to_numpy(dtype="float64", na_value=np.nan)
        return week_series.to_numpy(dtype="int64")

    week = weekofyear

    year = _field_accessor(
        "year",
        "Y",
        """
        The year of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="Y")
        ... )
        >>> datetime_series
        0   2000-12-31
        1   2001-12-31
        2   2002-12-31
        dtype: datetime64[ns]
        >>> datetime_series.dt.year
        0    2000
        1    2001
        2    2002
        dtype: int64
        """,
    )
    month = _field_accessor(
        "month",
        "M",
        """
        The month as January=1, December=12.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="M")
        ... )
        >>> datetime_series
        0   2000-01-31
        1   2000-02-29
        2   2000-03-31
        dtype: datetime64[ns]
        >>> datetime_series.dt.month
        0    1
        1    2
        2    3
        dtype: int64
        """,
    )
    day = _field_accessor(
        "day",
        "D",
        """
        The day of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="D")
        ... )
        >>> datetime_series
        0   2000-01-01
        1   2000-01-02
        2   2000-01-03
        dtype: datetime64[ns]
        >>> datetime_series.dt.day
        0    1
        1    2
        2    3
        dtype: int64
        """,
    )
    hour = _field_accessor(
        "hour",
        "h",
        """
        The hours of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="h")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 01:00:00
        2   2000-01-01 02:00:00
        dtype: datetime64[ns]
        >>> datetime_series.dt.hour
        0    0
        1    1
        2    2
        dtype: int64
        """,
    )
    minute = _field_accessor(
        "minute",
        "m",
        """
        The minutes of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="T")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 00:01:00
        2   2000-01-01 00:02:00
        dtype: datetime64[ns]
        >>> datetime_series.dt.minute
        0    0
        1    1
        2    2
        dtype: int64
        """,
    )
    second = _field_accessor(
        "second",
        "s",
        """
        The seconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="s")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 00:00:01
        2   2000-01-01 00:00:02
        dtype: datetime64[ns]
        >>> datetime_series.dt.second
        0    0
        1    1
        2    2
        dtype: int64
        """,
    )
    microsecond = _field_accessor(
        "microsecond",
        "us",
        """
        The microseconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="us")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00.000000
        1   2000-01-01 00:00:00.000001
        2   2000-01-01 00:00:00.000002
        dtype: datetime64[ns]
        >>> datetime_series.dt.microsecond
        0       0
        1       1
        2       2
        dtype: int64
        """,
    )
    nanosecond = _field_accessor(
        "nanosecond",
        "ns",
        """
        The nanoseconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="ns")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00.000000000
        1   2000-01-01 00:00:00.000000001
        2   2000-01-01 00:00:00.000000002
        dtype: datetime64[ns]
        >>> datetime_series.dt.nanosecond
        0       0
        1       1
        2       2
        dtype: int64
        """,
    )
    _dayofweek_doc = """
    The day of the week with Monday=0, Sunday=6.

    Return the day of the week. It is assumed the week starts on
    Monday, which is denoted by 0 and ends on Sunday which is denoted
    by 6. This method is available on both Series with datetime
    values (using the `dt` accessor) or DatetimeIndex.

    Returns
    -------
    Series or Index
        Containing integers indicating the day number.

    See Also
    --------
    Series.dt.dayofweek : Alias.
    Series.dt.weekday : Alias.
    Series.dt.day_name : Returns the name of the day of the week.

    Examples
    --------
    >>> s = pd.date_range('2016-12-31', '2017-01-08', freq='D').to_series()
    >>> s.dt.dayofweek
    2016-12-31    5
    2017-01-01    6
    2017-01-02    0
    2017-01-03    1
    2017-01-04    2
    2017-01-05    3
    2017-01-06    4
    2017-01-07    5
    2017-01-08    6
    Freq: D, dtype: int64
    """
    day_of_week = _field_accessor("day_of_week", "dow", _dayofweek_doc)
    dayofweek = day_of_week
    weekday = day_of_week

    day_of_year = _field_accessor(
        "dayofyear",
        "doy",
        """
        The ordinal day of the year.
        """,
    )
    dayofyear = day_of_year
    quarter = _field_accessor(
        "quarter",
        "q",
        """
        The quarter of the date.
        """,
    )
    days_in_month = _field_accessor(
        "days_in_month",
        "dim",
        """
        The number of days in the month.
        """,
    )
    daysinmonth = days_in_month
    _is_month_doc = """
        Indicates whether the date is the {first_or_last} day of the month.

        Returns
        -------
        Series or array
            For Series, returns a Series with boolean values.
            For DatetimeIndex, returns a boolean array.

        See Also
        --------
        is_month_start : Return a boolean indicating whether the date
            is the first day of the month.
        is_month_end : Return a boolean indicating whether the date
            is the last day of the month.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> s = pd.Series(pd.date_range("2018-02-27", periods=3))
        >>> s
        0   2018-02-27
        1   2018-02-28
        2   2018-03-01
        dtype: datetime64[ns]
        >>> s.dt.is_month_start
        0    False
        1    False
        2    True
        dtype: bool
        >>> s.dt.is_month_end
        0    False
        1    True
        2    False
        dtype: bool

        >>> idx = pd.date_range("2018-02-27", periods=3)
        >>> idx.is_month_start
        array([False, False, True])
        >>> idx.is_month_end
        array([False, True, False])
    """
    is_month_start = _field_accessor(
        "is_month_start", "is_month_start", _is_month_doc.format(first_or_last="first")
    )

    is_month_end = _field_accessor(
        "is_month_end", "is_month_end", _is_month_doc.format(first_or_last="last")
    )

    is_quarter_start = _field_accessor(
        "is_quarter_start",
        "is_quarter_start",
        """
        Indicator for whether the date is the first day of a quarter.

        Returns
        -------
        is_quarter_start : Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        quarter : Return the quarter of the date.
        is_quarter_end : Similar property for indicating the quarter start.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> df = pd.DataFrame({'dates': pd.date_range("2017-03-30",
        ...                   periods=4)})
        >>> df.assign(quarter=df.dates.dt.quarter,
        ...           is_quarter_start=df.dates.dt.is_quarter_start)
               dates  quarter  is_quarter_start
        0 2017-03-30        1             False
        1 2017-03-31        1             False
        2 2017-04-01        2              True
        3 2017-04-02        2             False

        >>> idx = pd.date_range('2017-03-30', periods=4)
        >>> idx
        DatetimeIndex(['2017-03-30', '2017-03-31', '2017-04-01', '2017-04-02'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_quarter_start
        array([False, False,  True, False])
        """,
    )
    is_quarter_end = _field_accessor(
        "is_quarter_end",
        "is_quarter_end",
        """
        Indicator for whether the date is the last day of a quarter.

        Returns
        -------
        is_quarter_end : Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        quarter : Return the quarter of the date.
        is_quarter_start : Similar property indicating the quarter start.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> df = pd.DataFrame({'dates': pd.date_range("2017-03-30",
        ...                    periods=4)})
        >>> df.assign(quarter=df.dates.dt.quarter,
        ...           is_quarter_end=df.dates.dt.is_quarter_end)
               dates  quarter    is_quarter_end
        0 2017-03-30        1             False
        1 2017-03-31        1              True
        2 2017-04-01        2             False
        3 2017-04-02        2             False

        >>> idx = pd.date_range('2017-03-30', periods=4)
        >>> idx
        DatetimeIndex(['2017-03-30', '2017-03-31', '2017-04-01', '2017-04-02'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_quarter_end
        array([False,  True, False, False])
        """,
    )
    is_year_start = _field_accessor(
        "is_year_start",
        "is_year_start",
        """
        Indicate whether the date is the first day of a year.

        Returns
        -------
        Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        is_year_end : Similar property indicating the last day of the year.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))
        >>> dates
        0   2017-12-30
        1   2017-12-31
        2   2018-01-01
        dtype: datetime64[ns]

        >>> dates.dt.is_year_start
        0    False
        1    False
        2    True
        dtype: bool

        >>> idx = pd.date_range("2017-12-30", periods=3)
        >>> idx
        DatetimeIndex(['2017-12-30', '2017-12-31', '2018-01-01'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_year_start
        array([False, False,  True])
        """,
    )
    is_year_end = _field_accessor(
        "is_year_end",
        "is_year_end",
        """
        Indicate whether the date is the last day of the year.

        Returns
        -------
        Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        is_year_start : Similar property indicating the start of the year.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))
        >>> dates
        0   2017-12-30
        1   2017-12-31
        2   2018-01-01
        dtype: datetime64[ns]

        >>> dates.dt.is_year_end
        0    False
        1     True
        2    False
        dtype: bool

        >>> idx = pd.date_range("2017-12-30", periods=3)
        >>> idx
        DatetimeIndex(['2017-12-30', '2017-12-31', '2018-01-01'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_year_end
        array([False,  True, False])
        """,
    )
    is_leap_year = _field_accessor(
        "is_leap_year",
        "is_leap_year",
        """
        Boolean indicator if the date belongs to a leap year.

        A leap year is a year, which has 366 days (instead of 365) including
        29th of February as an intercalary day.
        Leap years are years which are multiples of four with the exception
        of years divisible by 100 but not by 400.

        Returns
        -------
        Series or ndarray
             Booleans indicating if dates belong to a leap year.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> idx = pd.date_range("2012-01-01", "2015-01-01", freq="Y")
        >>> idx
        DatetimeIndex(['2012-12-31', '2013-12-31', '2014-12-31'],
                      dtype='datetime64[ns]', freq='A-DEC')
        >>> idx.is_leap_year
        array([ True, False, False])

        >>> dates_series = pd.Series(idx)
        >>> dates_series
        0   2012-12-31
        1   2013-12-31
        2   2014-12-31
        dtype: datetime64[ns]
        >>> dates_series.dt.is_leap_year
        0     True
        1    False
        2    False
        dtype: bool
        """,
    )

    def to_julian_date(self) -> np.ndarray:
        """
        Convert Datetime Array to float64 ndarray of Julian Dates.
        0 Julian date is noon January 1, 4713 BC.
        https://en.wikipedia.org/wiki/Julian_day
        """

        # http://mysite.verizon.net/aesir_research/date/jdalg2.htm
        year = np.asarray(self.year)
        month = np.asarray(self.month)
        day = np.asarray(self.day)
        testarr = month < 3
        year[testarr] -= 1
        month[testarr] += 12
        return (
            day
            + np.fix((153 * month - 457) / 5)
            + 365 * year
            + np.floor(year / 4)
            - np.floor(year / 100)
            + np.floor(year / 400)
            + 1_721_118.5
            + (
                self.hour
                + self.minute / 60
                + self.second / 3600
                + self.microsecond / 3600 / 10 ** 6
                + self.nanosecond / 3600 / 10 ** 9
            )
            / 24
        )

    # -----------------------------------------------------------------
    # Reductions

    def std(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # Because std is translation-invariant, we can get self.std
        #  by calculating (self - Timestamp(0)).std, and we can do it
        #  without creating a copy by using a view on self._ndarray
        from pandas.core.arrays import TimedeltaArray

        tda = TimedeltaArray(self._ndarray.view("i8"))
        return tda.std(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, skipna=skipna
        )


# -------------------------------------------------------------------
# Constructor Helpers


@overload
def sequence_to_datetimes(
    data, allow_object: Literal[False] = ..., require_iso8601: bool = ...
) -> DatetimeArray:
    ...


@overload
def sequence_to_datetimes(
    data, allow_object: Literal[True] = ..., require_iso8601: bool = ...
) -> np.ndarray | DatetimeArray:
    ...


def sequence_to_datetimes(
    data, allow_object: bool = False, require_iso8601: bool = False
) -> np.ndarray | DatetimeArray:
    """
    Parse/convert the passed data to either DatetimeArray or np.ndarray[object].
    """
    result, tz, freq = sequence_to_dt64ns(
        data,
        allow_object=allow_object,
        allow_mixed=True,
        require_iso8601=require_iso8601,
    )
    if result.dtype == object:
        return result

    dtype = tz_to_dtype(tz)
    dta = DatetimeArray._simple_new(result, freq=freq, dtype=dtype)
    return dta


def sequence_to_dt64ns(
    data,
    dtype=None,
    copy=False,
    tz=None,
    dayfirst=False,
    yearfirst=False,
    ambiguous="raise",
    *,
    allow_object: bool = False,
    allow_mixed: bool = False,
    require_iso8601: bool = False,
):
    """
    Parameters
    ----------
    data : list-like
    dtype : dtype, str, or None, default None
    copy : bool, default False
    tz : tzinfo, str, or None, default None
    dayfirst : bool, default False
    yearfirst : bool, default False
    ambiguous : str, bool, or arraylike, default 'raise'
        See pandas._libs.tslibs.tzconversion.tz_localize_to_utc.
    allow_object : bool, default False
        Whether to return an object-dtype ndarray instead of raising if the
        data contains more than one timezone.
    allow_mixed : bool, default False
        Interpret integers as timestamps when datetime objects are also present.
    require_iso8601 : bool, default False
        Only consider ISO-8601 formats when parsing strings.

    Returns
    -------
    result : numpy.ndarray
        The sequence converted to a numpy array with dtype ``datetime64[ns]``.
    tz : tzinfo or None
        Either the user-provided tzinfo or one inferred from the data.
    inferred_freq : Tick or None
        The inferred frequency of the sequence.

    Raises
    ------
    TypeError : PeriodDType data is passed
    """

    inferred_freq = None

    dtype = _validate_dt64_dtype(dtype)
    tz = timezones.maybe_get_tz(tz)

    # if dtype has an embedded tz, capture it
    tz = validate_tz_from_dtype(dtype, tz)

    if not hasattr(data, "dtype"):
        # e.g. list, tuple
        if np.ndim(data) == 0:
            # i.e. generator
            data = list(data)
        data = np.asarray(data)
        copy = False
    elif isinstance(data, ABCMultiIndex):
        raise TypeError("Cannot create a DatetimeArray from a MultiIndex.")
    else:
        data = extract_array(data, extract_numpy=True)

    if isinstance(data, IntegerArray):
        data = data.to_numpy("int64", na_value=iNaT)
    elif not isinstance(data, (np.ndarray, ExtensionArray)):
        # GH#24539 e.g. xarray, dask object
        data = np.asarray(data)

    if isinstance(data, DatetimeArray):
        inferred_freq = data.freq

    # By this point we are assured to have either a numpy array or Index
    data, copy = maybe_convert_dtype(data, copy)
    data_dtype = getattr(data, "dtype", None)

    if (
        is_object_dtype(data_dtype)
        or is_string_dtype(data_dtype)
        or is_sparse(data_dtype)
    ):
        # TODO: We do not have tests specific to string-dtypes,
        #  also complex or categorical or other extension
        copy = False
        if lib.infer_dtype(data, skipna=False) == "integer":
            data = data.astype(np.int64)
        else:
            # data comes back here as either i8 to denote UTC timestamps
            #  or M8[ns] to denote wall times
            data, inferred_tz = objects_to_datetime64ns(
                data,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                allow_object=allow_object,
                allow_mixed=allow_mixed,
                require_iso8601=require_iso8601,
            )
            if tz and inferred_tz:
                #  two timezones: convert to intended from base UTC repr
                if data.dtype == "i8":
                    # GH#42505
                    # by convention, these are _already_ UTC, e.g
                    return data.view(DT64NS_DTYPE), tz, None

                utc_vals = tzconversion.tz_convert_from_utc(data.view("i8"), tz)
                data = utc_vals.view(DT64NS_DTYPE)
            elif inferred_tz:
                tz = inferred_tz
            elif allow_object and data.dtype == object:
                # We encountered mixed-timezones.
                return data, None, None

        data_dtype = data.dtype

    # `data` may have originally been a Categorical[datetime64[ns, tz]],
    # so we need to handle these types.
    if is_datetime64tz_dtype(data_dtype):
        # DatetimeArray -> ndarray
        tz = _maybe_infer_tz(tz, data.tz)
        result = data._ndarray

    elif is_datetime64_dtype(data_dtype):
        # tz-naive DatetimeArray or ndarray[datetime64]
        data = getattr(data, "_ndarray", data)
        if data.dtype != DT64NS_DTYPE:
            data = conversion.ensure_datetime64ns(data)
            copy = False

        if tz is not None:
            # Convert tz-naive to UTC
            tz = timezones.maybe_get_tz(tz)
            data = tzconversion.tz_localize_to_utc(
                data.view("i8"), tz, ambiguous=ambiguous
            )
            data = data.view(DT64NS_DTYPE)

        assert data.dtype == DT64NS_DTYPE, data.dtype
        result = data

    else:
        # must be integer dtype otherwise
        # assume this data are epoch timestamps
        if tz:
            tz = timezones.maybe_get_tz(tz)

        if data.dtype != INT64_DTYPE:
            data = data.astype(np.int64, copy=False)
        result = data.view(DT64NS_DTYPE)

    if copy:
        result = result.copy()

    assert isinstance(result, np.ndarray), type(result)
    assert result.dtype == "M8[ns]", result.dtype

    # We have to call this again after possibly inferring a tz above
    validate_tz_from_dtype(dtype, tz)

    return result, tz, inferred_freq


def objects_to_datetime64ns(
    data: np.ndarray,
    dayfirst,
    yearfirst,
    utc=False,
    errors="raise",
    require_iso8601: bool = False,
    allow_object: bool = False,
    allow_mixed: bool = False,
):
    """
    Convert data to array of timestamps.

    Parameters
    ----------
    data : np.ndarray[object]
    dayfirst : bool
    yearfirst : bool
    utc : bool, default False
        Whether to convert timezone-aware timestamps to UTC.
    errors : {'raise', 'ignore', 'coerce'}
    require_iso8601 : bool, default False
    allow_object : bool
        Whether to return an object-dtype ndarray instead of raising if the
        data contains more than one timezone.
    allow_mixed : bool, default False
        Interpret integers as timestamps when datetime objects are also present.

    Returns
    -------
    result : ndarray
        np.int64 dtype if returned values represent UTC timestamps
        np.datetime64[ns] if returned values represent wall times
        object if mixed timezones
    inferred_tz : tzinfo or None

    Raises
    ------
    ValueError : if data cannot be converted to datetimes
    """
    assert errors in ["raise", "ignore", "coerce"]

    # if str-dtype, convert
    data = np.array(data, copy=False, dtype=np.object_)

    flags = data.flags
    order: Literal["F", "C"] = "F" if flags.f_contiguous else "C"
    try:
        result, tz_parsed = tslib.array_to_datetime(
            data.ravel("K"),
            errors=errors,
            utc=utc,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            require_iso8601=require_iso8601,
            allow_mixed=allow_mixed,
        )
        result = result.reshape(data.shape, order=order)
    except ValueError as err:
        try:
            values, tz_parsed = conversion.datetime_to_datetime64(data.ravel("K"))
            # If tzaware, these values represent unix timestamps, so we
            #  return them as i8 to distinguish from wall times
            values = values.reshape(data.shape, order=order)
            return values.view("i8"), tz_parsed
        except (ValueError, TypeError):
            raise err

    if tz_parsed is not None:
        # We can take a shortcut since the datetime64 numpy array
        #  is in UTC
        # Return i8 values to denote unix timestamps
        return result.view("i8"), tz_parsed
    elif is_datetime64_dtype(result):
        # returning M8[ns] denotes wall-times; since tz is None
        #  the distinction is a thin one
        return result, tz_parsed
    elif is_object_dtype(result):
        # GH#23675 when called via `pd.to_datetime`, returning an object-dtype
        #  array is allowed.  When called via `pd.DatetimeIndex`, we can
        #  only accept datetime64 dtype, so raise TypeError if object-dtype
        #  is returned, as that indicates the values can be recognized as
        #  datetimes but they have conflicting timezones/awareness
        if allow_object:
            return result, tz_parsed
        raise TypeError(result)
    else:  # pragma: no cover
        # GH#23675 this TypeError should never be hit, whereas the TypeError
        #  in the object-dtype branch above is reachable.
        raise TypeError(result)


def maybe_convert_dtype(data, copy: bool):
    """
    Convert data based on dtype conventions, issuing deprecation warnings
    or errors where appropriate.

    Parameters
    ----------
    data : np.ndarray or pd.Index
    copy : bool

    Returns
    -------
    data : np.ndarray or pd.Index
    copy : bool

    Raises
    ------
    TypeError : PeriodDType data is passed
    """
    if not hasattr(data, "dtype"):
        # e.g. collections.deque
        return data, copy

    if is_float_dtype(data.dtype):
        # Note: we must cast to datetime64[ns] here in order to treat these
        #  as wall-times instead of UTC timestamps.
        data = data.astype(DT64NS_DTYPE)
        copy = False
        # TODO: deprecate this behavior to instead treat symmetrically
        #  with integer dtypes.  See discussion in GH#23675

    elif is_timedelta64_dtype(data.dtype) or is_bool_dtype(data.dtype):
        # GH#29794 enforcing deprecation introduced in GH#23539
        raise TypeError(f"dtype {data.dtype} cannot be converted to datetime64[ns]")
    elif is_period_dtype(data.dtype):
        # Note: without explicitly raising here, PeriodIndex
        #  test_setops.test_join_does_not_recur fails
        raise TypeError(
            "Passing PeriodDtype data is invalid. Use `data.to_timestamp()` instead"
        )

    elif is_categorical_dtype(data.dtype):
        # GH#18664 preserve tz in going DTI->Categorical->DTI
        # TODO: cases where we need to do another pass through this func,
        #  e.g. the categories are timedelta64s
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False

    elif is_extension_array_dtype(data.dtype) and not is_datetime64tz_dtype(data.dtype):
        # Includes categorical
        # TODO: We have no tests for these
        data = np.array(data, dtype=np.object_)
        copy = False

    return data, copy


# -------------------------------------------------------------------
# Validation and Inference


def _maybe_infer_tz(tz: tzinfo | None, inferred_tz: tzinfo | None) -> tzinfo | None:
    """
    If a timezone is inferred from data, check that it is compatible with
    the user-provided timezone, if any.

    Parameters
    ----------
    tz : tzinfo or None
    inferred_tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if both timezones are present but do not match
    """
    if tz is None:
        tz = inferred_tz
    elif inferred_tz is None:
        pass
    elif not timezones.tz_compare(tz, inferred_tz):
        raise TypeError(
            f"data is already tz-aware {inferred_tz}, unable to "
            f"set specified tz: {tz}"
        )
    return tz


def _validate_dt64_dtype(dtype):
    """
    Check that a dtype, if passed, represents either a numpy datetime64[ns]
    dtype or a pandas DatetimeTZDtype.

    Parameters
    ----------
    dtype : object

    Returns
    -------
    dtype : None, numpy.dtype, or DatetimeTZDtype

    Raises
    ------
    ValueError : invalid dtype

    Notes
    -----
    Unlike validate_tz_from_dtype, this does _not_ allow non-existent
    tz errors to go through
    """
    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, np.dtype("M8")):
            # no precision, disallowed GH#24806
            msg = (
                "Passing in 'datetime64' dtype with no precision is not allowed. "
                "Please pass in 'datetime64[ns]' instead."
            )
            raise ValueError(msg)

        if (isinstance(dtype, np.dtype) and dtype != DT64NS_DTYPE) or not isinstance(
            dtype, (np.dtype, DatetimeTZDtype)
        ):
            raise ValueError(
                f"Unexpected value for 'dtype': '{dtype}'. "
                "Must be 'datetime64[ns]' or DatetimeTZDtype'."
            )
    return dtype


def validate_tz_from_dtype(dtype, tz: tzinfo | None) -> tzinfo | None:
    """
    If the given dtype is a DatetimeTZDtype, extract the implied
    tzinfo object from it and check that it does not conflict with the given
    tz.

    Parameters
    ----------
    dtype : dtype, str
    tz : None, tzinfo

    Returns
    -------
    tz : consensus tzinfo

    Raises
    ------
    ValueError : on tzinfo mismatch
    """
    if dtype is not None:
        if isinstance(dtype, str):
            try:
                dtype = DatetimeTZDtype.construct_from_string(dtype)
            except TypeError:
                # Things like `datetime64[ns]`, which is OK for the
                # constructors, but also nonsense, which should be validated
                # but not by us. We *do* allow non-existent tz errors to
                # go through
                pass
        dtz = getattr(dtype, "tz", None)
        if dtz is not None:
            if tz is not None and not timezones.tz_compare(tz, dtz):
                raise ValueError("cannot supply both a tz and a dtype with a tz")
            tz = dtz

        if tz is not None and is_datetime64_dtype(dtype):
            # We also need to check for the case where the user passed a
            #  tz-naive dtype (i.e. datetime64[ns])
            if tz is not None and not timezones.tz_compare(tz, dtz):
                raise ValueError(
                    "cannot supply both a tz and a "
                    "timezone-naive dtype (i.e. datetime64[ns])"
                )

    return tz


def _infer_tz_from_endpoints(
    start: Timestamp, end: Timestamp, tz: tzinfo | None
) -> tzinfo | None:
    """
    If a timezone is not explicitly given via `tz`, see if one can
    be inferred from the `start` and `end` endpoints.  If more than one
    of these inputs provides a timezone, require that they all agree.

    Parameters
    ----------
    start : Timestamp
    end : Timestamp
    tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if start and end timezones do not agree
    """
    try:
        inferred_tz = timezones.infer_tzinfo(start, end)
    except AssertionError as err:
        # infer_tzinfo raises AssertionError if passed mismatched timezones
        raise TypeError(
            "Start and end cannot both be tz-aware with different timezones"
        ) from err

    inferred_tz = timezones.maybe_get_tz(inferred_tz)
    tz = timezones.maybe_get_tz(tz)

    if tz is not None and inferred_tz is not None:
        if not timezones.tz_compare(inferred_tz, tz):
            raise AssertionError("Inferred time zone not equal to passed time zone")

    elif inferred_tz is not None:
        tz = inferred_tz

    return tz


def _maybe_normalize_endpoints(
    start: Timestamp | None, end: Timestamp | None, normalize: bool
):
    _normalized = True

    if start is not None:
        if normalize:
            start = start.normalize()
            _normalized = True
        else:
            _normalized = _normalized and start.time() == _midnight

    if end is not None:
        if normalize:
            end = end.normalize()
            _normalized = True
        else:
            _normalized = _normalized and end.time() == _midnight

    return start, end, _normalized


def _maybe_localize_point(ts, is_none, is_not_none, freq, tz, ambiguous, nonexistent):
    """
    Localize a start or end Timestamp to the timezone of the corresponding
    start or end Timestamp

    Parameters
    ----------
    ts : start or end Timestamp to potentially localize
    is_none : argument that should be None
    is_not_none : argument that should not be None
    freq : Tick, DateOffset, or None
    tz : str, timezone object or None
    ambiguous: str, localization behavior for ambiguous times
    nonexistent: str, localization behavior for nonexistent times

    Returns
    -------
    ts : Timestamp
    """
    # Make sure start and end are timezone localized if:
    # 1) freq = a Timedelta-like frequency (Tick)
    # 2) freq = None i.e. generating a linspaced range
    if is_none is None and is_not_none is not None:
        # Note: We can't ambiguous='infer' a singular ambiguous time; however,
        # we have historically defaulted ambiguous=False
        ambiguous = ambiguous if ambiguous != "infer" else False
        localize_args = {"ambiguous": ambiguous, "nonexistent": nonexistent, "tz": None}
        if isinstance(freq, Tick) or freq is None:
            localize_args["tz"] = tz
        ts = ts.tz_localize(**localize_args)
    return ts


def generate_range(start=None, end=None, periods=None, offset=BDay()):
    """
    Generates a sequence of dates corresponding to the specified time
    offset. Similar to dateutil.rrule except uses pandas DateOffset
    objects to represent time increments.

    Parameters
    ----------
    start : datetime, (default None)
    end : datetime, (default None)
    periods : int, (default None)
    offset : DateOffset, (default BDay())

    Notes
    -----
    * This method is faster for generating weekdays than dateutil.rrule
    * At least two of (start, end, periods) must be specified.
    * If both start and end are specified, the returned dates will
    satisfy start <= date <= end.

    Returns
    -------
    dates : generator object
    """
    offset = to_offset(offset)

    start = Timestamp(start)
    start = start if start is not NaT else None
    end = Timestamp(end)
    end = end if end is not NaT else None

    if start and not offset.is_on_offset(start):
        start = offset.rollforward(start)

    elif end and not offset.is_on_offset(end):
        end = offset.rollback(end)

    if periods is None and end < start and offset.n >= 0:
        end = None
        periods = 0

    if end is None:
        end = start + (periods - 1) * offset

    if start is None:
        start = end - (periods - 1) * offset

    cur = start
    if offset.n >= 0:
        while cur <= end:
            yield cur

            if cur == end:
                # GH#24252 avoid overflows by not performing the addition
                # in offset.apply unless we have to
                break

            # faster than cur + offset
            next_date = offset.apply(cur)
            if next_date <= cur:
                raise ValueError(f"Offset {offset} did not increment date")
            cur = next_date
    else:
        while cur >= end:
            yield cur

            if cur == end:
                # GH#24252 avoid overflows by not performing the addition
                # in offset.apply unless we have to
                break

            # faster than cur + offset
            next_date = offset.apply(cur)
            if next_date >= cur:
                raise ValueError(f"Offset {offset} did not decrement date")
            cur = next_date
