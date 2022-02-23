from __future__ import annotations

from datetime import timedelta
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Sequence,
)

import numpy as np

from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Timedelta,
    delta_to_nanoseconds,
    dt64arr_to_periodarr as c_dt64arr_to_periodarr,
    iNaT,
    parsing,
    period as libperiod,
    to_offset,
)
from pandas._libs.tslibs.dtypes import FreqGroup
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import (
    Tick,
    delta_to_tick,
)
from pandas._libs.tslibs.period import (
    DIFFERENT_FREQ,
    IncompatibleFrequency,
    Period,
    get_period_field_arr,
    period_asfreq_arr,
)
from pandas._typing import (
    AnyArrayLike,
    Dtype,
    NpDtype,
)
from pandas.util._decorators import (
    cache_readonly,
    doc,
)

from pandas.core.dtypes.common import (
    TD64NS_DTYPE,
    ensure_object,
    is_datetime64_dtype,
    is_dtype_equal,
    is_float_dtype,
    is_integer_dtype,
    is_period_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCPeriodIndex,
    ABCSeries,
    ABCTimedeltaArray,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

import pandas.core.algorithms as algos
from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com

if TYPE_CHECKING:
    from pandas.core.arrays import DatetimeArray

_shared_doc_kwargs = {
    "klass": "PeriodArray",
}


def _field_accessor(name: str, docstring=None):
    def f(self):
        base = self.freq._period_dtype_code
        result = get_period_field_arr(name, self.asi8, base)
        return result

    f.__name__ = name
    f.__doc__ = docstring
    return property(f)


class PeriodArray(dtl.DatelikeOps):
    """
    Pandas ExtensionArray for storing Period data.

    Users should use :func:`~pandas.period_array` to create new instances.
    Alternatively, :func:`~pandas.array` can be used to create new instances
    from a sequence of Period scalars.

    Parameters
    ----------
    values : Union[PeriodArray, Series[period], ndarray[int], PeriodIndex]
        The data to store. These should be arrays that can be directly
        converted to ordinals without inference or copy (PeriodArray,
        ndarray[int64]), or a box around such an array (Series[period],
        PeriodIndex).
    dtype : PeriodDtype, optional
        A PeriodDtype instance from which to extract a `freq`. If both
        `freq` and `dtype` are specified, then the frequencies must match.
    freq : str or DateOffset
        The `freq` to use for the array. Mostly applicable when `values`
        is an ndarray of integers, when `freq` is required. When `values`
        is a PeriodArray (or box around), it's checked that ``values.freq``
        matches `freq`.
    copy : bool, default False
        Whether to copy the ordinals before storing.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    Period: Represents a period of time.
    PeriodIndex : Immutable Index for period data.
    period_range: Create a fixed-frequency PeriodArray.
    array: Construct a pandas array.

    Notes
    -----
    There are two components to a PeriodArray

    - ordinals : integer ndarray
    - freq : pd.tseries.offsets.Offset

    The values are physically stored as a 1-D ndarray of integers. These are
    called "ordinals" and represent some kind of offset from a base.

    The `freq` indicates the span covered by each element of the array.
    All elements in the PeriodArray have the same `freq`.
    """

    # array priority higher than numpy scalars
    __array_priority__ = 1000
    _typ = "periodarray"  # ABCPeriodArray
    _scalar_type = Period
    _recognized_scalars = (Period,)
    _is_recognized_dtype = is_period_dtype
    _infer_matches = ("period",)

    # Names others delegate to us
    _other_ops: list[str] = []
    _bool_ops: list[str] = ["is_leap_year"]
    _object_ops: list[str] = ["start_time", "end_time", "freq"]
    _field_ops: list[str] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "weekofyear",
        "weekday",
        "week",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "qyear",
        "days_in_month",
        "daysinmonth",
    ]
    _datetimelike_ops: list[str] = _field_ops + _object_ops + _bool_ops
    _datetimelike_methods: list[str] = ["strftime", "to_timestamp", "asfreq"]

    _dtype: PeriodDtype

    # --------------------------------------------------------------------
    # Constructors

    def __init__(
        self, values, dtype: Dtype | None = None, freq=None, copy: bool = False
    ):
        freq = validate_dtype_freq(dtype, freq)

        if freq is not None:
            freq = Period._maybe_convert_freq(freq)

        if isinstance(values, ABCSeries):
            values = values._values
            if not isinstance(values, type(self)):
                raise TypeError("Incorrect dtype")

        elif isinstance(values, ABCPeriodIndex):
            values = values._values

        if isinstance(values, type(self)):
            if freq is not None and freq != values.freq:
                raise raise_on_incompatible(values, freq)
            values, freq = values._ndarray, values.freq

        values = np.array(values, dtype="int64", copy=copy)
        if freq is None:
            raise ValueError("freq is not specified and cannot be inferred")
        NDArrayBacked.__init__(self, values, PeriodDtype(freq))

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls,
        values: np.ndarray,
        freq: BaseOffset | None = None,
        dtype: Dtype | None = None,
    ) -> PeriodArray:
        # alias for PeriodArray.__init__
        assertion_msg = "Should be numpy array of type i8"
        assert isinstance(values, np.ndarray) and values.dtype == "i8", assertion_msg
        return cls(values, freq=freq, dtype=dtype)

    @classmethod
    def _from_sequence(
        cls: type[PeriodArray],
        scalars: Sequence[Period | None] | AnyArrayLike,
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> PeriodArray:
        if dtype and isinstance(dtype, PeriodDtype):
            freq = dtype.freq
        else:
            freq = None

        if isinstance(scalars, cls):
            validate_dtype_freq(scalars.dtype, freq)
            if copy:
                scalars = scalars.copy()
            return scalars

        periods = np.asarray(scalars, dtype=object)

        freq = freq or libperiod.extract_freq(periods)
        ordinals = libperiod.extract_ordinals(periods, freq)
        return cls(ordinals, freq=freq)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ) -> PeriodArray:
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_datetime64(cls, data, freq, tz=None) -> PeriodArray:
        """
        Construct a PeriodArray from a datetime64 array

        Parameters
        ----------
        data : ndarray[datetime64[ns], datetime64[ns, tz]]
        freq : str or Tick
        tz : tzinfo, optional

        Returns
        -------
        PeriodArray[freq]
        """
        data, freq = dt64arr_to_periodarr(data, freq, tz)
        return cls(data, freq=freq)

    @classmethod
    def _generate_range(cls, start, end, periods, freq, fields):
        periods = dtl.validate_periods(periods)

        if freq is not None:
            freq = Period._maybe_convert_freq(freq)

        field_count = len(fields)
        if start is not None or end is not None:
            if field_count > 0:
                raise ValueError(
                    "Can either instantiate from fields or endpoints, but not both"
                )
            subarr, freq = _get_ordinal_range(start, end, periods, freq)
        elif field_count > 0:
            subarr, freq = _range_from_fields(freq=freq, **fields)
        else:
            raise ValueError("Not enough parameters to construct Period range")

        return subarr, freq

    # -----------------------------------------------------------------
    # DatetimeLike Interface

    # error: Argument 1 of "_unbox_scalar" is incompatible with supertype
    # "DatetimeLikeArrayMixin"; supertype defines the argument type as
    # "Union[Union[Period, Any, Timedelta], NaTType]"
    def _unbox_scalar(  # type: ignore[override]
        self,
        value: Period | NaTType,
        setitem: bool = False,
    ) -> np.int64:
        if value is NaT:
            # error: Item "Period" of "Union[Period, NaTType]" has no attribute "value"
            return np.int64(value.value)  # type: ignore[union-attr]
        elif isinstance(value, self._scalar_type):
            self._check_compatible_with(value, setitem=setitem)
            return np.int64(value.ordinal)
        else:
            raise ValueError(f"'value' should be a Period. Got '{value}' instead.")

    def _scalar_from_string(self, value: str) -> Period:
        return Period(value, freq=self.freq)

    def _check_compatible_with(self, other, setitem: bool = False):
        if other is NaT:
            return
        self._require_matching_freq(other)

    # --------------------------------------------------------------------
    # Data / Attributes

    @cache_readonly
    def dtype(self) -> PeriodDtype:
        return self._dtype

    # error: Read-only property cannot override read-write property
    @property  # type: ignore[misc]
    def freq(self) -> BaseOffset:
        """
        Return the frequency object for this PeriodArray.
        """
        return self.dtype.freq

    def __array__(self, dtype: NpDtype | None = None) -> np.ndarray:
        if dtype == "i8":
            return self.asi8
        # error: Non-overlapping equality check (left operand type: "Optional[Union[str,
        # dtype[Any]]]", right operand type: "Type[bool]")
        elif dtype == bool:  # type: ignore[comparison-overlap]
            return ~self._isnan

        # This will raise TypeError for non-object dtypes
        return np.array(list(self), dtype=object)

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow

        from pandas.core.arrays._arrow_utils import ArrowPeriodType

        if type is not None:
            if pyarrow.types.is_integer(type):
                return pyarrow.array(self._ndarray, mask=self.isna(), type=type)
            elif isinstance(type, ArrowPeriodType):
                # ensure we have the same freq
                if self.freqstr != type.freq:
                    raise TypeError(
                        "Not supported to convert PeriodArray to array with different "
                        f"'freq' ({self.freqstr} vs {type.freq})"
                    )
            else:
                raise TypeError(
                    f"Not supported to convert PeriodArray to '{type}' type"
                )

        period_type = ArrowPeriodType(self.freqstr)
        storage_array = pyarrow.array(self._ndarray, mask=self.isna(), type="int64")
        return pyarrow.ExtensionArray.from_storage(period_type, storage_array)

    # --------------------------------------------------------------------
    # Vectorized analogues of Period properties

    year = _field_accessor(
        "year",
        """
        The year of the period.
        """,
    )
    month = _field_accessor(
        "month",
        """
        The month as January=1, December=12.
        """,
    )
    day = _field_accessor(
        "day",
        """
        The days of the period.
        """,
    )
    hour = _field_accessor(
        "hour",
        """
        The hour of the period.
        """,
    )
    minute = _field_accessor(
        "minute",
        """
        The minute of the period.
        """,
    )
    second = _field_accessor(
        "second",
        """
        The second of the period.
        """,
    )
    weekofyear = _field_accessor(
        "week",
        """
        The week ordinal of the year.
        """,
    )
    week = weekofyear
    day_of_week = _field_accessor(
        "day_of_week",
        """
        The day of the week with Monday=0, Sunday=6.
        """,
    )
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = day_of_year = _field_accessor(
        "day_of_year",
        """
        The ordinal day of the year.
        """,
    )
    quarter = _field_accessor(
        "quarter",
        """
        The quarter of the date.
        """,
    )
    qyear = _field_accessor("qyear")
    days_in_month = _field_accessor(
        "days_in_month",
        """
        The number of days in the month.
        """,
    )
    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> np.ndarray:
        """
        Logical indicating if the date belongs to a leap year.
        """
        return isleapyear_arr(np.asarray(self.year))

    def to_timestamp(self, freq=None, how: str = "start") -> DatetimeArray:
        """
        Cast to DatetimeArray/Index.

        Parameters
        ----------
        freq : str or DateOffset, optional
            Target frequency. The default is 'D' for week or longer,
            'S' otherwise.
        how : {'s', 'e', 'start', 'end'}
            Whether to use the start or end of the time period being converted.

        Returns
        -------
        DatetimeArray/Index
        """
        from pandas.core.arrays import DatetimeArray

        how = libperiod.validate_end_alias(how)

        end = how == "E"
        if end:
            if freq == "B" or self.freq == "B":
                # roll forward to ensure we land on B date
                adjust = Timedelta(1, "D") - Timedelta(1, "ns")
                return self.to_timestamp(how="start") + adjust
            else:
                adjust = Timedelta(1, "ns")
                return (self + self.freq).to_timestamp(how="start") - adjust

        if freq is None:
            freq = self._get_to_timestamp_base()
            base = freq
        else:
            freq = Period._maybe_convert_freq(freq)
            base = freq._period_dtype_code

        new_parr = self.asfreq(freq, how=how)

        new_data = libperiod.periodarr_to_dt64arr(new_parr.asi8, base)
        return DatetimeArray(new_data)._with_freq("infer")

    # --------------------------------------------------------------------

    def _time_shift(self, periods: int, freq=None) -> PeriodArray:
        """
        Shift each value by `periods`.

        Note this is different from ExtensionArray.shift, which
        shifts the *position* of each element, padding the end with
        missing values.

        Parameters
        ----------
        periods : int
            Number of periods to shift by.
        freq : pandas.DateOffset, pandas.Timedelta, or str
            Frequency increment to shift by.
        """
        if freq is not None:
            raise TypeError(
                "`freq` argument is not supported for "
                f"{type(self).__name__}._time_shift"
            )
        values = self.asi8 + periods * self.freq.n
        if self._hasnans:
            values[self._isnan] = iNaT
        return type(self)(values, freq=self.freq)

    def _box_func(self, x) -> Period | NaTType:
        return Period._from_ordinal(ordinal=x, freq=self.freq)

    @doc(**_shared_doc_kwargs, other="PeriodIndex", other_name="PeriodIndex")
    def asfreq(self, freq=None, how: str = "E") -> PeriodArray:
        """
        Convert the {klass} to the specified frequency `freq`.

        Equivalent to applying :meth:`pandas.Period.asfreq` with the given arguments
        to each :class:`~pandas.Period` in this {klass}.

        Parameters
        ----------
        freq : str
            A frequency.
        how : str {{'E', 'S'}}, default 'E'
            Whether the elements should be aligned to the end
            or start within pa period.

            * 'E', 'END', or 'FINISH' for end,
            * 'S', 'START', or 'BEGIN' for start.

            January 31st ('END') vs. January 1st ('START') for example.

        Returns
        -------
        {klass}
            The transformed {klass} with the new frequency.

        See Also
        --------
        {other}.asfreq: Convert each Period in a {other_name} to the given frequency.
        Period.asfreq : Convert a :class:`~pandas.Period` object to the given frequency.

        Examples
        --------
        >>> pidx = pd.period_range('2010-01-01', '2015-01-01', freq='A')
        >>> pidx
        PeriodIndex(['2010', '2011', '2012', '2013', '2014', '2015'],
        dtype='period[A-DEC]')

        >>> pidx.asfreq('M')
        PeriodIndex(['2010-12', '2011-12', '2012-12', '2013-12', '2014-12',
        '2015-12'], dtype='period[M]')

        >>> pidx.asfreq('M', how='S')
        PeriodIndex(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01',
        '2015-01'], dtype='period[M]')
        """
        how = libperiod.validate_end_alias(how)

        freq = Period._maybe_convert_freq(freq)

        base1 = self.freq._period_dtype_code
        base2 = freq._period_dtype_code

        asi8 = self.asi8
        # self.freq.n can't be negative or 0
        end = how == "E"
        if end:
            ordinal = asi8 + self.freq.n - 1
        else:
            ordinal = asi8

        new_data = period_asfreq_arr(ordinal, base1, base2, end)

        if self._hasnans:
            new_data[self._isnan] = iNaT

        return type(self)(new_data, freq=freq)

    # ------------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        if boxed:
            return str
        return "'{}'".format

    @dtl.ravel_compat
    def _format_native_types(
        self, na_rep="NaT", date_format=None, **kwargs
    ) -> np.ndarray:
        """
        actually format my specific types
        """
        values = self.astype(object)

        if date_format:
            formatter = lambda dt: dt.strftime(date_format)
        else:
            formatter = lambda dt: str(dt)

        if self._hasnans:
            mask = self._isnan
            values[mask] = na_rep
            imask = ~mask
            values[imask] = np.array([formatter(dt) for dt in values[imask]])
        else:
            values = np.array([formatter(dt) for dt in values])
        return values

    # ------------------------------------------------------------------

    def astype(self, dtype, copy: bool = True):
        # We handle Period[T] -> Period[U]
        # Our parent handles everything else.
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, self._dtype):
            if not copy:
                return self
            else:
                return self.copy()
        if is_period_dtype(dtype):
            return self.asfreq(dtype.freq)
        return super().astype(dtype, copy=copy)

    def searchsorted(self, value, side="left", sorter=None) -> np.ndarray:
        value = self._validate_searchsorted_value(value).view("M8[ns]")

        # Cast to M8 to get datetime-like NaT placement
        m8arr = self._ndarray.view("M8[ns]")
        return m8arr.searchsorted(value, side=side, sorter=sorter)

    def fillna(self, value=None, method=None, limit=None) -> PeriodArray:
        if method is not None:
            # view as dt64 so we get treated as timelike in core.missing
            dta = self.view("M8[ns]")
            result = dta.fillna(value=value, method=method, limit=limit)
            return result.view(self.dtype)
        return super().fillna(value=value, method=method, limit=limit)

    # ------------------------------------------------------------------
    # Arithmetic Methods

    def _sub_datelike(self, other):
        assert other is not NaT
        return NotImplemented

    def _sub_period(self, other):
        # If the operation is well-defined, we return an object-Index
        # of DateOffsets.  Null entries are filled with pd.NaT
        self._check_compatible_with(other)
        asi8 = self.asi8
        new_data = asi8 - other.ordinal
        new_data = np.array([self.freq * x for x in new_data])

        if self._hasnans:
            new_data[self._isnan] = NaT

        return new_data

    def _sub_period_array(self, other):
        """
        Subtract a Period Array/Index from self.  This is only valid if self
        is itself a Period Array/Index, raises otherwise.  Both objects must
        have the same frequency.

        Parameters
        ----------
        other : PeriodIndex or PeriodArray

        Returns
        -------
        result : np.ndarray[object]
            Array of DateOffset objects; nulls represented by NaT.
        """
        self._require_matching_freq(other)

        new_values = algos.checked_add_with_arr(
            self.asi8, -other.asi8, arr_mask=self._isnan, b_mask=other._isnan
        )

        new_values = np.array([self.freq.base * x for x in new_values])
        if self._hasnans or other._hasnans:
            mask = self._isnan | other._isnan
            new_values[mask] = NaT
        return new_values

    def _addsub_int_array(
        self, other: np.ndarray, op: Callable[[Any, Any], Any]
    ) -> PeriodArray:
        """
        Add or subtract array of integers; equivalent to applying
        `_time_shift` pointwise.

        Parameters
        ----------
        other : np.ndarray[integer-dtype]
        op : {operator.add, operator.sub}

        Returns
        -------
        result : PeriodArray
        """
        assert op in [operator.add, operator.sub]
        if op is operator.sub:
            other = -other
        res_values = algos.checked_add_with_arr(self.asi8, other, arr_mask=self._isnan)
        res_values = res_values.view("i8")
        np.putmask(res_values, self._isnan, iNaT)
        return type(self)(res_values, freq=self.freq)

    def _add_offset(self, other: BaseOffset):
        assert not isinstance(other, Tick)

        self._require_matching_freq(other, base=True)

        # Note: when calling parent class's _add_timedeltalike_scalar,
        #  it will call delta_to_nanoseconds(delta).  Because delta here
        #  is an integer, delta_to_nanoseconds will return it unchanged.
        result = super()._add_timedeltalike_scalar(other.n)
        return type(self)(result, freq=self.freq)

    def _add_timedeltalike_scalar(self, other):
        """
        Parameters
        ----------
        other : timedelta, Tick, np.timedelta64

        Returns
        -------
        PeriodArray
        """
        if not isinstance(self.freq, Tick):
            # We cannot add timedelta-like to non-tick PeriodArray
            raise raise_on_incompatible(self, other)

        if notna(other):
            # special handling for np.timedelta64("NaT"), avoid calling
            #  _check_timedeltalike_freq_compat as that would raise TypeError
            other = self._check_timedeltalike_freq_compat(other)

        # Note: when calling parent class's _add_timedeltalike_scalar,
        #  it will call delta_to_nanoseconds(delta).  Because delta here
        #  is an integer, delta_to_nanoseconds will return it unchanged.
        return super()._add_timedeltalike_scalar(other)

    def _add_timedelta_arraylike(self, other):
        """
        Parameters
        ----------
        other : TimedeltaArray or ndarray[timedelta64]

        Returns
        -------
        result : ndarray[int64]
        """
        if not isinstance(self.freq, Tick):
            # We cannot add timedelta-like to non-tick PeriodArray
            raise TypeError(
                f"Cannot add or subtract timedelta64[ns] dtype from {self.dtype}"
            )

        if not np.all(isna(other)):
            delta = self._check_timedeltalike_freq_compat(other)
        else:
            # all-NaT TimedeltaIndex is equivalent to a single scalar td64 NaT
            return self + np.timedelta64("NaT")

        ordinals = self._addsub_int_array(delta, operator.add).asi8
        return type(self)(ordinals, dtype=self.dtype)

    def _check_timedeltalike_freq_compat(self, other):
        """
        Arithmetic operations with timedelta-like scalars or array `other`
        are only valid if `other` is an integer multiple of `self.freq`.
        If the operation is valid, find that integer multiple.  Otherwise,
        raise because the operation is invalid.

        Parameters
        ----------
        other : timedelta, np.timedelta64, Tick,
                ndarray[timedelta64], TimedeltaArray, TimedeltaIndex

        Returns
        -------
        multiple : int or ndarray[int64]

        Raises
        ------
        IncompatibleFrequency
        """
        assert isinstance(self.freq, Tick)  # checked by calling function
        base_nanos = self.freq.base.nanos

        if isinstance(other, (timedelta, np.timedelta64, Tick)):
            nanos = delta_to_nanoseconds(other)

        elif isinstance(other, np.ndarray):
            # numpy timedelta64 array; all entries must be compatible
            assert other.dtype.kind == "m"
            if other.dtype != TD64NS_DTYPE:
                # i.e. non-nano unit
                # TODO: disallow unit-less timedelta64
                other = other.astype(TD64NS_DTYPE)
            nanos = other.view("i8")
        else:
            # TimedeltaArray/Index
            nanos = other.asi8

        if np.all(nanos % base_nanos == 0):
            # nanos being added is an integer multiple of the
            #  base-frequency to self.freq
            delta = nanos // base_nanos
            # delta is the integer (or integer-array) number of periods
            # by which will be added to self.
            return delta

        raise raise_on_incompatible(self, other)

    # ------------------------------------------------------------------
    # TODO: See if we can re-share this with Period

    def _get_to_timestamp_base(self) -> int:
        """
        Return frequency code group used for base of to_timestamp against
        frequency code.

        Return day freq code against longer freq than day.
        Return second freq code against hour between second.

        Returns
        -------
        int
        """
        base = self._dtype._dtype_code
        if base < FreqGroup.FR_BUS.value:
            return FreqGroup.FR_DAY.value
        elif FreqGroup.FR_HR.value <= base <= FreqGroup.FR_SEC.value:
            return FreqGroup.FR_SEC.value
        return base

    @property
    def start_time(self) -> DatetimeArray:
        return self.to_timestamp(how="start")

    @property
    def end_time(self) -> DatetimeArray:
        return self.to_timestamp(how="end")

    def _require_matching_freq(self, other, base: bool = False) -> None:
        # See also arrays.period.raise_on_incompatible
        if isinstance(other, BaseOffset):
            other_freq = other
        else:
            other_freq = other.freq

        if base:
            condition = self.freq.base != other_freq.base
        else:
            condition = self.freq != other_freq

        if condition:
            msg = DIFFERENT_FREQ.format(
                cls=type(self).__name__,
                own_freq=self.freqstr,
                other_freq=other_freq.freqstr,
            )
            raise IncompatibleFrequency(msg)


def raise_on_incompatible(left, right):
    """
    Helper function to render a consistent error message when raising
    IncompatibleFrequency.

    Parameters
    ----------
    left : PeriodArray
    right : None, DateOffset, Period, ndarray, or timedelta-like

    Returns
    -------
    IncompatibleFrequency
        Exception to be raised by the caller.
    """
    # GH#24283 error message format depends on whether right is scalar
    if isinstance(right, (np.ndarray, ABCTimedeltaArray)) or right is None:
        other_freq = None
    elif isinstance(right, (ABCPeriodIndex, PeriodArray, Period, BaseOffset)):
        other_freq = right.freqstr
    else:
        other_freq = delta_to_tick(Timedelta(right)).freqstr

    msg = DIFFERENT_FREQ.format(
        cls=type(left).__name__, own_freq=left.freqstr, other_freq=other_freq
    )
    return IncompatibleFrequency(msg)


# -------------------------------------------------------------------
# Constructor Helpers


def period_array(
    data: Sequence[Period | str | None] | AnyArrayLike,
    freq: str | Tick | None = None,
    copy: bool = False,
) -> PeriodArray:
    """
    Construct a new PeriodArray from a sequence of Period scalars.

    Parameters
    ----------
    data : Sequence of Period objects
        A sequence of Period objects. These are required to all have
        the same ``freq.`` Missing values can be indicated by ``None``
        or ``pandas.NaT``.
    freq : str, Tick, or Offset
        The frequency of every element of the array. This can be specified
        to avoid inferring the `freq` from `data`.
    copy : bool, default False
        Whether to ensure a copy of the data is made.

    Returns
    -------
    PeriodArray

    See Also
    --------
    PeriodArray
    pandas.PeriodIndex

    Examples
    --------
    >>> period_array([pd.Period('2017', freq='A'),
    ...               pd.Period('2018', freq='A')])
    <PeriodArray>
    ['2017', '2018']
    Length: 2, dtype: period[A-DEC]

    >>> period_array([pd.Period('2017', freq='A'),
    ...               pd.Period('2018', freq='A'),
    ...               pd.NaT])
    <PeriodArray>
    ['2017', '2018', 'NaT']
    Length: 3, dtype: period[A-DEC]

    Integers that look like years are handled

    >>> period_array([2000, 2001, 2002], freq='D')
    <PeriodArray>
    ['2000-01-01', '2001-01-01', '2002-01-01']
    Length: 3, dtype: period[D]

    Datetime-like strings may also be passed

    >>> period_array(['2000-Q1', '2000-Q2', '2000-Q3', '2000-Q4'], freq='Q')
    <PeriodArray>
    ['2000Q1', '2000Q2', '2000Q3', '2000Q4']
    Length: 4, dtype: period[Q-DEC]
    """
    data_dtype = getattr(data, "dtype", None)

    if is_datetime64_dtype(data_dtype):
        return PeriodArray._from_datetime64(data, freq)
    if is_period_dtype(data_dtype):
        return PeriodArray(data, freq=freq)

    # other iterable of some kind
    if not isinstance(data, (np.ndarray, list, tuple, ABCSeries)):
        data = list(data)

    arrdata = np.asarray(data)

    dtype: PeriodDtype | None
    if freq:
        dtype = PeriodDtype(freq)
    else:
        dtype = None

    if is_float_dtype(arrdata) and len(arrdata) > 0:
        raise TypeError("PeriodIndex does not allow floating point in construction")

    if is_integer_dtype(arrdata.dtype):
        arr = arrdata.astype(np.int64, copy=False)
        ordinals = libperiod.from_ordinals(arr, freq)
        return PeriodArray(ordinals, dtype=dtype)

    data = ensure_object(arrdata)

    return PeriodArray._from_sequence(data, dtype=dtype)


def validate_dtype_freq(dtype, freq):
    """
    If both a dtype and a freq are available, ensure they match.  If only
    dtype is available, extract the implied freq.

    Parameters
    ----------
    dtype : dtype
    freq : DateOffset or None

    Returns
    -------
    freq : DateOffset

    Raises
    ------
    ValueError : non-period dtype
    IncompatibleFrequency : mismatch between dtype and freq
    """
    if freq is not None:
        freq = to_offset(freq)

    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if not is_period_dtype(dtype):
            raise ValueError("dtype must be PeriodDtype")
        if freq is None:
            freq = dtype.freq
        elif freq != dtype.freq:
            raise IncompatibleFrequency("specified freq and dtype are different")
    return freq


def dt64arr_to_periodarr(data, freq, tz=None):
    """
    Convert an datetime-like array to values Period ordinals.

    Parameters
    ----------
    data : Union[Series[datetime64[ns]], DatetimeIndex, ndarray[datetime64ns]]
    freq : Optional[Union[str, Tick]]
        Must match the `freq` on the `data` if `data` is a DatetimeIndex
        or Series.
    tz : Optional[tzinfo]

    Returns
    -------
    ordinals : ndarray[int64]
    freq : Tick
        The frequency extracted from the Series or DatetimeIndex if that's
        used.

    """
    if data.dtype != np.dtype("M8[ns]"):
        raise ValueError(f"Wrong dtype: {data.dtype}")

    if freq is None:
        if isinstance(data, ABCIndex):
            data, freq = data._values, data.freq
        elif isinstance(data, ABCSeries):
            data, freq = data._values, data.dt.freq

    elif isinstance(data, (ABCIndex, ABCSeries)):
        data = data._values

    freq = Period._maybe_convert_freq(freq)
    base = freq._period_dtype_code
    return c_dt64arr_to_periodarr(data.view("i8"), base, tz), freq


def _get_ordinal_range(start, end, periods, freq, mult=1):
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )

    if freq is not None:
        freq = to_offset(freq)
        mult = freq.n

    if start is not None:
        start = Period(start, freq)
    if end is not None:
        end = Period(end, freq)

    is_start_per = isinstance(start, Period)
    is_end_per = isinstance(end, Period)

    if is_start_per and is_end_per and start.freq != end.freq:
        raise ValueError("start and end must have same freq")
    if start is NaT or end is NaT:
        raise ValueError("start and end must not be NaT")

    if freq is None:
        if is_start_per:
            freq = start.freq
        elif is_end_per:
            freq = end.freq
        else:  # pragma: no cover
            raise ValueError("Could not infer freq from start/end")

    if periods is not None:
        periods = periods * mult
        if start is None:
            data = np.arange(
                end.ordinal - periods + mult, end.ordinal + 1, mult, dtype=np.int64
            )
        else:
            data = np.arange(
                start.ordinal, start.ordinal + periods, mult, dtype=np.int64
            )
    else:
        data = np.arange(start.ordinal, end.ordinal + 1, mult, dtype=np.int64)

    return data, freq


def _range_from_fields(
    year=None,
    month=None,
    quarter=None,
    day=None,
    hour=None,
    minute=None,
    second=None,
    freq=None,
) -> tuple[np.ndarray, BaseOffset]:
    if hour is None:
        hour = 0
    if minute is None:
        minute = 0
    if second is None:
        second = 0
    if day is None:
        day = 1

    ordinals = []

    if quarter is not None:
        if freq is None:
            freq = to_offset("Q")
            base = FreqGroup.FR_QTR.value
        else:
            freq = to_offset(freq)
            base = libperiod.freq_to_dtype_code(freq)
            if base != FreqGroup.FR_QTR.value:
                raise AssertionError("base must equal FR_QTR")

        freqstr = freq.freqstr
        year, quarter = _make_field_arrays(year, quarter)
        for y, q in zip(year, quarter):
            y, m = parsing.quarter_to_myear(y, q, freqstr)
            val = libperiod.period_ordinal(y, m, 1, 1, 1, 1, 0, 0, base)
            ordinals.append(val)
    else:
        freq = to_offset(freq)
        base = libperiod.freq_to_dtype_code(freq)
        arrays = _make_field_arrays(year, month, day, hour, minute, second)
        for y, mth, d, h, mn, s in zip(*arrays):
            ordinals.append(libperiod.period_ordinal(y, mth, d, h, mn, s, 0, 0, base))

    return np.array(ordinals, dtype=np.int64), freq


def _make_field_arrays(*fields) -> list[np.ndarray]:
    length = None
    for x in fields:
        if isinstance(x, (list, np.ndarray, ABCSeries)):
            if length is not None and len(x) != length:
                raise ValueError("Mismatched Period array lengths")
            elif length is None:
                length = len(x)

    # error: Argument 2 to "repeat" has incompatible type "Optional[int]"; expected
    # "Union[Union[int, integer[Any]], Union[bool, bool_], ndarray, Sequence[Union[int,
    # integer[Any]]], Sequence[Union[bool, bool_]], Sequence[Sequence[Any]]]"
    return [
        np.asarray(x)
        if isinstance(x, (np.ndarray, list, ABCSeries))
        else np.repeat(x, length)  # type: ignore[arg-type]
        for x in fields
    ]
