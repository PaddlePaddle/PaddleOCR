from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import (
    lib,
    tslibs,
)
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Period,
    Tick,
    Timedelta,
    Timestamp,
    iNaT,
    to_offset,
)
from pandas._libs.tslibs.conversion import (
    ensure_timedelta64ns,
    precision_from_unit,
)
from pandas._libs.tslibs.fields import get_timedelta_field
from pandas._libs.tslibs.timedeltas import (
    array_to_timedelta64,
    ints_to_pytimedelta,
    parse_timedelta_unit,
)
from pandas._typing import (
    DtypeObj,
    NpDtype,
)
from pandas.compat.numpy import function as nv

from pandas.core.dtypes.cast import astype_td64_unit_conversion
from pandas.core.dtypes.common import (
    DT64NS_DTYPE,
    TD64NS_DTYPE,
    is_dtype_equal,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    is_timedelta64_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import (
    ABCCategorical,
    ABCMultiIndex,
)
from pandas.core.dtypes.missing import isna

from pandas.core import nanops
from pandas.core.algorithms import checked_add_with_arr
from pandas.core.arrays import (
    ExtensionArray,
    IntegerArray,
    datetimelike as dtl,
)
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.core.arrays import (
        DatetimeArray,
        PeriodArray,
    )


def _field_accessor(name: str, alias: str, docstring: str):
    def f(self) -> np.ndarray:
        values = self.asi8
        result = get_timedelta_field(values, alias)
        if self._hasnans:
            result = self._maybe_mask_results(
                result, fill_value=None, convert="float64"
            )

        return result

    f.__name__ = name
    f.__doc__ = f"\n{docstring}\n"
    return property(f)


class TimedeltaArray(dtl.TimelikeOps):
    """
    Pandas ExtensionArray for timedelta data.

    .. warning::

       TimedeltaArray is currently experimental, and its API may change
       without warning. In particular, :attr:`TimedeltaArray.dtype` is
       expected to change to be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    values : array-like
        The timedelta data.

    dtype : numpy.dtype
        Currently, only ``numpy.dtype("timedelta64[ns]")`` is accepted.
    freq : Offset, optional
    copy : bool, default False
        Whether to copy the underlying array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None
    """

    _typ = "timedeltaarray"
    _scalar_type = Timedelta
    _recognized_scalars = (timedelta, np.timedelta64, Tick)
    _is_recognized_dtype = is_timedelta64_dtype
    _infer_matches = ("timedelta", "timedelta64")

    __array_priority__ = 1000
    # define my properties & methods for delegation
    _other_ops: list[str] = []
    _bool_ops: list[str] = []
    _object_ops: list[str] = ["freq"]
    _field_ops: list[str] = ["days", "seconds", "microseconds", "nanoseconds"]
    _datetimelike_ops: list[str] = _field_ops + _object_ops + _bool_ops
    _datetimelike_methods: list[str] = [
        "to_pytimedelta",
        "total_seconds",
        "round",
        "floor",
        "ceil",
    ]

    # Note: ndim must be defined to ensure NaT.__richcmp__(TimedeltaArray)
    #  operates pointwise.

    def _box_func(self, x) -> Timedelta | NaTType:
        return Timedelta(x, unit="ns")

    @property
    # error: Return type "dtype" of "dtype" incompatible with return type
    # "ExtensionDtype" in supertype "ExtensionArray"
    def dtype(self) -> np.dtype:  # type: ignore[override]
        """
        The dtype for the TimedeltaArray.

        .. warning::

           A future version of pandas will change dtype to be an instance
           of a :class:`pandas.api.extensions.ExtensionDtype` subclass,
           not a ``numpy.dtype``.

        Returns
        -------
        numpy.dtype
        """
        return TD64NS_DTYPE

    # ----------------------------------------------------------------
    # Constructors

    _freq = None

    def __init__(
        self, values, dtype=TD64NS_DTYPE, freq=lib.no_default, copy: bool = False
    ):
        values = extract_array(values, extract_numpy=True)
        if isinstance(values, IntegerArray):
            values = values.to_numpy("int64", na_value=tslibs.iNaT)

        inferred_freq = getattr(values, "_freq", None)
        explicit_none = freq is None
        freq = freq if freq is not lib.no_default else None

        if isinstance(values, type(self)):
            if explicit_none:
                # dont inherit from values
                pass
            elif freq is None:
                freq = values.freq
            elif freq and values.freq:
                freq = to_offset(freq)
                freq, _ = dtl.validate_inferred_freq(freq, values.freq, False)
            values = values._ndarray

        if not isinstance(values, np.ndarray):
            msg = (
                f"Unexpected type '{type(values).__name__}'. 'values' must be a "
                "TimedeltaArray, ndarray, or Series or Index containing one of those."
            )
            raise ValueError(msg)
        if values.ndim not in [1, 2]:
            raise ValueError("Only 1-dimensional input arrays are supported.")

        if values.dtype == "i8":
            # for compat with datetime/timedelta/period shared methods,
            #  we can sometimes get here with int64 values.  These represent
            #  nanosecond UTC (or tz-naive) unix timestamps
            values = values.view(TD64NS_DTYPE)

        _validate_td64_dtype(values.dtype)
        dtype = _validate_td64_dtype(dtype)

        if freq == "infer":
            msg = (
                "Frequency inference not allowed in TimedeltaArray.__init__. "
                "Use 'pd.array()' instead."
            )
            raise ValueError(msg)

        if copy:
            values = values.copy()
        if freq:
            freq = to_offset(freq)

        NDArrayBacked.__init__(self, values=values, dtype=dtype)
        self._freq = freq

        if inferred_freq is None and freq is not None:
            type(self)._validate_frequency(self, freq)

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls, values: np.ndarray, freq: BaseOffset | None = None, dtype=TD64NS_DTYPE
    ) -> TimedeltaArray:
        assert dtype == TD64NS_DTYPE, dtype
        assert isinstance(values, np.ndarray), type(values)
        assert values.dtype == TD64NS_DTYPE

        result = super()._simple_new(values=values, dtype=TD64NS_DTYPE)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(
        cls, data, *, dtype=TD64NS_DTYPE, copy: bool = False
    ) -> TimedeltaArray:
        if dtype:
            _validate_td64_dtype(dtype)

        data, inferred_freq = sequence_to_td64ns(data, copy=copy, unit=None)
        freq, _ = dtl.validate_inferred_freq(None, inferred_freq, False)

        return cls._simple_new(data, freq=freq)

    @classmethod
    def _from_sequence_not_strict(
        cls,
        data,
        dtype=TD64NS_DTYPE,
        copy: bool = False,
        freq=lib.no_default,
        unit=None,
    ) -> TimedeltaArray:
        if dtype:
            _validate_td64_dtype(dtype)

        explicit_none = freq is None
        freq = freq if freq is not lib.no_default else None

        freq, freq_infer = dtl.maybe_infer_freq(freq)

        data, inferred_freq = sequence_to_td64ns(data, copy=copy, unit=unit)
        freq, freq_infer = dtl.validate_inferred_freq(freq, inferred_freq, freq_infer)
        if explicit_none:
            freq = None

        result = cls._simple_new(data, freq=freq)

        if inferred_freq is None and freq is not None:
            # this condition precludes `freq_infer`
            cls._validate_frequency(result, freq)

        elif freq_infer:
            # Set _freq directly to bypass duplicative _validate_frequency
            # check.
            result._freq = to_offset(result.inferred_freq)

        return result

    @classmethod
    def _generate_range(cls, start, end, periods, freq, closed=None):

        periods = dtl.validate_periods(periods)
        if freq is None and any(x is None for x in [periods, start, end]):
            raise ValueError("Must provide freq argument if no data is supplied")

        if com.count_not_none(start, end, periods, freq) != 3:
            raise ValueError(
                "Of the four parameters: start, end, periods, "
                "and freq, exactly three must be specified"
            )

        if start is not None:
            start = Timedelta(start)

        if end is not None:
            end = Timedelta(end)

        left_closed, right_closed = dtl.validate_endpoints(closed)

        if freq is not None:
            index = generate_regular_range(start, end, periods, freq)
        else:
            index = np.linspace(start.value, end.value, periods).astype("i8")

        if not left_closed:
            index = index[1:]
        if not right_closed:
            index = index[:-1]

        return cls._simple_new(index.view("m8[ns]"), freq=freq)

    # ----------------------------------------------------------------
    # DatetimeLike Interface

    def _unbox_scalar(self, value, setitem: bool = False) -> np.timedelta64:
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timedelta.")
        self._check_compatible_with(value, setitem=setitem)
        return np.timedelta64(value.value, "ns")

    def _scalar_from_string(self, value) -> Timedelta | NaTType:
        return Timedelta(value)

    def _check_compatible_with(self, other, setitem: bool = False) -> None:
        # we don't have anything to validate.
        pass

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def astype(self, dtype, copy: bool = True):
        # We handle
        #   --> timedelta64[ns]
        #   --> timedelta64
        # DatetimeLikeArrayMixin super call handles other cases
        dtype = pandas_dtype(dtype)

        if dtype.kind == "m":
            return astype_td64_unit_conversion(self._ndarray, dtype, copy=copy)

        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy=copy)

    def __iter__(self):
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            # convert in chunks of 10k for efficiency
            data = self.asi8
            length = len(self)
            chunksize = 10000
            chunks = (length // chunksize) + 1
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = ints_to_pytimedelta(data[start_i:end_i], box=True)
                yield from converted

    # ----------------------------------------------------------------
    # Reductions

    def sum(
        self,
        *,
        axis: int | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        initial=None,
        skipna: bool = True,
        min_count: int = 0,
    ):
        nv.validate_sum(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims, "initial": initial}
        )

        result = nanops.nansum(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def std(
        self,
        *,
        axis: int | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="std"
        )

        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        return self._from_backing_data(result)

    # ----------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        from pandas.io.formats.format import get_format_timedelta64

        return get_format_timedelta64(self, box=True)

    @dtl.ravel_compat
    def _format_native_types(
        self, na_rep="NaT", date_format=None, **kwargs
    ) -> np.ndarray:
        from pandas.io.formats.format import get_format_timedelta64

        formatter = get_format_timedelta64(self._ndarray, na_rep)
        return np.array([formatter(x) for x in self._ndarray])

    # ----------------------------------------------------------------
    # Arithmetic Methods

    def _add_offset(self, other):
        assert not isinstance(other, Tick)
        raise TypeError(
            f"cannot add the type {type(other).__name__} to a {type(self).__name__}"
        )

    def _add_period(self, other: Period) -> PeriodArray:
        """
        Add a Period object.
        """
        # We will wrap in a PeriodArray and defer to the reversed operation
        from pandas.core.arrays.period import PeriodArray

        i8vals = np.broadcast_to(other.ordinal, self.shape)
        oth = PeriodArray(i8vals, freq=other.freq)
        return oth + self

    def _add_datetime_arraylike(self, other):
        """
        Add DatetimeArray/Index or ndarray[datetime64] to TimedeltaArray.
        """
        if isinstance(other, np.ndarray):
            # At this point we have already checked that dtype is datetime64
            from pandas.core.arrays import DatetimeArray

            other = DatetimeArray(other)

        # defer to implementation in DatetimeArray
        return other + self

    def _add_datetimelike_scalar(self, other) -> DatetimeArray:
        # adding a timedeltaindex to a datetimelike
        from pandas.core.arrays import DatetimeArray

        assert other is not NaT
        other = Timestamp(other)
        if other is NaT:
            # In this case we specifically interpret NaT as a datetime, not
            # the timedelta interpretation we would get by returning self + NaT
            result = self.asi8.view("m8[ms]") + NaT.to_datetime64()
            return DatetimeArray(result)

        i8 = self.asi8
        result = checked_add_with_arr(i8, other.value, arr_mask=self._isnan)
        result = self._maybe_mask_results(result)
        dtype = DatetimeTZDtype(tz=other.tz) if other.tz else DT64NS_DTYPE
        return DatetimeArray(result, dtype=dtype, freq=self.freq)

    def _addsub_object_array(self, other, op):
        # Add or subtract Array-like of objects
        try:
            # TimedeltaIndex can only operate with a subset of DateOffset
            # subclasses.  Incompatible classes will raise AttributeError,
            # which we re-raise as TypeError
            return super()._addsub_object_array(other, op)
        except AttributeError as err:
            raise TypeError(
                f"Cannot add/subtract non-tick DateOffset to {type(self).__name__}"
            ) from err

    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other) -> TimedeltaArray:
        if is_scalar(other):
            # numpy will accept float and int, raise TypeError for others
            result = self._ndarray * other
            freq = None
            if self.freq is not None and not isna(other):
                freq = self.freq * other
            return type(self)(result, freq=freq)

        if not hasattr(other, "dtype"):
            # list, tuple
            other = np.array(other)
        if len(other) != len(self) and not is_timedelta64_dtype(other.dtype):
            # Exclude timedelta64 here so we correctly raise TypeError
            #  for that instead of ValueError
            raise ValueError("Cannot multiply with unequal lengths")

        if is_object_dtype(other.dtype):
            # this multiplication will succeed only if all elements of other
            #  are int or float scalars, so we will end up with
            #  timedelta64[ns]-dtyped result
            result = [self[n] * other[n] for n in range(len(self))]
            result = np.array(result)
            return type(self)(result)

        # numpy will accept float or int dtype, raise TypeError for others
        result = self._ndarray * other
        return type(self)(result)

    __rmul__ = __mul__

    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        # timedelta / X is well-defined for timedelta-like or numeric X

        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
            if other is NaT:
                # specifically timedelta64-NaT
                result = np.empty(self.shape, dtype=np.float64)
                result.fill(np.nan)
                return result

            # otherwise, dispatch to Timedelta implementation
            return self._ndarray / other

        elif lib.is_scalar(other):
            # assume it is numeric
            result = self._ndarray / other
            freq = None
            if self.freq is not None:
                # Tick division is not implemented, so operate on Timedelta
                freq = self.freq.delta / other
            return type(self)(result, freq=freq)

        if not hasattr(other, "dtype"):
            # e.g. list, tuple
            other = np.array(other)

        if len(other) != len(self):
            raise ValueError("Cannot divide vectors with unequal lengths")

        elif is_timedelta64_dtype(other.dtype):
            # let numpy handle it
            return self._ndarray / other

        elif is_object_dtype(other.dtype):
            # We operate on raveled arrays to avoid problems in inference
            #  on NaT
            srav = self.ravel()
            orav = other.ravel()
            result = [srav[n] / orav[n] for n in range(len(srav))]
            result = np.array(result).reshape(self.shape)

            # We need to do dtype inference in order to keep DataFrame ops
            #  behavior consistent with Series behavior
            inferred = lib.infer_dtype(result)
            if inferred == "timedelta":
                flat = result.ravel()
                result = type(self)._from_sequence(flat).reshape(result.shape)
            elif inferred == "floating":
                result = result.astype(float)

            return result

        else:
            result = self._ndarray / other
            return type(self)(result)

    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other):
        # X / timedelta is defined only for timedelta-like X
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
            if other is NaT:
                # specifically timedelta64-NaT
                result = np.empty(self.shape, dtype=np.float64)
                result.fill(np.nan)
                return result

            # otherwise, dispatch to Timedelta implementation
            return other / self._ndarray

        elif lib.is_scalar(other):
            raise TypeError(
                f"Cannot divide {type(other).__name__} by {type(self).__name__}"
            )

        if not hasattr(other, "dtype"):
            # e.g. list, tuple
            other = np.array(other)

        if len(other) != len(self):
            raise ValueError("Cannot divide vectors with unequal lengths")

        elif is_timedelta64_dtype(other.dtype):
            # let numpy handle it
            return other / self._ndarray

        elif is_object_dtype(other.dtype):
            # Note: unlike in __truediv__, we do not _need_ to do type
            #  inference on the result.  It does not raise, a numeric array
            #  is returned.  GH#23829
            result = [other[n] / self[n] for n in range(len(self))]
            return np.array(result)

        else:
            raise TypeError(
                f"Cannot divide {other.dtype} data by {type(self).__name__}"
            )

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):

        if is_scalar(other):
            if isinstance(other, self._recognized_scalars):
                other = Timedelta(other)
                if other is NaT:
                    # treat this specifically as timedelta-NaT
                    result = np.empty(self.shape, dtype=np.float64)
                    result.fill(np.nan)
                    return result

                # dispatch to Timedelta implementation
                result = other.__rfloordiv__(self._ndarray)
                return result

            # at this point we should only have numeric scalars; anything
            #  else will raise
            result = self.asi8 // other
            np.putmask(result, self._isnan, iNaT)
            freq = None
            if self.freq is not None:
                # Note: freq gets division, not floor-division
                freq = self.freq / other
                if freq.nanos == 0 and self.freq.nanos != 0:
                    # e.g. if self.freq is Nano(1) then dividing by 2
                    #  rounds down to zero
                    freq = None
            return type(self)(result.view("m8[ns]"), freq=freq)

        if not hasattr(other, "dtype"):
            # list, tuple
            other = np.array(other)
        if len(other) != len(self):
            raise ValueError("Cannot divide with unequal lengths")

        elif is_timedelta64_dtype(other.dtype):
            other = type(self)(other)

            # numpy timedelta64 does not natively support floordiv, so operate
            #  on the i8 values
            result = self.asi8 // other.asi8
            mask = self._isnan | other._isnan
            if mask.any():
                result = result.astype(np.float64)
                np.putmask(result, mask, np.nan)
            return result

        elif is_object_dtype(other.dtype):
            # error: Incompatible types in assignment (expression has type
            # "List[Any]", variable has type "ndarray")
            result = [  # type: ignore[assignment]
                self[n] // other[n] for n in range(len(self))
            ]
            result = np.array(result)
            if lib.infer_dtype(result, skipna=False) == "timedelta":
                result, _ = sequence_to_td64ns(result)
                return type(self)(result)
            return result

        elif is_integer_dtype(other.dtype) or is_float_dtype(other.dtype):
            result = self._ndarray // other
            return type(self)(result)

        else:
            dtype = getattr(other, "dtype", type(other).__name__)
            raise TypeError(f"Cannot divide {dtype} by {type(self).__name__}")

    @unpack_zerodim_and_defer("__rfloordiv__")
    def __rfloordiv__(self, other):

        if is_scalar(other):
            if isinstance(other, self._recognized_scalars):
                other = Timedelta(other)
                if other is NaT:
                    # treat this specifically as timedelta-NaT
                    result = np.empty(self.shape, dtype=np.float64)
                    result.fill(np.nan)
                    return result

                # dispatch to Timedelta implementation
                result = other.__floordiv__(self._ndarray)
                return result

            raise TypeError(
                f"Cannot divide {type(other).__name__} by {type(self).__name__}"
            )

        if not hasattr(other, "dtype"):
            # list, tuple
            other = np.array(other)

        if len(other) != len(self):
            raise ValueError("Cannot divide with unequal lengths")

        elif is_timedelta64_dtype(other.dtype):
            other = type(self)(other)
            # numpy timedelta64 does not natively support floordiv, so operate
            #  on the i8 values
            result = other.asi8 // self.asi8
            mask = self._isnan | other._isnan
            if mask.any():
                result = result.astype(np.float64)
                np.putmask(result, mask, np.nan)
            return result

        elif is_object_dtype(other.dtype):
            # error: Incompatible types in assignment (expression has type
            # "List[Any]", variable has type "ndarray")
            result = [  # type: ignore[assignment]
                other[n] // self[n] for n in range(len(self))
            ]
            result = np.array(result)
            return result

        else:
            dtype = getattr(other, "dtype", type(other).__name__)
            raise TypeError(f"Cannot divide {dtype} by {type(self).__name__}")

    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return self - (self // other) * other

    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return other - (other // self) * self

    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)

        res1 = self // other
        res2 = self - res1 * other
        return res1, res2

    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)

        res1 = other // self
        res2 = other - res1 * self
        return res1, res2

    def __neg__(self) -> TimedeltaArray:
        if self.freq is not None:
            return type(self)(-self._ndarray, freq=-self.freq)
        return type(self)(-self._ndarray)

    def __pos__(self) -> TimedeltaArray:
        return type(self)(self._ndarray, freq=self.freq)

    def __abs__(self) -> TimedeltaArray:
        # Note: freq is not preserved
        return type(self)(np.abs(self._ndarray))

    # ----------------------------------------------------------------
    # Conversion Methods - Vectorized analogues of Timedelta methods

    def total_seconds(self) -> np.ndarray:
        """
        Return total duration of each element expressed in seconds.

        This method is available directly on TimedeltaArray, TimedeltaIndex
        and on Series containing timedelta values under the ``.dt`` namespace.

        Returns
        -------
        seconds : [ndarray, Float64Index, Series]
            When the calling object is a TimedeltaArray, the return type
            is ndarray.  When the calling object is a TimedeltaIndex,
            the return type is a Float64Index. When the calling object
            is a Series, the return type is Series of type `float64` whose
            index is the same as the original.

        See Also
        --------
        datetime.timedelta.total_seconds : Standard library version
            of this method.
        TimedeltaIndex.components : Return a DataFrame with components of
            each Timedelta.

        Examples
        --------
        **Series**

        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='d'))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.total_seconds()
        0         0.0
        1     86400.0
        2    172800.0
        3    259200.0
        4    345600.0
        dtype: float64

        **TimedeltaIndex**

        >>> idx = pd.to_timedelta(np.arange(5), unit='d')
        >>> idx
        TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                       dtype='timedelta64[ns]', freq=None)

        >>> idx.total_seconds()
        Float64Index([0.0, 86400.0, 172800.0, 259200.00000000003, 345600.0],
                     dtype='float64')
        """
        return self._maybe_mask_results(1e-9 * self.asi8, fill_value=None)

    def to_pytimedelta(self) -> np.ndarray:
        """
        Return Timedelta Array/Index as object ndarray of datetime.timedelta
        objects.

        Returns
        -------
        timedeltas : ndarray[object]
        """
        return tslibs.ints_to_pytimedelta(self.asi8)

    days = _field_accessor("days", "days", "Number of days for each element.")
    seconds = _field_accessor(
        "seconds",
        "seconds",
        "Number of seconds (>= 0 and less than 1 day) for each element.",
    )
    microseconds = _field_accessor(
        "microseconds",
        "microseconds",
        "Number of microseconds (>= 0 and less than 1 second) for each element.",
    )
    nanoseconds = _field_accessor(
        "nanoseconds",
        "nanoseconds",
        "Number of nanoseconds (>= 0 and less than 1 microsecond) for each element.",
    )

    @property
    def components(self) -> DataFrame:
        """
        Return a dataframe of the components (days, hours, minutes,
        seconds, milliseconds, microseconds, nanoseconds) of the Timedeltas.

        Returns
        -------
        DataFrame
        """
        from pandas import DataFrame

        columns = [
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ]
        hasnans = self._hasnans
        if hasnans:

            def f(x):
                if isna(x):
                    return [np.nan] * len(columns)
                return x.components

        else:

            def f(x):
                return x.components

        result = DataFrame([f(x) for x in self], columns=columns)
        if not hasnans:
            result = result.astype("int64")
        return result


# ---------------------------------------------------------------------
# Constructor Helpers


def sequence_to_td64ns(
    data, copy: bool = False, unit=None, errors="raise"
) -> tuple[np.ndarray, Tick | None]:
    """
    Parameters
    ----------
    data : list-like
    copy : bool, default False
    unit : str, optional
        The timedelta unit to treat integers as multiples of. For numeric
        data this defaults to ``'ns'``.
        Must be un-specified if the data contains a str and ``errors=="raise"``.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    converted : numpy.ndarray
        The sequence converted to a numpy array with dtype ``timedelta64[ns]``.
    inferred_freq : Tick or None
        The inferred frequency of the sequence.

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting ``errors=ignore`` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    inferred_freq = None
    if unit is not None:
        unit = parse_timedelta_unit(unit)

    # Unwrap whatever we have into a np.ndarray
    if not hasattr(data, "dtype"):
        # e.g. list, tuple
        if np.ndim(data) == 0:
            # i.e. generator
            data = list(data)
        data = np.array(data, copy=False)
    elif isinstance(data, ABCMultiIndex):
        raise TypeError("Cannot create a DatetimeArray from a MultiIndex.")
    else:
        data = extract_array(data, extract_numpy=True)

    if isinstance(data, IntegerArray):
        data = data.to_numpy("int64", na_value=iNaT)
    elif not isinstance(data, (np.ndarray, ExtensionArray)):
        # GH#24539 e.g. xarray, dask object
        data = np.asarray(data)
    elif isinstance(data, ABCCategorical):
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False

    if isinstance(data, TimedeltaArray):
        inferred_freq = data.freq

    # Convert whatever we have into timedelta64[ns] dtype
    if is_object_dtype(data.dtype) or is_string_dtype(data.dtype):
        # no need to make a copy, need to convert if string-dtyped
        data = objects_to_td64ns(data, unit=unit, errors=errors)
        copy = False

    elif is_integer_dtype(data.dtype):
        # treat as multiples of the given unit
        data, copy_made = ints_to_td64ns(data, unit=unit)
        copy = copy and not copy_made

    elif is_float_dtype(data.dtype):
        # cast the unit, multiply base/frac separately
        # to avoid precision issues from float -> int
        mask = np.isnan(data)
        m, p = precision_from_unit(unit or "ns")
        base = data.astype(np.int64)
        frac = data - base
        if p:
            frac = np.round(frac, p)
        data = (base * m + (frac * m).astype(np.int64)).view("timedelta64[ns]")
        data[mask] = iNaT
        copy = False

    elif is_timedelta64_dtype(data.dtype):
        if data.dtype != TD64NS_DTYPE:
            # non-nano unit
            data = ensure_timedelta64ns(data)
            copy = False

    else:
        # This includes datetime64-dtype, see GH#23539, GH#29794
        raise TypeError(f"dtype {data.dtype} cannot be converted to timedelta64[ns]")

    data = np.array(data, copy=copy)

    assert data.dtype == "m8[ns]", data
    return data, inferred_freq


def ints_to_td64ns(data, unit="ns"):
    """
    Convert an ndarray with integer-dtype to timedelta64[ns] dtype, treating
    the integers as multiples of the given timedelta unit.

    Parameters
    ----------
    data : numpy.ndarray with integer-dtype
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data
    bool : whether a copy was made
    """
    copy_made = False
    unit = unit if unit is not None else "ns"

    if data.dtype != np.int64:
        # converting to int64 makes a copy, so we can avoid
        # re-copying later
        data = data.astype(np.int64)
        copy_made = True

    if unit != "ns":
        dtype_str = f"timedelta64[{unit}]"
        data = data.view(dtype_str)

        data = ensure_timedelta64ns(data)

        # the astype conversion makes a copy, so we can avoid re-copying later
        copy_made = True

    else:
        data = data.view("timedelta64[ns]")

    return data, copy_made


def objects_to_td64ns(data, unit=None, errors="raise"):
    """
    Convert a object-dtyped or string-dtyped array into an
    timedelta64[ns]-dtyped array.

    Parameters
    ----------
    data : ndarray or Index
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.
        Must not be specified if the data contains a str.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting `errors=ignore` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    # coerce Index to np.ndarray, converting string-dtype if necessary
    values = np.array(data, dtype=np.object_, copy=False)

    result = array_to_timedelta64(values, unit=unit, errors=errors)
    return result.view("timedelta64[ns]")


def _validate_td64_dtype(dtype) -> DtypeObj:
    dtype = pandas_dtype(dtype)
    if is_dtype_equal(dtype, np.dtype("timedelta64")):
        # no precision disallowed GH#24806
        msg = (
            "Passing in 'timedelta' dtype with no precision is not allowed. "
            "Please pass in 'timedelta64[ns]' instead."
        )
        raise ValueError(msg)

    if not is_dtype_equal(dtype, TD64NS_DTYPE):
        raise ValueError(f"dtype {dtype} cannot be converted to timedelta64[ns]")

    return dtype
