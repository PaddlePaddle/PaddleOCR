from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from typing import Hashable
import warnings

import numpy as np

from pandas._libs import (
    index as libindex,
    lib,
)
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    Period,
    Resolution,
    Tick,
)
from pandas._libs.tslibs.parsing import (
    DateParseError,
    parse_time_string,
)
from pandas._typing import (
    Dtype,
    DtypeObj,
)
from pandas.errors import InvalidIndexError
from pandas.util._decorators import doc

from pandas.core.dtypes.common import (
    is_datetime64_any_dtype,
    is_integer,
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.arrays.period import (
    PeriodArray,
    period_array,
    raise_on_incompatible,
    validate_dtype_freq,
)
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    Index,
)
from pandas.core.indexes.extension import inherit_names
from pandas.core.indexes.numeric import Int64Index

_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({"target_klass": "PeriodIndex or list of Periods"})
_shared_doc_kwargs = {
    "klass": "PeriodArray",
}

# --- Period index sketch


def _new_PeriodIndex(cls, **d):
    # GH13277 for unpickling
    values = d.pop("data")
    if values.dtype == "int64":
        freq = d.pop("freq", None)
        values = PeriodArray(values, freq=freq)
        return cls._simple_new(values, **d)
    else:
        return cls(values, **d)


@inherit_names(
    ["strftime", "start_time", "end_time"] + PeriodArray._field_ops,
    PeriodArray,
    wrap=True,
)
@inherit_names(["is_leap_year", "_format_native_types"], PeriodArray)
class PeriodIndex(DatetimeIndexOpsMixin):
    """
    Immutable ndarray holding ordinal values indicating regular periods in time.

    Index keys are boxed to Period objects which carries the metadata (eg,
    frequency information).

    Parameters
    ----------
    data : array-like (1d int np.ndarray or PeriodArray), optional
        Optional period-like data to construct index with.
    copy : bool
        Make a copy of input ndarray.
    freq : str or period object, optional
        One of pandas period strings or corresponding objects.
    year : int, array, or Series, default None
    month : int, array, or Series, default None
    quarter : int, array, or Series, default None
    day : int, array, or Series, default None
    hour : int, array, or Series, default None
    minute : int, array, or Series, default None
    second : int, array, or Series, default None
    dtype : str or PeriodDtype, default None

    Attributes
    ----------
    day
    dayofweek
    day_of_week
    dayofyear
    day_of_year
    days_in_month
    daysinmonth
    end_time
    freq
    freqstr
    hour
    is_leap_year
    minute
    month
    quarter
    qyear
    second
    start_time
    week
    weekday
    weekofyear
    year

    Methods
    -------
    asfreq
    strftime
    to_timestamp

    See Also
    --------
    Index : The base pandas Index type.
    Period : Represents a period of time.
    DatetimeIndex : Index with datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    period_range : Create a fixed-frequency PeriodIndex.

    Examples
    --------
    >>> idx = pd.PeriodIndex(year=[2000, 2002], quarter=[1, 3])
    >>> idx
    PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
    """

    _typ = "periodindex"
    _attributes = ["name"]

    _data: PeriodArray
    freq: BaseOffset

    _data_cls = PeriodArray
    _engine_type = libindex.PeriodEngine
    _supports_partial_string_indexing = True

    # --------------------------------------------------------------------
    # methods that dispatch to array and wrap result in Index
    # These are defined here instead of via inherit_names for mypy

    @doc(
        PeriodArray.asfreq,
        other="pandas.arrays.PeriodArray",
        other_name="PeriodArray",
        **_shared_doc_kwargs,
    )
    def asfreq(self, freq=None, how: str = "E") -> PeriodIndex:
        arr = self._data.asfreq(freq, how)
        return type(self)._simple_new(arr, name=self.name)

    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq=None, how="start") -> DatetimeIndex:
        arr = self._data.to_timestamp(freq, how)
        return DatetimeIndex._simple_new(arr, name=self.name)

    # https://github.com/python/mypy/issues/1362
    # error: Decorated property not supported
    @property  # type:ignore[misc]
    @doc(PeriodArray.hour.fget)
    def hour(self) -> Int64Index:
        return Int64Index(self._data.hour, name=self.name)

    # https://github.com/python/mypy/issues/1362
    # error: Decorated property not supported
    @property  # type:ignore[misc]
    @doc(PeriodArray.minute.fget)
    def minute(self) -> Int64Index:
        return Int64Index(self._data.minute, name=self.name)

    # https://github.com/python/mypy/issues/1362
    # error: Decorated property not supported
    @property  # type:ignore[misc]
    @doc(PeriodArray.second.fget)
    def second(self) -> Int64Index:
        return Int64Index(self._data.second, name=self.name)

    # ------------------------------------------------------------------------
    # Index Constructors

    def __new__(
        cls,
        data=None,
        ordinal=None,
        freq=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable = None,
        **fields,
    ) -> PeriodIndex:

        valid_field_set = {
            "year",
            "month",
            "day",
            "quarter",
            "hour",
            "minute",
            "second",
        }

        if not set(fields).issubset(valid_field_set):
            argument = list(set(fields) - valid_field_set)[0]
            raise TypeError(f"__new__() got an unexpected keyword argument {argument}")

        name = maybe_extract_name(name, data, cls)

        if data is None and ordinal is None:
            # range-based.
            data, freq2 = PeriodArray._generate_range(None, None, None, freq, fields)
            # PeriodArray._generate range does validation that fields is
            # empty when really using the range-based constructor.
            freq = freq2

            data = PeriodArray(data, freq=freq)
        else:
            freq = validate_dtype_freq(dtype, freq)

            # PeriodIndex allow PeriodIndex(period_index, freq=different)
            # Let's not encourage that kind of behavior in PeriodArray.

            if freq and isinstance(data, cls) and data.freq != freq:
                # TODO: We can do some of these with no-copy / coercion?
                # e.g. D -> 2D seems to be OK
                data = data.asfreq(freq)

            if data is None and ordinal is not None:
                # we strangely ignore `ordinal` if data is passed.
                ordinal = np.asarray(ordinal, dtype=np.int64)
                data = PeriodArray(ordinal, freq=freq)
            else:
                # don't pass copy here, since we copy later.
                data = period_array(data=data, freq=freq)

        if copy:
            data = data.copy()

        return cls._simple_new(data, name=name)

    # ------------------------------------------------------------------------
    # Data

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self, dtype=object)

    def _maybe_convert_timedelta(self, other):
        """
        Convert timedelta-like input to an integer multiple of self.freq

        Parameters
        ----------
        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray

        Returns
        -------
        converted : int, np.ndarray[int64]

        Raises
        ------
        IncompatibleFrequency : if the input cannot be written as a multiple
            of self.freq.  Note IncompatibleFrequency subclasses ValueError.
        """
        if isinstance(other, (timedelta, np.timedelta64, Tick, np.ndarray)):
            if isinstance(self.freq, Tick):
                # _check_timedeltalike_freq_compat will raise if incompatible
                delta = self._data._check_timedeltalike_freq_compat(other)
                return delta
        elif isinstance(other, BaseOffset):
            if other.base == self.freq.base:
                return other.n

            raise raise_on_incompatible(self, other)
        elif is_integer(other):
            # integer is passed to .shift via
            # _add_datetimelike_methods basically
            # but ufunc may pass integer to _add_delta
            return other

        # raise when input doesn't have freq
        raise raise_on_incompatible(self, None)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        if not isinstance(dtype, PeriodDtype):
            return False
        return dtype.freq == self.freq

    # ------------------------------------------------------------------------
    # Index Methods

    def asof_locs(self, where: Index, mask: np.ndarray) -> np.ndarray:
        """
        where : array of timestamps
        mask : np.ndarray[bool]
            Array of booleans where data is not NA.
        """
        if isinstance(where, DatetimeIndex):
            where = PeriodIndex(where._values, freq=self.freq)
        elif not isinstance(where, PeriodIndex):
            raise TypeError("asof_locs `where` must be DatetimeIndex or PeriodIndex")

        return super().asof_locs(where, mask)

    @doc(Index.astype)
    def astype(self, dtype, copy: bool = True, how=lib.no_default):
        dtype = pandas_dtype(dtype)

        if how is not lib.no_default:
            # GH#37982
            warnings.warn(
                "The 'how' keyword in PeriodIndex.astype is deprecated and "
                "will be removed in a future version. "
                "Use index.to_timestamp(how=how) instead",
                FutureWarning,
                stacklevel=2,
            )
        else:
            how = "start"

        if is_datetime64_any_dtype(dtype):
            # 'how' is index-specific, isn't part of the EA interface.
            tz = getattr(dtype, "tz", None)
            return self.to_timestamp(how=how).tz_localize(tz)

        return super().astype(dtype, copy=copy)

    @property
    def is_full(self) -> bool:
        """
        Returns True if this PeriodIndex is range-like in that all Periods
        between start and end are present, in order.
        """
        if len(self) == 0:
            return True
        if not self.is_monotonic_increasing:
            raise ValueError("Index is not monotonic")
        values = self.asi8
        return ((values[1:] - values[:-1]) < 2).all()

    @property
    def inferred_type(self) -> str:
        # b/c data is represented as ints make sure we can't have ambiguous
        # indexing
        return "period"

    # ------------------------------------------------------------------------
    # Indexing Methods

    def _convert_tolerance(self, tolerance, target):
        # Returned tolerance must be in dtype/units so that
        #  `|self._get_engine_target() - target._engine_target()| <= tolerance`
        #  is meaningful.  Since PeriodIndex returns int64 for engine_target,
        #  we may need to convert timedelta64 tolerance to int64.
        tolerance = super()._convert_tolerance(tolerance, target)

        if self.dtype == target.dtype:
            # convert tolerance to i8
            tolerance = self._maybe_convert_timedelta(tolerance)

        return tolerance

    def get_loc(self, key, method=None, tolerance=None):
        """
        Get integer location for requested label.

        Parameters
        ----------
        key : Period, NaT, str, or datetime
            String or datetime key must be parsable as Period.

        Returns
        -------
        loc : int or ndarray[int64]

        Raises
        ------
        KeyError
            Key is not present in the index.
        TypeError
            If key is listlike or otherwise not hashable.
        """
        orig_key = key

        if not is_scalar(key):
            raise InvalidIndexError(key)

        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT

        elif isinstance(key, str):

            try:
                loc = self._get_string_slice(key)
                return loc
            except (TypeError, ValueError):
                pass

            try:
                asdt, reso_str = parse_time_string(key, self.freq)
            except (ValueError, DateParseError) as err:
                # A string with invalid format
                raise KeyError(f"Cannot interpret '{key}' as period") from err

            reso = Resolution.from_attrname(reso_str)
            grp = reso.freq_group.value
            freqn = self.dtype.freq_group_code

            # _get_string_slice will handle cases where grp < freqn
            assert grp >= freqn

            # BusinessDay is a bit strange. It has a *lower* code, but we never parse
            # a string as "BusinessDay" resolution, just Day.
            if grp == freqn or (
                reso == Resolution.RESO_DAY and self.dtype.freq.name == "B"
            ):
                key = Period(asdt, freq=self.freq)
                loc = self.get_loc(key, method=method, tolerance=tolerance)
                return loc
            elif method is None:
                raise KeyError(key)
            else:
                key = asdt

        elif isinstance(key, Period):
            sfreq = self.freq
            kfreq = key.freq
            if not (
                sfreq.n == kfreq.n
                and sfreq._period_dtype_code == kfreq._period_dtype_code
            ):
                # GH#42247 For the subset of DateOffsets that can be Period freqs,
                #  checking these two attributes is sufficient to check equality,
                #  and much more performant than `self.freq == key.freq`
                raise KeyError(key)
        elif isinstance(key, datetime):
            try:
                key = Period(key, freq=self.freq)
            except ValueError as err:
                # we cannot construct the Period
                raise KeyError(orig_key) from err
        else:
            # in particular integer, which Period constructor would cast to string
            raise KeyError(key)

        try:
            key = Period(key, freq=self.freq)
        except ValueError as err:
            # we cannot construct the Period
            raise KeyError(orig_key) from err

        try:
            return Index.get_loc(self, key, method, tolerance)
        except KeyError as err:
            raise KeyError(orig_key) from err

    def _maybe_cast_slice_bound(self, label, side: str, kind=lib.no_default):
        """
        If label is a string or a datetime, cast it to Period.ordinal according
        to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'loc', 'getitem'}, or None

        Returns
        -------
        bound : Period or object

        Notes
        -----
        Value of `side` parameter should be validated in caller.

        """
        assert kind in ["loc", "getitem", None, lib.no_default]
        self._deprecated_arg(kind, "kind", "_maybe_cast_slice_bound")

        if isinstance(label, datetime):
            return Period(label, freq=self.freq)
        elif isinstance(label, str):
            try:
                parsed, reso_str = parse_time_string(label, self.freq)
            except ValueError as err:
                # string cannot be parsed as datetime-like
                raise self._invalid_indexer("slice", label) from err

            reso = Resolution.from_attrname(reso_str)
            lower, upper = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == "left" else upper
        elif not isinstance(label, self._data._recognized_scalars):
            raise self._invalid_indexer("slice", label)

        return label

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime):
        grp = reso.freq_group
        iv = Period(parsed, freq=grp.value)
        return (iv.asfreq(self.freq, how="start"), iv.asfreq(self.freq, how="end"))

    def _validate_partial_date_slice(self, reso: Resolution):
        assert isinstance(reso, Resolution), (type(reso), reso)
        grp = reso.freq_group
        freqn = self.dtype.freq_group_code

        if not grp.value < freqn:
            # TODO: we used to also check for
            #  reso in ["day", "hour", "minute", "second"]
            #  why is that check not needed?
            raise ValueError

    def _get_string_slice(self, key: str):
        parsed, reso_str = parse_time_string(key, self.freq)
        reso = Resolution.from_attrname(reso_str)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err


def period_range(
    start=None, end=None, periods: int | None = None, freq=None, name=None
) -> PeriodIndex:
    """
    Return a fixed frequency PeriodIndex.

    The day (calendar) is the default frequency.

    Parameters
    ----------
    start : str or period-like, default None
        Left bound for generating periods.
    end : str or period-like, default None
        Right bound for generating periods.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.
    name : str, default None
        Name of the resulting PeriodIndex.

    Returns
    -------
    PeriodIndex

    Notes
    -----
    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two
    must be specified.

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
    PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
             '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
             '2018-01'],
            dtype='period[M]')

    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor
    endpoints for a ``PeriodIndex`` with frequency matching that of the
    ``period_range`` constructor.

    >>> pd.period_range(start=pd.Period('2017Q1', freq='Q'),
    ...                 end=pd.Period('2017Q2', freq='Q'), freq='M')
    PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'],
                dtype='period[M]')
    """
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
    if freq is None and (not isinstance(start, Period) and not isinstance(end, Period)):
        freq = "D"

    data, freq = PeriodArray._generate_range(start, end, periods, freq, fields={})
    data = PeriodArray(data, freq=freq)
    return PeriodIndex(data, name=name)
