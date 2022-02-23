""" implement the TimedeltaIndex """
from __future__ import annotations

from pandas._libs import (
    index as libindex,
    lib,
)
from pandas._libs.tslibs import (
    Timedelta,
    to_offset,
)
from pandas._typing import (
    DtypeObj,
    Optional,
)
from pandas.errors import InvalidIndexError

from pandas.core.dtypes.common import (
    TD64NS_DTYPE,
    is_scalar,
    is_timedelta64_dtype,
)

from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names


@inherit_names(
    ["__neg__", "__pos__", "__abs__", "total_seconds", "round", "floor", "ceil"]
    + TimedeltaArray._field_ops,
    TimedeltaArray,
    wrap=True,
)
@inherit_names(
    [
        "components",
        "to_pytimedelta",
        "sum",
        "std",
        "median",
        "_format_native_types",
    ],
    TimedeltaArray,
)
class TimedeltaIndex(DatetimeTimedeltaMixin):
    """
    Immutable ndarray of timedelta64 data, represented internally as int64, and
    which can be boxed to timedelta objects.

    Parameters
    ----------
    data  : array-like (1-dimensional), optional
        Optional timedelta-like data to construct index with.
    unit : unit of the arg (D,h,m,s,ms,us,ns) denote the unit, optional
        Which is an integer/float number.
    freq : str or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        'infer' can be passed in order to set the frequency of the index as the
        inferred frequency upon creation.
    copy  : bool
        Make a copy of input ndarray.
    name : object
        Name to be stored in the index.

    Attributes
    ----------
    days
    seconds
    microseconds
    nanoseconds
    components
    inferred_freq

    Methods
    -------
    to_pytimedelta
    to_series
    round
    floor
    ceil
    to_frame
    mean

    See Also
    --------
    Index : The base pandas Index type.
    Timedelta : Represents a duration between two dates or times.
    DatetimeIndex : Index of datetime64 data.
    PeriodIndex : Index of Period data.
    timedelta_range : Create a fixed-frequency TimedeltaIndex.

    Notes
    -----
    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.
    """

    _typ = "timedeltaindex"

    _data_cls = TimedeltaArray
    _engine_type = libindex.TimedeltaEngine

    _data: TimedeltaArray

    # -------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        unit=None,
        freq=lib.no_default,
        closed=None,
        dtype=TD64NS_DTYPE,
        copy=False,
        name=None,
    ):
        name = maybe_extract_name(name, data, cls)

        if is_scalar(data):
            raise cls._scalar_data_error(data)

        if unit in {"Y", "y", "M"}:
            raise ValueError(
                "Units 'M', 'Y', and 'y' are no longer supported, as they do not "
                "represent unambiguous timedelta values durations."
            )

        if isinstance(data, TimedeltaArray) and freq is lib.no_default:
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name)

        if isinstance(data, TimedeltaIndex) and freq is lib.no_default and name is None:
            if copy:
                return data.copy()
            else:
                return data._view()

        # - Cases checked above all return/raise before reaching here - #

        tdarr = TimedeltaArray._from_sequence_not_strict(
            data, freq=freq, unit=unit, dtype=dtype, copy=copy
        )
        return cls._simple_new(tdarr, name=name)

    # -------------------------------------------------------------------

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        return is_timedelta64_dtype(dtype)  # aka self._data._is_recognized_dtype

    # -------------------------------------------------------------------
    # Indexing Methods

    def get_loc(self, key, method=None, tolerance=None):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int, slice, or ndarray[int]
        """
        if not is_scalar(key):
            raise InvalidIndexError(key)

        try:
            key = self._data._validate_scalar(key, unbox=False)
        except TypeError as err:
            raise KeyError(key) from err

        return Index.get_loc(self, key, method, tolerance)

    def _maybe_cast_slice_bound(self, label, side: str, kind=lib.no_default):
        """
        If label is a string, cast it to timedelta according to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'loc', 'getitem'} or None

        Returns
        -------
        label : object
        """
        assert kind in ["loc", "getitem", None, lib.no_default]
        self._deprecated_arg(kind, "kind", "_maybe_cast_slice_bound")

        if isinstance(label, str):
            try:
                parsed = Timedelta(label)
            except ValueError as err:
                # e.g. 'unit abbreviation w/o a number'
                raise self._invalid_indexer("slice", label) from err

            # The next two lines are analogous to DTI/PI._parsed_str_to_bounds
            lower = parsed.round(parsed.resolution_string)
            upper = lower + to_offset(parsed.resolution_string) - Timedelta(1, "ns")
            return lower if side == "left" else upper
        elif not isinstance(label, self._data._recognized_scalars):
            raise self._invalid_indexer("slice", label)

        return label

    # -------------------------------------------------------------------

    @property
    def inferred_type(self) -> str:
        return "timedelta64"


def timedelta_range(
    start=None,
    end=None,
    periods: Optional[int] = None,
    freq=None,
    name=None,
    closed=None,
) -> TimedeltaIndex:
    """
    Return a fixed frequency TimedeltaIndex, with day as the default
    frequency.

    Parameters
    ----------
    start : str or timedelta-like, default None
        Left bound for generating timedeltas.
    end : str or timedelta-like, default None
        Right bound for generating timedeltas.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5H'.
    name : str, default None
        Name of the resulting TimedeltaIndex.
    closed : str, default None
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None).

    Returns
    -------
    TimedeltaIndex

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``TimedeltaIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.timedelta_range(start='1 day', periods=4)
    TimedeltaIndex(['1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')

    The ``closed`` parameter specifies which endpoint is included.  The default
    behavior is to include both endpoints.

    >>> pd.timedelta_range(start='1 day', periods=4, closed='right')
    TimedeltaIndex(['2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')

    The ``freq`` parameter specifies the frequency of the TimedeltaIndex.
    Only fixed frequencies can be passed, non-fixed frequencies such as
    'M' (month end) will raise.

    >>> pd.timedelta_range(start='1 day', end='2 days', freq='6H')
    TimedeltaIndex(['1 days 00:00:00', '1 days 06:00:00', '1 days 12:00:00',
                    '1 days 18:00:00', '2 days 00:00:00'],
                   dtype='timedelta64[ns]', freq='6H')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).

    >>> pd.timedelta_range(start='1 day', end='5 days', periods=4)
    TimedeltaIndex(['1 days 00:00:00', '2 days 08:00:00', '3 days 16:00:00',
                    '5 days 00:00:00'],
                   dtype='timedelta64[ns]', freq=None)
    """
    if freq is None and com.any_none(periods, start, end):
        freq = "D"

    freq, _ = dtl.maybe_infer_freq(freq)
    tdarr = TimedeltaArray._generate_range(start, end, periods, freq, closed=closed)
    return TimedeltaIndex._simple_new(tdarr, name=name)
