"""
Base and utility classes for tseries type pandas objects.
"""
from __future__ import annotations

from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np

from pandas._libs import (
    NaT,
    Timedelta,
    iNaT,
    lib,
)
from pandas._libs.tslibs import (
    BaseOffset,
    NaTType,
    Resolution,
    Tick,
)
from pandas._typing import (
    Callable,
    final,
)
from pandas.compat.numpy import function as nv
from pandas.util._decorators import (
    Appender,
    cache_readonly,
    doc,
)

from pandas.core.dtypes.common import (
    is_categorical_dtype,
    is_dtype_equal,
    is_integer,
    is_list_like,
    is_period_dtype,
)
from pandas.core.dtypes.concat import concat_compat

from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
)
from pandas.core.indexes.extension import (
    NDArrayBackedExtensionIndex,
    inherit_names,
    make_wrapped_arith_op,
)
from pandas.core.indexes.numeric import Int64Index
from pandas.core.tools.timedeltas import to_timedelta

if TYPE_CHECKING:
    from pandas import CategoricalIndex

_index_doc_kwargs = dict(ibase._index_doc_kwargs)

_T = TypeVar("_T", bound="DatetimeIndexOpsMixin")


@inherit_names(
    ["inferred_freq", "_resolution_obj", "resolution"],
    DatetimeLikeArrayMixin,
    cache=True,
)
@inherit_names(["mean", "asi8", "freq", "freqstr"], DatetimeLikeArrayMixin)
class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """

    _is_numeric_dtype = False
    _can_hold_strings = False
    _data: DatetimeArray | TimedeltaArray | PeriodArray
    freq: BaseOffset | None
    freqstr: str | None
    _resolution_obj: Resolution
    _bool_ops: list[str] = []
    _field_ops: list[str] = []

    # error: "Callable[[Any], Any]" has no attribute "fget"
    hasnans = cache_readonly(
        DatetimeLikeArrayMixin._hasnans.fget  # type: ignore[attr-defined]
    )
    _hasnans = hasnans  # for index / array -agnostic code

    @property
    def _is_all_dates(self) -> bool:
        return True

    # ------------------------------------------------------------------------
    # Abstract data attributes

    @property
    def values(self) -> np.ndarray:
        # Note: PeriodArray overrides this to return an ndarray of objects.
        return self._data._ndarray

    def __array_wrap__(self, result, context=None):
        """
        Gets called after a ufunc and other functions.
        """
        out = super().__array_wrap__(result, context=context)
        if isinstance(out, DatetimeTimedeltaMixin) and self.freq is not None:
            out = out._with_freq("infer")
        return out

    # ------------------------------------------------------------------------

    def equals(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False
        elif other.dtype.kind in ["f", "i", "u", "c"]:
            return False
        elif not isinstance(other, type(self)):
            should_try = False
            inferable = self._data._infer_matches
            if other.dtype == object:
                should_try = other.inferred_type in inferable
            elif is_categorical_dtype(other.dtype):
                other = cast("CategoricalIndex", other)
                should_try = other.categories.inferred_type in inferable

            if should_try:
                try:
                    other = type(self)(other)
                except (ValueError, TypeError, OverflowError):
                    # e.g.
                    #  ValueError -> cannot parse str entry, or OutOfBoundsDatetime
                    #  TypeError  -> trying to convert IntervalIndex to DatetimeIndex
                    #  OverflowError -> Index([very_large_timedeltas])
                    return False

        if not is_dtype_equal(self.dtype, other.dtype):
            # have different timezone
            return False

        return np.array_equal(self.asi8, other.asi8)

    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(self, indices, axis=0, allow_fill=True, fill_value=None, **kwargs):
        nv.validate_take((), kwargs)
        indices = np.asarray(indices, dtype=np.intp)

        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))

        result = NDArrayBackedExtensionIndex.take(
            self, indices, axis, allow_fill, fill_value, **kwargs
        )
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result

    _can_hold_na = True

    _na_value: NaTType = NaT
    """The expected NA value to use with this index."""

    def _convert_tolerance(self, tolerance, target):
        tolerance = np.asarray(to_timedelta(tolerance).to_numpy())
        return super()._convert_tolerance(tolerance, target)

    def tolist(self) -> list:
        """
        Return a list of the underlying data.
        """
        return list(self.astype(object))

    def min(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return the minimum value of the Index or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Series.min : Return the minimum value in a Series.
        """
        nv.validate_min(args, kwargs)
        nv.validate_minmax_axis(axis)

        if not len(self):
            return self._na_value

        i8 = self.asi8

        if len(i8) and self.is_monotonic_increasing:
            # quick check
            if i8[0] != iNaT:
                return self._data._box_func(i8[0])

        if self.hasnans:
            if not skipna:
                return self._na_value
            i8 = i8[~self._isnan]

        if not len(i8):
            return self._na_value

        min_stamp = i8.min()
        return self._data._box_func(min_stamp)

    def argmin(self, axis=None, skipna=True, *args, **kwargs):
        """
        Returns the indices of the minimum values along an axis.

        See `numpy.ndarray.argmin` for more information on the
        `axis` parameter.

        See Also
        --------
        numpy.ndarray.argmin
        """
        nv.validate_argmin(args, kwargs)
        nv.validate_minmax_axis(axis)

        i8 = self.asi8
        if self.hasnans:
            mask = self._isnan
            if mask.all() or not skipna:
                return -1
            i8 = i8.copy()
            i8[mask] = np.iinfo("int64").max
        return i8.argmin()

    def max(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return the maximum value of the Index or maximum along
        an axis.

        See Also
        --------
        numpy.ndarray.max
        Series.max : Return the maximum value in a Series.
        """
        nv.validate_max(args, kwargs)
        nv.validate_minmax_axis(axis)

        if not len(self):
            return self._na_value

        i8 = self.asi8

        if len(i8) and self.is_monotonic:
            # quick check
            if i8[-1] != iNaT:
                return self._data._box_func(i8[-1])

        if self.hasnans:
            if not skipna:
                return self._na_value
            i8 = i8[~self._isnan]

        if not len(i8):
            return self._na_value

        max_stamp = i8.max()
        return self._data._box_func(max_stamp)

    def argmax(self, axis=None, skipna=True, *args, **kwargs):
        """
        Returns the indices of the maximum values along an axis.

        See `numpy.ndarray.argmax` for more information on the
        `axis` parameter.

        See Also
        --------
        numpy.ndarray.argmax
        """
        nv.validate_argmax(args, kwargs)
        nv.validate_minmax_axis(axis)

        i8 = self.asi8
        if self.hasnans:
            mask = self._isnan
            if mask.all() or not skipna:
                return -1
            i8 = i8.copy()
            i8[mask] = 0
        return i8.argmax()

    # --------------------------------------------------------------------
    # Rendering Methods

    def format(
        self,
        name: bool = False,
        formatter: Callable | None = None,
        na_rep: str = "NaT",
        date_format: str | None = None,
    ) -> list[str]:
        """
        Render a string representation of the Index.
        """
        header = []
        if name:
            header.append(
                ibase.pprint_thing(self.name, escape_chars=("\t", "\r", "\n"))
                if self.name is not None
                else ""
            )

        if formatter is not None:
            return header + list(self.map(formatter))

        return self._format_with_header(header, na_rep=na_rep, date_format=date_format)

    def _format_with_header(
        self, header: list[str], na_rep: str = "NaT", date_format: str | None = None
    ) -> list[str]:
        return header + list(
            self._format_native_types(na_rep=na_rep, date_format=date_format)
        )

    @property
    def _formatter_func(self):
        return self._data._formatter()

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs = super()._format_attrs()
        for attrib in self._attributes:
            if attrib == "freq":
                freq = self.freqstr
                if freq is not None:
                    freq = repr(freq)
                # Argument 1 to "append" of "list" has incompatible type
                # "Tuple[str, Optional[str]]"; expected "Tuple[str, Union[str, int]]"
                attrs.append(("freq", freq))  # type: ignore[arg-type]
        return attrs

    def _summary(self, name=None) -> str:
        """
        Return a summarized representation.

        Parameters
        ----------
        name : str
            Name to use in the summary representation.

        Returns
        -------
        str
            Summarized representation of the index.
        """
        formatter = self._formatter_func
        if len(self) > 0:
            index_summary = f", {formatter(self[0])} to {formatter(self[-1])}"
        else:
            index_summary = ""

        if name is None:
            name = type(self).__name__
        result = f"{name}: {len(self)} entries{index_summary}"
        if self.freq:
            result += f"\nFreq: {self.freqstr}"

        # display as values, not quoted
        result = result.replace("'", "")
        return result

    # --------------------------------------------------------------------
    # Indexing Methods

    def _validate_partial_date_slice(self, reso: Resolution):
        raise NotImplementedError

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime):
        raise NotImplementedError

    @final
    def _partial_date_slice(
        self,
        reso: Resolution,
        parsed: datetime,
    ):
        """
        Parameters
        ----------
        reso : Resolution
        parsed : datetime

        Returns
        -------
        slice or ndarray[intp]
        """
        self._validate_partial_date_slice(reso)

        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        vals = self._data._ndarray
        unbox = self._data._unbox

        if self.is_monotonic_increasing:

            if len(self) and (
                (t1 < self[0] and t2 < self[0]) or (t1 > self[-1] and t2 > self[-1])
            ):
                # we are out of range
                raise KeyError

            # TODO: does this depend on being monotonic _increasing_?

            # a monotonic (sorted) series can be sliced
            left = vals.searchsorted(unbox(t1), side="left")
            right = vals.searchsorted(unbox(t2), side="right")
            return slice(left, right)

        else:
            lhs_mask = vals >= unbox(t1)
            rhs_mask = vals <= unbox(t2)

            # try to find the dates
            return (lhs_mask & rhs_mask).nonzero()[0]

    # --------------------------------------------------------------------
    # Arithmetic Methods

    __add__ = make_wrapped_arith_op("__add__")
    __sub__ = make_wrapped_arith_op("__sub__")
    __radd__ = make_wrapped_arith_op("__radd__")
    __rsub__ = make_wrapped_arith_op("__rsub__")
    __pow__ = make_wrapped_arith_op("__pow__")
    __rpow__ = make_wrapped_arith_op("__rpow__")
    __mul__ = make_wrapped_arith_op("__mul__")
    __rmul__ = make_wrapped_arith_op("__rmul__")
    __floordiv__ = make_wrapped_arith_op("__floordiv__")
    __rfloordiv__ = make_wrapped_arith_op("__rfloordiv__")
    __mod__ = make_wrapped_arith_op("__mod__")
    __rmod__ = make_wrapped_arith_op("__rmod__")
    __divmod__ = make_wrapped_arith_op("__divmod__")
    __rdivmod__ = make_wrapped_arith_op("__rdivmod__")
    __truediv__ = make_wrapped_arith_op("__truediv__")
    __rtruediv__ = make_wrapped_arith_op("__rtruediv__")

    def shift(self: _T, periods: int = 1, freq=None) -> _T:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        arr = self._data.view()
        arr._freq = self.freq
        result = arr._time_shift(periods, freq=freq)
        return type(self)(result, name=self.name)

    # --------------------------------------------------------------------
    # List-like Methods

    def _get_delete_freq(self, loc: int | slice | Sequence[int]):
        """
        Find the `freq` for self.delete(loc).
        """
        freq = None
        if is_period_dtype(self.dtype):
            freq = self.freq
        elif self.freq is not None:
            if is_integer(loc):
                if loc in (0, -len(self), -1, len(self) - 1):
                    freq = self.freq
            else:
                if is_list_like(loc):
                    # error: Incompatible types in assignment (expression has
                    # type "Union[slice, ndarray]", variable has type
                    # "Union[int, slice, Sequence[int]]")
                    loc = lib.maybe_indices_to_slice(  # type: ignore[assignment]
                        np.asarray(loc, dtype=np.intp), len(self)
                    )
                if isinstance(loc, slice) and loc.step in (1, None):
                    if loc.start in (0, None) or loc.stop in (len(self), None):
                        freq = self.freq
        return freq

    def _get_insert_freq(self, loc: int, item):
        """
        Find the `freq` for self.insert(loc, item).
        """
        value = self._data._validate_scalar(item)
        item = self._data._box_func(value)

        freq = None
        if is_period_dtype(self.dtype):
            freq = self.freq
        elif self.freq is not None:
            # freq can be preserved on edge cases
            if self.size:
                if item is NaT:
                    pass
                elif (loc == 0 or loc == -len(self)) and item + self.freq == self[0]:
                    freq = self.freq
                elif (loc == len(self)) and item - self.freq == self[-1]:
                    freq = self.freq
            else:
                # Adding a single item to an empty index may preserve freq
                if self.freq.is_on_offset(item):
                    freq = self.freq
        return freq

    @doc(NDArrayBackedExtensionIndex.delete)
    def delete(self: _T, loc) -> _T:
        result = super().delete(loc)
        result._data._freq = self._get_delete_freq(loc)
        return result

    @doc(NDArrayBackedExtensionIndex.insert)
    def insert(self, loc: int, item):
        result = super().insert(loc, item)
        if isinstance(result, type(self)):
            # i.e. parent class method did not cast
            result._data._freq = self._get_insert_freq(loc, item)
        return result

    # --------------------------------------------------------------------
    # Join/Set Methods

    def _get_join_freq(self, other):
        """
        Get the freq to attach to the result of a join operation.
        """
        if is_period_dtype(self.dtype):
            freq = self.freq
        else:
            self = cast(DatetimeTimedeltaMixin, self)
            freq = self.freq if self._can_fast_union(other) else None
        return freq

    def _wrap_joined_index(self, joined, other):
        assert other.dtype == self.dtype, (other.dtype, self.dtype)
        result = super()._wrap_joined_index(joined, other)
        result._data._freq = self._get_join_freq(other)
        return result

    def _get_join_target(self) -> np.ndarray:
        return self._data._ndarray.view("i8")

    def _from_join_target(self, result: np.ndarray):
        # view e.g. i8 back to M8[ns]
        result = result.view(self._data._ndarray.dtype)
        return self._data._from_backing_data(result)

    # --------------------------------------------------------------------

    @doc(Index._maybe_cast_listlike_indexer)
    def _maybe_cast_listlike_indexer(self, keyarr):
        try:
            res = self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            res = com.asarray_tuplesafe(keyarr)
        return Index(res, dtype=res.dtype)


class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin):
    """
    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,
    but not PeriodIndex
    """

    _data: DatetimeArray | TimedeltaArray
    _comparables = ["name", "freq"]
    _attributes = ["name", "freq"]

    # Compat for frequency inference, see GH#23789
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    def _with_freq(self, freq):
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self._name)

    @property
    def _has_complex_internals(self) -> bool:
        # used to avoid libreduction code paths, which raise or require conversion
        return True

    def is_type_compatible(self, kind: str) -> bool:
        return kind in self._data._infer_matches

    # --------------------------------------------------------------------
    # Set Operation Methods

    def _intersection(self, other: Index, sort=False) -> Index:
        """
        intersection specialized to the case with matching dtypes.
        """
        other = cast("DatetimeTimedeltaMixin", other)
        if len(self) == 0:
            return self.copy()._get_reconciled_name_object(other)
        if len(other) == 0:
            return other.copy()._get_reconciled_name_object(self)

        elif not self._can_fast_intersect(other):
            result = Index._intersection(self, other, sort=sort)
            # We need to invalidate the freq because Index._intersection
            #  uses _shallow_copy on a view of self._data, which will preserve
            #  self.freq if we're not careful.
            # At this point we should have result.dtype == self.dtype
            #  and type(result) is type(self._data)
            result = self._wrap_setop_result(other, result)
            return result._with_freq(None)._with_freq("infer")

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        # after sorting, the intersection always starts with the right index
        # and ends with the index of which the last elements is smallest
        end = min(left[-1], right[-1])
        start = right[0]

        if end < start:
            result = self[:0]
        else:
            lslice = slice(*left.slice_locs(start, end))
            result = left._values[lslice]

        return result

    def _can_fast_intersect(self: _T, other: _T) -> bool:
        # Note: we only get here with len(self) > 0 and len(other) > 0
        if self.freq is None:
            return False

        elif other.freq != self.freq:
            return False

        elif not self.is_monotonic_increasing:
            # Because freq is not None, we must then be monotonic decreasing
            return False

        elif self.freq.is_anchored():
            # this along with matching freqs ensure that we "line up",
            #  so intersection will preserve freq
            # GH#42104
            return self.freq.n == 1

        elif isinstance(self.freq, Tick):
            # We "line up" if and only if the difference between two of our points
            #  is a multiple of our freq
            diff = self[0] - other[0]
            remainder = diff % self.freq.delta
            return remainder == Timedelta(0)

        # GH#42104
        return self.freq.n == 1

    def _can_fast_union(self: _T, other: _T) -> bool:
        # Assumes that type(self) == type(other), as per the annotation
        # The ability to fast_union also implies that `freq` should be
        #  retained on union.
        freq = self.freq

        if freq is None or freq != other.freq:
            return False

        if not self.is_monotonic_increasing:
            # Because freq is not None, we must then be monotonic decreasing
            # TODO: do union on the reversed indexes?
            return False

        if len(self) == 0 or len(other) == 0:
            return True

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        right_start = right[0]
        left_end = left[-1]

        # Only need to "adjoin", not overlap
        return (right_start == left_end + freq) or right_start in left

    def _fast_union(self: _T, other: _T, sort=None) -> _T:
        if len(other) == 0:
            return self.view(type(self))

        if len(self) == 0:
            return other.view(type(self))

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        elif sort is False:
            # TDIs are not in the "correct" order and we don't want
            #  to sort but want to remove overlaps
            left, right = self, other
            left_start = left[0]
            loc = right.searchsorted(left_start, side="left")
            # error: Slice index must be an integer or None
            right_chunk = right._values[:loc]  # type: ignore[misc]
            dates = concat_compat((left._values, right_chunk))
            # With sort being False, we can't infer that result.freq == self.freq
            # TODO: no tests rely on the _with_freq("infer"); needed?
            result = type(self)._simple_new(dates, name=self.name)
            result = result._with_freq("infer")
            return result
        else:
            left, right = other, self

        left_end = left[-1]
        right_end = right[-1]

        # concatenate
        if left_end < right_end:
            loc = right.searchsorted(left_end, side="right")
            # error: Slice index must be an integer or None
            right_chunk = right._values[loc:]  # type: ignore[misc]
            dates = concat_compat([left._values, right_chunk])
            # The can_fast_union check ensures that the result.freq
            #  should match self.freq
            dates = type(self._data)(dates, freq=self.freq)
            result = type(self)._simple_new(dates)
            return result
        else:
            return left

    def _union(self, other, sort):
        # We are called by `union`, which is responsible for this validation
        assert isinstance(other, type(self))
        assert self.dtype == other.dtype

        if self._can_fast_union(other):
            result = self._fast_union(other, sort=sort)
            # in the case with sort=None, the _can_fast_union check ensures
            #  that result.freq == self.freq
            return result
        else:
            i8self = Int64Index._simple_new(self.asi8)
            i8other = Int64Index._simple_new(other.asi8)
            i8result = i8self._union(i8other, sort=sort)
            result = type(self)(i8result, dtype=self.dtype, freq="infer")
            return result

    # --------------------------------------------------------------------
    # Join Methods
    _join_precedence = 10

    def join(
        self,
        other,
        how: str = "left",
        level=None,
        return_indexers: bool = False,
        sort: bool = False,
    ):
        """
        See Index.join
        """
        pself, pother = self._maybe_promote(other)
        if pself is not self or pother is not other:
            return pself.join(
                pother, how=how, level=level, return_indexers=return_indexers, sort=sort
            )

        self._maybe_utc_convert(other)  # raises if we dont have tzawareness compat
        return Index.join(
            self,
            other,
            how=how,
            level=level,
            return_indexers=return_indexers,
            sort=sort,
        )

    def _maybe_utc_convert(self: _T, other: Index) -> tuple[_T, Index]:
        # Overridden by DatetimeIndex
        return self, other
