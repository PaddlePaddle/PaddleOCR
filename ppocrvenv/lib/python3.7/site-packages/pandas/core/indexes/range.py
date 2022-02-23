from __future__ import annotations

from datetime import timedelta
import operator
from sys import getsizeof
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    List,
    cast,
)
import warnings

import numpy as np

from pandas._libs import index as libindex
from pandas._libs.lib import no_default
from pandas._typing import Dtype
from pandas.compat.numpy import function as nv
from pandas.util._decorators import (
    cache_readonly,
    doc,
)
from pandas.util._exceptions import rewrite_exception

from pandas.core.dtypes.common import (
    ensure_platform_int,
    ensure_python_int,
    is_float,
    is_integer,
    is_scalar,
    is_signed_integer_dtype,
    is_timedelta64_dtype,
)
from pandas.core.dtypes.generic import ABCTimedeltaIndex

from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.numeric import (
    Float64Index,
    Int64Index,
    NumericIndex,
)
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from pandas import Index

_empty_range = range(0)


class RangeIndex(NumericIndex):
    """
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of Int64Index limited to
    representing monotonic ranges. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), range, or other RangeIndex instance
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int (default: 0)
    step : int (default: 1)
    dtype : np.int64
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.
    Int64Index : Index of int64 data.
    """

    _typ = "rangeindex"
    _engine_type = libindex.Int64Engine
    _dtype_validation_metadata = (is_signed_integer_dtype, "signed integer")
    _range: range

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        start=None,
        stop=None,
        step=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable = None,
    ) -> RangeIndex:
        cls._validate_dtype(dtype)
        name = maybe_extract_name(name, start, cls)

        # RangeIndex
        if isinstance(start, RangeIndex):
            return start.copy(name=name)
        elif isinstance(start, range):
            return cls._simple_new(start, name=name)

        # validate the arguments
        if com.all_none(start, stop, step):
            raise TypeError("RangeIndex(...) must be called with integers")

        start = ensure_python_int(start) if start is not None else 0

        if stop is None:
            start, stop = 0, start
        else:
            stop = ensure_python_int(stop)

        step = ensure_python_int(step) if step is not None else 1
        if step == 0:
            raise ValueError("Step must not be zero")

        rng = range(start, stop, step)
        return cls._simple_new(rng, name=name)

    @classmethod
    def from_range(
        cls, data: range, name=None, dtype: Dtype | None = None
    ) -> RangeIndex:
        """
        Create RangeIndex from a range object.

        Returns
        -------
        RangeIndex
        """
        if not isinstance(data, range):
            raise TypeError(
                f"{cls.__name__}(...) must be called with object coercible to a "
                f"range, {repr(data)} was passed"
            )
        cls._validate_dtype(dtype)
        return cls._simple_new(data, name=name)

    @classmethod
    def _simple_new(cls, values: range, name: Hashable = None) -> RangeIndex:
        result = object.__new__(cls)

        assert isinstance(values, range)

        result._range = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        return result

    # --------------------------------------------------------------------

    @cache_readonly
    def _constructor(self) -> type[Int64Index]:
        """return the class to use for construction"""
        return Int64Index

    @cache_readonly
    def _data(self) -> np.ndarray:
        """
        An int array that for performance reasons is created only when needed.

        The constructed array is saved in ``_cache``.
        """
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    @cache_readonly
    def _cached_int64index(self) -> Int64Index:
        return Int64Index._simple_new(self._data, name=self.name)

    @property
    def _int64index(self) -> Int64Index:
        # wrap _cached_int64index so we can be sure its name matches self.name
        res = self._cached_int64index
        res._name = self._name
        return res

    def _get_data_as_items(self):
        """return a list of tuples of start, stop, step"""
        rng = self._range
        return [("start", rng.start), ("stop", rng.stop), ("step", rng.step)]

    def __reduce__(self):
        d = self._get_attributes_dict()
        d.update(dict(self._get_data_as_items()))
        return ibase._new_Index, (type(self), d), None

    # --------------------------------------------------------------------
    # Rendering Methods

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr, formatted_value)
        """
        attrs = self._get_data_as_items()
        if self.name is not None:
            attrs.append(("name", ibase.default_pprint(self.name)))
        return attrs

    def _format_data(self, name=None):
        # we are formatting thru the attributes
        return None

    def _format_with_header(self, header: list[str], na_rep: str = "NaN") -> list[str]:
        if not len(self._range):
            return header
        first_val_str = str(self._range[0])
        last_val_str = str(self._range[-1])
        max_length = max(len(first_val_str), len(last_val_str))

        return header + [f"{x:<{max_length}}" for x in self._range]

    # --------------------------------------------------------------------
    _deprecation_message = (
        "RangeIndex.{} is deprecated and will be "
        "removed in a future version. Use RangeIndex.{} "
        "instead"
    )

    @property
    def start(self) -> int:
        """
        The value of the `start` parameter (``0`` if this was not supplied).
        """
        # GH 25710
        return self._range.start

    @property
    def _start(self) -> int:
        """
        The value of the `start` parameter (``0`` if this was not supplied).

         .. deprecated:: 0.25.0
            Use ``start`` instead.
        """
        warnings.warn(
            self._deprecation_message.format("_start", "start"),
            FutureWarning,
            stacklevel=2,
        )
        return self.start

    @property
    def stop(self) -> int:
        """
        The value of the `stop` parameter.
        """
        return self._range.stop

    @property
    def _stop(self) -> int:
        """
        The value of the `stop` parameter.

         .. deprecated:: 0.25.0
            Use ``stop`` instead.
        """
        # GH 25710
        warnings.warn(
            self._deprecation_message.format("_stop", "stop"),
            FutureWarning,
            stacklevel=2,
        )
        return self.stop

    @property
    def step(self) -> int:
        """
        The value of the `step` parameter (``1`` if this was not supplied).
        """
        # GH 25710
        return self._range.step

    @property
    def _step(self) -> int:
        """
        The value of the `step` parameter (``1`` if this was not supplied).

         .. deprecated:: 0.25.0
            Use ``step`` instead.
        """
        # GH 25710
        warnings.warn(
            self._deprecation_message.format("_step", "step"),
            FutureWarning,
            stacklevel=2,
        )
        return self.step

    @cache_readonly
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
        rng = self._range
        return getsizeof(rng) + sum(
            getsizeof(getattr(rng, attr_name))
            for attr_name in ["start", "stop", "step"]
        )

    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self.nbytes

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.int64)

    @property
    def is_unique(self) -> bool:
        """return if the index has unique values"""
        return True

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        return self._range.step > 0 or len(self) <= 1

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        return self._range.step < 0 or len(self) <= 1

    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            key = ensure_python_int(key)
        except TypeError:
            return False
        return key in self._range

    @property
    def inferred_type(self) -> str:
        return "integer"

    # --------------------------------------------------------------------
    # Indexing Methods

    @doc(Int64Index.get_loc)
    def get_loc(self, key, method=None, tolerance=None):
        if method is None and tolerance is None:
            if is_integer(key) or (is_float(key) and key.is_integer()):
                new_key = int(key)
                try:
                    return self._range.index(new_key)
                except ValueError as err:
                    raise KeyError(key) from err
            raise KeyError(key)
        return super().get_loc(key, method=method, tolerance=tolerance)

    def _get_indexer(
        self,
        target: Index,
        method: str | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> np.ndarray:
        # -> np.ndarray[np.intp]
        if com.any_not_none(method, tolerance, limit):
            return super()._get_indexer(
                target, method=method, tolerance=tolerance, limit=limit
            )

        if self.step > 0:
            start, stop, step = self.start, self.stop, self.step
        else:
            # GH 28678: work on reversed range for simplicity
            reverse = self._range[::-1]
            start, stop, step = reverse.start, reverse.stop, reverse.step

        if not is_signed_integer_dtype(target):
            # checks/conversions/roundings are delegated to general method
            return super()._get_indexer(target, method=method, tolerance=tolerance)

        target_array = np.asarray(target)
        locs = target_array - start
        valid = (locs % step == 0) & (locs >= 0) & (target_array < stop)
        locs[~valid] = -1
        locs[valid] = locs[valid] / step

        if step != self.step:
            # We reversed this range: transform to original locs
            locs[valid] = len(self) - 1 - locs[valid]
        return ensure_platform_int(locs)

    # --------------------------------------------------------------------

    def repeat(self, repeats, axis=None) -> Int64Index:
        return self._int64index.repeat(repeats, axis=axis)

    def delete(self, loc) -> Int64Index:  # type: ignore[override]
        return self._int64index.delete(loc)

    def take(
        self, indices, axis: int = 0, allow_fill: bool = True, fill_value=None, **kwargs
    ) -> Int64Index:
        with rewrite_exception("Int64Index", type(self).__name__):
            return self._int64index.take(
                indices,
                axis=axis,
                allow_fill=allow_fill,
                fill_value=fill_value,
                **kwargs,
            )

    def tolist(self) -> list[int]:
        return list(self._range)

    @doc(Int64Index.__iter__)
    def __iter__(self):
        yield from self._range

    @doc(Int64Index._shallow_copy)
    def _shallow_copy(self, values, name: Hashable = no_default):
        name = self.name if name is no_default else name

        if values.dtype.kind == "f":
            return Float64Index(values, name=name)
        return Int64Index._simple_new(values, name=name)

    def _view(self: RangeIndex) -> RangeIndex:
        result = type(self)._simple_new(self._range, name=self._name)
        result._cache = self._cache
        return result

    @doc(Int64Index.copy)
    def copy(
        self,
        name: Hashable = None,
        deep: bool = False,
        dtype: Dtype | None = None,
        names=None,
    ):
        name = self._validate_names(name=name, names=names, deep=deep)[0]
        new_index = self._rename(name=name)

        if dtype:
            warnings.warn(
                "parameter dtype is deprecated and will be removed in a future "
                "version. Use the astype method instead.",
                FutureWarning,
                stacklevel=2,
            )
            new_index = new_index.astype(dtype)
        return new_index

    def _minmax(self, meth: str):
        no_steps = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (meth == "max" and self.step < 0):
            return self.start

        return self.start + self.step * no_steps

    def min(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        """The minimum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return self._minmax("min")

    def max(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        """The maximum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_max(args, kwargs)
        return self._minmax("max")

    def argsort(self, *args, **kwargs) -> np.ndarray:
        """
        Returns the indices that would sort the index and its
        underlying data.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort
        """
        ascending = kwargs.pop("ascending", True)  # EA compat
        nv.validate_argsort(args, kwargs)

        if self._range.step > 0:
            result = np.arange(len(self), dtype=np.intp)
        else:
            result = np.arange(len(self) - 1, -1, -1, dtype=np.intp)

        if not ascending:
            result = result[::-1]
        return result

    def factorize(
        self, sort: bool = False, na_sentinel: int | None = -1
    ) -> tuple[np.ndarray, RangeIndex]:
        codes = np.arange(len(self), dtype=np.intp)
        uniques = self
        if sort and self.step < 0:
            codes = codes[::-1]
            uniques = uniques[::-1]
        return codes, uniques

    def equals(self, other: object) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if isinstance(other, RangeIndex):
            return self._range == other._range
        return super().equals(other)

    # --------------------------------------------------------------------
    # Set Operations

    def _intersection(self, other: Index, sort=False):

        if not isinstance(other, RangeIndex):
            # Int64Index
            return super()._intersection(other, sort=sort)

        if not len(self) or not len(other):
            return self._simple_new(_empty_range)

        first = self._range[::-1] if self.step < 0 else self._range
        second = other._range[::-1] if other.step < 0 else other._range

        # check whether intervals intersect
        # deals with in- and decreasing ranges
        int_low = max(first.start, second.start)
        int_high = min(first.stop, second.stop)
        if int_high <= int_low:
            return self._simple_new(_empty_range)

        # Method hint: linear Diophantine equation
        # solve intersection problem
        # performance hint: for identical step sizes, could use
        # cheaper alternative
        gcd, s, _ = self._extended_gcd(first.step, second.step)

        # check whether element sets intersect
        if (first.start - second.start) % gcd:
            return self._simple_new(_empty_range)

        # calculate parameters for the RangeIndex describing the
        # intersection disregarding the lower bounds
        tmp_start = first.start + (second.start - first.start) * first.step // gcd * s
        new_step = first.step * second.step // gcd
        new_range = range(tmp_start, int_high, new_step)
        new_index = self._simple_new(new_range)

        # adjust index to limiting interval
        new_start = new_index._min_fitting_element(int_low)
        new_range = range(new_start, new_index.stop, new_index.step)
        new_index = self._simple_new(new_range)

        if (self.step < 0 and other.step < 0) is not (new_index.step < 0):
            new_index = new_index[::-1]
        if sort is None:
            new_index = new_index.sort_values()

        return new_index

    def _min_fitting_element(self, lower_limit: int) -> int:
        """Returns the smallest element greater than or equal to the limit"""
        no_steps = -(-(lower_limit - self.start) // abs(self.step))
        return self.start + abs(self.step) * no_steps

    def _extended_gcd(self, a: int, b: int) -> tuple[int, int, int]:
        """
        Extended Euclidean algorithms to solve Bezout's identity:
           a*x + b*y = gcd(x, y)
        Finds one particular solution for x, y: s, t
        Returns: gcd, s, t
        """
        s, old_s = 0, 1
        t, old_t = 1, 0
        r, old_r = b, a
        while r:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
        return old_r, old_s, old_t

    def _union(self, other: Index, sort):
        """
        Form the union of two Index objects and sorts if possible

        Parameters
        ----------
        other : Index or array-like

        sort : False or None, default None
            Whether to sort resulting index. ``sort=None`` returns a
            monotonically increasing ``RangeIndex`` if possible or a sorted
            ``Int64Index`` if not. ``sort=False`` always returns an
            unsorted ``Int64Index``

            .. versionadded:: 0.25.0

        Returns
        -------
        union : Index
        """
        if isinstance(other, RangeIndex) and sort is None:
            start_s, step_s = self.start, self.step
            end_s = self.start + self.step * (len(self) - 1)
            start_o, step_o = other.start, other.step
            end_o = other.start + other.step * (len(other) - 1)
            if self.step < 0:
                start_s, step_s, end_s = end_s, -step_s, start_s
            if other.step < 0:
                start_o, step_o, end_o = end_o, -step_o, start_o
            if len(self) == 1 and len(other) == 1:
                step_s = step_o = abs(self.start - other.start)
            elif len(self) == 1:
                step_s = step_o
            elif len(other) == 1:
                step_o = step_s
            start_r = min(start_s, start_o)
            end_r = max(end_s, end_o)
            if step_o == step_s:
                if (
                    (start_s - start_o) % step_s == 0
                    and (start_s - end_o) <= step_s
                    and (start_o - end_s) <= step_s
                ):
                    return type(self)(start_r, end_r + step_s, step_s)
                if (
                    (step_s % 2 == 0)
                    and (abs(start_s - start_o) <= step_s / 2)
                    and (abs(end_s - end_o) <= step_s / 2)
                ):
                    return type(self)(start_r, end_r + step_s / 2, step_s / 2)
            elif step_o % step_s == 0:
                if (
                    (start_o - start_s) % step_s == 0
                    and (start_o + step_s >= start_s)
                    and (end_o - step_s <= end_s)
                ):
                    return type(self)(start_r, end_r + step_s, step_s)
            elif step_s % step_o == 0:
                if (
                    (start_s - start_o) % step_o == 0
                    and (start_s + step_o >= start_o)
                    and (end_s - step_o <= end_o)
                ):
                    return type(self)(start_r, end_r + step_o, step_o)
        return self._int64index._union(other, sort=sort)

    def _difference(self, other, sort=None):
        # optimized set operation if we have another RangeIndex
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)

        if not isinstance(other, RangeIndex):
            return super()._difference(other, sort=sort)

        res_name = ops.get_op_result_name(self, other)

        first = self._range[::-1] if self.step < 0 else self._range
        overlap = self.intersection(other)
        if overlap.step < 0:
            overlap = overlap[::-1]

        if len(overlap) == 0:
            return self.rename(name=res_name)
        if len(overlap) == len(self):
            return self[:0].rename(res_name)
        if not isinstance(overlap, RangeIndex):
            # We won't end up with RangeIndex, so fall back
            return super()._difference(other, sort=sort)
        if overlap.step != first.step:
            # In some cases we might be able to get a RangeIndex back,
            #  but not worth the effort.
            return super()._difference(other, sort=sort)

        if overlap[0] == first.start:
            # The difference is everything after the intersection
            new_rng = range(overlap[-1] + first.step, first.stop, first.step)
        elif overlap[-1] == first[-1]:
            # The difference is everything before the intersection
            new_rng = range(first.start, overlap[0], first.step)
        else:
            # The difference is not range-like
            return super()._difference(other, sort=sort)

        new_index = type(self)._simple_new(new_rng, name=res_name)
        if first is not self._range:
            new_index = new_index[::-1]
        return new_index

    def symmetric_difference(self, other, result_name: Hashable = None, sort=None):
        if not isinstance(other, RangeIndex) or sort is not None:
            return super().symmetric_difference(other, result_name, sort)

        left = self.difference(other)
        right = other.difference(self)
        result = left.union(right)

        if result_name is not None:
            result = result.rename(result_name)
        return result

    # --------------------------------------------------------------------

    def _concat(self, indexes: list[Index], name: Hashable) -> Index:
        """
        Overriding parent method for the case of all RangeIndex instances.

        When all members of "indexes" are of type RangeIndex: result will be
        RangeIndex if possible, Int64Index otherwise. E.g.:
        indexes = [RangeIndex(3), RangeIndex(3, 6)] -> RangeIndex(6)
        indexes = [RangeIndex(3), RangeIndex(4, 6)] -> Int64Index([0,1,2,4,5])
        """
        if not all(isinstance(x, RangeIndex) for x in indexes):
            return super()._concat(indexes, name)

        elif len(indexes) == 1:
            return indexes[0]

        rng_indexes = cast(List[RangeIndex], indexes)

        start = step = next_ = None

        # Filter the empty indexes
        non_empty_indexes = [obj for obj in rng_indexes if len(obj)]

        for obj in non_empty_indexes:
            rng = obj._range

            if start is None:
                # This is set by the first non-empty index
                start = rng.start
                if step is None and len(rng) > 1:
                    step = rng.step
            elif step is None:
                # First non-empty index had only one element
                if rng.start == start:
                    values = np.concatenate([x._values for x in rng_indexes])
                    result = Int64Index(values)
                    return result.rename(name)

                step = rng.start - start

            non_consecutive = (step != rng.step and len(rng) > 1) or (
                next_ is not None and rng.start != next_
            )
            if non_consecutive:
                result = Int64Index(np.concatenate([x._values for x in rng_indexes]))
                return result.rename(name)

            if step is not None:
                next_ = rng[-1] + step

        if non_empty_indexes:
            # Get the stop value from "next" or alternatively
            # from the last non-empty index
            stop = non_empty_indexes[-1].stop if next_ is None else next_
            return RangeIndex(start, stop, step).rename(name)

        # Here all "indexes" had 0 length, i.e. were empty.
        # In this case return an empty range index.
        return RangeIndex(0, 0).rename(name)

    def __len__(self) -> int:
        """
        return the length of the RangeIndex
        """
        return len(self._range)

    @property
    def size(self) -> int:
        return len(self)

    def __getitem__(self, key):
        """
        Conserve RangeIndex type for scalar and slice keys.
        """
        if isinstance(key, slice):
            new_range = self._range[key]
            return self._simple_new(new_range, name=self._name)
        elif is_integer(key):
            new_key = int(key)
            try:
                return self._range[new_key]
            except IndexError as err:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {len(self)}"
                ) from err
        elif is_scalar(key):
            raise IndexError(
                "only integers, slices (`:`), "
                "ellipsis (`...`), numpy.newaxis (`None`) "
                "and integer or boolean "
                "arrays are valid indices"
            )
        # fall back to Int64Index
        return super().__getitem__(key)

    def _getitem_slice(self: RangeIndex, slobj: slice) -> RangeIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._range[slobj]
        return type(self)._simple_new(res, name=self._name)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):

        if is_integer(other) and other != 0:
            if len(self) == 0 or self.start % other == 0 and self.step % other == 0:
                start = self.start // other
                step = self.step // other
                stop = start + len(self) * step
                new_range = range(start, stop, step or 1)
                return self._simple_new(new_range, name=self.name)
            if len(self) == 1:
                start = self.start // other
                new_range = range(start, start + 1, 1)
                return self._simple_new(new_range, name=self.name)
        return self._int64index // other

    # --------------------------------------------------------------------
    # Reductions

    def all(self, *args, **kwargs) -> bool:
        return 0 not in self._range

    def any(self, *args, **kwargs) -> bool:
        return any(self._range)

    # --------------------------------------------------------------------

    def _cmp_method(self, other, op):
        if isinstance(other, RangeIndex) and self._range == other._range:
            # Both are immutable so if ._range attr. are equal, shortcut is possible
            return super()._cmp_method(self, op)
        return super()._cmp_method(other, op)

    def _arith_method(self, other, op):
        """
        Parameters
        ----------
        other : Any
        op : callable that accepts 2 params
            perform the binary op
        """

        if isinstance(other, ABCTimedeltaIndex):
            # Defer to TimedeltaIndex implementation
            return NotImplemented
        elif isinstance(other, (timedelta, np.timedelta64)):
            # GH#19333 is_integer evaluated True on timedelta64,
            # so we need to catch these explicitly
            return op(self._int64index, other)
        elif is_timedelta64_dtype(other):
            # Must be an np.ndarray; GH#22390
            return op(self._int64index, other)

        if op in [
            operator.pow,
            ops.rpow,
            operator.mod,
            ops.rmod,
            ops.rfloordiv,
            divmod,
            ops.rdivmod,
        ]:
            return op(self._int64index, other)

        step: Callable | None = None
        if op in [operator.mul, ops.rmul, operator.truediv, ops.rtruediv]:
            step = op

        # TODO: if other is a RangeIndex we may have more efficient options
        other = extract_array(other, extract_numpy=True, extract_range=True)
        attrs = self._get_attributes_dict()

        left, right = self, other

        try:
            # apply if we have an override
            if step:
                with np.errstate(all="ignore"):
                    rstep = step(left.step, right)

                # we don't have a representable op
                # so return a base index
                if not is_integer(rstep) or not rstep:
                    raise ValueError

            else:
                rstep = left.step

            with np.errstate(all="ignore"):
                rstart = op(left.start, right)
                rstop = op(left.stop, right)

            result = type(self)(rstart, rstop, rstep, **attrs)

            # for compat with numpy / Int64Index
            # even if we can represent as a RangeIndex, return
            # as a Float64Index if we have float-like descriptors
            if not all(is_integer(x) for x in [rstart, rstop, rstep]):
                result = result.astype("float64")

            return result

        except (ValueError, TypeError, ZeroDivisionError):
            # Defer to Int64Index implementation
            return op(self._int64index, other)
            # TODO: Do attrs get handled reliably?
