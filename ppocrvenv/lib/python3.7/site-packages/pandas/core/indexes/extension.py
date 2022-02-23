"""
Shared methods for Index subclasses backed by ExtensionArray.
"""
from __future__ import annotations

from typing import (
    Hashable,
    TypeVar,
)

import numpy as np

from pandas._typing import ArrayLike
from pandas.compat.numpy import function as nv
from pandas.util._decorators import (
    cache_readonly,
    doc,
)
from pandas.util._exceptions import rewrite_exception

from pandas.core.dtypes.common import (
    is_dtype_equal,
    is_object_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.indexers import deprecate_ndim_indexing
from pandas.core.indexes.base import Index
from pandas.core.ops import get_op_result_name

_T = TypeVar("_T", bound="NDArrayBackedExtensionIndex")


def inherit_from_data(name: str, delegate, cache: bool = False, wrap: bool = False):
    """
    Make an alias for a method of the underlying ExtensionArray.

    Parameters
    ----------
    name : str
        Name of an attribute the class should inherit from its EA parent.
    delegate : class
    cache : bool, default False
        Whether to convert wrapped properties into cache_readonly
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.

    Returns
    -------
    attribute, method, property, or cache_readonly
    """
    attr = getattr(delegate, name)

    if isinstance(attr, property) or type(attr).__name__ == "getset_descriptor":
        # getset_descriptor i.e. property defined in cython class
        if cache:

            def cached(self):
                return getattr(self._data, name)

            cached.__name__ = name
            cached.__doc__ = attr.__doc__
            method = cache_readonly(cached)

        else:

            def fget(self):
                result = getattr(self._data, name)
                if wrap:
                    if isinstance(result, type(self._data)):
                        return type(self)._simple_new(result, name=self.name)
                    elif isinstance(result, ABCDataFrame):
                        return result.set_index(self)
                    return Index(result, name=self.name)
                return result

            def fset(self, value):
                setattr(self._data, name, value)

            fget.__name__ = name
            fget.__doc__ = attr.__doc__

            method = property(fget, fset)

    elif not callable(attr):
        # just a normal attribute, no wrapping
        method = attr

    else:

        def method(self, *args, **kwargs):
            result = attr(self._data, *args, **kwargs)
            if wrap:
                if isinstance(result, type(self._data)):
                    return type(self)._simple_new(result, name=self.name)
                elif isinstance(result, ABCDataFrame):
                    return result.set_index(self)
                return Index(result, name=self.name)
            return result

        method.__name__ = name
        method.__doc__ = attr.__doc__
    return method


def inherit_names(names: list[str], delegate, cache: bool = False, wrap: bool = False):
    """
    Class decorator to pin attributes from an ExtensionArray to a Index subclass.

    Parameters
    ----------
    names : List[str]
    delegate : class
    cache : bool, default False
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.
    """

    def wrapper(cls):
        for name in names:
            meth = inherit_from_data(name, delegate, cache=cache, wrap=wrap)
            setattr(cls, name, meth)

        return cls

    return wrapper


def _make_wrapped_comparison_op(opname: str):
    """
    Create a comparison method that dispatches to ``._data``.
    """

    def wrapper(self, other):
        if isinstance(other, ABCSeries):
            # the arrays defer to Series for comparison ops but the indexes
            #  don't, so we have to unwrap here.
            other = other._values

        other = _maybe_unwrap_index(other)

        op = getattr(self._data, opname)
        return op(other)

    wrapper.__name__ = opname
    return wrapper


def make_wrapped_arith_op(opname: str):
    def method(self, other):
        if (
            isinstance(other, Index)
            and is_object_dtype(other.dtype)
            and type(other) is not Index
        ):
            # We return NotImplemented for object-dtype index *subclasses* so they have
            # a chance to implement ops before we unwrap them.
            # See https://github.com/pandas-dev/pandas/issues/31109
            return NotImplemented
        meth = getattr(self._data, opname)
        result = meth(_maybe_unwrap_index(other))
        return _wrap_arithmetic_op(self, other, result)

    method.__name__ = opname
    return method


def _wrap_arithmetic_op(self, other, result):
    if result is NotImplemented:
        return NotImplemented

    if isinstance(result, tuple):
        # divmod, rdivmod
        assert len(result) == 2
        return (
            _wrap_arithmetic_op(self, other, result[0]),
            _wrap_arithmetic_op(self, other, result[1]),
        )

    if not isinstance(result, Index):
        # Index.__new__ will choose appropriate subclass for dtype
        result = Index(result)

    res_name = get_op_result_name(self, other)
    result.name = res_name
    return result


def _maybe_unwrap_index(obj):
    """
    If operating against another Index object, we need to unwrap the underlying
    data before deferring to the DatetimeArray/TimedeltaArray/PeriodArray
    implementation, otherwise we will incorrectly return NotImplemented.

    Parameters
    ----------
    obj : object

    Returns
    -------
    unwrapped object
    """
    if isinstance(obj, Index):
        return obj._data
    return obj


class ExtensionIndex(Index):
    """
    Index subclass for indexes backed by ExtensionArray.
    """

    # The base class already passes through to _data:
    #  size, __len__, dtype

    _data: IntervalArray | NDArrayBackedExtensionArray

    _data_cls: (
        type[Categorical]
        | type[DatetimeArray]
        | type[TimedeltaArray]
        | type[PeriodArray]
        | type[IntervalArray]
    )

    @classmethod
    def _simple_new(
        cls,
        array: IntervalArray | NDArrayBackedExtensionArray,
        name: Hashable = None,
    ):
        """
        Construct from an ExtensionArray of the appropriate type.

        Parameters
        ----------
        array : ExtensionArray
        name : Label, default None
            Attached as result.name
        """
        assert isinstance(array, cls._data_cls), type(array)

        result = object.__new__(cls)
        result._data = array
        result._name = name
        result._cache = {}
        result._reset_identity()
        return result

    __eq__ = _make_wrapped_comparison_op("__eq__")
    __ne__ = _make_wrapped_comparison_op("__ne__")
    __lt__ = _make_wrapped_comparison_op("__lt__")
    __gt__ = _make_wrapped_comparison_op("__gt__")
    __le__ = _make_wrapped_comparison_op("__le__")
    __ge__ = _make_wrapped_comparison_op("__ge__")

    @property
    def _has_complex_internals(self) -> bool:
        # used to avoid libreduction code paths, which raise or require conversion
        return True

    # ---------------------------------------------------------------------
    # NDarray-Like Methods

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, type(self._data)):
            if result.ndim == 1:
                return type(self)(result, name=self._name)
            # Unpack to ndarray for MPL compat

            result = result._ndarray

        # Includes cases where we get a 2D ndarray back for MPL compat
        deprecate_ndim_indexing(result)
        return result

    def searchsorted(self, value, side="left", sorter=None) -> np.ndarray:
        # overriding IndexOpsMixin improves performance GH#38083
        return self._data.searchsorted(value, side=side, sorter=sorter)

    def putmask(self, mask, value) -> Index:
        mask, noop = validate_putmask(self._data, mask)
        if noop:
            return self.copy()

        try:
            self._validate_fill_value(value)
        except (ValueError, TypeError):
            dtype = self._find_common_type_compat(value)
            return self.astype(dtype).putmask(mask, value)

        arr = self._data.copy()
        arr.putmask(mask, value)
        return type(self)._simple_new(arr, name=self.name)

    # ---------------------------------------------------------------------

    def _get_engine_target(self) -> np.ndarray:
        return np.asarray(self._data)

    def _from_join_target(self, result: np.ndarray) -> ArrayLike:
        # ATM this is only for IntervalIndex, implicit assumption
        #  about _get_engine_target
        return type(self._data)._from_sequence(result, dtype=self.dtype)

    def delete(self, loc):
        """
        Make new Index with passed location(-s) deleted

        Returns
        -------
        new_index : Index
        """
        arr = self._data.delete(loc)
        return type(self)._simple_new(arr, name=self.name)

    def repeat(self, repeats, axis=None):
        nv.validate_repeat((), {"axis": axis})
        result = self._data.repeat(repeats, axis=axis)
        return type(self)._simple_new(result, name=self.name)

    def insert(self, loc: int, item) -> Index:
        """
        Make new Index inserting new item at location. Follows
        Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        new_index : Index
        """
        try:
            result = self._data.insert(loc, item)
        except (ValueError, TypeError):
            # e.g. trying to insert an integer into a DatetimeIndex
            #  We cannot keep the same dtype, so cast to the (often object)
            #  minimal shared dtype before doing the insert.
            dtype = self._find_common_type_compat(item)
            return self.astype(dtype).insert(loc, item)
        else:
            return type(self)._simple_new(result, name=self.name)

    def _validate_fill_value(self, value):
        """
        Convert value to be insertable to underlying array.
        """
        return self._data._validate_setitem_value(value)

    def _get_unique_index(self):
        if self.is_unique:
            return self

        result = self._data.unique()
        return type(self)._simple_new(result, name=self.name)

    @doc(Index.map)
    def map(self, mapper, na_action=None):
        # Try to run function on index first, and then on elements of index
        # Especially important for group-by functionality
        try:
            result = mapper(self)

            # Try to use this result if we can
            if isinstance(result, np.ndarray):
                result = Index(result)

            if not isinstance(result, Index):
                raise TypeError("The map function must return an Index object")
            return result
        except Exception:
            return self.astype(object).map(mapper)

    @doc(Index.astype)
    def astype(self, dtype, copy: bool = True) -> Index:
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(self.dtype, dtype):
            if not copy:
                # Ensure that self.astype(self.dtype) is self
                return self
            return self.copy()

        # error: Non-overlapping equality check (left operand type: "dtype[Any]", right
        # operand type: "Literal['M8[ns]']")
        if (
            isinstance(self.dtype, np.dtype)
            and isinstance(dtype, np.dtype)
            and dtype.kind == "M"
            and dtype != "M8[ns]"  # type: ignore[comparison-overlap]
        ):
            # For now Datetime supports this by unwrapping ndarray, but DTI doesn't
            raise TypeError(f"Cannot cast {type(self).__name__} to dtype")

        with rewrite_exception(type(self._data).__name__, type(self).__name__):
            new_values = self._data.astype(dtype, copy=copy)

        # pass copy=False because any copying will be done in the
        #  _data.astype call above
        return Index(new_values, dtype=new_values.dtype, name=self.name, copy=False)

    @cache_readonly
    def _isnan(self) -> np.ndarray:
        # error: Incompatible return value type (got "ExtensionArray", expected
        # "ndarray")
        return self._data.isna()  # type: ignore[return-value]

    @doc(Index.equals)
    def equals(self, other) -> bool:
        # Dispatch to the ExtensionArray's .equals method.
        if self.is_(other):
            return True

        if not isinstance(other, type(self)):
            return False

        return self._data.equals(other._data)


class NDArrayBackedExtensionIndex(ExtensionIndex):
    """
    Index subclass for indexes backed by NDArrayBackedExtensionArray.
    """

    _data: NDArrayBackedExtensionArray

    @classmethod
    def _simple_new(
        cls,
        values: NDArrayBackedExtensionArray,
        name: Hashable = None,
    ):
        result = super()._simple_new(values, name)

        # For groupby perf. See note in indexes/base about _index_data
        result._index_data = values._ndarray

        return result

    def _get_engine_target(self) -> np.ndarray:
        return self._data._ndarray

    def _from_join_target(self, result: np.ndarray) -> ArrayLike:
        assert result.dtype == self._data._ndarray.dtype
        return self._data._from_backing_data(result)
