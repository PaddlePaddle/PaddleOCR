"""
Experimental manager based on storing a collection of 1D arrays
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)
from pandas._typing import (
    ArrayLike,
    DtypeObj,
    Hashable,
)
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.cast import (
    astype_array_safe,
    ensure_dtype_can_hold_na,
    infer_dtype_from_scalar,
    soft_convert_objects,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_datetime64_ns_dtype,
    is_dtype_equal,
    is_extension_array_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_timedelta64_ns_dtype,
)
from pandas.core.dtypes.dtypes import (
    ExtensionDtype,
    PandasDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCPandasArray,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_inferred_bool_dtype
from pandas.core.dtypes.missing import (
    array_equals,
    isna,
    na_value_for_dtype,
)

import pandas.core.algorithms as algos
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.take import take_1d
from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    PandasArray,
    TimedeltaArray,
)
from pandas.core.arrays.sparse import SparseDtype
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
    sanitize_array,
)
from pandas.core.indexers import (
    maybe_convert_indices,
    validate_indices,
)
from pandas.core.indexes.api import (
    Index,
    ensure_index,
)
from pandas.core.internals.base import (
    DataManager,
    SingleDataManager,
    interleaved_dtype,
)
from pandas.core.internals.blocks import (
    ensure_block_shape,
    external_values,
    maybe_coerce_values,
    new_block,
    to_native_types,
)

if TYPE_CHECKING:
    from pandas import Float64Index


T = TypeVar("T", bound="BaseArrayManager")


class BaseArrayManager(DataManager):
    """
    Core internal data structure to implement DataFrame and Series.

    Alternative to the BlockManager, storing a list of 1D arrays instead of
    Blocks.

    This is *not* a public API class

    Parameters
    ----------
    arrays : Sequence of arrays
    axes : Sequence of Index
    verify_integrity : bool, default True

    """

    __slots__ = [
        "_axes",  # private attribute, because 'axes' has different order, see below
        "arrays",
    ]

    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]

    def __init__(
        self,
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[Index],
        verify_integrity: bool = True,
    ):
        raise NotImplementedError

    def make_empty(self: T, axes=None) -> T:
        """Return an empty ArrayManager with the items axis of len 0 (no columns)"""
        if axes is None:
            axes = [self.axes[1:], Index([])]

        arrays: list[np.ndarray | ExtensionArray] = []
        return type(self)(arrays, axes)

    @property
    def items(self) -> Index:
        return self._axes[-1]

    @property
    # error: Signature of "axes" incompatible with supertype "DataManager"
    def axes(self) -> list[Index]:  # type: ignore[override]
        # mypy doesn't work to override attribute with property
        # see https://github.com/python/mypy/issues/4125
        """Axes is BlockManager-compatible order (columns, rows)"""
        return [self._axes[1], self._axes[0]]

    @property
    def shape_proper(self) -> tuple[int, ...]:
        # this returns (n_rows, n_columns)
        return tuple(len(ax) for ax in self._axes)

    @staticmethod
    def _normalize_axis(axis: int) -> int:
        # switch axis
        axis = 1 if axis == 0 else 0
        return axis

    def set_axis(self, axis: int, new_labels: Index) -> None:
        # Caller is responsible for ensuring we have an Index object.
        self._validate_set_axis(axis, new_labels)
        axis = self._normalize_axis(axis)
        self._axes[axis] = new_labels

    def consolidate(self: T) -> T:
        return self

    def is_consolidated(self) -> bool:
        return True

    def _consolidate_inplace(self) -> None:
        pass

    def get_dtypes(self):
        return np.array([arr.dtype for arr in self.arrays], dtype="object")

    # TODO setstate getstate

    def __repr__(self) -> str:
        output = type(self).__name__
        output += f"\nIndex: {self._axes[0]}"
        if self.ndim == 2:
            output += f"\nColumns: {self._axes[1]}"
        output += f"\n{len(self.arrays)} arrays:"
        for arr in self.arrays:
            output += f"\n{arr.dtype}"
        return output

    def apply(
        self: T,
        f,
        align_keys: list[str] | None = None,
        ignore_failures: bool = False,
        **kwargs,
    ) -> T:
        """
        Iterate over the arrays, collect and create a new ArrayManager.

        Parameters
        ----------
        f : str or callable
            Name of the Array method to apply.
        align_keys: List[str] or None, default None
        ignore_failures: bool, default False
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        ArrayManager
        """
        assert "filter" not in kwargs

        align_keys = align_keys or []
        result_arrays: list[np.ndarray] = []
        result_indices: list[int] = []
        # fillna: Series/DataFrame is responsible for making sure value is aligned

        aligned_args = {k: kwargs[k] for k in align_keys}

        if f == "apply":
            f = kwargs.pop("func")

        for i, arr in enumerate(self.arrays):

            if aligned_args:

                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        # The caller is responsible for ensuring that
                        #  obj.axes[-1].equals(self.items)
                        if obj.ndim == 1:
                            kwargs[k] = obj.iloc[i]
                        else:
                            kwargs[k] = obj.iloc[:, i]._values
                    else:
                        # otherwise we have an array-like
                        kwargs[k] = obj[i]

            try:
                if callable(f):
                    applied = f(arr, **kwargs)
                else:
                    applied = getattr(arr, f)(**kwargs)
            except (TypeError, NotImplementedError):
                if not ignore_failures:
                    raise
                continue
            # if not isinstance(applied, ExtensionArray):
            #     # TODO not all EA operations return new EAs (eg astype)
            #     applied = array(applied)
            result_arrays.append(applied)
            result_indices.append(i)

        new_axes: list[Index]
        if ignore_failures:
            # TODO copy?
            new_axes = [self._axes[0], self._axes[1][result_indices]]
        else:
            new_axes = self._axes

        # error: Argument 1 to "ArrayManager" has incompatible type "List[ndarray]";
        # expected "List[Union[ndarray, ExtensionArray]]"
        return type(self)(result_arrays, new_axes)  # type: ignore[arg-type]

    def apply_with_block(self: T, f, align_keys=None, swap_axis=True, **kwargs) -> T:
        # switch axis to follow BlockManager logic
        if swap_axis and "axis" in kwargs and self.ndim == 2:
            kwargs["axis"] = 1 if kwargs["axis"] == 0 else 0

        align_keys = align_keys or []
        aligned_args = {k: kwargs[k] for k in align_keys}

        result_arrays = []

        for i, arr in enumerate(self.arrays):

            if aligned_args:
                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        # The caller is responsible for ensuring that
                        #  obj.axes[-1].equals(self.items)
                        if obj.ndim == 1:
                            if self.ndim == 2:
                                kwargs[k] = obj.iloc[slice(i, i + 1)]._values
                            else:
                                kwargs[k] = obj.iloc[:]._values
                        else:
                            kwargs[k] = obj.iloc[:, [i]]._values
                    else:
                        # otherwise we have an ndarray
                        if obj.ndim == 2:
                            kwargs[k] = obj[[i]]

            # error: Item "ExtensionArray" of "Union[Any, ExtensionArray]" has no
            # attribute "tz"
            if hasattr(arr, "tz") and arr.tz is None:  # type: ignore[union-attr]
                # DatetimeArray needs to be converted to ndarray for DatetimeLikeBlock

                # error: Item "ExtensionArray" of "Union[Any, ExtensionArray]" has no
                # attribute "_data"
                arr = arr._data  # type: ignore[union-attr]
            elif arr.dtype.kind == "m" and not isinstance(arr, np.ndarray):
                # TimedeltaArray needs to be converted to ndarray for TimedeltaBlock

                # error: "ExtensionArray" has no attribute "_data"
                arr = arr._data  # type: ignore[attr-defined]

            if self.ndim == 2:
                arr = ensure_block_shape(arr, 2)
                block = new_block(arr, placement=slice(0, 1, 1), ndim=2)
            else:
                block = new_block(arr, placement=slice(0, len(self), 1), ndim=1)

            applied = getattr(block, f)(**kwargs)
            if isinstance(applied, list):
                applied = applied[0]
            arr = applied.values
            if self.ndim == 2 and arr.ndim == 2:
                # 2D for np.ndarray or DatetimeArray/TimedeltaArray
                assert len(arr) == 1
                # error: Invalid index type "Tuple[int, slice]" for
                # "Union[ndarray, ExtensionArray]"; expected type
                # "Union[int, slice, ndarray]"
                arr = arr[0, :]  # type: ignore[index]
            result_arrays.append(arr)

        return type(self)(result_arrays, self._axes)

    def where(self: T, other, cond, align: bool, errors: str) -> T:
        if align:
            align_keys = ["other", "cond"]
        else:
            align_keys = ["cond"]
            other = extract_array(other, extract_numpy=True)

        return self.apply_with_block(
            "where",
            align_keys=align_keys,
            other=other,
            cond=cond,
            errors=errors,
        )

    # TODO what is this used for?
    # def setitem(self, indexer, value) -> ArrayManager:
    #     return self.apply_with_block("setitem", indexer=indexer, value=value)

    def putmask(self, mask, new, align: bool = True):
        if align:
            align_keys = ["new", "mask"]
        else:
            align_keys = ["mask"]
            new = extract_array(new, extract_numpy=True)

        return self.apply_with_block(
            "putmask",
            align_keys=align_keys,
            mask=mask,
            new=new,
        )

    def diff(self: T, n: int, axis: int) -> T:
        if axis == 1:
            # DataFrame only calls this for n=0, in which case performing it
            # with axis=0 is equivalent
            assert n == 0
            axis = 0
        return self.apply(algos.diff, n=n, axis=axis, stacklevel=5)

    def interpolate(self: T, **kwargs) -> T:
        return self.apply_with_block("interpolate", swap_axis=False, **kwargs)

    def shift(self: T, periods: int, axis: int, fill_value) -> T:
        if fill_value is lib.no_default:
            fill_value = None

        if axis == 1 and self.ndim == 2:
            # TODO column-wise shift
            raise NotImplementedError

        return self.apply_with_block(
            "shift", periods=periods, axis=axis, fill_value=fill_value
        )

    def fillna(self: T, value, limit, inplace: bool, downcast) -> T:
        return self.apply_with_block(
            "fillna", value=value, limit=limit, inplace=inplace, downcast=downcast
        )

    def downcast(self: T) -> T:
        return self.apply_with_block("downcast")

    def astype(self: T, dtype, copy: bool = False, errors: str = "raise") -> T:
        return self.apply(astype_array_safe, dtype=dtype, copy=copy, errors=errors)

    def convert(
        self: T,
        copy: bool = True,
        datetime: bool = True,
        numeric: bool = True,
        timedelta: bool = True,
    ) -> T:
        def _convert(arr):
            if is_object_dtype(arr.dtype):
                return soft_convert_objects(
                    arr,
                    datetime=datetime,
                    numeric=numeric,
                    timedelta=timedelta,
                    copy=copy,
                )
            else:
                return arr.copy() if copy else arr

        return self.apply(_convert)

    def replace(self: T, value, **kwargs) -> T:
        assert np.ndim(value) == 0, value
        # TODO "replace" is right now implemented on the blocks, we should move
        # it to general array algos so it can be reused here
        return self.apply_with_block("replace", value=value, **kwargs)

    def replace_list(
        self: T,
        src_list: list[Any],
        dest_list: list[Any],
        inplace: bool = False,
        regex: bool = False,
    ) -> T:
        """do a list replace"""
        inplace = validate_bool_kwarg(inplace, "inplace")

        return self.apply_with_block(
            "_replace_list",
            src_list=src_list,
            dest_list=dest_list,
            inplace=inplace,
            regex=regex,
        )

    def to_native_types(self, **kwargs):
        return self.apply(to_native_types, **kwargs)

    @property
    def is_mixed_type(self) -> bool:
        return True

    @property
    def is_numeric_mixed_type(self) -> bool:
        return all(is_numeric_dtype(t) for t in self.get_dtypes())

    @property
    def any_extension_types(self) -> bool:
        """Whether any of the blocks in this manager are extension blocks"""
        return False  # any(block.is_extension for block in self.blocks)

    @property
    def is_view(self) -> bool:
        """return a boolean if we are a single block and are a view"""
        # TODO what is this used for?
        return False

    @property
    def is_single_block(self) -> bool:
        return False

    def _get_data_subset(self: T, predicate: Callable) -> T:
        indices = [i for i, arr in enumerate(self.arrays) if predicate(arr)]
        arrays = [self.arrays[i] for i in indices]
        # TODO copy?
        new_axes = [self._axes[0], self._axes[1][np.array(indices, dtype="intp")]]
        return type(self)(arrays, new_axes, verify_integrity=False)

    def get_bool_data(self: T, copy: bool = False) -> T:
        """
        Select columns that are bool-dtype and object-dtype columns that are all-bool.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        return self._get_data_subset(is_inferred_bool_dtype)

    def get_numeric_data(self: T, copy: bool = False) -> T:
        """
        Select columns that have a numeric dtype.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        return self._get_data_subset(
            lambda arr: is_numeric_dtype(arr.dtype)
            or getattr(arr.dtype, "_is_numeric", False)
        )

    def copy(self: T, deep=True) -> T:
        """
        Make deep or shallow copy of ArrayManager

        Parameters
        ----------
        deep : bool or string, default True
            If False, return shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
        # this preserves the notion of view copying of axes
        if deep:
            # hit in e.g. tests.io.json.test_pandas

            def copy_func(ax):
                return ax.copy(deep=True) if deep == "all" else ax.view()

            new_axes = [copy_func(ax) for ax in self._axes]
        else:
            new_axes = list(self._axes)

        if deep:
            new_arrays = [arr.copy() for arr in self.arrays]
        else:
            new_arrays = self.arrays
        return type(self)(new_arrays, new_axes)

    def reindex_indexer(
        self: T,
        new_axis,
        indexer,
        axis: int,
        fill_value=None,
        allow_dups: bool = False,
        copy: bool = True,
        # ignored keywords
        consolidate: bool = True,
        only_slice: bool = False,
        # ArrayManager specific keywords
        use_na_proxy: bool = False,
    ) -> T:
        axis = self._normalize_axis(axis)
        return self._reindex_indexer(
            new_axis,
            indexer,
            axis,
            fill_value,
            allow_dups,
            copy,
            use_na_proxy,
        )

    def _reindex_indexer(
        self: T,
        new_axis,
        indexer,
        axis: int,
        fill_value=None,
        allow_dups: bool = False,
        copy: bool = True,
        use_na_proxy: bool = False,
    ) -> T:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray of int64 or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool, default True


        pandas-indexer with -1's only.
        """
        if indexer is None:
            if new_axis is self._axes[axis] and not copy:
                return self

            result = self.copy(deep=copy)
            result._axes = list(self._axes)
            result._axes[axis] = new_axis
            return result

        # some axes don't allow reindexing with dups
        if not allow_dups:
            self._axes[axis]._validate_can_reindex(indexer)

        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        if axis == 1:
            new_arrays = []
            for i in indexer:
                if i == -1:
                    arr = self._make_na_array(
                        fill_value=fill_value, use_na_proxy=use_na_proxy
                    )
                else:
                    arr = self.arrays[i]
                new_arrays.append(arr)

        else:
            validate_indices(indexer, len(self._axes[0]))
            indexer = ensure_platform_int(indexer)
            if (indexer == -1).any():
                allow_fill = True
            else:
                allow_fill = False
            new_arrays = [
                take_1d(
                    arr,
                    indexer,
                    allow_fill=allow_fill,
                    fill_value=fill_value,
                    # if fill_value is not None else blk.fill_value
                )
                for arr in self.arrays
            ]

        new_axes = list(self._axes)
        new_axes[axis] = new_axis

        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def take(self: T, indexer, axis: int = 1, verify: bool = True) -> T:
        """
        Take items along any axis.
        """
        axis = self._normalize_axis(axis)

        indexer = (
            np.arange(indexer.start, indexer.stop, indexer.step, dtype="int64")
            if isinstance(indexer, slice)
            else np.asanyarray(indexer, dtype="int64")
        )

        if not indexer.ndim == 1:
            raise ValueError("indexer should be 1-dimensional")

        n = self.shape_proper[axis]
        indexer = maybe_convert_indices(indexer, n, verify=verify)

        new_labels = self._axes[axis].take(indexer)
        return self._reindex_indexer(
            new_axis=new_labels, indexer=indexer, axis=axis, allow_dups=True
        )

    def _make_na_array(self, fill_value=None, use_na_proxy=False):
        if use_na_proxy:
            assert fill_value is None
            return NullArrayProxy(self.shape_proper[0])

        if fill_value is None:
            fill_value = np.nan

        dtype, fill_value = infer_dtype_from_scalar(fill_value)
        # error: Argument "dtype" to "empty" has incompatible type "Union[dtype[Any],
        # ExtensionDtype]"; expected "Union[dtype[Any], None, type, _SupportsDType, str,
        # Union[Tuple[Any, int], Tuple[Any, Union[int, Sequence[int]]], List[Any],
        # _DTypeDict, Tuple[Any, Any]]]"
        values = np.empty(self.shape_proper[0], dtype=dtype)  # type: ignore[arg-type]
        values.fill(fill_value)
        return values

    def _equal_values(self, other) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        for left, right in zip(self.arrays, other.arrays):
            if not array_equals(left, right):
                return False
        else:
            return True

    # TODO
    # to_dict


class ArrayManager(BaseArrayManager):
    ndim = 2

    def __init__(
        self,
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[Index],
        verify_integrity: bool = True,
    ):
        # Note: we are storing the axes in "_axes" in the (row, columns) order
        # which contrasts the order how it is stored in BlockManager
        self._axes = axes
        self.arrays = arrays

        if verify_integrity:
            self._axes = [ensure_index(ax) for ax in axes]
            self.arrays = [maybe_coerce_values(arr) for arr in arrays]
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        n_rows, n_columns = self.shape_proper
        if not len(self.arrays) == n_columns:
            raise ValueError(
                "Number of passed arrays must equal the size of the column Index: "
                f"{len(self.arrays)} arrays vs {n_columns} columns."
            )
        for arr in self.arrays:
            if not len(arr) == n_rows:
                raise ValueError(
                    "Passed arrays should have the same length as the rows Index: "
                    f"{len(arr)} vs {n_rows} rows"
                )
            if not isinstance(arr, (np.ndarray, ExtensionArray)):
                raise ValueError(
                    "Passed arrays should be np.ndarray or ExtensionArray instances, "
                    f"got {type(arr)} instead"
                )
            if not arr.ndim == 1:
                raise ValueError(
                    "Passed arrays should be 1-dimensional, got array with "
                    f"{arr.ndim} dimensions instead."
                )

    # --------------------------------------------------------------------
    # Indexing

    def fast_xs(self, loc: int) -> ArrayLike:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        dtype = interleaved_dtype([arr.dtype for arr in self.arrays])

        values = [arr[loc] for arr in self.arrays]
        if isinstance(dtype, ExtensionDtype):
            result = dtype.construct_array_type()._from_sequence(values, dtype=dtype)
        # for datetime64/timedelta64, the np.ndarray constructor cannot handle pd.NaT
        elif is_datetime64_ns_dtype(dtype):
            result = DatetimeArray._from_sequence(values, dtype=dtype)._data
        elif is_timedelta64_ns_dtype(dtype):
            result = TimedeltaArray._from_sequence(values, dtype=dtype)._data
        else:
            result = np.array(values, dtype=dtype)
        return result

    def get_slice(self, slobj: slice, axis: int = 0) -> ArrayManager:
        axis = self._normalize_axis(axis)

        if axis == 0:
            arrays = [arr[slobj] for arr in self.arrays]
        elif axis == 1:
            arrays = self.arrays[slobj]

        new_axes = list(self._axes)
        new_axes[axis] = new_axes[axis]._getitem_slice(slobj)

        return type(self)(arrays, new_axes, verify_integrity=False)

    def iget(self, i: int) -> SingleArrayManager:
        """
        Return the data as a SingleArrayManager.
        """
        values = self.arrays[i]
        return SingleArrayManager([values], [self._axes[0]])

    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).
        """
        return self.arrays[i]

    @property
    def column_arrays(self) -> list[ArrayLike]:
        """
        Used in the JSON C code to access column arrays.
        """
        return self.arrays

    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike):
        """
        Set new column(s).

        This changes the ArrayManager in-place, but replaces (an) existing
        column(s), not changing column values in-place).

        Parameters
        ----------
        loc : integer, slice or boolean mask
            Positional location (already bounds checked)
        value : np.ndarray or ExtensionArray
        """
        # single column -> single integer index
        if lib.is_integer(loc):

            # TODO can we avoid needing to unpack this here? That means converting
            # DataFrame into 1D array when loc is an integer
            if isinstance(value, np.ndarray) and value.ndim == 2:
                assert value.shape[1] == 1
                value = value[:, 0]

            # TODO we receive a datetime/timedelta64 ndarray from DataFrame._iset_item
            # but we should avoid that and pass directly the proper array
            value = maybe_coerce_values(value)

            assert isinstance(value, (np.ndarray, ExtensionArray))
            assert value.ndim == 1
            assert len(value) == len(self._axes[0])
            # error: Invalid index type "Union[int, slice, ndarray]" for
            # "List[Union[ndarray, ExtensionArray]]"; expected type "int"
            self.arrays[loc] = value  # type: ignore[index]
            return

        # multiple columns -> convert slice or array to integer indices
        elif isinstance(loc, slice):
            indices = range(
                loc.start if loc.start is not None else 0,
                loc.stop if loc.stop is not None else self.shape_proper[1],
                loc.step if loc.step is not None else 1,
            )
        else:
            assert isinstance(loc, np.ndarray)
            assert loc.dtype == "bool"
            # error: Incompatible types in assignment (expression has type "ndarray",
            # variable has type "range")
            indices = np.nonzero(loc)[0]  # type: ignore[assignment]

        assert value.ndim == 2
        assert value.shape[0] == len(self._axes[0])

        for value_idx, mgr_idx in enumerate(indices):
            # error: Invalid index type "Tuple[slice, int]" for
            # "Union[ExtensionArray, ndarray]"; expected type
            # "Union[int, slice, ndarray]"
            value_arr = value[:, value_idx]  # type: ignore[index]
            self.arrays[mgr_idx] = value_arr
        return

    def insert(self, loc: int, item: Hashable, value: ArrayLike) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        """
        # insert to the axis; this could possibly raise a TypeError
        new_axis = self.items.insert(loc, item)

        value = extract_array(value, extract_numpy=True)
        if value.ndim == 2:
            if value.shape[0] == 1:
                # error: Invalid index type "Tuple[int, slice]" for
                # "Union[Any, ExtensionArray, ndarray]"; expected type
                # "Union[int, slice, ndarray]"
                value = value[0, :]  # type: ignore[index]
            else:
                raise ValueError(
                    f"Expected a 1D array, got an array with shape {value.shape}"
                )
        value = maybe_coerce_values(value)

        # TODO self.arrays can be empty
        # assert len(value) == len(self.arrays[0])

        # TODO is this copy needed?
        arrays = self.arrays.copy()
        arrays.insert(loc, value)

        self.arrays = arrays
        self._axes[1] = new_axis

    def idelete(self, indexer):
        """
        Delete selected locations in-place (new block and array, same BlockManager)
        """
        to_keep = np.ones(self.shape[0], dtype=np.bool_)
        to_keep[indexer] = False

        self.arrays = [self.arrays[i] for i in np.nonzero(to_keep)[0]]
        self._axes = [self._axes[0], self._axes[1][to_keep]]
        return self

    # --------------------------------------------------------------------
    # Array-wise Operation

    def grouped_reduce(self: T, func: Callable, ignore_failures: bool = False) -> T:
        """
        Apply grouped reduction function columnwise, returning a new ArrayManager.

        Parameters
        ----------
        func : grouped reduction function
        ignore_failures : bool, default False
            Whether to drop columns where func raises TypeError.

        Returns
        -------
        ArrayManager
        """
        result_arrays: list[np.ndarray] = []
        result_indices: list[int] = []

        for i, arr in enumerate(self.arrays):
            # grouped_reduce functions all expect 2D arrays
            arr = ensure_block_shape(arr, ndim=2)
            try:
                res = func(arr)
            except (TypeError, NotImplementedError):
                if not ignore_failures:
                    raise
                continue

            if res.ndim == 2:
                # reverse of ensure_block_shape
                assert res.shape[0] == 1
                res = res[0]

            result_arrays.append(res)
            result_indices.append(i)

        if len(result_arrays) == 0:
            index = Index([None])  # placeholder
        else:
            index = Index(range(result_arrays[0].shape[0]))

        if ignore_failures:
            columns = self.items[np.array(result_indices, dtype="int64")]
        else:
            columns = self.items

        # error: Argument 1 to "ArrayManager" has incompatible type "List[ndarray]";
        # expected "List[Union[ndarray, ExtensionArray]]"
        return type(self)(result_arrays, [index, columns])  # type: ignore[arg-type]

    def reduce(
        self: T, func: Callable, ignore_failures: bool = False
    ) -> tuple[T, np.ndarray]:
        """
        Apply reduction function column-wise, returning a single-row ArrayManager.

        Parameters
        ----------
        func : reduction function
        ignore_failures : bool, default False
            Whether to drop columns where func raises TypeError.

        Returns
        -------
        ArrayManager
        np.ndarray
            Indexer of column indices that are retained.
        """
        result_arrays: list[np.ndarray] = []
        result_indices: list[int] = []
        for i, arr in enumerate(self.arrays):
            try:
                res = func(arr, axis=0)
            except TypeError:
                if not ignore_failures:
                    raise
            else:
                # TODO NaT doesn't preserve dtype, so we need to ensure to create
                # a timedelta result array if original was timedelta
                # what if datetime results in timedelta? (eg std)
                if res is NaT and is_timedelta64_ns_dtype(arr.dtype):
                    result_arrays.append(np.array(["NaT"], dtype="timedelta64[ns]"))
                else:
                    # error: Argument 1 to "append" of "list" has incompatible type
                    # "ExtensionArray"; expected "ndarray"
                    result_arrays.append(
                        sanitize_array([res], None)  # type: ignore[arg-type]
                    )
                result_indices.append(i)

        index = Index._simple_new(np.array([None], dtype=object))  # placeholder
        if ignore_failures:
            indexer = np.array(result_indices)
            columns = self.items[result_indices]
        else:
            indexer = np.arange(self.shape[0])
            columns = self.items

        # error: Argument 1 to "ArrayManager" has incompatible type "List[ndarray]";
        # expected "List[Union[ndarray, ExtensionArray]]"
        new_mgr = type(self)(result_arrays, [index, columns])  # type: ignore[arg-type]
        return new_mgr, indexer

    def operate_blockwise(self, other: ArrayManager, array_op) -> ArrayManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        # TODO what if `other` is BlockManager ?
        left_arrays = self.arrays
        right_arrays = other.arrays
        result_arrays = [
            array_op(left, right) for left, right in zip(left_arrays, right_arrays)
        ]
        return type(self)(result_arrays, self._axes)

    def quantile(
        self,
        *,
        qs: Float64Index,
        axis: int = 0,
        transposed: bool = False,
        interpolation="linear",
    ) -> ArrayManager:

        arrs = [ensure_block_shape(x, 2) for x in self.arrays]
        assert axis == 1
        new_arrs = [
            quantile_compat(x, np.asarray(qs._values), interpolation) for x in arrs
        ]
        for i, arr in enumerate(new_arrs):
            if arr.ndim == 2:
                assert arr.shape[0] == 1, arr.shape
                new_arrs[i] = arr[0]

        axes = [qs, self._axes[1]]
        return type(self)(new_arrs, axes)

    def apply_2d(
        self: ArrayManager, f, ignore_failures: bool = False, **kwargs
    ) -> ArrayManager:
        """
        Variant of `apply`, but where the function should not be applied to
        each column independently, but to the full data as a 2D array.
        """
        values = self.as_array()
        try:
            result = f(values, **kwargs)
        except (TypeError, NotImplementedError):
            if not ignore_failures:
                raise
            result_arrays = []
            new_axes = [self._axes[0], self.axes[1].take([])]
        else:
            result_arrays = [result[:, i] for i in range(len(self._axes[1]))]
            new_axes = self._axes

        return type(self)(result_arrays, new_axes)

    # ----------------------------------------------------------------

    def unstack(self, unstacker, fill_value) -> ArrayManager:
        """
        Return a BlockManager with all blocks unstacked..

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        indexer, _ = unstacker._indexer_and_to_sort
        if unstacker.mask.all():
            new_indexer = indexer
            allow_fill = False
        else:
            new_indexer = np.full(unstacker.mask.shape, -1)
            new_indexer[unstacker.mask] = indexer
            allow_fill = True
        new_indexer2D = new_indexer.reshape(*unstacker.full_shape)
        new_indexer2D = ensure_platform_int(new_indexer2D)

        new_arrays = []
        for arr in self.arrays:
            for i in range(unstacker.full_shape[1]):
                new_arr = take_1d(
                    arr,
                    new_indexer2D[:, i],
                    allow_fill=allow_fill,
                    fill_value=fill_value,
                )
                new_arrays.append(new_arr)

        new_index = unstacker.new_index
        new_columns = unstacker.get_new_columns(self._axes[1])
        new_axes = [new_index, new_columns]

        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def as_array(
        self,
        transpose: bool = False,
        dtype=None,
        copy: bool = False,
        na_value=lib.no_default,
    ) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        transpose : bool, default False
            If True, transpose the return array.
        dtype : object, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        if len(self.arrays) == 0:
            arr = np.empty(self.shape, dtype=float)
            return arr.transpose() if transpose else arr

        # We want to copy when na_value is provided to avoid
        # mutating the original object
        copy = copy or na_value is not lib.no_default

        if not dtype:
            dtype = interleaved_dtype([arr.dtype for arr in self.arrays])

        if isinstance(dtype, SparseDtype):
            dtype = dtype.subtype
        elif isinstance(dtype, PandasDtype):
            dtype = dtype.numpy_dtype
        elif is_extension_array_dtype(dtype):
            dtype = "object"
        elif is_dtype_equal(dtype, str):
            dtype = "object"

        result = np.empty(self.shape_proper, dtype=dtype)

        # error: Incompatible types in assignment (expression has type "Union[ndarray,
        # ExtensionArray]", variable has type "ndarray")
        for i, arr in enumerate(self.arrays):  # type: ignore[assignment]
            arr = arr.astype(dtype, copy=copy)
            result[:, i] = arr

        if na_value is not lib.no_default:
            result[isna(result)] = na_value

        return result
        # return arr.transpose() if transpose else arr


class SingleArrayManager(BaseArrayManager, SingleDataManager):

    __slots__ = [
        "_axes",  # private attribute, because 'axes' has different order, see below
        "arrays",
    ]

    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]

    ndim = 1

    def __init__(
        self,
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[Index],
        verify_integrity: bool = True,
    ):
        self._axes = axes
        self.arrays = arrays

        if verify_integrity:
            assert len(axes) == 1
            assert len(arrays) == 1
            self._axes = [ensure_index(ax) for ax in self._axes]
            arr = arrays[0]
            arr = maybe_coerce_values(arr)
            if isinstance(arr, ABCPandasArray):
                arr = arr.to_numpy()
            self.arrays = [arr]
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        (n_rows,) = self.shape
        assert len(self.arrays) == 1
        arr = self.arrays[0]
        assert len(arr) == n_rows
        if not arr.ndim == 1:
            raise ValueError(
                "Passed array should be 1-dimensional, got array with "
                f"{arr.ndim} dimensions instead."
            )

    @staticmethod
    def _normalize_axis(axis):
        return axis

    def make_empty(self, axes=None) -> SingleArrayManager:
        """Return an empty ArrayManager with index/array of length 0"""
        if axes is None:
            axes = [Index([], dtype=object)]
        array = np.array([], dtype=self.dtype)
        return type(self)([array], axes)

    @classmethod
    def from_array(cls, array, index):
        return cls([array], [index])

    @property
    def axes(self):
        return self._axes

    @property
    def index(self) -> Index:
        return self._axes[0]

    @property
    def dtype(self):
        return self.array.dtype

    def external_values(self):
        """The array that Series.values returns"""
        return external_values(self.array)

    def internal_values(self):
        """The array that Series._values returns"""
        return self.array

    def array_values(self):
        """The array that Series.array returns"""
        arr = self.array
        if isinstance(arr, np.ndarray):
            arr = PandasArray(arr)
        return arr

    @property
    def _can_hold_na(self) -> bool:
        if isinstance(self.array, np.ndarray):
            return self.array.dtype.kind not in ["b", "i", "u"]
        else:
            # ExtensionArray
            return self.array._can_hold_na

    @property
    def is_single_block(self) -> bool:
        return True

    def _consolidate_check(self):
        pass

    def fast_xs(self, loc: int) -> ArrayLike:
        raise NotImplementedError("Use series._values[loc] instead")

    def get_slice(self, slobj: slice, axis: int = 0) -> SingleArrayManager:
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        new_array = self.array[slobj]
        new_index = self.index._getitem_slice(slobj)
        return type(self)([new_array], [new_index], verify_integrity=False)

    def getitem_mgr(self, indexer) -> SingleArrayManager:
        new_array = self.array[indexer]
        new_index = self.index[indexer]
        return type(self)([new_array], [new_index])

    def apply(self, func, **kwargs):
        if callable(func):
            new_array = func(self.array, **kwargs)
        else:
            new_array = getattr(self.array, func)(**kwargs)
        return type(self)([new_array], self._axes)

    def setitem(self, indexer, value):
        return self.apply_with_block("setitem", indexer=indexer, value=value)

    def idelete(self, indexer) -> SingleArrayManager:
        """
        Delete selected locations in-place (new array, same ArrayManager)
        """
        to_keep = np.ones(self.shape[0], dtype=np.bool_)
        to_keep[indexer] = False

        self.arrays = [self.arrays[0][to_keep]]
        self._axes = [self._axes[0][to_keep]]
        return self

    def _get_data_subset(self, predicate: Callable) -> SingleArrayManager:
        # used in get_numeric_data / get_bool_data
        if predicate(self.array):
            return type(self)(self.arrays, self._axes, verify_integrity=False)
        else:
            return self.make_empty()

    def set_values(self, values: ArrayLike):
        """
        Set (replace) the values of the SingleArrayManager in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current SingleArrayManager (length, dtype, etc).
        """
        self.arrays[0] = values


class NullArrayProxy:
    """
    Proxy object for an all-NA array.

    Only stores the length of the array, and not the dtype. The dtype
    will only be known when actually concatenating (after determining the
    common dtype, for which this proxy is ignored).
    Using this object avoids that the internals/concat.py needs to determine
    the proper dtype and array type.
    """

    ndim = 1

    def __init__(self, n: int):
        self.n = n

    @property
    def shape(self):
        return (self.n,)

    def to_array(self, dtype: DtypeObj) -> ArrayLike:
        """
        Helper function to create the actual all-NA array from the NullArrayProxy
        object.

        Parameters
        ----------
        arr : NullArrayProxy
        dtype : the dtype for the resulting array

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if isinstance(dtype, ExtensionDtype):
            empty = dtype.construct_array_type()._from_sequence([], dtype=dtype)
            indexer = -np.ones(self.n, dtype=np.intp)
            return empty.take(indexer, allow_fill=True)
        else:
            # when introducing missing values, int becomes float, bool becomes object
            dtype = ensure_dtype_can_hold_na(dtype)
            fill_value = na_value_for_dtype(dtype)
            arr = np.empty(self.n, dtype=dtype)
            arr.fill(fill_value)
            return ensure_wrapped_if_datetimelike(arr)
