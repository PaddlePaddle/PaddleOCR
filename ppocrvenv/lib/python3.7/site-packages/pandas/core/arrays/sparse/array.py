"""
SparseArray data structure
"""
from __future__ import annotations

from collections import abc
import numbers
import operator
from typing import (
    Any,
    Callable,
    Sequence,
    TypeVar,
)
import warnings

import numpy as np

from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
    BlockIndex,
    IntIndex,
    SparseIndex,
)
from pandas._libs.tslibs import NaT
from pandas._typing import (
    Dtype,
    NpDtype,
    Scalar,
)
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning

from pandas.core.dtypes.cast import (
    astype_nansafe,
    construct_1d_arraylike_from_scalar,
    find_common_type,
    maybe_box_datetimelike,
)
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_dtype_equal,
    is_integer,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.sparse.dtype import SparseDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
    extract_array,
    sanitize_array,
)
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import interpolate_2d
from pandas.core.nanops import check_below_min_count
import pandas.core.ops as ops

import pandas.io.formats.printing as printing

# ----------------------------------------------------------------------------
# Array

SparseArrayT = TypeVar("SparseArrayT", bound="SparseArray")

_sparray_doc_kwargs = {"klass": "SparseArray"}


def _get_fill(arr: SparseArray) -> np.ndarray:
    """
    Create a 0-dim ndarray containing the fill value

    Parameters
    ----------
    arr : SparseArray

    Returns
    -------
    fill_value : ndarray
        0-dim ndarray with just the fill value.

    Notes
    -----
    coerce fill_value to arr dtype if possible
    int64 SparseArray can have NaN as fill_value if there is no missing
    """
    try:
        return np.asarray(arr.fill_value, dtype=arr.dtype.subtype)
    except ValueError:
        return np.asarray(arr.fill_value)


def _sparse_array_op(
    left: SparseArray, right: SparseArray, op: Callable, name: str
) -> Any:
    """
    Perform a binary operation between two arrays.

    Parameters
    ----------
    left : Union[SparseArray, ndarray]
    right : Union[SparseArray, ndarray]
    op : Callable
        The binary operation to perform
    name str
        Name of the callable.

    Returns
    -------
    SparseArray
    """
    if name.startswith("__"):
        # For lookups in _libs.sparse we need non-dunder op name
        name = name[2:-2]

    # dtype used to find corresponding sparse method
    ltype = left.dtype.subtype
    rtype = right.dtype.subtype

    if not is_dtype_equal(ltype, rtype):
        subtype = find_common_type([ltype, rtype])
        ltype = SparseDtype(subtype, left.fill_value)
        rtype = SparseDtype(subtype, right.fill_value)

        # TODO(GH-23092): pass copy=False. Need to fix astype_nansafe
        left = left.astype(ltype)
        right = right.astype(rtype)
        dtype = ltype.subtype
    else:
        dtype = ltype

    # dtype the result must have
    result_dtype = None

    if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
        with np.errstate(all="ignore"):
            result = op(left.to_dense(), right.to_dense())
            fill = op(_get_fill(left), _get_fill(right))

        if left.sp_index.ngaps == 0:
            index = left.sp_index
        else:
            index = right.sp_index
    elif left.sp_index.equals(right.sp_index):
        with np.errstate(all="ignore"):
            result = op(left.sp_values, right.sp_values)
            fill = op(_get_fill(left), _get_fill(right))
        index = left.sp_index
    else:
        if name[0] == "r":
            left, right = right, left
            name = name[1:]

        if name in ("and", "or", "xor") and dtype == "bool":
            opname = f"sparse_{name}_uint8"
            # to make template simple, cast here
            left_sp_values = left.sp_values.view(np.uint8)
            right_sp_values = right.sp_values.view(np.uint8)
            result_dtype = bool
        else:
            opname = f"sparse_{name}_{dtype}"
            left_sp_values = left.sp_values
            right_sp_values = right.sp_values

        sparse_op = getattr(splib, opname)

        with np.errstate(all="ignore"):
            result, index, fill = sparse_op(
                left_sp_values,
                left.sp_index,
                left.fill_value,
                right_sp_values,
                right.sp_index,
                right.fill_value,
            )

    if result_dtype is None:
        result_dtype = result.dtype

    return _wrap_result(name, result, index, fill, dtype=result_dtype)


def _wrap_result(name, data, sparse_index, fill_value, dtype: Dtype | None = None):
    """
    wrap op result to have correct dtype
    """
    if name.startswith("__"):
        # e.g. __eq__ --> eq
        name = name[2:-2]

    if name in ("eq", "ne", "lt", "gt", "le", "ge"):
        dtype = bool

    fill_value = lib.item_from_zerodim(fill_value)

    if is_bool_dtype(dtype):
        # fill_value may be np.bool_
        fill_value = bool(fill_value)
    return SparseArray(
        data, sparse_index=sparse_index, fill_value=fill_value, dtype=dtype
    )


class SparseArray(OpsMixin, PandasObject, ExtensionArray):
    """
    An ExtensionArray for storing sparse data.

    Parameters
    ----------
    data : array-like
        A dense array of values to store in the SparseArray. This may contain
        `fill_value`.
    sparse_index : SparseIndex, optional
    index : Index
    fill_value : scalar, optional
        Elements in `data` that are `fill_value` are not stored in the
        SparseArray. For memory savings, this should be the most common value
        in `data`. By default, `fill_value` depends on the dtype of `data`:

        =========== ==========
        data.dtype  na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        False
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The fill value is potentially specified in three ways. In order of
        precedence, these are

        1. The `fill_value` argument
        2. ``dtype.fill_value`` if `fill_value` is None and `dtype` is
           a ``SparseDtype``
        3. ``data.dtype.fill_value`` if `fill_value` is None and `dtype`
           is not a ``SparseDtype`` and `data` is a ``SparseArray``.

    kind : {'integer', 'block'}, default 'integer'
        The type of storage for sparse locations.

        * 'block': Stores a `block` and `block_length` for each
          contiguous *span* of sparse values. This is best when
          sparse data tends to be clumped together, with large
          regions of ``fill-value`` values between sparse values.
        * 'integer': uses an integer to store the location of
          each sparse value.

    dtype : np.dtype or SparseDtype, optional
        The dtype to use for the SparseArray. For numpy dtypes, this
        determines the dtype of ``self.sp_values``. For SparseDtype,
        this determines ``self.sp_values`` and ``self.fill_value``.
    copy : bool, default False
        Whether to explicitly copy the incoming `data` array.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> from pandas.arrays import SparseArray
    >>> arr = SparseArray([0, 0, 1, 2])
    >>> arr
    [0, 0, 1, 2]
    Fill: 0
    IntIndex
    Indices: array([2, 3], dtype=int32)
    """

    _subtyp = "sparse_array"  # register ABCSparseArray
    _hidden_attrs = PandasObject._hidden_attrs | frozenset(["get_values"])
    _sparse_index: SparseIndex

    def __init__(
        self,
        data,
        sparse_index=None,
        index=None,
        fill_value=None,
        kind="integer",
        dtype: Dtype | None = None,
        copy=False,
    ):

        if fill_value is None and isinstance(dtype, SparseDtype):
            fill_value = dtype.fill_value

        if isinstance(data, type(self)):
            # disable normal inference on dtype, sparse_index, & fill_value
            if sparse_index is None:
                sparse_index = data.sp_index
            if fill_value is None:
                fill_value = data.fill_value
            if dtype is None:
                dtype = data.dtype
            # TODO: make kind=None, and use data.kind?
            data = data.sp_values

        # Handle use-provided dtype
        if isinstance(dtype, str):
            # Two options: dtype='int', regular numpy dtype
            # or dtype='Sparse[int]', a sparse dtype
            try:
                dtype = SparseDtype.construct_from_string(dtype)
            except TypeError:
                dtype = pandas_dtype(dtype)

        if isinstance(dtype, SparseDtype):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype

        if index is not None and not is_scalar(data):
            raise Exception("must only pass scalars with an index")

        if is_scalar(data):
            if index is not None and data is None:
                data = np.nan

            if index is not None:
                npoints = len(index)
            elif sparse_index is None:
                npoints = 1
            else:
                npoints = sparse_index.length

            data = construct_1d_arraylike_from_scalar(data, npoints, dtype=None)
            dtype = data.dtype

        if dtype is not None:
            dtype = pandas_dtype(dtype)

        # TODO: disentangle the fill_value dtype inference from
        # dtype inference
        if data is None:
            # TODO: What should the empty dtype be? Object or float?

            # error: Argument "dtype" to "array" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "Union[dtype[Any],
            # None, type, _SupportsDType, str, Union[Tuple[Any, int], Tuple[Any,
            # Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any, Any]]]"
            data = np.array([], dtype=dtype)  # type: ignore[arg-type]

        if not is_array_like(data):
            try:
                # probably shared code in sanitize_series

                data = sanitize_array(data, index=None)
            except ValueError:
                # NumPy may raise a ValueError on data like [1, []]
                # we retry with object dtype here.
                if dtype is None:
                    dtype = object
                    data = np.atleast_1d(np.asarray(data, dtype=dtype))
                else:
                    raise

        if copy:
            # TODO: avoid double copy when dtype forces cast.
            data = data.copy()

        if fill_value is None:
            fill_value_dtype = data.dtype if dtype is None else dtype
            if fill_value_dtype is None:
                fill_value = np.nan
            else:
                fill_value = na_value_for_dtype(fill_value_dtype)

        if isinstance(data, type(self)) and sparse_index is None:
            sparse_index = data._sparse_index
            # error: Argument "dtype" to "asarray" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], Type[object], None]"; expected
            # "Union[dtype[Any], None, type, _SupportsDType, str, Union[Tuple[Any, int],
            # Tuple[Any, Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any,
            # Any]]]"
            sparse_values = np.asarray(
                data.sp_values, dtype=dtype  # type: ignore[arg-type]
            )
        elif sparse_index is None:
            data = extract_array(data, extract_numpy=True)
            if not isinstance(data, np.ndarray):
                # EA
                if is_datetime64tz_dtype(data.dtype):
                    warnings.warn(
                        f"Creating SparseArray from {data.dtype} data "
                        "loses timezone information.  Cast to object before "
                        "sparse to retain timezone information.",
                        UserWarning,
                        stacklevel=2,
                    )
                    data = np.asarray(data, dtype="datetime64[ns]")
                    if fill_value is NaT:
                        fill_value = np.datetime64("NaT", "ns")
                data = np.asarray(data)
            sparse_values, sparse_index, fill_value = make_sparse(
                # error: Argument "dtype" to "make_sparse" has incompatible type
                # "Union[ExtensionDtype, dtype[Any], Type[object], None]"; expected
                # "Union[str, dtype[Any], None]"
                data,
                kind=kind,
                fill_value=fill_value,
                dtype=dtype,  # type: ignore[arg-type]
            )
        else:
            # error: Argument "dtype" to "asarray" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], Type[object], None]"; expected
            # "Union[dtype[Any], None, type, _SupportsDType, str, Union[Tuple[Any, int],
            # Tuple[Any, Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any,
            # Any]]]"
            sparse_values = np.asarray(data, dtype=dtype)  # type: ignore[arg-type]
            if len(sparse_values) != sparse_index.npoints:
                raise AssertionError(
                    f"Non array-like type {type(sparse_values)} must "
                    "have the same length as the index"
                )
        self._sparse_index = sparse_index
        self._sparse_values = sparse_values
        self._dtype = SparseDtype(sparse_values.dtype, fill_value)

    @classmethod
    def _simple_new(
        cls: type[SparseArrayT],
        sparse_array: np.ndarray,
        sparse_index: SparseIndex,
        dtype: SparseDtype,
    ) -> SparseArrayT:
        new = object.__new__(cls)
        new._sparse_index = sparse_index
        new._sparse_values = sparse_array
        new._dtype = dtype
        return new

    @classmethod
    def from_spmatrix(cls, data):
        """
        Create a SparseArray from a scipy.sparse matrix.

        .. versionadded:: 0.25.0

        Parameters
        ----------
        data : scipy.sparse.sp_matrix
            This should be a SciPy sparse matrix where the size
            of the second dimension is 1. In other words, a
            sparse matrix with a single column.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.coo_matrix((4, 1))
        >>> pd.arrays.SparseArray.from_spmatrix(mat)
        [0.0, 0.0, 0.0, 0.0]
        Fill: 0.0
        IntIndex
        Indices: array([], dtype=int32)
        """
        length, ncol = data.shape

        if ncol != 1:
            raise ValueError(f"'data' must have a single column, not '{ncol}'")

        # our sparse index classes require that the positions be strictly
        # increasing. So we need to sort loc, and arr accordingly.
        data = data.tocsc()
        data.sort_indices()
        arr = data.data
        idx = data.indices

        zero = np.array(0, dtype=arr.dtype).item()
        dtype = SparseDtype(arr.dtype, zero)
        index = IntIndex(length, idx)

        return cls._simple_new(arr, index, dtype)

    def __array__(self, dtype: NpDtype | None = None) -> np.ndarray:
        fill_value = self.fill_value

        if self.sp_index.ngaps == 0:
            # Compat for na dtype and int values.
            return self.sp_values
        if dtype is None:
            # Can NumPy represent this type?
            # If not, `np.result_type` will raise. We catch that
            # and return object.
            if is_datetime64_any_dtype(self.sp_values.dtype):
                # However, we *do* special-case the common case of
                # a datetime64 with pandas NaT.
                if fill_value is NaT:
                    # Can't put pd.NaT in a datetime64[ns]
                    fill_value = np.datetime64("NaT")
            try:
                dtype = np.result_type(self.sp_values.dtype, type(fill_value))
            except TypeError:
                # error: Incompatible types in assignment (expression has type
                # "Type[object]", variable has type "Union[str, dtype[Any], None]")
                dtype = object  # type: ignore[assignment]

        out = np.full(self.shape, fill_value, dtype=dtype)
        out[self.sp_index.to_int_index().indices] = self.sp_values
        return out

    def __setitem__(self, key, value):
        # I suppose we could allow setting of non-fill_value elements.
        # TODO(SparseArray.__setitem__): remove special cases in
        # ExtensionBlock.where
        msg = "SparseArray does not support item assignment via setitem"
        raise TypeError(msg)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy=False):
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    @property
    def sp_index(self) -> SparseIndex:
        """
        The SparseIndex containing the location of non- ``fill_value`` points.
        """
        return self._sparse_index

    @property
    def sp_values(self) -> np.ndarray:
        """
        An ndarray containing the non- ``fill_value`` values.

        Examples
        --------
        >>> s = SparseArray([0, 0, 1, 0, 2], fill_value=0)
        >>> s.sp_values
        array([1, 2])
        """
        return self._sparse_values

    @property
    def dtype(self) -> SparseDtype:
        return self._dtype

    @property
    def fill_value(self):
        """
        Elements in `data` that are `fill_value` are not stored.

        For memory savings, this should be the most common value in the array.
        """
        return self.dtype.fill_value

    @fill_value.setter
    def fill_value(self, value):
        self._dtype = SparseDtype(self.dtype.subtype, value)

    @property
    def kind(self) -> str:
        """
        The kind of sparse index for this array. One of {'integer', 'block'}.
        """
        if isinstance(self.sp_index, IntIndex):
            return "integer"
        else:
            return "block"

    @property
    def _valid_sp_values(self) -> np.ndarray:
        sp_vals = self.sp_values
        mask = notna(sp_vals)
        return sp_vals[mask]

    def __len__(self) -> int:
        return self.sp_index.length

    @property
    def _null_fill_value(self):
        return self._dtype._is_na_fill_value

    def _fill_value_matches(self, fill_value):
        if self._null_fill_value:
            return isna(fill_value)
        else:
            return self.fill_value == fill_value

    @property
    def nbytes(self) -> int:
        return self.sp_values.nbytes + self.sp_index.nbytes

    @property
    def density(self) -> float:
        """
        The percent of non- ``fill_value`` points, as decimal.

        Examples
        --------
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.density
        0.6
        """
        return self.sp_index.npoints / self.sp_index.length

    @property
    def npoints(self) -> int:
        """
        The number of non- ``fill_value`` points.

        Examples
        --------
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.npoints
        3
        """
        return self.sp_index.npoints

    def isna(self):
        # If null fill value, we want SparseDtype[bool, true]
        # to preserve the same memory usage.
        dtype = SparseDtype(bool, self._null_fill_value)
        return type(self)._simple_new(isna(self.sp_values), self.sp_index, dtype)

    def fillna(self, value=None, method=None, limit=None):
        """
        Fill missing values with `value`.

        Parameters
        ----------
        value : scalar, optional
        method : str, optional

            .. warning::

               Using 'method' will result in high memory use,
               as all `fill_value` methods will be converted to
               an in-memory ndarray

        limit : int, optional

        Returns
        -------
        SparseArray

        Notes
        -----
        When `value` is specified, the result's ``fill_value`` depends on
        ``self.fill_value``. The goal is to maintain low-memory use.

        If ``self.fill_value`` is NA, the result dtype will be
        ``SparseDtype(self.dtype, fill_value=value)``. This will preserve
        amount of memory used before and after filling.

        When ``self.fill_value`` is not NA, the result dtype will be
        ``self.dtype``. Again, this preserves the amount of memory used.
        """
        if (method is None and value is None) or (
            method is not None and value is not None
        ):
            raise ValueError("Must specify one of 'method' or 'value'.")

        elif method is not None:
            msg = "fillna with 'method' requires high memory usage."
            warnings.warn(msg, PerformanceWarning)
            filled = interpolate_2d(np.asarray(self), method=method, limit=limit)
            return type(self)(filled, fill_value=self.fill_value)

        else:
            new_values = np.where(isna(self.sp_values), value, self.sp_values)

            if self._null_fill_value:
                # This is essentially just updating the dtype.
                new_dtype = SparseDtype(self.dtype.subtype, fill_value=value)
            else:
                new_dtype = self.dtype

        return self._simple_new(new_values, self._sparse_index, new_dtype)

    def shift(self, periods=1, fill_value=None):

        if not len(self) or periods == 0:
            return self.copy()

        if isna(fill_value):
            fill_value = self.dtype.na_value

        subtype = np.result_type(fill_value, self.dtype.subtype)

        if subtype != self.dtype.subtype:
            # just coerce up front
            arr = self.astype(SparseDtype(subtype, self.fill_value))
        else:
            arr = self

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=arr.dtype
        )

        if periods > 0:
            a = empty
            b = arr[:-periods]
        else:
            a = arr[abs(periods) :]
            b = empty
        return arr._concat_same_type([a, b])

    def _first_fill_value_loc(self):
        """
        Get the location of the first missing value.

        Returns
        -------
        int
        """
        if len(self) == 0 or self.sp_index.npoints == len(self):
            return -1

        indices = self.sp_index.to_int_index().indices
        if not len(indices) or indices[0] > 0:
            return 0

        diff = indices[1:] - indices[:-1]
        return np.searchsorted(diff, 2) + 1

    def unique(self):
        uniques = list(algos.unique(self.sp_values))
        fill_loc = self._first_fill_value_loc()
        if fill_loc >= 0:
            uniques.insert(fill_loc, self.fill_value)
        return type(self)._from_sequence(uniques, dtype=self.dtype)

    def _values_for_factorize(self):
        # Still override this for hash_pandas_object
        return np.asarray(self), self.fill_value

    def factorize(self, na_sentinel=-1):
        # Currently, ExtensionArray.factorize -> Tuple[ndarray, EA]
        # The sparsity on this is backwards from what Sparse would want. Want
        # ExtensionArray.factorize -> Tuple[EA, EA]
        # Given that we have to return a dense array of codes, why bother
        # implementing an efficient factorize?
        codes, uniques = algos.factorize(np.asarray(self), na_sentinel=na_sentinel)
        # error: Incompatible types in assignment (expression has type "SparseArray",
        # variable has type "Union[ndarray, Index]")
        uniques = SparseArray(uniques, dtype=self.dtype)  # type: ignore[assignment]
        return codes, uniques

    def value_counts(self, dropna: bool = True):
        """
        Returns a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN, even if NaN is in sp_values.

        Returns
        -------
        counts : Series
        """
        from pandas import (
            Index,
            Series,
        )

        keys, counts = algos.value_counts_arraylike(self.sp_values, dropna=dropna)
        fcounts = self.sp_index.ngaps
        if fcounts > 0 and (not self._null_fill_value or not dropna):
            mask = isna(keys) if self._null_fill_value else keys == self.fill_value
            if mask.any():
                counts[mask] += fcounts
            else:
                keys = np.insert(keys, 0, self.fill_value)
                counts = np.insert(counts, 0, fcounts)

        if not isinstance(keys, ABCIndex):
            keys = Index(keys)
        return Series(counts, index=keys)

    # --------
    # Indexing
    # --------

    def __getitem__(self, key):

        if isinstance(key, tuple):
            if len(key) > 1:
                if key[0] is Ellipsis:
                    key = key[1:]
                elif key[-1] is Ellipsis:
                    key = key[:-1]
            if len(key) > 1:
                raise IndexError("too many indices for array.")
            key = key[0]

        if is_integer(key):
            return self._get_val_at(key)
        elif isinstance(key, tuple):
            data_slice = self.to_dense()[key]
        elif isinstance(key, slice):
            # special case to preserve dtypes
            if key == slice(None):
                return self.copy()
            # TODO: this logic is surely elsewhere
            # TODO: this could be more efficient
            indices = np.arange(len(self), dtype=np.int32)[key]
            return self.take(indices)
        else:
            # TODO: I think we can avoid densifying when masking a
            # boolean SparseArray with another. Need to look at the
            # key's fill_value for True / False, and then do an intersection
            # on the indices of the sp_values.
            if isinstance(key, SparseArray):
                if is_bool_dtype(key):
                    key = key.to_dense()
                else:
                    key = np.asarray(key)

            key = check_array_indexer(self, key)

            if com.is_bool_indexer(key):

                return self.take(np.arange(len(key), dtype=np.int32)[key])
            elif hasattr(key, "__len__"):
                return self.take(key)
            else:
                raise ValueError(f"Cannot slice with '{key}'")

        return type(self)(data_slice, kind=self.kind)

    def _get_val_at(self, loc):
        n = len(self)
        if loc < 0:
            loc += n

        if loc >= n or loc < 0:
            raise IndexError("Out of bounds access")

        sp_loc = self.sp_index.lookup(loc)
        if sp_loc == -1:
            return self.fill_value
        else:
            val = self.sp_values[sp_loc]
            val = maybe_box_datetimelike(val, self.sp_values.dtype)
            return val

    def take(self, indices, *, allow_fill=False, fill_value=None) -> SparseArray:
        if is_scalar(indices):
            raise ValueError(f"'indices' must be an array, not a scalar '{indices}'.")
        indices = np.asarray(indices, dtype=np.int32)

        if indices.size == 0:
            result = np.array([], dtype="object")
            kwargs = {"dtype": self.dtype}
        elif allow_fill:
            result = self._take_with_fill(indices, fill_value=fill_value)
            kwargs = {}
        else:
            # error: Incompatible types in assignment (expression has type
            # "Union[ndarray, SparseArray]", variable has type "ndarray")
            result = self._take_without_fill(indices)  # type: ignore[assignment]
            kwargs = {"dtype": self.dtype}

        return type(self)(result, fill_value=self.fill_value, kind=self.kind, **kwargs)

    def _take_with_fill(self, indices, fill_value=None) -> np.ndarray:
        if fill_value is None:
            fill_value = self.dtype.na_value

        if indices.min() < -1:
            raise ValueError(
                "Invalid value in 'indices'. Must be between -1 "
                "and the length of the array."
            )

        if indices.max() >= len(self):
            raise IndexError("out of bounds value in 'indices'.")

        if len(self) == 0:
            # Empty... Allow taking only if all empty
            if (indices == -1).all():
                dtype = np.result_type(self.sp_values, type(fill_value))
                taken = np.empty_like(indices, dtype=dtype)
                taken.fill(fill_value)
                return taken
            else:
                raise IndexError("cannot do a non-empty take from an empty axes.")

        # sp_indexer may be -1 for two reasons
        # 1.) we took for an index of -1 (new)
        # 2.) we took a value that was self.fill_value (old)
        sp_indexer = self.sp_index.lookup_array(indices)
        new_fill_indices = indices == -1
        old_fill_indices = (sp_indexer == -1) & ~new_fill_indices

        if self.sp_index.npoints == 0 and old_fill_indices.all():
            # We've looked up all valid points on an all-sparse array.
            taken = np.full(
                sp_indexer.shape, fill_value=self.fill_value, dtype=self.dtype.subtype
            )

        elif self.sp_index.npoints == 0:
            # Avoid taking from the empty self.sp_values
            _dtype = np.result_type(self.dtype.subtype, type(fill_value))
            taken = np.full(sp_indexer.shape, fill_value=fill_value, dtype=_dtype)
        else:
            taken = self.sp_values.take(sp_indexer)

            # Fill in two steps.
            # Old fill values
            # New fill values
            # potentially coercing to a new dtype at each stage.

            m0 = sp_indexer[old_fill_indices] < 0
            m1 = sp_indexer[new_fill_indices] < 0

            result_type = taken.dtype

            if m0.any():
                result_type = np.result_type(result_type, type(self.fill_value))
                taken = taken.astype(result_type)
                taken[old_fill_indices] = self.fill_value

            if m1.any():
                result_type = np.result_type(result_type, type(fill_value))
                taken = taken.astype(result_type)
                taken[new_fill_indices] = fill_value

        return taken

    def _take_without_fill(self, indices) -> np.ndarray | SparseArray:
        to_shift = indices < 0
        indices = indices.copy()

        n = len(self)

        if (indices.max() >= n) or (indices.min() < -n):
            if n == 0:
                raise IndexError("cannot do a non-empty take from an empty axes.")
            else:
                raise IndexError("out of bounds value in 'indices'.")

        if to_shift.any():
            indices[to_shift] += n

        if self.sp_index.npoints == 0:
            # edge case in take...
            # I think just return
            out = np.full(
                indices.shape,
                self.fill_value,
                dtype=np.result_type(type(self.fill_value)),
            )
            arr, sp_index, fill_value = make_sparse(out, fill_value=self.fill_value)
            return type(self)(arr, sparse_index=sp_index, fill_value=fill_value)

        sp_indexer = self.sp_index.lookup_array(indices)
        taken = self.sp_values.take(sp_indexer)
        fillable = sp_indexer < 0

        if fillable.any():
            # TODO: may need to coerce array to fill value
            result_type = np.result_type(taken, type(self.fill_value))
            taken = taken.astype(result_type)
            taken[fillable] = self.fill_value

        return taken

    def searchsorted(self, v, side="left", sorter=None):
        msg = "searchsorted requires high memory usage."
        warnings.warn(msg, PerformanceWarning, stacklevel=2)
        if not is_scalar(v):
            v = np.asarray(v)
        v = np.asarray(v)
        return np.asarray(self, dtype=self.dtype.subtype).searchsorted(v, side, sorter)

    def copy(self: SparseArrayT) -> SparseArrayT:
        values = self.sp_values.copy()
        return self._simple_new(values, self.sp_index, self.dtype)

    @classmethod
    def _concat_same_type(
        cls: type[SparseArrayT], to_concat: Sequence[SparseArrayT]
    ) -> SparseArrayT:
        fill_value = to_concat[0].fill_value

        values = []
        length = 0

        if to_concat:
            sp_kind = to_concat[0].kind
        else:
            sp_kind = "integer"

        if sp_kind == "integer":
            indices = []

            for arr in to_concat:
                idx = arr.sp_index.to_int_index().indices.copy()
                idx += length  # TODO: wraparound
                length += arr.sp_index.length

                values.append(arr.sp_values)
                indices.append(idx)

            data = np.concatenate(values)
            indices = np.concatenate(indices)
            sp_index = IntIndex(length, indices)

        else:
            # when concatenating block indices, we don't claim that you'll
            # get an identical index as concatenating the values and then
            # creating a new index. We don't want to spend the time trying
            # to merge blocks across arrays in `to_concat`, so the resulting
            # BlockIndex may have more blocks.
            blengths = []
            blocs = []

            for arr in to_concat:
                idx = arr.sp_index.to_block_index()

                values.append(arr.sp_values)
                blocs.append(idx.blocs.copy() + length)
                blengths.append(idx.blengths)
                length += arr.sp_index.length

            data = np.concatenate(values)
            blocs = np.concatenate(blocs)
            blengths = np.concatenate(blengths)

            sp_index = BlockIndex(length, blocs, blengths)

        return cls(data, sparse_index=sp_index, fill_value=fill_value)

    def astype(self, dtype: Dtype | None = None, copy=True):
        """
        Change the dtype of a SparseArray.

        The output will always be a SparseArray. To convert to a dense
        ndarray with a certain dtype, use :meth:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
            For SparseDtype, this changes the dtype of
            ``self.sp_values`` and the ``self.fill_value``.

            For other dtypes, this only changes the dtype of
            ``self.sp_values``.

        copy : bool, default True
            Whether to ensure a copy is made, even if not necessary.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 0, 1, 2])
        >>> arr
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        >>> arr.astype(np.dtype('int32'))
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a NumPy dtype with a different kind (e.g. float) will coerce
        just ``self.sp_values``.

        >>> arr.astype(np.dtype('float64'))
        ... # doctest: +NORMALIZE_WHITESPACE
        [0.0, 0.0, 1.0, 2.0]
        Fill: 0.0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Use a SparseDtype if you wish to be change the fill value as well.

        >>> arr.astype(SparseDtype("float64", fill_value=np.nan))
        ... # doctest: +NORMALIZE_WHITESPACE
        [nan, nan, 1.0, 2.0]
        Fill: nan
        IntIndex
        Indices: array([2, 3], dtype=int32)
        """
        if is_dtype_equal(dtype, self._dtype):
            if not copy:
                return self
            else:
                return self.copy()
        dtype = self.dtype.update_dtype(dtype)
        # error: Item "ExtensionDtype" of "Union[ExtensionDtype, str, dtype[Any],
        # Type[str], Type[float], Type[int], Type[complex], Type[bool], Type[object],
        # None]" has no attribute "_subtype_with_str"
        # error: Item "str" of "Union[ExtensionDtype, str, dtype[Any], Type[str],
        # Type[float], Type[int], Type[complex], Type[bool], Type[object], None]" has no
        # attribute "_subtype_with_str"
        # error: Item "dtype[Any]" of "Union[ExtensionDtype, str, dtype[Any], Type[str],
        # Type[float], Type[int], Type[complex], Type[bool], Type[object], None]" has no
        # attribute "_subtype_with_str"
        # error: Item "ABCMeta" of "Union[ExtensionDtype, str, dtype[Any], Type[str],
        # Type[float], Type[int], Type[complex], Type[bool], Type[object], None]" has no
        # attribute "_subtype_with_str"
        # error: Item "type" of "Union[ExtensionDtype, str, dtype[Any], Type[str],
        # Type[float], Type[int], Type[complex], Type[bool], Type[object], None]" has no
        # attribute "_subtype_with_str"
        # error: Item "None" of "Union[ExtensionDtype, str, dtype[Any], Type[str],
        # Type[float], Type[int], Type[complex], Type[bool], Type[object], None]" has no
        # attribute "_subtype_with_str"
        subtype = pandas_dtype(dtype._subtype_with_str)  # type: ignore[union-attr]
        # TODO copy=False is broken for astype_nansafe with int -> float, so cannot
        # passthrough copy keyword: https://github.com/pandas-dev/pandas/issues/34456
        sp_values = astype_nansafe(self.sp_values, subtype, copy=True)
        if sp_values is self.sp_values and copy:
            sp_values = sp_values.copy()

        # error: Argument 1 to "_simple_new" of "SparseArray" has incompatible type
        # "ExtensionArray"; expected "ndarray"
        return self._simple_new(
            sp_values, self.sp_index, dtype  # type: ignore[arg-type]
        )

    def map(self, mapper):
        """
        Map categories using input correspondence (dict, Series, or function).

        Parameters
        ----------
        mapper : dict, Series, callable
            The correspondence from old values to new.

        Returns
        -------
        SparseArray
            The output array will have the same density as the input.
            The output fill value will be the result of applying the
            mapping to ``self.fill_value``

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 1, 2])
        >>> arr.map(lambda x: x + 10)
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map({0: 10, 1: 11, 2: 12})
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map(pd.Series([10, 11, 12], index=[0, 1, 2]))
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)
        """
        # this is used in apply.
        # We get hit since we're an "is_extension_type" but regular extension
        # types are not hit. This may be worth adding to the interface.
        if isinstance(mapper, ABCSeries):
            mapper = mapper.to_dict()

        if isinstance(mapper, abc.Mapping):
            fill_value = mapper.get(self.fill_value, self.fill_value)
            sp_values = [mapper.get(x, None) for x in self.sp_values]
        else:
            fill_value = mapper(self.fill_value)
            sp_values = [mapper(x) for x in self.sp_values]

        return type(self)(sp_values, sparse_index=self.sp_index, fill_value=fill_value)

    def to_dense(self):
        """
        Convert SparseArray to a NumPy array.

        Returns
        -------
        arr : NumPy array
        """
        return np.asarray(self, dtype=self.sp_values.dtype)

    _internal_get_values = to_dense

    # ------------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------------
    def __setstate__(self, state):
        """Necessary for making this object picklable"""
        if isinstance(state, tuple):
            # Compat for pandas < 0.24.0
            nd_state, (fill_value, sp_index) = state
            sparse_values = np.array([])
            sparse_values.__setstate__(nd_state)

            self._sparse_values = sparse_values
            self._sparse_index = sp_index
            self._dtype = SparseDtype(sparse_values.dtype, fill_value)
        else:
            self.__dict__.update(state)

    def nonzero(self):
        if self.fill_value == 0:
            return (self.sp_index.to_int_index().indices,)
        else:
            return (self.sp_index.to_int_index().indices[self.sp_values != 0],)

    # ------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------

    def _reduce(self, name: str, *, skipna: bool = True, **kwargs):
        method = getattr(self, name, None)

        if method is None:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if skipna:
            arr = self
        else:
            arr = self.dropna()

        # we don't support these kwargs.
        # They should only be present when called via pandas, so do it here.
        # instead of in `any` / `all` (which will raise if they're present,
        # thanks to nv.validate
        kwargs.pop("filter_type", None)
        kwargs.pop("numeric_only", None)
        kwargs.pop("op", None)
        return getattr(arr, name)(**kwargs)

    def all(self, axis=None, *args, **kwargs):
        """
        Tests whether all elements evaluate True

        Returns
        -------
        all : bool

        See Also
        --------
        numpy.all
        """
        nv.validate_all(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and not np.all(self.fill_value):
            return False

        return values.all()

    def any(self, axis=0, *args, **kwargs):
        """
        Tests whether at least one of elements evaluate True

        Returns
        -------
        any : bool

        See Also
        --------
        numpy.any
        """
        nv.validate_any(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and np.any(self.fill_value):
            return True

        return values.any().item()

    def sum(self, axis: int = 0, min_count: int = 0, *args, **kwargs) -> Scalar:
        """
        Sum of non-NA/null values

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        min_count : int, default 0
            The required number of valid values to perform the summation. If fewer
            than ``min_count`` valid values are present, the result will be the missing
            value indicator for subarray type.
        *args, **kwargs
            Not Used. NumPy compatibility.

        Returns
        -------
        scalar
        """
        nv.validate_sum(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        if self._null_fill_value:
            if check_below_min_count(valid_vals.shape, None, min_count):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum
        else:
            nsparse = self.sp_index.ngaps
            if check_below_min_count(valid_vals.shape, None, min_count - nsparse):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum + self.fill_value * nsparse

    def cumsum(self, axis=0, *args, **kwargs):
        """
        Cumulative sum of non-NA/null values.

        When performing the cumulative summation, any non-NA/null values will
        be skipped. The resulting SparseArray will preserve the locations of
        NaN values, but the fill value will be `np.nan` regardless.

        Parameters
        ----------
        axis : int or None
            Axis over which to perform the cumulative summation. If None,
            perform cumulative summation over flattened array.

        Returns
        -------
        cumsum : SparseArray
        """
        nv.validate_cumsum(args, kwargs)

        if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
            raise ValueError(f"axis(={axis}) out of bounds")

        if not self._null_fill_value:
            return SparseArray(self.to_dense()).cumsum()

        return SparseArray(
            self.sp_values.cumsum(),
            sparse_index=self.sp_index,
            fill_value=self.fill_value,
        )

    def mean(self, axis=0, *args, **kwargs):
        """
        Mean of non-NA/null values

        Returns
        -------
        mean : float
        """
        nv.validate_mean(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        ct = len(valid_vals)

        if self._null_fill_value:
            return sp_sum / ct
        else:
            nsparse = self.sp_index.ngaps
            return (sp_sum + self.fill_value * nsparse) / (ct + nsparse)

    def max(self, axis=0, *args, **kwargs):
        nv.validate_max(args, kwargs)

        # This condition returns a nan if there are no valid values in the array.
        if self.size > 0 and self._valid_sp_values.size == 0:
            return self.fill_value
        else:
            return np.nanmax(self, axis)

    def min(self, axis=0, *args, **kwargs):
        nv.validate_min(args, kwargs)

        # This condition returns a nan if there are no valid values in the array.
        if self.size > 0 and self._valid_sp_values.size == 0:
            return self.fill_value
        else:
            return np.nanmin(self, axis)

    # ------------------------------------------------------------------------
    # Ufuncs
    # ------------------------------------------------------------------------

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (SparseArray,)):
                return NotImplemented

        # for binary ops, use our custom dunder methods
        result = ops.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if len(inputs) == 1:
            # No alignment necessary.
            sp_values = getattr(ufunc, method)(self.sp_values, **kwargs)
            fill_value = getattr(ufunc, method)(self.fill_value, **kwargs)

            if isinstance(sp_values, tuple):
                # multiple outputs. e.g. modf
                arrays = tuple(
                    self._simple_new(
                        sp_value, self.sp_index, SparseDtype(sp_value.dtype, fv)
                    )
                    for sp_value, fv in zip(sp_values, fill_value)
                )
                return arrays
            elif is_scalar(sp_values):
                # e.g. reductions
                return sp_values

            return self._simple_new(
                sp_values, self.sp_index, SparseDtype(sp_values.dtype, fill_value)
            )

        result = getattr(ufunc, method)(*[np.asarray(x) for x in inputs], **kwargs)
        if out:
            if len(out) == 1:
                out = out[0]
            return out

        if type(result) is tuple:
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            return type(self)(result)

    def __abs__(self):
        return np.abs(self)

    # ------------------------------------------------------------------------
    # Ops
    # ------------------------------------------------------------------------

    def _arith_method(self, other, op):
        op_name = op.__name__

        if isinstance(other, SparseArray):
            return _sparse_array_op(self, other, op, op_name)

        elif is_scalar(other):
            with np.errstate(all="ignore"):
                fill = op(_get_fill(self), np.asarray(other))
                result = op(self.sp_values, other)

            if op_name == "divmod":
                left, right = result
                lfill, rfill = fill
                return (
                    _wrap_result(op_name, left, self.sp_index, lfill),
                    _wrap_result(op_name, right, self.sp_index, rfill),
                )

            return _wrap_result(op_name, result, self.sp_index, fill)

        else:
            other = np.asarray(other)
            with np.errstate(all="ignore"):
                # TODO: look into _wrap_result
                if len(self) != len(other):
                    raise AssertionError(
                        f"length mismatch: {len(self)} vs. {len(other)}"
                    )
                if not isinstance(other, SparseArray):
                    dtype = getattr(other, "dtype", None)
                    other = SparseArray(other, fill_value=self.fill_value, dtype=dtype)
                return _sparse_array_op(self, other, op, op_name)

    def _cmp_method(self, other, op) -> SparseArray:
        if not is_scalar(other) and not isinstance(other, type(self)):
            # convert list-like to ndarray
            other = np.asarray(other)

        if isinstance(other, np.ndarray):
            # TODO: make this more flexible than just ndarray...
            if len(self) != len(other):
                raise AssertionError(f"length mismatch: {len(self)} vs. {len(other)}")
            other = SparseArray(other, fill_value=self.fill_value)

        if isinstance(other, SparseArray):
            op_name = op.__name__.strip("_")
            return _sparse_array_op(self, other, op, op_name)
        else:
            with np.errstate(all="ignore"):
                fill_value = op(self.fill_value, other)
                result = op(self.sp_values, other)

            return type(self)(
                result,
                sparse_index=self.sp_index,
                fill_value=fill_value,
                dtype=np.bool_,
            )

    _logical_method = _cmp_method

    def _unary_method(self, op) -> SparseArray:
        fill_value = op(np.array(self.fill_value)).item()
        values = op(self.sp_values)
        dtype = SparseDtype(values.dtype, fill_value)
        return type(self)._simple_new(values, self.sp_index, dtype)

    def __pos__(self) -> SparseArray:
        return self._unary_method(operator.pos)

    def __neg__(self) -> SparseArray:
        return self._unary_method(operator.neg)

    def __invert__(self) -> SparseArray:
        return self._unary_method(operator.invert)

    # ----------
    # Formatting
    # -----------
    def __repr__(self) -> str:
        pp_str = printing.pprint_thing(self)
        pp_fill = printing.pprint_thing(self.fill_value)
        pp_index = printing.pprint_thing(self.sp_index)
        return f"{pp_str}\nFill: {pp_fill}\n{pp_index}"

    def _formatter(self, boxed=False):
        # Defer to the formatter from the GenericArrayFormatter calling us.
        # This will infer the correct formatter from the dtype of the values.
        return None


def make_sparse(
    arr: np.ndarray, kind="block", fill_value=None, dtype: NpDtype | None = None
):
    """
    Convert ndarray to sparse format

    Parameters
    ----------
    arr : ndarray
    kind : {'block', 'integer'}
    fill_value : NaN or another value
    dtype : np.dtype, optional
    copy : bool, default False

    Returns
    -------
    (sparse_values, index, fill_value) : (ndarray, SparseIndex, Scalar)
    """
    assert isinstance(arr, np.ndarray)

    if arr.ndim > 1:
        raise TypeError("expected dimension <= 1 data")

    if fill_value is None:
        fill_value = na_value_for_dtype(arr.dtype)

    if isna(fill_value):
        mask = notna(arr)
    else:
        # cast to object comparison to be safe
        if is_string_dtype(arr.dtype):
            arr = arr.astype(object)

        if is_object_dtype(arr.dtype):
            # element-wise equality check method in numpy doesn't treat
            # each element type, eg. 0, 0.0, and False are treated as
            # same. So we have to check the both of its type and value.
            mask = splib.make_mask_object_ndarray(arr, fill_value)
        else:
            mask = arr != fill_value

    length = len(arr)
    if length != len(mask):
        # the arr is a SparseArray
        indices = mask.sp_index.indices
    else:
        indices = mask.nonzero()[0].astype(np.int32)

    index = make_sparse_index(length, indices, kind)
    sparsified_values = arr[mask]
    if dtype is not None:
        # error: Argument "dtype" to "astype_nansafe" has incompatible type "Union[str,
        # dtype[Any]]"; expected "Union[dtype[Any], ExtensionDtype]"
        sparsified_values = astype_nansafe(
            sparsified_values, dtype=dtype  # type: ignore[arg-type]
        )
    # TODO: copy
    return sparsified_values, index, fill_value


def make_sparse_index(length, indices, kind):

    if kind == "block" or isinstance(kind, BlockIndex):
        locs, lens = splib.get_blocks(indices)
        index = BlockIndex(length, locs, lens)
    elif kind == "integer" or isinstance(kind, IntIndex):
        index = IntIndex(length, indices)
    else:  # pragma: no cover
        raise ValueError("must be block or integer type")
    return index
