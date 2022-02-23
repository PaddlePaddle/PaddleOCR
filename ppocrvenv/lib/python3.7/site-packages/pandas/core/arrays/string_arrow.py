from __future__ import annotations

from collections.abc import Callable  # noqa: PDF001
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Sequence,
    cast,
)

import numpy as np

from pandas._libs import lib
from pandas._typing import (
    Dtype,
    NpDtype,
    PositionalIndexer,
    Scalar,
)
from pandas.compat import (
    pa_version_under1p0,
    pa_version_under2p0,
    pa_version_under3p0,
    pa_version_under4p0,
)
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_dtype_equal,
    is_integer,
    is_integer_dtype,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.missing import isna

from pandas.core import missing
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import (
    BaseStringArray,
    StringDtype,
)
from pandas.core.indexers import (
    check_array_indexer,
    validate_indices,
)
from pandas.core.strings.object_array import ObjectStringArrayMixin

# PyArrow backed StringArrays are available starting at 1.0.0, but this
# file is imported from even if pyarrow is < 1.0.0, before pyarrow.compute
# and its compute functions existed. GH38801
if not pa_version_under1p0:
    import pyarrow as pa
    import pyarrow.compute as pc

    ARROW_CMP_FUNCS = {
        "eq": pc.equal,
        "ne": pc.not_equal,
        "lt": pc.less,
        "gt": pc.greater,
        "le": pc.less_equal,
        "ge": pc.greater_equal,
    }


if TYPE_CHECKING:
    from pandas import Series


def _chk_pyarrow_available() -> None:
    if pa_version_under1p0:
        msg = "pyarrow>=1.0.0 is required for PyArrow backed StringArray."
        raise ImportError(msg)


# TODO: Inherit directly from BaseStringArrayMethods. Currently we inherit from
# ObjectStringArrayMixin because we want to have the object-dtype based methods as
# fallback for the ones that pyarrow doesn't yet support


class ArrowStringArray(OpsMixin, BaseStringArray, ObjectStringArrayMixin):
    """
    Extension array for string data in a ``pyarrow.ChunkedArray``.

    .. versionadded:: 1.2.0

    .. warning::

       ArrowStringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    array
        The recommended function for creating a ArrowStringArray.
    Series.str
        The string methods are available on Series backed by
        a ArrowStringArray.

    Notes
    -----
    ArrowStringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(['This is', 'some text', None, 'data.'], dtype="string[pyarrow]")
    <ArrowStringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string
    """

    def __init__(self, values):
        self._dtype = StringDtype(storage="pyarrow")
        if isinstance(values, pa.Array):
            self._data = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._data = values
        else:
            raise ValueError(f"Unsupported type '{type(values)}' for ArrowStringArray")

        if not pa.types.is_string(self._data.type):
            raise ValueError(
                "ArrowStringArray requires a PyArrow (chunked) array of string type"
            )

    @classmethod
    def _from_sequence(cls, scalars, dtype: Dtype | None = None, copy: bool = False):
        from pandas.core.arrays.masked import BaseMaskedArray

        _chk_pyarrow_available()

        if dtype and not (isinstance(dtype, str) and dtype == "string"):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == "pyarrow"

        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype in ensure_string_array and
            # numerical issues with Float32Dtype
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            return cls(pa.array(result, mask=na_values, type=pa.string()))

        # convert non-na-likes to str
        result = lib.ensure_string_array(scalars, copy=copy)
        return cls(pa.array(result, type=pa.string(), from_pandas=True))

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, dtype: Dtype | None = None, copy: bool = False
    ):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @property
    def dtype(self) -> StringDtype:
        """
        An instance of 'string[pyarrow]'.
        """
        return self._dtype

    def __array__(self, dtype: NpDtype | None = None) -> np.ndarray:
        """Correctly construct numpy arrays when passed to `np.asarray()`."""
        return self.to_numpy(dtype=dtype)

    def __arrow_array__(self, type=None):
        """Convert myself to a pyarrow Array or ChunkedArray."""
        return self._data

    # error: Argument 1 of "to_numpy" is incompatible with supertype "ExtensionArray";
    # supertype defines the argument type as "Union[ExtensionDtype, str, dtype[Any],
    # Type[str], Type[float], Type[int], Type[complex], Type[bool], Type[object], None]"
    def to_numpy(  # type: ignore[override]
        self,
        dtype: NpDtype | None = None,
        copy: bool = False,
        na_value=lib.no_default,
    ) -> np.ndarray:
        """
        Convert to a NumPy ndarray.
        """
        # TODO: copy argument is ignored

        result = np.array(self._data, dtype=dtype)
        if self._data.null_count > 0:
            if na_value is lib.no_default:
                if dtype and np.issubdtype(dtype, np.floating):
                    return result
                na_value = self._dtype.na_value
            mask = self.isna()
            result[mask] = na_value
        return result

    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return len(self._data)

    @doc(ExtensionArray.factorize)
    def factorize(self, na_sentinel: int = -1) -> tuple[np.ndarray, ExtensionArray]:
        encoded = self._data.dictionary_encode()
        indices = pa.chunked_array(
            [c.indices for c in encoded.chunks], type=encoded.type.index_type
        ).to_pandas()
        if indices.dtype.kind == "f":
            indices[np.isnan(indices)] = na_sentinel
        indices = indices.astype(np.int64, copy=False)

        if encoded.num_chunks:
            uniques = type(self)(encoded.chunk(0).dictionary)
        else:
            uniques = type(self)(pa.array([], type=encoded.type.value_type))

        return indices.values, uniques

    @classmethod
    def _concat_same_type(cls, to_concat) -> ArrowStringArray:
        """
        Concatenate multiple ArrowStringArray.

        Parameters
        ----------
        to_concat : sequence of ArrowStringArray

        Returns
        -------
        ArrowStringArray
        """
        return cls(
            pa.chunked_array(
                [array for ea in to_concat for array in ea._data.iterchunks()]
            )
        )

    def __getitem__(self, item: PositionalIndexer) -> Any:
        """Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.
        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.
        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        item = check_array_indexer(self, item)

        if isinstance(item, np.ndarray):
            if not len(item):
                return type(self)(pa.chunked_array([], type=pa.string()))
            elif is_integer_dtype(item.dtype):
                # error: Argument 1 to "take" of "ArrowStringArray" has incompatible
                # type "ndarray"; expected "Sequence[int]"
                return self.take(item)  # type: ignore[arg-type]
            elif is_bool_dtype(item.dtype):
                return type(self)(self._data.filter(item))
            else:
                raise IndexError(
                    "Only integers, slices and integer or "
                    "boolean arrays are valid indices."
                )
        elif isinstance(item, tuple):
            # possibly unpack arr[..., n] to arr[n]
            if len(item) == 1:
                item = item[0]
            elif len(item) == 2:
                if item[0] is Ellipsis:
                    item = item[1]
                elif item[1] is Ellipsis:
                    item = item[0]

        # We are not an array indexer, so maybe e.g. a slice or integer
        # indexer. We dispatch to pyarrow.
        value = self._data[item]
        if isinstance(value, pa.ChunkedArray):
            return type(self)(value)
        else:
            return self._as_pandas_scalar(value)

    def _as_pandas_scalar(self, arrow_scalar: pa.Scalar):
        scalar = arrow_scalar.as_py()
        if scalar is None:
            return self._dtype.na_value
        else:
            return scalar

    def fillna(self, value=None, method=None, limit=None):
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap.
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.
        """
        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()
        value = missing.check_value_size(value, mask, len(self))

        if mask.any():
            if method is not None:
                func = missing.get_fill_func(method)
                new_values, _ = func(
                    self.to_numpy("object"),
                    limit=limit,
                    mask=mask,
                )
                new_values = self._from_sequence(new_values)
            else:
                # fill with value
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

    def _reduce(self, name: str, skipna: bool = True, **kwargs):
        if name in ["min", "max"]:
            return getattr(self, name)(skipna=skipna)

        raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self._data.nbytes

    def isna(self) -> np.ndarray:
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
        # TODO: Implement .to_numpy for ChunkedArray
        return self._data.is_null().to_pandas().values

    def copy(self) -> ArrowStringArray:
        """
        Return a shallow copy of the array.

        Returns
        -------
        ArrowStringArray
        """
        return type(self)(self._data)

    def _cmp_method(self, other, op):
        from pandas.arrays import BooleanArray

        pc_func = ARROW_CMP_FUNCS[op.__name__]
        if isinstance(other, ArrowStringArray):
            result = pc_func(self._data, other._data)
        elif isinstance(other, np.ndarray):
            result = pc_func(self._data, other)
        elif is_scalar(other):
            try:
                result = pc_func(self._data, pa.scalar(other))
            except (pa.lib.ArrowNotImplementedError, pa.lib.ArrowInvalid):
                mask = isna(self) | isna(other)
                valid = ~mask
                result = np.zeros(len(self), dtype="bool")
                result[valid] = op(np.array(self)[valid], other)
                return BooleanArray(result, mask)
        else:
            return NotImplemented

        # TODO(ARROW-9429): Add a .to_numpy() to ChunkedArray
        return BooleanArray._from_sequence(result.to_pandas().values)

    def __setitem__(self, key: int | slice | np.ndarray, value: Any) -> None:
        """Set one or more values inplace.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        key = check_array_indexer(self, key)

        if is_integer(key):
            key = cast(int, key)

            if not is_scalar(value):
                raise ValueError("Must pass scalars with scalar indexer")
            elif isna(value):
                value = None
            elif not isinstance(value, str):
                raise ValueError("Scalar must be NA or str")

            # Slice data and insert in-between
            new_data = [
                *self._data[0:key].chunks,
                pa.array([value], type=pa.string()),
                *self._data[(key + 1) :].chunks,
            ]
            self._data = pa.chunked_array(new_data)
        else:
            # Convert to integer indices and iteratively assign.
            # TODO: Make a faster variant of this in Arrow upstream.
            #       This is probably extremely slow.

            # Convert all possible input key types to an array of integers
            if isinstance(key, slice):
                key_array = np.array(range(len(self))[key])
            elif is_bool_dtype(key):
                # TODO(ARROW-9430): Directly support setitem(booleans)
                key_array = np.argwhere(key).flatten()
            else:
                # TODO(ARROW-9431): Directly support setitem(integers)
                key_array = np.asanyarray(key)

            if is_scalar(value):
                value = np.broadcast_to(value, len(key_array))
            else:
                value = np.asarray(value)

            if len(key_array) != len(value):
                raise ValueError("Length of indexer and values mismatch")

            for k, v in zip(key_array, value):
                self[k] = v

    def take(
        self, indices: Sequence[int], allow_fill: bool = False, fill_value: Any = None
    ):
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        Returns
        -------
        ExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take
        api.extensions.take

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignment, with a `fill_value`.
        """
        # TODO: Remove once we got rid of the (indices < 0) check
        if not is_array_like(indices):
            indices_array = np.asanyarray(indices)
        else:
            # error: Incompatible types in assignment (expression has type
            # "Sequence[int]", variable has type "ndarray")
            indices_array = indices  # type: ignore[assignment]

        if len(self._data) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take")
        if indices_array.size > 0 and indices_array.max() >= len(self._data):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_mask = indices_array < 0
            if fill_mask.any():
                validate_indices(indices_array, len(self._data))
                # TODO(ARROW-9433): Treat negative indices as NULL
                indices_array = pa.array(indices_array, mask=fill_mask)
                result = self._data.take(indices_array)
                if isna(fill_value):
                    return type(self)(result)
                # TODO: ArrowNotImplementedError: Function fill_null has no
                # kernel matching input types (array[string], scalar[string])
                result = type(self)(result)
                result[fill_mask] = fill_value
                return result
                # return type(self)(pc.fill_null(result, pa.scalar(fill_value)))
            else:
                # Nothing to fill
                return type(self)(self._data.take(indices))
        else:  # allow_fill=False
            # TODO(ARROW-9432): Treat negative indices as indices from the right.
            if (indices_array < 0).any():
                # Don't modify in-place
                indices_array = np.copy(indices_array)
                indices_array[indices_array < 0] += len(self._data)
            return type(self)(self._data.take(indices_array))

    def isin(self, values):
        if pa_version_under2p0:
            return super().isin(values)

        value_set = [
            pa_scalar.as_py()
            for pa_scalar in [pa.scalar(value, from_pandas=True) for value in values]
            if pa_scalar.type in (pa.string(), pa.null())
        ]

        # for an empty value_set pyarrow 3.0.0 segfaults and pyarrow 2.0.0 returns True
        # for null values, so we short-circuit to return all False array.
        if not len(value_set):
            return np.zeros(len(self), dtype=bool)

        kwargs = {}
        if pa_version_under3p0:
            # in pyarrow 2.0.0 skip_null is ignored but is a required keyword and raises
            # with unexpected keyword argument in pyarrow 3.0.0+
            kwargs["skip_null"] = True

        result = pc.is_in(self._data, value_set=pa.array(value_set), **kwargs)
        # pyarrow 2.0.0 returned nulls, so we explicily specify dtype to convert nulls
        # to False
        return np.array(result, dtype=np.bool_)

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        from pandas import (
            Index,
            Series,
        )

        vc = self._data.value_counts()

        values = vc.field(0)
        counts = vc.field(1)
        if dropna and self._data.null_count > 0:
            mask = values.is_valid()
            values = values.filter(mask)
            counts = counts.filter(mask)

        # No missing values so we can adhere to the interface and return a numpy array.
        counts = np.array(counts)

        # Index cannot hold ExtensionArrays yet
        index = Index(type(self)(values)).astype(object)

        return Series(counts, index=index).astype("Int64")

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)

        if is_dtype_equal(dtype, self.dtype):
            if copy:
                return self.copy()
            return self

        elif isinstance(dtype, NumericDtype):
            data = self._data.cast(pa.from_numpy_dtype(dtype.numpy_dtype))
            return dtype.__from_arrow__(data)

        elif isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            return cls._from_sequence(self, dtype=dtype, copy=copy)

        return super().astype(dtype, copy)

    # ------------------------------------------------------------------------
    # String methods interface

    # error: Cannot determine type of 'na_value'
    _str_na_value = StringDtype.na_value  # type: ignore[has-type]

    def _str_map(
        self, f, na_value=None, dtype: Dtype | None = None, convert: bool = True
    ):
        # TODO: de-duplicate with StringArray method. This method is moreless copy and
        # paste.

        from pandas.arrays import (
            BooleanArray,
            IntegerArray,
        )

        if dtype is None:
            dtype = self.dtype
        if na_value is None:
            na_value = self.dtype.na_value

        mask = isna(self)
        arr = np.asarray(self)

        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            constructor: type[IntegerArray] | type[BooleanArray]
            if is_integer_dtype(dtype):
                constructor = IntegerArray
            else:
                constructor = BooleanArray

            na_value_is_na = isna(na_value)
            if na_value_is_na:
                na_value = 1
            result = lib.map_infer_mask(
                arr,
                f,
                mask.view("uint8"),
                convert=False,
                na_value=na_value,
                # error: Value of type variable "_DTypeScalar" of "dtype" cannot be
                # "object"
                # error: Argument 1 to "dtype" has incompatible type
                # "Union[ExtensionDtype, str, dtype[Any], Type[object]]"; expected
                # "Type[object]"
                dtype=np.dtype(dtype),  # type: ignore[type-var,arg-type]
            )

            if not na_value_is_na:
                mask[:] = False

            return constructor(result, mask)

        elif is_string_dtype(dtype) and not is_object_dtype(dtype):
            # i.e. StringDtype
            result = lib.map_infer_mask(
                arr, f, mask.view("uint8"), convert=False, na_value=na_value
            )
            result = pa.array(result, mask=mask, type=pa.string(), from_pandas=True)
            return type(self)(result)
        else:
            # This is when the result type is object. We reach this when
            # -> We know the result type is truly object (e.g. .encode returns bytes
            #    or .findall returns a list).
            # -> We don't know the result type. E.g. `.get` can return anything.
            return lib.map_infer_mask(arr, f, mask.view("uint8"))

    def _str_contains(self, pat, case=True, flags=0, na=np.nan, regex: bool = True):
        if flags:
            return super()._str_contains(pat, case, flags, na, regex)

        if regex:
            if pa_version_under4p0 or case is False:
                return super()._str_contains(pat, case, flags, na, regex)
            else:
                result = pc.match_substring_regex(self._data, pat)
        else:
            if case:
                result = pc.match_substring(self._data, pat)
            else:
                result = pc.match_substring(pc.utf8_upper(self._data), pat.upper())
        result = BooleanDtype().__from_arrow__(result)
        if not isna(na):
            result[isna(result)] = bool(na)
        return result

    def _str_startswith(self, pat: str, na=None):
        if pa_version_under4p0:
            return super()._str_startswith(pat, na)

        pat = "^" + re.escape(pat)
        return self._str_contains(pat, na=na, regex=True)

    def _str_endswith(self, pat: str, na=None):
        if pa_version_under4p0:
            return super()._str_endswith(pat, na)

        pat = re.escape(pat) + "$"
        return self._str_contains(pat, na=na, regex=True)

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        if (
            pa_version_under4p0
            or isinstance(pat, re.Pattern)
            or callable(repl)
            or not case
            or flags
        ):
            return super()._str_replace(pat, repl, n, case, flags, regex)

        func = pc.replace_substring_regex if regex else pc.replace_substring
        result = func(self._data, pattern=pat, replacement=repl, max_replacements=n)
        return type(self)(result)

    def _str_match(
        self, pat: str, case: bool = True, flags: int = 0, na: Scalar = None
    ):
        if pa_version_under4p0:
            return super()._str_match(pat, case, flags, na)

        if not pat.startswith("^"):
            pat = "^" + pat
        return self._str_contains(pat, case, flags, na, regex=True)

    def _str_fullmatch(self, pat, case: bool = True, flags: int = 0, na: Scalar = None):
        if pa_version_under4p0:
            return super()._str_fullmatch(pat, case, flags, na)

        if not pat.endswith("$") or pat.endswith("//$"):
            pat = pat + "$"
        return self._str_match(pat, case, flags, na)

    def _str_isalnum(self):
        result = pc.utf8_is_alnum(self._data)
        return BooleanDtype().__from_arrow__(result)

    def _str_isalpha(self):
        result = pc.utf8_is_alpha(self._data)
        return BooleanDtype().__from_arrow__(result)

    def _str_isdecimal(self):
        result = pc.utf8_is_decimal(self._data)
        return BooleanDtype().__from_arrow__(result)

    def _str_isdigit(self):
        result = pc.utf8_is_digit(self._data)
        return BooleanDtype().__from_arrow__(result)

    def _str_islower(self):
        result = pc.utf8_is_lower(self._data)
        return BooleanDtype().__from_arrow__(result)

    def _str_isnumeric(self):
        result = pc.utf8_is_numeric(self._data)
        return BooleanDtype().__from_arrow__(result)

    def _str_isspace(self):
        if pa_version_under2p0:
            return super()._str_isspace()

        result = pc.utf8_is_space(self._data)
        return BooleanDtype().__from_arrow__(result)

    def _str_istitle(self):
        result = pc.utf8_is_title(self._data)
        return BooleanDtype().__from_arrow__(result)

    def _str_isupper(self):
        result = pc.utf8_is_upper(self._data)
        return BooleanDtype().__from_arrow__(result)

    def _str_len(self):
        if pa_version_under4p0:
            return super()._str_len()

        result = pc.utf8_length(self._data)
        return Int64Dtype().__from_arrow__(result)

    def _str_lower(self):
        return type(self)(pc.utf8_lower(self._data))

    def _str_upper(self):
        return type(self)(pc.utf8_upper(self._data))

    def _str_strip(self, to_strip=None):
        if pa_version_under4p0:
            return super()._str_strip(to_strip)

        if to_strip is None:
            result = pc.utf8_trim_whitespace(self._data)
        else:
            result = pc.utf8_trim(self._data, characters=to_strip)
        return type(self)(result)

    def _str_lstrip(self, to_strip=None):
        if pa_version_under4p0:
            return super()._str_lstrip(to_strip)

        if to_strip is None:
            result = pc.utf8_ltrim_whitespace(self._data)
        else:
            result = pc.utf8_ltrim(self._data, characters=to_strip)
        return type(self)(result)

    def _str_rstrip(self, to_strip=None):
        if pa_version_under4p0:
            return super()._str_rstrip(to_strip)

        if to_strip is None:
            result = pc.utf8_rtrim_whitespace(self._data)
        else:
            result = pc.utf8_rtrim(self._data, characters=to_strip)
        return type(self)(result)
