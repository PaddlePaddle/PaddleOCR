from __future__ import annotations

import warnings

import numpy as np

from pandas._libs import (
    iNaT,
    lib,
    missing as libmissing,
)
from pandas._typing import (
    ArrayLike,
    Dtype,
    DtypeObj,
)
from pandas.compat.numpy import function as nv
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.base import (
    ExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_float,
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays.masked import (
    BaseMaskedArray,
    BaseMaskedDtype,
)
from pandas.core.arrays.numeric import (
    NumericArray,
    NumericDtype,
)
from pandas.core.ops import invalid_comparison
from pandas.core.tools.numeric import to_numeric


class _IntegerDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size & kind of integer dtype.

    These specific implementations are subclasses of the non-public
    _IntegerDtype. For example we have Int8Dtype to represent signed int 8s.

    The attributes name & type are set when these subclasses are created.
    """

    def __repr__(self) -> str:
        sign = "U" if self.is_unsigned_integer else ""
        return f"{sign}Int{8 * self.itemsize}Dtype()"

    @cache_readonly
    def is_signed_integer(self) -> bool:
        return self.kind == "i"

    @cache_readonly
    def is_unsigned_integer(self) -> bool:
        return self.kind == "u"

    @property
    def _is_numeric(self) -> bool:
        return True

    @classmethod
    def construct_array_type(cls) -> type[IntegerArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return IntegerArray

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # we only handle nullable EA dtypes and numeric numpy dtypes
        if not all(
            isinstance(t, BaseMaskedDtype)
            or (
                isinstance(t, np.dtype)
                and (np.issubdtype(t, np.number) or np.issubdtype(t, np.bool_))
            )
            for t in dtypes
        ):
            return None
        np_dtype = np.find_common_type(
            # error: List comprehension has incompatible type List[Union[Any,
            # dtype, ExtensionDtype]]; expected List[Union[dtype, None, type,
            # _SupportsDtype, str, Tuple[Any, Union[int, Sequence[int]]],
            # List[Any], _DtypeDict, Tuple[Any, Any]]]
            [
                t.numpy_dtype  # type: ignore[misc]
                if isinstance(t, BaseMaskedDtype)
                else t
                for t in dtypes
            ],
            [],
        )
        if np.issubdtype(np_dtype, np.integer):
            return INT_STR_TO_DTYPE[str(np_dtype)]
        elif np.issubdtype(np_dtype, np.floating):
            from pandas.core.arrays.floating import FLOAT_STR_TO_DTYPE

            return FLOAT_STR_TO_DTYPE[str(np_dtype)]
        return None


def safe_cast(values, dtype, copy: bool):
    """
    Safely cast the values to the dtype if they
    are equivalent, meaning floats must be equivalent to the
    ints.

    """
    try:
        return values.astype(dtype, casting="safe", copy=copy)
    except TypeError as err:

        casted = values.astype(dtype, copy=copy)
        if (casted == values).all():
            return casted

        raise TypeError(
            f"cannot safely cast non-equivalent {values.dtype} to {np.dtype(dtype)}"
        ) from err


def coerce_to_array(
    values, dtype, mask=None, copy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask

    Parameters
    ----------
    values : 1D list-like
    dtype : integer dtype
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    # if values is integer numpy array, preserve its dtype
    if dtype is None and hasattr(values, "dtype"):
        if is_integer_dtype(values.dtype):
            dtype = values.dtype

    if dtype is not None:
        if isinstance(dtype, str) and (
            dtype.startswith("Int") or dtype.startswith("UInt")
        ):
            # Avoid DeprecationWarning from NumPy about np.dtype("Int64")
            # https://github.com/numpy/numpy/pull/7476
            dtype = dtype.lower()

        if not issubclass(type(dtype), _IntegerDtype):
            try:
                dtype = INT_STR_TO_DTYPE[str(np.dtype(dtype))]
            except KeyError as err:
                raise ValueError(f"invalid dtype specified {dtype}") from err

    if isinstance(values, IntegerArray):
        values, mask = values._data, values._mask
        if dtype is not None:
            values = values.astype(dtype.numpy_dtype, copy=False)

        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask

    values = np.array(values, copy=copy)
    if is_object_dtype(values):
        inferred_type = lib.infer_dtype(values, skipna=True)
        if inferred_type == "empty":
            values = np.empty(len(values))
            values.fill(np.nan)
        elif inferred_type not in [
            "floating",
            "integer",
            "mixed-integer",
            "integer-na",
            "mixed-integer-float",
        ]:
            raise TypeError(f"{values.dtype} cannot be converted to an IntegerDtype")

    elif is_bool_dtype(values) and is_integer_dtype(dtype):
        values = np.array(values, dtype=int, copy=copy)

    elif not (is_integer_dtype(values) or is_float_dtype(values)):
        raise TypeError(f"{values.dtype} cannot be converted to an IntegerDtype")

    if mask is None:
        mask = isna(values)
    else:
        assert len(mask) == len(values)

    if not values.ndim == 1:
        raise TypeError("values must be a 1D list-like")
    if not mask.ndim == 1:
        raise TypeError("mask must be a 1D list-like")

    # infer dtype if needed
    if dtype is None:
        dtype = np.dtype("int64")
    else:
        dtype = dtype.type

    # if we are float, let's make sure that we can
    # safely cast

    # we copy as need to coerce here
    if mask.any():
        values = values.copy()
        values[mask] = 1
        values = safe_cast(values, dtype, copy=False)
    else:
        values = safe_cast(values, dtype, copy=False)

    return values, mask


class IntegerArray(NumericArray):
    """
    Array of integer (optional missing) values.

    .. versionchanged:: 1.0.0

       Now uses :attr:`pandas.NA` as the missing value rather
       than :attr:`numpy.nan`.

    .. warning::

       IntegerArray is currently experimental, and its API or internal
       implementation may change without warning.

    We represent an IntegerArray with 2 numpy arrays:

    - data: contains a numpy integer array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an IntegerArray from generic array-like input, use
    :func:`pandas.array` with one of the integer dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d integer-dtype array.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values.
    copy : bool, default False
        Whether to copy the `values` and `mask`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    IntegerArray

    Examples
    --------
    Create an IntegerArray with :func:`pandas.array`.

    >>> int_array = pd.array([1, None, 3], dtype=pd.Int32Dtype())
    >>> int_array
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([1, None, 3], dtype='Int32')
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32

    >>> pd.array([1, None, 3], dtype='UInt16')
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: UInt16
    """

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = 1

    @cache_readonly
    def dtype(self) -> _IntegerDtype:
        return INT_STR_TO_DTYPE[str(self._data.dtype)]

    def __init__(self, values: np.ndarray, mask: np.ndarray, copy: bool = False):
        if not (isinstance(values, np.ndarray) and values.dtype.kind in ["i", "u"]):
            raise TypeError(
                "values should be integer numpy array. Use "
                "the 'pd.array' function instead"
            )
        super().__init__(values, mask, copy=copy)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> IntegerArray:
        values, mask = coerce_to_array(scalars, dtype=dtype, copy=copy)
        return IntegerArray(values, mask)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ) -> IntegerArray:
        scalars = to_numeric(strings, errors="raise")
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    def _coerce_to_array(self, value) -> tuple[np.ndarray, np.ndarray]:
        return coerce_to_array(value, dtype=self.dtype)

    def astype(self, dtype, copy: bool = True) -> ArrayLike:
        """
        Cast to a NumPy array or ExtensionArray with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        ndarray or ExtensionArray
            NumPy ndarray, BooleanArray or IntegerArray with 'dtype' for its dtype.

        Raises
        ------
        TypeError
            if incompatible type with an IntegerDtype, equivalent of same_kind
            casting
        """
        dtype = pandas_dtype(dtype)

        if isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)

        na_value: float | np.datetime64 | lib.NoDefault

        # coerce
        if is_float_dtype(dtype):
            # In astype, we consider dtype=float to also mean na_value=np.nan
            na_value = np.nan
        elif is_datetime64_dtype(dtype):
            na_value = np.datetime64("NaT")
        else:
            na_value = lib.no_default

        return self.to_numpy(dtype=dtype, na_value=na_value, copy=False)

    def _values_for_argsort(self) -> np.ndarray:
        """
        Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.
        """
        data = self._data.copy()
        if self._mask.any():
            data[self._mask] = data.min() - 1
        return data

    def _cmp_method(self, other, op):
        from pandas.core.arrays import BooleanArray

        mask = None

        if isinstance(other, BaseMaskedArray):
            other, mask = other._data, other._mask

        elif is_list_like(other):
            other = np.asarray(other)
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")

        if other is libmissing.NA:
            # numpy does not handle pd.NA well as "other" scalar (it returns
            # a scalar False instead of an array)
            # This may be fixed by NA.__array_ufunc__. Revisit this check
            # once that's implemented.
            result = np.zeros(self._data.shape, dtype="bool")
            mask = np.ones(self._data.shape, dtype="bool")
        else:
            with warnings.catch_warnings():
                # numpy may show a FutureWarning:
                #     elementwise comparison failed; returning scalar instead,
                #     but in the future will perform elementwise comparison
                # before returning NotImplemented. We fall back to the correct
                # behavior today, so that should be fine to ignore.
                warnings.filterwarnings("ignore", "elementwise", FutureWarning)
                with np.errstate(all="ignore"):
                    method = getattr(self._data, f"__{op.__name__}__")
                    result = method(other)

                if result is NotImplemented:
                    result = invalid_comparison(self._data, other, op)

        # nans propagate
        if mask is None:
            mask = self._mask.copy()
        else:
            mask = self._mask | mask

        return BooleanArray(result, mask)

    def sum(self, *, skipna=True, min_count=0, **kwargs):
        nv.validate_sum((), kwargs)
        return super()._reduce("sum", skipna=skipna, min_count=min_count)

    def prod(self, *, skipna=True, min_count=0, **kwargs):
        nv.validate_prod((), kwargs)
        return super()._reduce("prod", skipna=skipna, min_count=min_count)

    def min(self, *, skipna=True, **kwargs):
        nv.validate_min((), kwargs)
        return super()._reduce("min", skipna=skipna)

    def max(self, *, skipna=True, **kwargs):
        nv.validate_max((), kwargs)
        return super()._reduce("max", skipna=skipna)

    def _maybe_mask_result(self, result, mask, other, op_name: str):
        """
        Parameters
        ----------
        result : array-like
        mask : array-like bool
        other : scalar or array-like
        op_name : str
        """
        # if we have a float operand we are by-definition
        # a float result
        # or our op is a divide
        if (is_float_dtype(other) or is_float(other)) or (
            op_name in ["rtruediv", "truediv"]
        ):
            from pandas.core.arrays import FloatingArray

            return FloatingArray(result, mask, copy=False)

        if result.dtype == "timedelta64[ns]":
            from pandas.core.arrays import TimedeltaArray

            result[mask] = iNaT
            return TimedeltaArray._simple_new(result)

        return type(self)(result, mask, copy=False)


_dtype_docstring = """
An ExtensionDtype for {dtype} integer data.

.. versionchanged:: 1.0.0

   Now uses :attr:`pandas.NA` as its missing value,
   rather than :attr:`numpy.nan`.

Attributes
----------
None

Methods
-------
None
"""

# create the Dtype


@register_extension_dtype
class Int8Dtype(_IntegerDtype):
    type = np.int8
    name = "Int8"
    __doc__ = _dtype_docstring.format(dtype="int8")


@register_extension_dtype
class Int16Dtype(_IntegerDtype):
    type = np.int16
    name = "Int16"
    __doc__ = _dtype_docstring.format(dtype="int16")


@register_extension_dtype
class Int32Dtype(_IntegerDtype):
    type = np.int32
    name = "Int32"
    __doc__ = _dtype_docstring.format(dtype="int32")


@register_extension_dtype
class Int64Dtype(_IntegerDtype):
    type = np.int64
    name = "Int64"
    __doc__ = _dtype_docstring.format(dtype="int64")


@register_extension_dtype
class UInt8Dtype(_IntegerDtype):
    type = np.uint8
    name = "UInt8"
    __doc__ = _dtype_docstring.format(dtype="uint8")


@register_extension_dtype
class UInt16Dtype(_IntegerDtype):
    type = np.uint16
    name = "UInt16"
    __doc__ = _dtype_docstring.format(dtype="uint16")


@register_extension_dtype
class UInt32Dtype(_IntegerDtype):
    type = np.uint32
    name = "UInt32"
    __doc__ = _dtype_docstring.format(dtype="uint32")


@register_extension_dtype
class UInt64Dtype(_IntegerDtype):
    type = np.uint64
    name = "UInt64"
    __doc__ = _dtype_docstring.format(dtype="uint64")


INT_STR_TO_DTYPE: dict[str, _IntegerDtype] = {
    "int8": Int8Dtype(),
    "int16": Int16Dtype(),
    "int32": Int32Dtype(),
    "int64": Int64Dtype(),
    "uint8": UInt8Dtype(),
    "uint16": UInt16Dtype(),
    "uint32": UInt32Dtype(),
    "uint64": UInt64Dtype(),
}
