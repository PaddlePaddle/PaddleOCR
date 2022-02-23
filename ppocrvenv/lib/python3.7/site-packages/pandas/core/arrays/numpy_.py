from __future__ import annotations

import numbers

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from pandas._libs import lib
from pandas._typing import (
    Dtype,
    NpDtype,
    Scalar,
)
from pandas.compat.numpy import function as nv

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.dtypes import PandasDtype
from pandas.core.dtypes.missing import isna

from pandas.core import (
    nanops,
    ops,
)
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.strings.object_array import ObjectStringArrayMixin


class PandasArray(
    OpsMixin,
    NDArrayBackedExtensionArray,
    NDArrayOperatorsMixin,
    ObjectStringArrayMixin,
):
    """
    A pandas ExtensionArray for NumPy data.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    values : ndarray
        The NumPy ndarray to wrap. Must be 1-dimensional.
    copy : bool, default False
        Whether to copy `values`.

    Attributes
    ----------
    None

    Methods
    -------
    None
    """

    # If you're wondering why pd.Series(cls) doesn't put the array in an
    # ExtensionBlock, search for `ABCPandasArray`. We check for
    # that _typ to ensure that users don't unnecessarily use EAs inside
    # pandas internals, which turns off things like block consolidation.
    _typ = "npy_extension"
    __array_priority__ = 1000
    _ndarray: np.ndarray
    _dtype: PandasDtype

    # ------------------------------------------------------------------------
    # Constructors

    def __init__(self, values: np.ndarray | PandasArray, copy: bool = False):
        if isinstance(values, type(self)):
            values = values._ndarray
        if not isinstance(values, np.ndarray):
            raise ValueError(
                f"'values' must be a NumPy array, not {type(values).__name__}"
            )

        if values.ndim == 0:
            # Technically we support 2, but do not advertise that fact.
            raise ValueError("PandasArray must be 1-dimensional.")

        if copy:
            values = values.copy()

        dtype = PandasDtype(values.dtype)
        super().__init__(values, dtype)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> PandasArray:
        if isinstance(dtype, PandasDtype):
            dtype = dtype._dtype

        # error: Argument "dtype" to "asarray" has incompatible type
        # "Union[ExtensionDtype, str, dtype[Any], dtype[floating[_64Bit]], Type[object],
        # None]"; expected "Union[dtype[Any], None, type, _SupportsDType, str,
        # Union[Tuple[Any, int], Tuple[Any, Union[int, Sequence[int]]], List[Any],
        # _DTypeDict, Tuple[Any, Any]]]"
        result = np.asarray(scalars, dtype=dtype)  # type: ignore[arg-type]
        if (
            result.ndim > 1
            and not hasattr(scalars, "dtype")
            and (dtype is None or dtype == object)
        ):
            # e.g. list-of-tuples
            result = construct_1d_object_array_from_listlike(scalars)

        if copy and result is scalars:
            result = result.copy()
        return cls(result)

    @classmethod
    def _from_factorized(cls, values, original) -> PandasArray:
        return cls(values)

    def _from_backing_data(self, arr: np.ndarray) -> PandasArray:
        return type(self)(arr)

    # ------------------------------------------------------------------------
    # Data

    @property
    def dtype(self) -> PandasDtype:
        return self._dtype

    # ------------------------------------------------------------------------
    # NumPy Array Interface

    def __array__(self, dtype: NpDtype | None = None) -> np.ndarray:
        return np.asarray(self._ndarray, dtype=dtype)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # Lightly modified version of
        # https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
        # The primary modification is not boxing scalar return values
        # in PandasArray, since pandas' ExtensionArrays are 1-d.
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use PandasArray instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle PandasArray objects.
            if not isinstance(x, self._HANDLED_TYPES + (PandasArray,)):
                return NotImplemented

        if ufunc not in [np.logical_or, np.bitwise_or, np.bitwise_xor]:
            # For binary ops, use our custom dunder methods
            # We haven't implemented logical dunder funcs, so exclude these
            #  to avoid RecursionError
            result = ops.maybe_dispatch_ufunc_to_dunder_op(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x._ndarray if isinstance(x, PandasArray) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(
                x._ndarray if isinstance(x, PandasArray) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple and len(result):
            # multiple return values
            if not lib.is_scalar(result[0]):
                # re-box array-like results
                return tuple(type(self)(x) for x in result)
            else:
                # but not scalar reductions
                return result
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            if not lib.is_scalar(result):
                # re-box array-like results, but not scalar reductions
                result = type(self)(result)
            return result

    # ------------------------------------------------------------------------
    # Pandas ExtensionArray Interface

    def isna(self) -> np.ndarray:
        return isna(self._ndarray)

    def _validate_scalar(self, fill_value):
        if fill_value is None:
            # Primarily for subclasses
            fill_value = self.dtype.na_value
        return fill_value

    def _values_for_factorize(self) -> tuple[np.ndarray, int]:
        return self._ndarray, -1

    # ------------------------------------------------------------------------
    # Reductions

    def any(
        self,
        *,
        axis: int | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_any((), {"out": out, "keepdims": keepdims})
        result = nanops.nanany(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def all(
        self,
        *,
        axis: int | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_all((), {"out": out, "keepdims": keepdims})
        result = nanops.nanall(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def min(self, *, axis: int | None = None, skipna: bool = True, **kwargs) -> Scalar:
        nv.validate_min((), kwargs)
        result = nanops.nanmin(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def max(self, *, axis: int | None = None, skipna: bool = True, **kwargs) -> Scalar:
        nv.validate_max((), kwargs)
        result = nanops.nanmax(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def sum(
        self, *, axis: int | None = None, skipna: bool = True, min_count=0, **kwargs
    ) -> Scalar:
        nv.validate_sum((), kwargs)
        result = nanops.nansum(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def prod(
        self, *, axis: int | None = None, skipna: bool = True, min_count=0, **kwargs
    ) -> Scalar:
        nv.validate_prod((), kwargs)
        result = nanops.nanprod(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def mean(
        self,
        *,
        axis: int | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_mean((), {"dtype": dtype, "out": out, "keepdims": keepdims})
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def median(
        self,
        *,
        axis: int | None = None,
        out=None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_median(
            (), {"out": out, "overwrite_input": overwrite_input, "keepdims": keepdims}
        )
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def std(
        self,
        *,
        axis: int | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof=1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="std"
        )
        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def var(
        self,
        *,
        axis: int | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof=1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="var"
        )
        result = nanops.nanvar(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def sem(
        self,
        *,
        axis: int | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof=1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="sem"
        )
        result = nanops.nansem(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def kurt(
        self,
        *,
        axis: int | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="kurt"
        )
        result = nanops.nankurt(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def skew(
        self,
        *,
        axis: int | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="skew"
        )
        result = nanops.nanskew(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    # ------------------------------------------------------------------------
    # Additional Methods

    # error: Argument 1 of "to_numpy" is incompatible with supertype "ExtensionArray";
    # supertype defines the argument type as "Union[ExtensionDtype, str, dtype[Any],
    # Type[str], Type[float], Type[int], Type[complex], Type[bool], Type[object], None]"
    def to_numpy(  # type: ignore[override]
        self,
        dtype: NpDtype | None = None,
        copy: bool = False,
        na_value=lib.no_default,
    ) -> np.ndarray:
        result = np.asarray(self._ndarray, dtype=dtype)

        if (copy or na_value is not lib.no_default) and result is self._ndarray:
            result = result.copy()

        if na_value is not lib.no_default:
            result[self.isna()] = na_value

        return result

    # ------------------------------------------------------------------------
    # Ops

    def __invert__(self) -> PandasArray:
        return type(self)(~self._ndarray)

    def _cmp_method(self, other, op):
        if isinstance(other, PandasArray):
            other = other._ndarray

        other = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        pd_op = ops.get_array_op(op)
        other = ensure_wrapped_if_datetimelike(other)
        with np.errstate(all="ignore"):
            result = pd_op(self._ndarray, other)

        if op is divmod or op is ops.rdivmod:
            a, b = result
            if isinstance(a, np.ndarray):
                # for e.g. op vs TimedeltaArray, we may already
                #  have an ExtensionArray, in which case we do not wrap
                return self._wrap_ndarray_result(a), self._wrap_ndarray_result(b)
            return a, b

        if isinstance(result, np.ndarray):
            # for e.g. multiplication vs TimedeltaArray, we may already
            #  have an ExtensionArray, in which case we do not wrap
            return self._wrap_ndarray_result(result)
        return result

    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray):
        # If we have timedelta64[ns] result, return a TimedeltaArray instead
        #  of a PandasArray
        if result.dtype == "timedelta64[ns]":
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._simple_new(result)
        return type(self)(result)

    # ------------------------------------------------------------------------
    # String methods interface
    _str_na_value = np.nan
