from __future__ import annotations

import datetime
import numbers
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import numpy as np

from pandas._libs import (
    Timedelta,
    missing as libmissing,
)
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError

from pandas.core.dtypes.common import (
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
)

from pandas.core import ops
from pandas.core.arrays.masked import (
    BaseMaskedArray,
    BaseMaskedDtype,
)

if TYPE_CHECKING:
    import pyarrow

T = TypeVar("T", bound="NumericArray")


class NumericDtype(BaseMaskedDtype):
    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseMaskedArray:
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays._arrow_utils import pyarrow_array_to_numpy_and_mask

        array_class = self.construct_array_type()

        pyarrow_type = pyarrow.from_numpy_dtype(self.type)
        if not array.type.equals(pyarrow_type):
            array = array.cast(pyarrow_type)

        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            # pyarrow.ChunkedArray
            chunks = array.chunks

        results = []
        for arr in chunks:
            data, mask = pyarrow_array_to_numpy_and_mask(arr, dtype=self.type)
            num_arr = array_class(data.copy(), ~mask, copy=False)
            results.append(num_arr)

        if not results:
            return array_class(
                np.array([], dtype=self.numpy_dtype), np.array([], dtype=np.bool_)
            )
        elif len(results) == 1:
            # avoid additional copy in _concat_same_type
            return results[0]
        else:
            return array_class._concat_same_type(results)


class NumericArray(BaseMaskedArray):
    """
    Base class for IntegerArray and FloatingArray.
    """

    def _maybe_mask_result(self, result, mask, other, op_name: str):
        raise AbstractMethodError(self)

    def _arith_method(self, other, op):
        op_name = op.__name__
        omask = None

        if getattr(other, "ndim", 0) > 1:
            raise NotImplementedError("can only perform ops with 1-d structures")

        if isinstance(other, NumericArray):
            other, omask = other._data, other._mask

        elif is_list_like(other):
            other = np.asarray(other)
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")
            if len(self) != len(other):
                raise ValueError("Lengths must match")
            if not (is_float_dtype(other) or is_integer_dtype(other)):
                raise TypeError("can only perform ops with numeric values")

        elif isinstance(other, (datetime.timedelta, np.timedelta64)):
            other = Timedelta(other)

        else:
            if not (is_float(other) or is_integer(other) or other is libmissing.NA):
                raise TypeError("can only perform ops with numeric values")

        if omask is None:
            mask = self._mask.copy()
            if other is libmissing.NA:
                mask |= True
        else:
            mask = self._mask | omask

        if op_name == "pow":
            # 1 ** x is 1.
            mask = np.where((self._data == 1) & ~self._mask, False, mask)
            # x ** 0 is 1.
            if omask is not None:
                mask = np.where((other == 0) & ~omask, False, mask)
            elif other is not libmissing.NA:
                mask = np.where(other == 0, False, mask)

        elif op_name == "rpow":
            # 1 ** x is 1.
            if omask is not None:
                mask = np.where((other == 1) & ~omask, False, mask)
            elif other is not libmissing.NA:
                mask = np.where(other == 1, False, mask)
            # x ** 0 is 1.
            mask = np.where((self._data == 0) & ~self._mask, False, mask)

        if other is libmissing.NA:
            result = np.ones_like(self._data)
        else:
            with np.errstate(all="ignore"):
                result = op(self._data, other)

        # divmod returns a tuple
        if op_name == "divmod":
            div, mod = result
            return (
                self._maybe_mask_result(div, mask, other, "floordiv"),
                self._maybe_mask_result(mod, mask, other, "mod"),
            )

        return self._maybe_mask_result(result, mask, other, op_name)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # For NumericArray inputs, we apply the ufunc to ._data
        # and mask the result.
        if method == "reduce":
            # Not clear how to handle missing values in reductions. Raise.
            raise NotImplementedError("The 'reduce' method is not supported.")
        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (NumericArray,)):
                return NotImplemented

        # for binary ops, use our custom dunder methods
        result = ops.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        mask = np.zeros(len(self), dtype=bool)
        inputs2: list[Any] = []
        for x in inputs:
            if isinstance(x, NumericArray):
                mask |= x._mask
                inputs2.append(x._data)
            else:
                inputs2.append(x)

        def reconstruct(x):
            # we don't worry about scalar `x` here, since we
            # raise for reduce up above.

            if is_integer_dtype(x.dtype):
                from pandas.core.arrays import IntegerArray

                m = mask.copy()
                return IntegerArray(x, m)
            elif is_float_dtype(x.dtype):
                from pandas.core.arrays import FloatingArray

                m = mask.copy()
                return FloatingArray(x, m)
            else:
                x[mask] = np.nan
            return x

        result = getattr(ufunc, method)(*inputs2, **kwargs)
        if isinstance(result, tuple):
            return tuple(reconstruct(x) for x in result)
        else:
            return reconstruct(result)

    def __neg__(self):
        return type(self)(-self._data, self._mask.copy())

    def __pos__(self):
        return self

    def __abs__(self):
        return type(self)(abs(self._data), self._mask.copy())

    def round(self: T, decimals: int = 0, *args, **kwargs) -> T:
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        NumericArray
            Rounded values of the NumericArray.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
        nv.validate_round(args, kwargs)
        values = np.round(self._data, decimals=decimals, **kwargs)
        return type(self)(values, self._mask.copy())
