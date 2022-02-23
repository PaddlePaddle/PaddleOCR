"""
Methods that can be shared by many array-like classes or subclasses:
    Series
    Index
    ExtensionArray
"""
import operator
from typing import Any
import warnings

import numpy as np

from pandas._libs import lib

from pandas.core.construction import extract_array
from pandas.core.ops import (
    maybe_dispatch_ufunc_to_dunder_op,
    roperator,
)
from pandas.core.ops.common import unpack_zerodim_and_defer


class OpsMixin:
    # -------------------------------------------------------------
    # Comparisons

    def _cmp_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    # -------------------------------------------------------------
    # Logical Methods

    def _logical_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__and__")
    def __and__(self, other):
        return self._logical_method(other, operator.and_)

    @unpack_zerodim_and_defer("__rand__")
    def __rand__(self, other):
        return self._logical_method(other, roperator.rand_)

    @unpack_zerodim_and_defer("__or__")
    def __or__(self, other):
        return self._logical_method(other, operator.or_)

    @unpack_zerodim_and_defer("__ror__")
    def __ror__(self, other):
        return self._logical_method(other, roperator.ror_)

    @unpack_zerodim_and_defer("__xor__")
    def __xor__(self, other):
        return self._logical_method(other, operator.xor)

    @unpack_zerodim_and_defer("__rxor__")
    def __rxor__(self, other):
        return self._logical_method(other, roperator.rxor)

    # -------------------------------------------------------------
    # Arithmetic Methods

    def _arith_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other):
        return self._arith_method(other, operator.add)

    @unpack_zerodim_and_defer("__radd__")
    def __radd__(self, other):
        return self._arith_method(other, roperator.radd)

    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):
        return self._arith_method(other, operator.sub)

    @unpack_zerodim_and_defer("__rsub__")
    def __rsub__(self, other):
        return self._arith_method(other, roperator.rsub)

    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other):
        return self._arith_method(other, operator.mul)

    @unpack_zerodim_and_defer("__rmul__")
    def __rmul__(self, other):
        return self._arith_method(other, roperator.rmul)

    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        return self._arith_method(other, operator.truediv)

    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other):
        return self._arith_method(other, roperator.rtruediv)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        return self._arith_method(other, operator.floordiv)

    @unpack_zerodim_and_defer("__rfloordiv")
    def __rfloordiv__(self, other):
        return self._arith_method(other, roperator.rfloordiv)

    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        return self._arith_method(other, operator.mod)

    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other):
        return self._arith_method(other, roperator.rmod)

    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other):
        return self._arith_method(other, divmod)

    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other):
        return self._arith_method(other, roperator.rdivmod)

    @unpack_zerodim_and_defer("__pow__")
    def __pow__(self, other):
        return self._arith_method(other, operator.pow)

    @unpack_zerodim_and_defer("__rpow__")
    def __rpow__(self, other):
        return self._arith_method(other, roperator.rpow)


# -----------------------------------------------------------------------------
# Helpers to implement __array_ufunc__


def _is_aligned(frame, other):
    """
    Helper to check if a DataFrame is aligned with another DataFrame or Series.
    """
    from pandas import DataFrame

    if isinstance(other, DataFrame):
        return frame._indexed_same(other)
    else:
        # Series -> match index
        return frame.columns.equals(other.index)


def _maybe_fallback(ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
    """
    In the future DataFrame, inputs to ufuncs will be aligned before applying
    the ufunc, but for now we ignore the index but raise a warning if behaviour
    would change in the future.
    This helper detects the case where a warning is needed and then fallbacks
    to applying the ufunc on arrays to avoid alignment.

    See https://github.com/pandas-dev/pandas/pull/39239
    """
    from pandas import DataFrame
    from pandas.core.generic import NDFrame

    n_alignable = sum(isinstance(x, NDFrame) for x in inputs)
    n_frames = sum(isinstance(x, DataFrame) for x in inputs)

    if n_alignable >= 2 and n_frames >= 1:
        # if there are 2 alignable inputs (Series or DataFrame), of which at least 1
        # is a DataFrame -> we would have had no alignment before -> warn that this
        # will align in the future

        # the first frame is what determines the output index/columns in pandas < 1.2
        first_frame = next(x for x in inputs if isinstance(x, DataFrame))

        # check if the objects are aligned or not
        non_aligned = sum(
            not _is_aligned(first_frame, x) for x in inputs if isinstance(x, NDFrame)
        )

        # if at least one is not aligned -> warn and fallback to array behaviour
        if non_aligned:
            warnings.warn(
                "Calling a ufunc on non-aligned DataFrames (or DataFrame/Series "
                "combination). Currently, the indices are ignored and the result "
                "takes the index/columns of the first DataFrame. In the future , "
                "the DataFrames/Series will be aligned before applying the ufunc.\n"
                "Convert one of the arguments to a NumPy array "
                "(eg 'ufunc(df1, np.asarray(df2)') to keep the current behaviour, "
                "or align manually (eg 'df1, df2 = df1.align(df2)') before passing to "
                "the ufunc to obtain the future behaviour and silence this warning.",
                FutureWarning,
                stacklevel=4,
            )

            # keep the first dataframe of the inputs, other DataFrame/Series is
            # converted to array for fallback behaviour
            new_inputs = []
            for x in inputs:
                if x is first_frame:
                    new_inputs.append(x)
                elif isinstance(x, NDFrame):
                    new_inputs.append(np.asarray(x))
                else:
                    new_inputs.append(x)

            # call the ufunc on those transformed inputs
            return getattr(ufunc, method)(*new_inputs, **kwargs)

    # signal that we didn't fallback / execute the ufunc yet
    return NotImplemented


def array_ufunc(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
    """
    Compatibility with numpy ufuncs.

    See also
    --------
    numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    """
    from pandas.core.generic import NDFrame
    from pandas.core.internals import BlockManager

    cls = type(self)

    # for backwards compatibility check and potentially fallback for non-aligned frames
    result = _maybe_fallback(ufunc, method, *inputs, **kwargs)
    if result is not NotImplemented:
        return result

    # for binary ops, use our custom dunder methods
    result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
    if result is not NotImplemented:
        return result

    # Determine if we should defer.

    # error: "Type[ndarray]" has no attribute "__array_ufunc__"
    no_defer = (
        np.ndarray.__array_ufunc__,  # type: ignore[attr-defined]
        cls.__array_ufunc__,
    )

    for item in inputs:
        higher_priority = (
            hasattr(item, "__array_priority__")
            and item.__array_priority__ > self.__array_priority__
        )
        has_array_ufunc = (
            hasattr(item, "__array_ufunc__")
            and type(item).__array_ufunc__ not in no_defer
            and not isinstance(item, self._HANDLED_TYPES)
        )
        if higher_priority or has_array_ufunc:
            return NotImplemented

    # align all the inputs.
    types = tuple(type(x) for x in inputs)
    alignable = [x for x, t in zip(inputs, types) if issubclass(t, NDFrame)]

    if len(alignable) > 1:
        # This triggers alignment.
        # At the moment, there aren't any ufuncs with more than two inputs
        # so this ends up just being x1.index | x2.index, but we write
        # it to handle *args.

        if len(set(types)) > 1:
            # We currently don't handle ufunc(DataFrame, Series)
            # well. Previously this raised an internal ValueError. We might
            # support it someday, so raise a NotImplementedError.
            raise NotImplementedError(
                "Cannot apply ufunc {} to mixed DataFrame and Series "
                "inputs.".format(ufunc)
            )
        axes = self.axes
        for obj in alignable[1:]:
            # this relies on the fact that we aren't handling mixed
            # series / frame ufuncs.
            for i, (ax1, ax2) in enumerate(zip(axes, obj.axes)):
                axes[i] = ax1.union(ax2)

        reconstruct_axes = dict(zip(self._AXIS_ORDERS, axes))
        inputs = tuple(
            x.reindex(**reconstruct_axes) if issubclass(t, NDFrame) else x
            for x, t in zip(inputs, types)
        )
    else:
        reconstruct_axes = dict(zip(self._AXIS_ORDERS, self.axes))

    if self.ndim == 1:
        names = [getattr(x, "name") for x in inputs if hasattr(x, "name")]
        name = names[0] if len(set(names)) == 1 else None
        reconstruct_kwargs = {"name": name}
    else:
        reconstruct_kwargs = {}

    def reconstruct(result):
        if lib.is_scalar(result):
            return result
        if result.ndim != self.ndim:
            if method == "outer":
                if self.ndim == 2:
                    # we already deprecated for Series
                    msg = (
                        "outer method for ufunc {} is not implemented on "
                        "pandas objects. Returning an ndarray, but in the "
                        "future this will raise a 'NotImplementedError'. "
                        "Consider explicitly converting the DataFrame "
                        "to an array with '.to_numpy()' first."
                    )
                    warnings.warn(msg.format(ufunc), FutureWarning, stacklevel=4)
                    return result
                raise NotImplementedError
            return result
        if isinstance(result, BlockManager):
            # we went through BlockManager.apply
            result = self._constructor(result, **reconstruct_kwargs, copy=False)
        else:
            # we converted an array, lost our axes
            result = self._constructor(
                result, **reconstruct_axes, **reconstruct_kwargs, copy=False
            )
        # TODO: When we support multiple values in __finalize__, this
        # should pass alignable to `__finalize__` instead of self.
        # Then `np.add(a, b)` would consider attrs from both a and b
        # when a and b are NDFrames.
        if len(alignable) == 1:
            result = result.__finalize__(self)
        return result

    if self.ndim > 1 and (len(inputs) > 1 or ufunc.nout > 1):
        # Just give up on preserving types in the complex case.
        # In theory we could preserve them for them.
        # * nout>1 is doable if BlockManager.apply took nout and
        #   returned a Tuple[BlockManager].
        # * len(inputs) > 1 is doable when we know that we have
        #   aligned blocks / dtypes.
        inputs = tuple(np.asarray(x) for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
    elif self.ndim == 1:
        # ufunc(series, ...)
        inputs = tuple(extract_array(x, extract_numpy=True) for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
    else:
        # ufunc(dataframe)
        if method == "__call__" and not kwargs:
            # for np.<ufunc>(..) calls
            # kwargs cannot necessarily be handled block-by-block, so only
            # take this path if there are no kwargs
            mgr = inputs[0]._mgr
            result = mgr.apply(getattr(ufunc, method))
        else:
            # otherwise specific ufunc methods (eg np.<ufunc>.accumulate(..))
            # Those can have an axis keyword and thus can't be called block-by-block
            result = getattr(ufunc, method)(np.asarray(inputs[0]), **kwargs)

    if ufunc.nout > 1:
        result = tuple(reconstruct(x) for x in result)
    else:
        result = reconstruct(result)
    return result
