"""
EA-compatible analogue to to np.putmask
"""
from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from pandas._libs import lib
from pandas._typing import ArrayLike

from pandas.core.dtypes.cast import (
    convert_scalar_for_putitemlike,
    find_common_type,
    infer_dtype_from,
)
from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
)
from pandas.core.dtypes.missing import isna_compat

from pandas.core.arrays import ExtensionArray


def putmask_inplace(values: ArrayLike, mask: np.ndarray, value: Any) -> None:
    """
    ExtensionArray-compatible implementation of np.putmask.  The main
    difference is we do not handle repeating or truncating like numpy.

    Parameters
    ----------
    mask : np.ndarray[bool]
        We assume extract_bool_array has already been called.
    value : Any
    """

    if lib.is_scalar(value) and isinstance(values, np.ndarray):
        value = convert_scalar_for_putitemlike(value, values.dtype)

    if (
        not isinstance(values, np.ndarray)
        or (values.dtype == object and not lib.is_scalar(value))
        # GH#43424: np.putmask raises TypeError if we cannot cast between types with
        # rule = "safe", a stricter guarantee we may not have here
        or (
            isinstance(value, np.ndarray) and not np.can_cast(value.dtype, values.dtype)
        )
    ):
        # GH#19266 using np.putmask gives unexpected results with listlike value
        if is_list_like(value) and len(value) == len(values):
            values[mask] = value[mask]
        else:
            values[mask] = value
    else:
        # GH#37833 np.putmask is more performant than __setitem__
        np.putmask(values, mask, value)


def putmask_smart(values: np.ndarray, mask: np.ndarray, new) -> np.ndarray:
    """
    Return a new ndarray, try to preserve dtype if possible.

    Parameters
    ----------
    values : np.ndarray
        `values`, updated in-place.
    mask : np.ndarray[bool]
        Applies to both sides (array like).
    new : `new values` either scalar or an array like aligned with `values`

    Returns
    -------
    values : ndarray with updated values
        this *may* be a copy of the original

    See Also
    --------
    ndarray.putmask
    """
    # we cannot use np.asarray() here as we cannot have conversions
    # that numpy does when numeric are mixed with strings

    # n should be the length of the mask or a scalar here
    if not is_list_like(new):
        new = np.broadcast_to(new, mask.shape)

    # see if we are only masking values that if putted
    # will work in the current dtype
    try:
        nn = new[mask]
    except TypeError:
        # TypeError: only integer scalar arrays can be converted to a scalar index
        pass
    else:
        # make sure that we have a nullable type if we have nulls
        if not isna_compat(values, nn[0]):
            pass
        elif not (is_float_dtype(nn.dtype) or is_integer_dtype(nn.dtype)):
            # only compare integers/floats
            pass
        elif not (is_float_dtype(values.dtype) or is_integer_dtype(values.dtype)):
            # only compare integers/floats
            pass
        else:

            # we ignore ComplexWarning here
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore", np.ComplexWarning)
                nn_at = nn.astype(values.dtype)

            comp = nn == nn_at
            if is_list_like(comp) and comp.all():
                nv = values.copy()
                nv[mask] = nn_at
                return nv

    new = np.asarray(new)

    if values.dtype.kind == new.dtype.kind:
        # preserves dtype if possible
        return _putmask_preserve(values, new, mask)

    dtype = find_common_type([values.dtype, new.dtype])
    # error: Argument 1 to "astype" of "_ArrayOrScalarCommon" has incompatible type
    # "Union[dtype[Any], ExtensionDtype]"; expected "Union[dtype[Any], None, type,
    # _SupportsDType, str, Union[Tuple[Any, int], Tuple[Any, Union[int, Sequence[int]]],
    # List[Any], _DTypeDict, Tuple[Any, Any]]]"
    values = values.astype(dtype)  # type: ignore[arg-type]

    return _putmask_preserve(values, new, mask)


def _putmask_preserve(new_values: np.ndarray, new, mask: np.ndarray):
    try:
        new_values[mask] = new[mask]
    except (IndexError, ValueError):
        new_values[mask] = new
    return new_values


def putmask_without_repeat(values: np.ndarray, mask: np.ndarray, new: Any) -> None:
    """
    np.putmask will truncate or repeat if `new` is a listlike with
    len(new) != len(values).  We require an exact match.

    Parameters
    ----------
    values : np.ndarray
    mask : np.ndarray[bool]
    new : Any
    """
    if getattr(new, "ndim", 0) >= 1:
        new = new.astype(values.dtype, copy=False)

    # TODO: this prob needs some better checking for 2D cases
    nlocs = mask.sum()
    if nlocs > 0 and is_list_like(new) and getattr(new, "ndim", 1) == 1:
        if nlocs == len(new):
            # GH#30567
            # If length of ``new`` is less than the length of ``values``,
            # `np.putmask` would first repeat the ``new`` array and then
            # assign the masked values hence produces incorrect result.
            # `np.place` on the other hand uses the ``new`` values at it is
            # to place in the masked locations of ``values``
            np.place(values, mask, new)
            # i.e. values[mask] = new
        elif mask.shape[-1] == len(new) or len(new) == 1:
            np.putmask(values, mask, new)
        else:
            raise ValueError("cannot assign mismatch length to masked array")
    else:
        np.putmask(values, mask, new)


def validate_putmask(values: ArrayLike, mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Validate mask and check if this putmask operation is a no-op.
    """
    mask = extract_bool_array(mask)
    if mask.shape != values.shape:
        raise ValueError("putmask: mask and data must be the same size")

    noop = not mask.any()
    return mask, noop


def extract_bool_array(mask: ArrayLike) -> np.ndarray:
    """
    If we have a SparseArray or BooleanArray, convert it to ndarray[bool].
    """
    if isinstance(mask, ExtensionArray):
        # We could have BooleanArray, Sparse[bool], ...
        #  Except for BooleanArray, this is equivalent to just
        #  np.asarray(mask, dtype=bool)
        mask = mask.to_numpy(dtype=bool, na_value=False)

    mask = np.asarray(mask, dtype=bool)
    return mask


def setitem_datetimelike_compat(values: np.ndarray, num_set: int, other):
    """
    Parameters
    ----------
    values : np.ndarray
    num_set : int
        For putmask, this is mask.sum()
    other : Any
    """
    if values.dtype == object:
        dtype, _ = infer_dtype_from(other, pandas_dtype=True)

        if isinstance(dtype, np.dtype) and dtype.kind in ["m", "M"]:
            # https://github.com/numpy/numpy/issues/12550
            #  timedelta64 will incorrectly cast to int
            if not is_list_like(other):
                other = [other] * num_set
            else:
                other = list(other)

    return other
