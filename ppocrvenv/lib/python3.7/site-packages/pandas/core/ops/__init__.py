"""
Arithmetic operations for PandasObjects

This is not a public API.
"""
from __future__ import annotations

import operator
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op  # noqa:F401
from pandas._typing import Level
from pandas.util._decorators import Appender

from pandas.core.dtypes.common import (
    is_array_like,
    is_list_like,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core import (
    algorithms,
    roperator,
)
from pandas.core.ops.array_ops import (  # noqa:F401
    arithmetic_op,
    comp_method_OBJECT_ARRAY,
    comparison_op,
    get_array_op,
    logical_op,
    maybe_prepare_scalar_for_op,
)
from pandas.core.ops.common import (  # noqa:F401
    get_op_result_name,
    unpack_zerodim_and_defer,
)
from pandas.core.ops.docstrings import (
    _flex_comp_doc_FRAME,
    _op_descriptions,
    make_flex_doc,
)
from pandas.core.ops.invalid import invalid_comparison  # noqa:F401
from pandas.core.ops.mask_ops import (  # noqa: F401
    kleene_and,
    kleene_or,
    kleene_xor,
)
from pandas.core.ops.methods import add_flex_arithmetic_methods  # noqa:F401
from pandas.core.roperator import (  # noqa:F401
    radd,
    rand_,
    rdiv,
    rdivmod,
    rfloordiv,
    rmod,
    rmul,
    ror_,
    rpow,
    rsub,
    rtruediv,
    rxor,
)

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        Series,
    )

# -----------------------------------------------------------------------------
# constants
ARITHMETIC_BINOPS: set[str] = {
    "add",
    "sub",
    "mul",
    "pow",
    "mod",
    "floordiv",
    "truediv",
    "divmod",
    "radd",
    "rsub",
    "rmul",
    "rpow",
    "rmod",
    "rfloordiv",
    "rtruediv",
    "rdivmod",
}


COMPARISON_BINOPS: set[str] = {"eq", "ne", "lt", "gt", "le", "ge"}


# -----------------------------------------------------------------------------
# Masking NA values and fallbacks for operations numpy does not support


def fill_binop(left, right, fill_value):
    """
    If a non-None fill_value is given, replace null entries in left and right
    with this value, but only in positions where _one_ of left/right is null,
    not both.

    Parameters
    ----------
    left : array-like
    right : array-like
    fill_value : object

    Returns
    -------
    left : array-like
    right : array-like

    Notes
    -----
    Makes copies if fill_value is not None and NAs are present.
    """
    if fill_value is not None:
        left_mask = isna(left)
        right_mask = isna(right)

        # one but not both
        mask = left_mask ^ right_mask

        if left_mask.any():
            # Avoid making a copy if we can
            left = left.copy()
            left[left_mask & mask] = fill_value

        if right_mask.any():
            # Avoid making a copy if we can
            right = right.copy()
            right[right_mask & mask] = fill_value

    return left, right


# -----------------------------------------------------------------------------
# Series


def align_method_SERIES(left: Series, right, align_asobject: bool = False):
    """align lhs and rhs Series"""
    # ToDo: Different from align_method_FRAME, list, tuple and ndarray
    # are not coerced here
    # because Series has inconsistencies described in #13637

    if isinstance(right, ABCSeries):
        # avoid repeated alignment
        if not left.index.equals(right.index):

            if align_asobject:
                # to keep original value's dtype for bool ops
                left = left.astype(object)
                right = right.astype(object)

            left, right = left.align(right, copy=False)

    return left, right


def flex_method_SERIES(op):
    name = op.__name__.strip("_")
    doc = make_flex_doc(name, "series")

    @Appender(doc)
    def flex_wrapper(self, other, level=None, fill_value=None, axis=0):
        # validate axis
        if axis is not None:
            self._get_axis_number(axis)

        res_name = get_op_result_name(self, other)

        if isinstance(other, ABCSeries):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError("Lengths must be equal")
            other = self._constructor(other, self.index)
            result = self._binop(other, op, level=level, fill_value=fill_value)
            result.name = res_name
            return result
        else:
            if fill_value is not None:
                self = self.fillna(fill_value)

            return op(self, other)

    flex_wrapper.__name__ = name
    return flex_wrapper


# -----------------------------------------------------------------------------
# DataFrame


def align_method_FRAME(
    left, right, axis, flex: bool | None = False, level: Level = None
):
    """
    Convert rhs to meet lhs dims if input is list, tuple or np.ndarray.

    Parameters
    ----------
    left : DataFrame
    right : Any
    axis : int, str, or None
    flex : bool or None, default False
        Whether this is a flex op, in which case we reindex.
        None indicates not to check for alignment.
    level : int or level name, default None

    Returns
    -------
    left : DataFrame
    right : Any
    """

    def to_series(right):
        msg = "Unable to coerce to Series, length must be {req_len}: given {given_len}"
        if axis is not None and left._get_axis_name(axis) == "index":
            if len(left.index) != len(right):
                raise ValueError(
                    msg.format(req_len=len(left.index), given_len=len(right))
                )
            right = left._constructor_sliced(right, index=left.index)
        else:
            if len(left.columns) != len(right):
                raise ValueError(
                    msg.format(req_len=len(left.columns), given_len=len(right))
                )
            right = left._constructor_sliced(right, index=left.columns)
        return right

    if isinstance(right, np.ndarray):

        if right.ndim == 1:
            right = to_series(right)

        elif right.ndim == 2:
            if right.shape == left.shape:
                right = left._constructor(right, index=left.index, columns=left.columns)

            elif right.shape[0] == left.shape[0] and right.shape[1] == 1:
                # Broadcast across columns
                right = np.broadcast_to(right, left.shape)
                right = left._constructor(right, index=left.index, columns=left.columns)

            elif right.shape[1] == left.shape[1] and right.shape[0] == 1:
                # Broadcast along rows
                right = to_series(right[0, :])

            else:
                raise ValueError(
                    "Unable to coerce to DataFrame, shape "
                    f"must be {left.shape}: given {right.shape}"
                )

        elif right.ndim > 2:
            raise ValueError(
                "Unable to coerce to Series/DataFrame, "
                f"dimension must be <= 2: {right.shape}"
            )

    elif is_list_like(right) and not isinstance(right, (ABCSeries, ABCDataFrame)):
        # GH 36702. Raise when attempting arithmetic with list of array-like.
        if any(is_array_like(el) for el in right):
            raise ValueError(
                f"Unable to coerce list of {type(right[0])} to Series/DataFrame"
            )
        # GH17901
        right = to_series(right)

    if flex is not None and isinstance(right, ABCDataFrame):
        if not left._indexed_same(right):
            if flex:
                left, right = left.align(right, join="outer", level=level, copy=False)
            else:
                raise ValueError(
                    "Can only compare identically-labeled DataFrame objects"
                )
    elif isinstance(right, ABCSeries):
        # axis=1 is default for DataFrame-with-Series op
        axis = left._get_axis_number(axis) if axis is not None else 1

        if not flex:
            if not left.axes[axis].equals(right.index):
                warnings.warn(
                    "Automatic reindexing on DataFrame vs Series comparisons "
                    "is deprecated and will raise ValueError in a future version.  "
                    "Do `left, right = left.align(right, axis=1, copy=False)` "
                    "before e.g. `left == right`",
                    FutureWarning,
                    stacklevel=5,
                )

        left, right = left.align(
            right, join="outer", axis=axis, level=level, copy=False
        )
        right = _maybe_align_series_as_frame(left, right, axis)

    return left, right


def should_reindex_frame_op(
    left: DataFrame, right, op, axis, default_axis, fill_value, level
) -> bool:
    """
    Check if this is an operation between DataFrames that will need to reindex.
    """
    assert isinstance(left, ABCDataFrame)

    if op is operator.pow or op is roperator.rpow:
        # GH#32685 pow has special semantics for operating with null values
        return False

    if not isinstance(right, ABCDataFrame):
        return False

    if fill_value is None and level is None and axis is default_axis:
        # TODO: any other cases we should handle here?

        # Intersection is always unique so we have to check the unique columns
        left_uniques = left.columns.unique()
        right_uniques = right.columns.unique()
        cols = left_uniques.intersection(right_uniques)
        if len(cols) and not (cols.equals(left_uniques) and cols.equals(right_uniques)):
            # TODO: is there a shortcut available when len(cols) == 0?
            return True

    return False


def frame_arith_method_with_reindex(left: DataFrame, right: DataFrame, op) -> DataFrame:
    """
    For DataFrame-with-DataFrame operations that require reindexing,
    operate only on shared columns, then reindex.

    Parameters
    ----------
    left : DataFrame
    right : DataFrame
    op : binary operator

    Returns
    -------
    DataFrame
    """
    # GH#31623, only operate on shared columns
    cols, lcols, rcols = left.columns.join(
        right.columns, how="inner", level=None, return_indexers=True
    )

    new_left = left.iloc[:, lcols]
    new_right = right.iloc[:, rcols]
    result = op(new_left, new_right)

    # Do the join on the columns instead of using align_method_FRAME
    #  to avoid constructing two potentially large/sparse DataFrames
    join_columns, _, _ = left.columns.join(
        right.columns, how="outer", level=None, return_indexers=True
    )

    if result.columns.has_duplicates:
        # Avoid reindexing with a duplicate axis.
        # https://github.com/pandas-dev/pandas/issues/35194
        indexer, _ = result.columns.get_indexer_non_unique(join_columns)
        indexer = algorithms.unique1d(indexer)
        result = result._reindex_with_indexers(
            {1: [join_columns, indexer]}, allow_dups=True
        )
    else:
        result = result.reindex(join_columns, axis=1)

    return result


def _maybe_align_series_as_frame(frame: DataFrame, series: Series, axis: int):
    """
    If the Series operand is not EA-dtype, we can broadcast to 2D and operate
    blockwise.
    """
    rvalues = series._values
    if not isinstance(rvalues, np.ndarray):
        # TODO(EA2D): no need to special-case with 2D EAs
        if rvalues.dtype == "datetime64[ns]" or rvalues.dtype == "timedelta64[ns]":
            # We can losslessly+cheaply cast to ndarray
            rvalues = np.asarray(rvalues)
        else:
            return series

    if axis == 0:
        rvalues = rvalues.reshape(-1, 1)
    else:
        rvalues = rvalues.reshape(1, -1)

    rvalues = np.broadcast_to(rvalues, frame.shape)
    return type(frame)(rvalues, index=frame.index, columns=frame.columns)


def flex_arith_method_FRAME(op):
    op_name = op.__name__.strip("_")
    default_axis = "columns"

    na_op = get_array_op(op)
    doc = make_flex_doc(op_name, "dataframe")

    @Appender(doc)
    def f(self, other, axis=default_axis, level=None, fill_value=None):

        if should_reindex_frame_op(
            self, other, op, axis, default_axis, fill_value, level
        ):
            return frame_arith_method_with_reindex(self, other, op)

        if isinstance(other, ABCSeries) and fill_value is not None:
            # TODO: We could allow this in cases where we end up going
            #  through the DataFrame path
            raise NotImplementedError(f"fill_value {fill_value} not supported.")

        axis = self._get_axis_number(axis) if axis is not None else 1

        other = maybe_prepare_scalar_for_op(other, self.shape)
        self, other = align_method_FRAME(self, other, axis, flex=True, level=level)

        if isinstance(other, ABCDataFrame):
            # Another DataFrame
            new_data = self._combine_frame(other, na_op, fill_value)

        elif isinstance(other, ABCSeries):
            new_data = self._dispatch_frame_op(other, op, axis=axis)
        else:
            # in this case we always have `np.ndim(other) == 0`
            if fill_value is not None:
                self = self.fillna(fill_value)

            new_data = self._dispatch_frame_op(other, op)

        return self._construct_result(new_data)

    f.__name__ = op_name

    return f


def flex_comp_method_FRAME(op):
    op_name = op.__name__.strip("_")
    default_axis = "columns"  # because we are "flex"

    doc = _flex_comp_doc_FRAME.format(
        op_name=op_name, desc=_op_descriptions[op_name]["desc"]
    )

    @Appender(doc)
    def f(self, other, axis=default_axis, level=None):
        axis = self._get_axis_number(axis) if axis is not None else 1

        self, other = align_method_FRAME(self, other, axis, flex=True, level=level)

        new_data = self._dispatch_frame_op(other, op, axis=axis)
        return self._construct_result(new_data)

    f.__name__ = op_name

    return f
