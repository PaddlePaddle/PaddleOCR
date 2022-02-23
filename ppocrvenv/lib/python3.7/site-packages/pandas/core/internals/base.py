"""
Base class for the internal managers. Both BlockManager and ArrayManager
inherit from this class.
"""
from __future__ import annotations

from typing import TypeVar

from pandas._typing import (
    DtypeObj,
    Shape,
    final,
)
from pandas.errors import AbstractMethodError

from pandas.core.dtypes.cast import find_common_type

from pandas.core.base import PandasObject
from pandas.core.indexes.api import Index

T = TypeVar("T", bound="DataManager")


class DataManager(PandasObject):

    # TODO share more methods/attributes

    axes: list[Index]

    @property
    def items(self) -> Index:
        raise AbstractMethodError(self)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> Shape:
        return tuple(len(ax) for ax in self.axes)

    @final
    def _validate_set_axis(self, axis: int, new_labels: Index) -> None:
        # Caller is responsible for ensuring we have an Index object.
        old_len = len(self.axes[axis])
        new_len = len(new_labels)

        if axis == 1 and len(self.items) == 0:
            # If we are setting the index on a DataFrame with no columns,
            #  it is OK to change the length.
            pass

        elif new_len != old_len:
            raise ValueError(
                f"Length mismatch: Expected axis has {old_len} elements, new "
                f"values have {new_len} elements"
            )

    def reindex_indexer(
        self: T,
        new_axis,
        indexer,
        axis: int,
        fill_value=None,
        allow_dups: bool = False,
        copy: bool = True,
        consolidate: bool = True,
        only_slice: bool = False,
    ) -> T:
        raise AbstractMethodError(self)

    @final
    def reindex_axis(
        self: T,
        new_index: Index,
        axis: int,
        fill_value=None,
        consolidate: bool = True,
        only_slice: bool = False,
    ) -> T:
        """
        Conform data manager to new index.
        """
        new_index, indexer = self.axes[axis].reindex(new_index)

        return self.reindex_indexer(
            new_index,
            indexer,
            axis=axis,
            fill_value=fill_value,
            copy=False,
            consolidate=consolidate,
            only_slice=only_slice,
        )

    def _equal_values(self: T, other: T) -> bool:
        """
        To be implemented by the subclasses. Only check the column values
        assuming shape and indexes have already been checked.
        """
        raise AbstractMethodError(self)

    def equals(self, other: object) -> bool:
        """
        Implementation for DataFrame.equals
        """
        if not isinstance(other, DataManager):
            return False

        self_axes, other_axes = self.axes, other.axes
        if len(self_axes) != len(other_axes):
            return False
        if not all(ax1.equals(ax2) for ax1, ax2 in zip(self_axes, other_axes)):
            return False

        return self._equal_values(other)

    def apply(
        self: T,
        f,
        align_keys: list[str] | None = None,
        ignore_failures: bool = False,
        **kwargs,
    ) -> T:
        raise AbstractMethodError(self)

    def isna(self: T, func) -> T:
        return self.apply("apply", func=func)


class SingleDataManager(DataManager):
    ndim = 1

    @property
    def array(self):
        """
        Quick access to the backing array of the Block or SingleArrayManager.
        """
        return self.arrays[0]  # type: ignore[attr-defined]


def interleaved_dtype(dtypes: list[DtypeObj]) -> DtypeObj | None:
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    blocks : List[DtypeObj]

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        None is returned when `blocks` is empty.
    """
    if not len(dtypes):
        return None

    return find_common_type(dtypes)
