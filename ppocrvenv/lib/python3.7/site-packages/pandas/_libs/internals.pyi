from typing import (
    Iterator,
    Sequence,
    overload,
)

import numpy as np

from pandas._typing import (
    ArrayLike,
    T,
)

from pandas import Index
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.internals.blocks import Block as B

def slice_len(slc: slice, objlen: int = ...) -> int: ...
def get_blkno_indexers(
    blknos: np.ndarray,  # int64_t[:]
    group: bool = ...,
) -> list[tuple[int, slice | np.ndarray]]: ...
def get_blkno_placements(
    blknos: np.ndarray,
    group: bool = ...,
) -> Iterator[tuple[int, BlockPlacement]]: ...

class BlockPlacement:
    def __init__(self, val: int | slice | np.ndarray): ...
    @property
    def indexer(self) -> np.ndarray | slice: ...
    @property
    def as_array(self) -> np.ndarray: ...
    @property
    def is_slice_like(self) -> bool: ...
    @overload
    def __getitem__(self, loc: slice | Sequence[int]) -> BlockPlacement: ...
    @overload
    def __getitem__(self, loc: int) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...
    def delete(self, loc) -> BlockPlacement: ...
    def append(self, others: list[BlockPlacement]) -> BlockPlacement: ...

class SharedBlock:
    _mgr_locs: BlockPlacement
    ndim: int
    values: ArrayLike
    def __init__(self, values: ArrayLike, placement: BlockPlacement, ndim: int): ...

class NumpyBlock(SharedBlock):
    values: np.ndarray
    def getitem_block_index(self: T, slicer: slice) -> T: ...

class NDArrayBackedBlock(SharedBlock):
    values: NDArrayBackedExtensionArray
    def getitem_block_index(self: T, slicer: slice) -> T: ...

class Block(SharedBlock): ...

class BlockManager:
    blocks: tuple[B, ...]
    axes: list[Index]
    _known_consolidated: bool
    _is_consolidated: bool
    _blknos: np.ndarray
    _blklocs: np.ndarray
    def __init__(
        self, blocks: tuple[B, ...], axes: list[Index], verify_integrity=True
    ): ...
    def get_slice(self: T, slobj: slice, axis: int = ...) -> T: ...
