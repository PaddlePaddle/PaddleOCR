import sys
from typing import overload, Tuple, Union, Sequence, Any

from numpy import ndarray
from numpy.typing import ArrayLike, DTypeLike, _SupportsArray, _NumberLike_co

if sys.version_info >= (3, 8):
    from typing import SupportsIndex, Literal
else:
    from typing_extensions import SupportsIndex, Literal

# TODO: wait for support for recursive types
_ArrayLikeNested = Sequence[Sequence[Any]]
_ArrayLikeNumber = Union[
    _NumberLike_co, Sequence[_NumberLike_co], ndarray, _SupportsArray, _ArrayLikeNested
]
@overload
def linspace(
    start: _ArrayLikeNumber,
    stop: _ArrayLikeNumber,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: Literal[False] = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> ndarray: ...
@overload
def linspace(
    start: _ArrayLikeNumber,
    stop: _ArrayLikeNumber,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: Literal[True] = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> Tuple[ndarray, Any]: ...

def logspace(
    start: _ArrayLikeNumber,
    stop: _ArrayLikeNumber,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeNumber = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> ndarray: ...

def geomspace(
    start: _ArrayLikeNumber,
    stop: _ArrayLikeNumber,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> ndarray: ...
