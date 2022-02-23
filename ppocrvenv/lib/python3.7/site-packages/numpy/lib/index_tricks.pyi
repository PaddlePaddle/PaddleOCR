import sys
from typing import (
    Any,
    Tuple,
    TypeVar,
    Generic,
    overload,
    List,
    Union,
    Sequence,
)

from numpy import (
    # Circumvent a naming conflict with `AxisConcatenator.matrix`
    matrix as _Matrix,
    ndenumerate as ndenumerate,
    ndindex as ndindex,
    ndarray,
    dtype,
    integer,
    str_,
    bytes_,
    bool_,
    int_,
    float_,
    complex_,
    intp,
    _OrderCF,
    _ModeKind,
)
from numpy.typing import (
    # Arrays
    ArrayLike,
    _NestedSequence,
    _RecursiveSequence,
    NDArray,
    _ArrayLikeInt,

    # DTypes
    DTypeLike,
    _SupportsDType,

    # Shapes
    _ShapeLike,
)

if sys.version_info >= (3, 8):
    from typing import Literal, SupportsIndex
else:
    from typing_extensions import Literal, SupportsIndex

_T = TypeVar("_T")
_DType = TypeVar("_DType", bound=dtype[Any])
_BoolType = TypeVar("_BoolType", Literal[True], Literal[False])
_TupType = TypeVar("_TupType", bound=Tuple[Any, ...])
_ArrayType = TypeVar("_ArrayType", bound=ndarray[Any, Any])

__all__: List[str]

@overload
def unravel_index(  # type: ignore[misc]
    indices: Union[int, integer[Any]],
    shape: _ShapeLike,
    order: _OrderCF = ...
) -> Tuple[intp, ...]: ...
@overload
def unravel_index(
    indices: _ArrayLikeInt,
    shape: _ShapeLike,
    order: _OrderCF = ...
) -> Tuple[NDArray[intp], ...]: ...

@overload
def ravel_multi_index(  # type: ignore[misc]
    multi_index: Sequence[Union[int, integer[Any]]],
    dims: _ShapeLike,
    mode: Union[_ModeKind, Tuple[_ModeKind, ...]] = ...,
    order: _OrderCF = ...
) -> intp: ...
@overload
def ravel_multi_index(
    multi_index: Sequence[_ArrayLikeInt],
    dims: _ShapeLike,
    mode: Union[_ModeKind, Tuple[_ModeKind, ...]] = ...,
    order: _OrderCF = ...
) -> NDArray[intp]: ...

@overload
def ix_(*args: _NestedSequence[_SupportsDType[_DType]]) -> Tuple[ndarray[Any, _DType], ...]: ...
@overload
def ix_(*args: _NestedSequence[str]) -> Tuple[NDArray[str_], ...]: ...
@overload
def ix_(*args: _NestedSequence[bytes]) -> Tuple[NDArray[bytes_], ...]: ...
@overload
def ix_(*args: _NestedSequence[bool]) -> Tuple[NDArray[bool_], ...]: ...
@overload
def ix_(*args: _NestedSequence[int]) -> Tuple[NDArray[int_], ...]: ...
@overload
def ix_(*args: _NestedSequence[float]) -> Tuple[NDArray[float_], ...]: ...
@overload
def ix_(*args: _NestedSequence[complex]) -> Tuple[NDArray[complex_], ...]: ...
@overload
def ix_(*args: _RecursiveSequence) -> Tuple[NDArray[Any], ...]: ...

class nd_grid(Generic[_BoolType]):
    sparse: _BoolType
    def __init__(self, sparse: _BoolType = ...) -> None: ...
    @overload
    def __getitem__(
        self: nd_grid[Literal[False]],
        key: Union[slice, Sequence[slice]],
    ) -> NDArray[Any]: ...
    @overload
    def __getitem__(
        self: nd_grid[Literal[True]],
        key: Union[slice, Sequence[slice]],
    ) -> List[NDArray[Any]]: ...

class MGridClass(nd_grid[Literal[False]]):
    def __init__(self) -> None: ...

mgrid: MGridClass

class OGridClass(nd_grid[Literal[True]]):
    def __init__(self) -> None: ...

ogrid: OGridClass

class AxisConcatenator:
    axis: int
    matrix: bool
    ndmin: int
    trans1d: int
    def __init__(
        self,
        axis: int = ...,
        matrix: bool = ...,
        ndmin: int = ...,
        trans1d: int = ...,
    ) -> None: ...
    @staticmethod
    @overload
    def concatenate(  # type: ignore[misc]
        *a: ArrayLike, axis: SupportsIndex = ..., out: None = ...
    ) -> NDArray[Any]: ...
    @staticmethod
    @overload
    def concatenate(
        *a: ArrayLike, axis: SupportsIndex = ..., out: _ArrayType = ...
    ) -> _ArrayType: ...
    @staticmethod
    def makemat(
        data: ArrayLike, dtype: DTypeLike = ..., copy: bool = ...
    ) -> _Matrix: ...

    # TODO: Sort out this `__getitem__` method
    def __getitem__(self, key: Any) -> Any: ...

class RClass(AxisConcatenator):
    axis: Literal[0]
    matrix: Literal[False]
    ndmin: Literal[1]
    trans1d: Literal[-1]
    def __init__(self) -> None: ...

r_: RClass

class CClass(AxisConcatenator):
    axis: Literal[-1]
    matrix: Literal[False]
    ndmin: Literal[2]
    trans1d: Literal[0]
    def __init__(self) -> None: ...

c_: CClass

class IndexExpression(Generic[_BoolType]):
    maketuple: _BoolType
    def __init__(self, maketuple: _BoolType) -> None: ...
    @overload
    def __getitem__(self, item: _TupType) -> _TupType: ...  # type: ignore[misc]
    @overload
    def __getitem__(self: IndexExpression[Literal[True]], item: _T) -> Tuple[_T]: ...
    @overload
    def __getitem__(self: IndexExpression[Literal[False]], item: _T) -> _T: ...

index_exp: IndexExpression[Literal[True]]
s_: IndexExpression[Literal[False]]

def fill_diagonal(a: ndarray[Any, Any], val: Any, wrap: bool = ...) -> None: ...
def diag_indices(n: int, ndim: int = ...) -> Tuple[NDArray[int_], ...]: ...
def diag_indices_from(arr: ArrayLike) -> Tuple[NDArray[int_], ...]: ...

# NOTE: see `numpy/__init__.pyi` for `ndenumerate` and `ndindex`
