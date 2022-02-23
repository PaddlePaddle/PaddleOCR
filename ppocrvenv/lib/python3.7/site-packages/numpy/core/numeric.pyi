import sys
from typing import (
    Any,
    Optional,
    Union,
    Sequence,
    Tuple,
    Callable,
    List,
    overload,
    TypeVar,
    Iterable,
)

from numpy import ndarray, generic, dtype, bool_, signedinteger, _OrderKACF, _OrderCF
from numpy.typing import ArrayLike, DTypeLike, _ShapeLike

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_T = TypeVar("_T")
_ArrayType = TypeVar("_ArrayType", bound=ndarray)

_CorrelateMode = Literal["valid", "same", "full"]

@overload
def zeros_like(
    a: _ArrayType,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: Literal[True] = ...,
    shape: None = ...,
) -> _ArrayType: ...
@overload
def zeros_like(
    a: ArrayLike,
    dtype: DTypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> ndarray: ...

def ones(
    shape: _ShapeLike,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> ndarray: ...

@overload
def ones_like(
    a: _ArrayType,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: Literal[True] = ...,
    shape: None = ...,
) -> _ArrayType: ...
@overload
def ones_like(
    a: ArrayLike,
    dtype: DTypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> ndarray: ...

@overload
def empty_like(
    a: _ArrayType,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: Literal[True] = ...,
    shape: None = ...,
) -> _ArrayType: ...
@overload
def empty_like(
    a: ArrayLike,
    dtype: DTypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> ndarray: ...

def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> ndarray: ...

@overload
def full_like(
    a: _ArrayType,
    fill_value: Any,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: Literal[True] = ...,
    shape: None = ...,
) -> _ArrayType: ...
@overload
def full_like(
    a: ArrayLike,
    fill_value: Any,
    dtype: DTypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> ndarray: ...

@overload
def count_nonzero(
    a: ArrayLike,
    axis: None = ...,
    *,
    keepdims: Literal[False] = ...,
) -> int: ...
@overload
def count_nonzero(
    a: ArrayLike,
    axis: _ShapeLike = ...,
    *,
    keepdims: bool = ...,
) -> Any: ...  # TODO: np.intp or ndarray[np.intp]

def isfortran(a: Union[ndarray, generic]) -> bool: ...

def argwhere(a: ArrayLike) -> ndarray: ...

def flatnonzero(a: ArrayLike) -> ndarray: ...

def correlate(
    a: ArrayLike,
    v: ArrayLike,
    mode: _CorrelateMode = ...,
) -> ndarray: ...

def convolve(
    a: ArrayLike,
    v: ArrayLike,
    mode: _CorrelateMode = ...,
) -> ndarray: ...

@overload
def outer(
    a: ArrayLike,
    b: ArrayLike,
    out: None = ...,
) -> ndarray: ...
@overload
def outer(
    a: ArrayLike,
    b: ArrayLike,
    out: _ArrayType = ...,
) -> _ArrayType: ...

def tensordot(
    a: ArrayLike,
    b: ArrayLike,
    axes: Union[int, Tuple[_ShapeLike, _ShapeLike]] = ...,
) -> ndarray: ...

def roll(
    a: ArrayLike,
    shift: _ShapeLike,
    axis: Optional[_ShapeLike] = ...,
) -> ndarray: ...

def rollaxis(a: ndarray, axis: int, start: int = ...) -> ndarray: ...

def moveaxis(
    a: ndarray,
    source: _ShapeLike,
    destination: _ShapeLike,
) -> ndarray: ...

def cross(
    a: ArrayLike,
    b: ArrayLike,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: Optional[int] = ...,
) -> ndarray: ...

@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike = ...,
    sparse: Literal[False] = ...,
) -> ndarray: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike = ...,
    sparse: Literal[True] = ...,
) -> Tuple[ndarray, ...]: ...

def fromfunction(
    function: Callable[..., _T],
    shape: Sequence[int],
    *,
    dtype: DTypeLike = ...,
    like: ArrayLike = ...,
    **kwargs: Any,
) -> _T: ...

def isscalar(element: Any) -> bool: ...

def binary_repr(num: int, width: Optional[int] = ...) -> str: ...

def base_repr(number: int, base: int = ..., padding: int = ...) -> str: ...

def identity(
    n: int,
    dtype: DTypeLike = ...,
    *,
    like: ArrayLike = ...,
) -> ndarray: ...

def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> bool: ...

def isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> Any: ...

def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: bool = ...) -> bool: ...

def array_equiv(a1: ArrayLike, a2: ArrayLike) -> bool: ...
