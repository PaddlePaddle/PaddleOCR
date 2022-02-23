"""A module with private type-check-only `numpy.ufunc` subclasses.

The signatures of the ufuncs are too varied to reasonably type
with a single class. So instead, `ufunc` has been expanded into
four private subclasses, one for each combination of
`~ufunc.nin` and `~ufunc.nout`.

"""

from typing import (
    Any,
    Generic,
    List,
    Optional,
    overload,
    Tuple,
    TypeVar,
    Union,
)

from numpy import ufunc, _Casting, _OrderKACF
from numpy.typing import NDArray

from ._shape import _ShapeLike
from ._scalars import _ScalarLike_co
from ._array_like import ArrayLike, _ArrayLikeBool_co, _ArrayLikeInt_co
from ._dtype_like import DTypeLike

from typing_extensions import Literal, SupportsIndex

_T = TypeVar("_T")
_2Tuple = Tuple[_T, _T]
_3Tuple = Tuple[_T, _T, _T]
_4Tuple = Tuple[_T, _T, _T, _T]

_NTypes = TypeVar("_NTypes", bound=int)
_IDType = TypeVar("_IDType", bound=Any)
_NameType = TypeVar("_NameType", bound=str)

# NOTE: In reality `extobj` should be a length of list 3 containing an
# int, an int, and a callable, but there's no way to properly express
# non-homogenous lists.
# Use `Any` over `Union` to avoid issues related to lists invariance.

# NOTE: `reduce`, `accumulate`, `reduceat` and `outer` raise a ValueError for
# ufuncs that don't accept two input arguments and return one output argument.
# In such cases the respective methods are simply typed as `None`.

# NOTE: Similarly, `at` won't be defined for ufuncs that return
# multiple outputs; in such cases `at` is typed as `None`

# NOTE: If 2 output types are returned then `out` must be a
# 2-tuple of arrays. Otherwise `None` or a plain array are also acceptable

class _UFunc_Nin1_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[1]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[2]: ...
    @property
    def signature(self) -> None: ...
    @property
    def reduce(self) -> None: ...
    @property
    def accumulate(self) -> None: ...
    @property
    def reduceat(self) -> None: ...
    @property
    def outer(self) -> None: ...

    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        out: None = ...,
        *,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _2Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> Any: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        out: Union[None, NDArray[Any], Tuple[NDArray[Any]]] = ...,
        *,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _2Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> NDArray[Any]: ...

    def at(
        self,
        __a: NDArray[Any],
        __indices: _ArrayLikeInt_co,
    ) -> None: ...

class _UFunc_Nin2_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def signature(self) -> None: ...

    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __x2: _ScalarLike_co,
        out: None = ...,
        *,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _3Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> Any: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: Union[None, NDArray[Any], Tuple[NDArray[Any]]] = ...,
        *,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _3Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> NDArray[Any]: ...

    def at(
        self,
        __a: NDArray[Any],
        __indices: _ArrayLikeInt_co,
        __b: ArrayLike,
    ) -> None: ...

    def reduce(
        self,
        array: ArrayLike,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: Optional[NDArray[Any]] = ...,
        keepdims: bool = ...,
        initial: Any = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...

    def accumulate(
        self,
        array: ArrayLike,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: Optional[NDArray[Any]] = ...,
    ) -> NDArray[Any]: ...

    def reduceat(
        self,
        array: ArrayLike,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: Optional[NDArray[Any]] = ...,
    ) -> NDArray[Any]: ...

    # Expand `**kwargs` into explicit keyword-only arguments
    @overload
    def outer(
        self,
        __A: _ScalarLike_co,
        __B: _ScalarLike_co,
        *,
        out: None = ...,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _3Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> Any: ...
    @overload
    def outer(  # type: ignore[misc]
        self,
        __A: ArrayLike,
        __B: ArrayLike,
        *,
        out: Union[None, NDArray[Any], Tuple[NDArray[Any]]] = ...,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _3Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> NDArray[Any]: ...

class _UFunc_Nin1_Nout2(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[1]: ...
    @property
    def nout(self) -> Literal[2]: ...
    @property
    def nargs(self) -> Literal[3]: ...
    @property
    def signature(self) -> None: ...
    @property
    def at(self) -> None: ...
    @property
    def reduce(self) -> None: ...
    @property
    def accumulate(self) -> None: ...
    @property
    def reduceat(self) -> None: ...
    @property
    def outer(self) -> None: ...

    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __out1: None = ...,
        __out2: None = ...,
        *,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _3Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> _2Tuple[Any]: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __out1: Optional[NDArray[Any]] = ...,
        __out2: Optional[NDArray[Any]] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _3Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> _2Tuple[NDArray[Any]]: ...

class _UFunc_Nin2_Nout2(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[2]: ...
    @property
    def nargs(self) -> Literal[4]: ...
    @property
    def signature(self) -> None: ...
    @property
    def at(self) -> None: ...
    @property
    def reduce(self) -> None: ...
    @property
    def accumulate(self) -> None: ...
    @property
    def reduceat(self) -> None: ...
    @property
    def outer(self) -> None: ...

    @overload
    def __call__(
        self,
        __x1: _ScalarLike_co,
        __x2: _ScalarLike_co,
        __out1: None = ...,
        __out2: None = ...,
        *,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _4Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> _2Tuple[Any]: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        __out1: Optional[NDArray[Any]] = ...,
        __out2: Optional[NDArray[Any]] = ...,
        *,
        out: _2Tuple[NDArray[Any]] = ...,
        where: Optional[_ArrayLikeBool_co] = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _4Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
    ) -> _2Tuple[NDArray[Any]]: ...

class _GUFunc_Nin2_Nout1(ufunc, Generic[_NameType, _NTypes, _IDType]):
    @property
    def __name__(self) -> _NameType: ...
    @property
    def ntypes(self) -> _NTypes: ...
    @property
    def identity(self) -> _IDType: ...
    @property
    def nin(self) -> Literal[2]: ...
    @property
    def nout(self) -> Literal[1]: ...
    @property
    def nargs(self) -> Literal[3]: ...

    # NOTE: In practice the only gufunc in the main name is `matmul`,
    # so we can use its signature here
    @property
    def signature(self) -> Literal["(n?,k),(k,m?)->(n?,m?)"]: ...
    @property
    def reduce(self) -> None: ...
    @property
    def accumulate(self) -> None: ...
    @property
    def reduceat(self) -> None: ...
    @property
    def outer(self) -> None: ...
    @property
    def at(self) -> None: ...

    # Scalar for 1D array-likes; ndarray otherwise
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: None = ...,
        *,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _3Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
        axes: List[_2Tuple[SupportsIndex]] = ...,
    ) -> Any: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: Union[NDArray[Any], Tuple[NDArray[Any]]],
        *,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, _3Tuple[Optional[str]]] = ...,
        extobj: List[Any] = ...,
        axes: List[_2Tuple[SupportsIndex]] = ...,
    ) -> NDArray[Any]: ...
