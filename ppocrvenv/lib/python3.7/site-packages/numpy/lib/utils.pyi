import sys
from ast import AST
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from numpy import ndarray, generic

from numpy.core.numerictypes import (
    issubclass_ as issubclass_,
    issubdtype as issubdtype,
    issubsctype as issubsctype,
)

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

_T_contra = TypeVar("_T_contra", contravariant=True)
_FuncType = TypeVar("_FuncType", bound=Callable[..., Any])

# A file-like object opened in `w` mode
class _SupportsWrite(Protocol[_T_contra]):
    def write(self, __s: _T_contra) -> Any: ...

__all__: List[str]

class _Deprecate:
    old_name: Optional[str]
    new_name: Optional[str]
    message: Optional[str]
    def __init__(
        self,
        old_name: Optional[str] = ...,
        new_name: Optional[str] = ...,
        message: Optional[str] = ...,
    ) -> None: ...
    # NOTE: `__call__` can in principle take arbitrary `*args` and `**kwargs`,
    # even though they aren't used for anything
    def __call__(self, func: _FuncType) -> _FuncType: ...

def get_include() -> str: ...

@overload
def deprecate(
    *,
    old_name: Optional[str] = ...,
    new_name: Optional[str] = ...,
    message: Optional[str] = ...,
) -> _Deprecate: ...
@overload
def deprecate(
    __func: _FuncType,
    old_name: Optional[str] = ...,
    new_name: Optional[str] = ...,
    message: Optional[str] = ...,
) -> _FuncType: ...

def deprecate_with_doc(msg: Optional[str]) -> _Deprecate: ...

# NOTE: In practice `byte_bounds` can (potentially) take any object
# implementing the `__array_interface__` protocol. The caveat is
# that certain keys, marked as optional in the spec, must be present for
#  `byte_bounds`. This concerns `"strides"` and `"data"`.
def byte_bounds(a: Union[generic, ndarray[Any, Any]]) -> Tuple[int, int]: ...

def who(vardict: Optional[Mapping[str, ndarray[Any, Any]]] = ...) -> None: ...

def info(
    object: object = ...,
    maxwidth: int = ...,
    output: Optional[_SupportsWrite[str]] = ...,
    toplevel: str = ...,
) -> None: ...

def source(
    object: object,
    output: Optional[_SupportsWrite[str]] = ...,
) -> None: ...

def lookfor(
    what: str,
    module: Union[None, str, Sequence[str]] = ...,
    import_modules: bool = ...,
    regenerate: bool = ...,
    output: Optional[_SupportsWrite[str]] =...,
) -> None: ...

def safe_eval(source: Union[str, AST]) -> Any: ...
