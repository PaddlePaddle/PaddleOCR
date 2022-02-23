import sys
from typing import Optional, Union, Callable, Any

if sys.version_info >= (3, 8):
    from typing import Literal, Protocol, TypedDict
else:
    from typing_extensions import Literal, Protocol, TypedDict

_ErrKind = Literal["ignore", "warn", "raise", "call", "print", "log"]
_ErrFunc = Callable[[str, int], Any]

class _SupportsWrite(Protocol):
    def write(self, __msg: str) -> Any: ...

class _ErrDict(TypedDict):
    divide: _ErrKind
    over: _ErrKind
    under: _ErrKind
    invalid: _ErrKind

class _ErrDictOptional(TypedDict, total=False):
    all: Optional[_ErrKind]
    divide: Optional[_ErrKind]
    over: Optional[_ErrKind]
    under: Optional[_ErrKind]
    invalid: Optional[_ErrKind]

def seterr(
    all: Optional[_ErrKind] = ...,
    divide: Optional[_ErrKind] = ...,
    over: Optional[_ErrKind] = ...,
    under: Optional[_ErrKind] = ...,
    invalid: Optional[_ErrKind] = ...,
) -> _ErrDict: ...
def geterr() -> _ErrDict: ...
def setbufsize(size: int) -> int: ...
def getbufsize() -> int: ...
def seterrcall(
    func: Union[None, _ErrFunc, _SupportsWrite]
) -> Union[None, _ErrFunc, _SupportsWrite]: ...
def geterrcall() -> Union[None, _ErrFunc, _SupportsWrite]: ...

# See `numpy/__init__.pyi` for the `errstate` class
