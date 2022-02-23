from typing import Any, TypeVar, Type, overload, Optional, Generic
import ctypes as ct

from numpy import ndarray

_CastT = TypeVar("_CastT", bound=ct._CanCastTo)  # Copied from `ctypes.cast`
_CT = TypeVar("_CT", bound=ct._CData)
_PT = TypeVar("_PT", bound=Optional[int])

# TODO: Let the likes of `shape_as` and `strides_as` return `None`
# for 0D arrays once we've got shape-support

class _ctypes(Generic[_PT]):
    @overload
    def __new__(cls, array: ndarray[Any, Any], ptr: None = ...) -> _ctypes[None]: ...
    @overload
    def __new__(cls, array: ndarray[Any, Any], ptr: _PT) -> _ctypes[_PT]: ...

    # NOTE: In practice `shape` and `strides` return one of the concrete
    # platform dependant array-types (`c_int`, `c_long` or `c_longlong`)
    # corresponding to C's `int_ptr_t`, as determined by `_getintp_ctype`
    # TODO: Hook this in to the mypy plugin so that a more appropiate
    # `ctypes._SimpleCData[int]` sub-type can be returned
    @property
    def data(self) -> _PT: ...
    @property
    def shape(self) -> ct.Array[ct.c_int64]: ...
    @property
    def strides(self) -> ct.Array[ct.c_int64]: ...
    @property
    def _as_parameter_(self) -> ct.c_void_p: ...

    def data_as(self, obj: Type[_CastT]) -> _CastT: ...
    def shape_as(self, obj: Type[_CT]) -> ct.Array[_CT]: ...
    def strides_as(self, obj: Type[_CT]) -> ct.Array[_CT]: ...
