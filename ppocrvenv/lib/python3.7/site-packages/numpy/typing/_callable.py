"""
A module with various ``typing.Protocol`` subclasses that implement
the ``__call__`` magic method.

See the `Mypy documentation`_ on protocols for more details.

.. _`Mypy documentation`: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols

"""

from __future__ import annotations

import sys
from typing import (
    Union,
    TypeVar,
    overload,
    Any,
    Tuple,
    NoReturn,
    TYPE_CHECKING,
)

from numpy import (
    ndarray,
    dtype,
    generic,
    bool_,
    timedelta64,
    number,
    integer,
    unsignedinteger,
    signedinteger,
    int8,
    int_,
    floating,
    float64,
    complexfloating,
    complex128,
)
from ._nbit import _NBitInt, _NBitDouble
from ._scalars import (
    _BoolLike_co,
    _IntLike_co,
    _FloatLike_co,
    _ComplexLike_co,
    _NumberLike_co,
)
from . import NBitBase
from ._array_like import ArrayLike
from ._generic_alias import NDArray

if sys.version_info >= (3, 8):
    from typing import Protocol
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

if TYPE_CHECKING or HAVE_PROTOCOL:
    _T1 = TypeVar("_T1")
    _T2 = TypeVar("_T2")
    _2Tuple = Tuple[_T1, _T1]

    _NBit1 = TypeVar("_NBit1", bound=NBitBase)
    _NBit2 = TypeVar("_NBit2", bound=NBitBase)

    _IntType = TypeVar("_IntType", bound=integer)
    _FloatType = TypeVar("_FloatType", bound=floating)
    _NumberType = TypeVar("_NumberType", bound=number)
    _NumberType_co = TypeVar("_NumberType_co", covariant=True, bound=number)
    _GenericType_co = TypeVar("_GenericType_co", covariant=True, bound=generic)

    class _BoolOp(Protocol[_GenericType_co]):
        @overload
        def __call__(self, __other: _BoolLike_co) -> _GenericType_co: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> int_: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: _NumberType) -> _NumberType: ...

    class _BoolBitOp(Protocol[_GenericType_co]):
        @overload
        def __call__(self, __other: _BoolLike_co) -> _GenericType_co: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> int_: ...
        @overload
        def __call__(self, __other: _IntType) -> _IntType: ...

    class _BoolSub(Protocol):
        # Note that `__other: bool_` is absent here
        @overload
        def __call__(self, __other: bool) -> NoReturn: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> int_: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: _NumberType) -> _NumberType: ...

    class _BoolTrueDiv(Protocol):
        @overload
        def __call__(self, __other: Union[float, _IntLike_co]) -> float64: ...
        @overload
        def __call__(self, __other: complex) -> complex128: ...
        @overload
        def __call__(self, __other: _NumberType) -> _NumberType: ...

    class _BoolMod(Protocol):
        @overload
        def __call__(self, __other: _BoolLike_co) -> int8: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> int_: ...
        @overload
        def __call__(self, __other: float) -> float64: ...
        @overload
        def __call__(self, __other: _IntType) -> _IntType: ...
        @overload
        def __call__(self, __other: _FloatType) -> _FloatType: ...

    class _BoolDivMod(Protocol):
        @overload
        def __call__(self, __other: _BoolLike_co) -> _2Tuple[int8]: ...
        @overload  # platform dependent
        def __call__(self, __other: int) -> _2Tuple[int_]: ...
        @overload
        def __call__(self, __other: float) -> _2Tuple[floating[Union[_NBit1, _NBitDouble]]]: ...
        @overload
        def __call__(self, __other: _IntType) -> _2Tuple[_IntType]: ...
        @overload
        def __call__(self, __other: _FloatType) -> _2Tuple[_FloatType]: ...

    class _TD64Div(Protocol[_NumberType_co]):
        @overload
        def __call__(self, __other: timedelta64) -> _NumberType_co: ...
        @overload
        def __call__(self, __other: _BoolLike_co) -> NoReturn: ...
        @overload
        def __call__(self, __other: _FloatLike_co) -> timedelta64: ...

    class _IntTrueDiv(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> floating[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> floating[Union[_NBit1, _NBitInt]]: ...
        @overload
        def __call__(self, __other: float) -> floating[Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: complex
        ) -> complexfloating[Union[_NBit1, _NBitDouble], Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(self, __other: integer[_NBit2]) -> floating[Union[_NBit1, _NBit2]]: ...

    class _UnsignedIntOp(Protocol[_NBit1]):
        # NOTE: `uint64 + signedinteger -> float64`
        @overload
        def __call__(self, __other: bool) -> unsignedinteger[_NBit1]: ...
        @overload
        def __call__(
            self, __other: Union[int, signedinteger[Any]]
        ) -> Any: ...
        @overload
        def __call__(self, __other: float) -> floating[Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: complex
        ) -> complexfloating[Union[_NBit1, _NBitDouble], Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: unsignedinteger[_NBit2]
        ) -> unsignedinteger[Union[_NBit1, _NBit2]]: ...

    class _UnsignedIntBitOp(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> unsignedinteger[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> signedinteger[Any]: ...
        @overload
        def __call__(self, __other: signedinteger[Any]) -> signedinteger[Any]: ...
        @overload
        def __call__(
            self, __other: unsignedinteger[_NBit2]
        ) -> unsignedinteger[Union[_NBit1, _NBit2]]: ...

    class _UnsignedIntMod(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> unsignedinteger[_NBit1]: ...
        @overload
        def __call__(
            self, __other: Union[int, signedinteger[Any]]
        ) -> Any: ...
        @overload
        def __call__(self, __other: float) -> floating[Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: unsignedinteger[_NBit2]
        ) -> unsignedinteger[Union[_NBit1, _NBit2]]: ...

    class _UnsignedIntDivMod(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> _2Tuple[signedinteger[_NBit1]]: ...
        @overload
        def __call__(
            self, __other: Union[int, signedinteger[Any]]
        ) -> _2Tuple[Any]: ...
        @overload
        def __call__(self, __other: float) -> _2Tuple[floating[Union[_NBit1, _NBitDouble]]]: ...
        @overload
        def __call__(
            self, __other: unsignedinteger[_NBit2]
        ) -> _2Tuple[unsignedinteger[Union[_NBit1, _NBit2]]]: ...

    class _SignedIntOp(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> signedinteger[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> signedinteger[Union[_NBit1, _NBitInt]]: ...
        @overload
        def __call__(self, __other: float) -> floating[Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: complex
        ) -> complexfloating[Union[_NBit1, _NBitDouble], Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: signedinteger[_NBit2]
        ) -> signedinteger[Union[_NBit1, _NBit2]]: ...

    class _SignedIntBitOp(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> signedinteger[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> signedinteger[Union[_NBit1, _NBitInt]]: ...
        @overload
        def __call__(
            self, __other: signedinteger[_NBit2]
        ) -> signedinteger[Union[_NBit1, _NBit2]]: ...

    class _SignedIntMod(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> signedinteger[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> signedinteger[Union[_NBit1, _NBitInt]]: ...
        @overload
        def __call__(self, __other: float) -> floating[Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: signedinteger[_NBit2]
        ) -> signedinteger[Union[_NBit1, _NBit2]]: ...

    class _SignedIntDivMod(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> _2Tuple[signedinteger[_NBit1]]: ...
        @overload
        def __call__(self, __other: int) -> _2Tuple[signedinteger[Union[_NBit1, _NBitInt]]]: ...
        @overload
        def __call__(self, __other: float) -> _2Tuple[floating[Union[_NBit1, _NBitDouble]]]: ...
        @overload
        def __call__(
            self, __other: signedinteger[_NBit2]
        ) -> _2Tuple[signedinteger[Union[_NBit1, _NBit2]]]: ...

    class _FloatOp(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> floating[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> floating[Union[_NBit1, _NBitInt]]: ...
        @overload
        def __call__(self, __other: float) -> floating[Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: complex
        ) -> complexfloating[Union[_NBit1, _NBitDouble], Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: Union[integer[_NBit2], floating[_NBit2]]
        ) -> floating[Union[_NBit1, _NBit2]]: ...

    class _FloatMod(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> floating[_NBit1]: ...
        @overload
        def __call__(self, __other: int) -> floating[Union[_NBit1, _NBitInt]]: ...
        @overload
        def __call__(self, __other: float) -> floating[Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self, __other: Union[integer[_NBit2], floating[_NBit2]]
        ) -> floating[Union[_NBit1, _NBit2]]: ...

    class _FloatDivMod(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> _2Tuple[floating[_NBit1]]: ...
        @overload
        def __call__(self, __other: int) -> _2Tuple[floating[Union[_NBit1, _NBitInt]]]: ...
        @overload
        def __call__(self, __other: float) -> _2Tuple[floating[Union[_NBit1, _NBitDouble]]]: ...
        @overload
        def __call__(
            self, __other: Union[integer[_NBit2], floating[_NBit2]]
        ) -> _2Tuple[floating[Union[_NBit1, _NBit2]]]: ...

    class _ComplexOp(Protocol[_NBit1]):
        @overload
        def __call__(self, __other: bool) -> complexfloating[_NBit1, _NBit1]: ...
        @overload
        def __call__(self, __other: int) -> complexfloating[Union[_NBit1, _NBitInt], Union[_NBit1, _NBitInt]]: ...
        @overload
        def __call__(
            self, __other: Union[float, complex]
        ) -> complexfloating[Union[_NBit1, _NBitDouble], Union[_NBit1, _NBitDouble]]: ...
        @overload
        def __call__(
            self,
            __other: Union[
                integer[_NBit2],
                floating[_NBit2],
                complexfloating[_NBit2, _NBit2],
            ]
        ) -> complexfloating[Union[_NBit1, _NBit2], Union[_NBit1, _NBit2]]: ...

    class _NumberOp(Protocol):
        def __call__(self, __other: _NumberLike_co) -> Any: ...

    class _ComparisonOp(Protocol[_T1, _T2]):
        @overload
        def __call__(self, __other: _T1) -> bool_: ...
        @overload
        def __call__(self, __other: _T2) -> NDArray[bool_]: ...

else:
    _BoolOp = Any
    _BoolBitOp = Any
    _BoolSub = Any
    _BoolTrueDiv = Any
    _BoolMod = Any
    _BoolDivMod = Any
    _TD64Div = Any
    _IntTrueDiv = Any
    _UnsignedIntOp = Any
    _UnsignedIntBitOp = Any
    _UnsignedIntMod = Any
    _UnsignedIntDivMod = Any
    _SignedIntOp = Any
    _SignedIntBitOp = Any
    _SignedIntMod = Any
    _SignedIntDivMod = Any
    _FloatOp = Any
    _FloatMod = Any
    _FloatDivMod = Any
    _ComplexOp = Any
    _NumberOp = Any
    _ComparisonOp = Any
