import sys
from typing import (
    Any,
    List,
    Sequence,
    Tuple,
    Union,
    Type,
    TypeVar,
    Generic,
    TYPE_CHECKING,
)

import numpy as np
from ._shape import _ShapeLike

if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol, TypedDict
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

from ._char_codes import (
    _BoolCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Complex64Codes,
    _Complex128Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntPCodes,
    _IntCodes,
    _LongLongCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
    _DT64Codes,
    _TD64Codes,
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _ObjectCodes,
)

_DType_co = TypeVar("_DType_co", covariant=True, bound=np.dtype)

_DTypeLikeNested = Any  # TODO: wait for support for recursive types

if TYPE_CHECKING or HAVE_PROTOCOL:
    # Mandatory keys
    class _DTypeDictBase(TypedDict):
        names: Sequence[str]
        formats: Sequence[_DTypeLikeNested]

    # Mandatory + optional keys
    class _DTypeDict(_DTypeDictBase, total=False):
        offsets: Sequence[int]
        titles: Sequence[Any]  # Only `str` elements are usable as indexing aliases, but all objects are legal
        itemsize: int
        aligned: bool


    # A protocol for anything with the dtype attribute
    class _SupportsDType(Protocol[_DType_co]):
        @property
        def dtype(self) -> _DType_co: ...

else:
    _DTypeDict = Any

    class _SupportsDType(Generic[_DType_co]):
        pass


# Would create a dtype[np.void]
_VoidDTypeLike = Union[
    # (flexible_dtype, itemsize)
    Tuple[_DTypeLikeNested, int],
    # (fixed_dtype, shape)
    Tuple[_DTypeLikeNested, _ShapeLike],
    # [(field_name, field_dtype, field_shape), ...]
    #
    # The type here is quite broad because NumPy accepts quite a wide
    # range of inputs inside the list; see the tests for some
    # examples.
    List[Any],
    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ...,
    #  'itemsize': ...}
    _DTypeDict,
    # (base_dtype, new_dtype)
    Tuple[_DTypeLikeNested, _DTypeLikeNested],
]

# Anything that can be coerced into numpy.dtype.
# Reference: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
DTypeLike = Union[
    np.dtype,
    # default data type (float64)
    None,
    # array-scalar types and generic types
    type,  # TODO: enumerate these when we add type hints for numpy scalars
    # anything with a dtype attribute
    _SupportsDType[np.dtype],
    # character codes, type strings or comma-separated fields, e.g., 'float64'
    str,
    _VoidDTypeLike,
]

# NOTE: while it is possible to provide the dtype as a dict of
# dtype-like objects (e.g. `{'field1': ..., 'field2': ..., ...}`),
# this syntax is officially discourged and
# therefore not included in the Union defining `DTypeLike`.
#
# See https://github.com/numpy/numpy/issues/16891 for more details.

# Aliases for commonly used dtype-like objects.
# Note that the precision of `np.number` subclasses is ignored herein.
_DTypeLikeBool = Union[
    Type[bool],
    Type[np.bool_],
    "np.dtype[np.bool_]",
    "_SupportsDType[np.dtype[np.bool_]]",
    _BoolCodes,
]
_DTypeLikeUInt = Union[
    Type[np.unsignedinteger],
    "np.dtype[np.unsignedinteger]",
    "_SupportsDType[np.dtype[np.unsignedinteger]]",
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
]
_DTypeLikeInt = Union[
    Type[int],
    Type[np.signedinteger],
    "np.dtype[np.signedinteger]",
    "_SupportsDType[np.dtype[np.signedinteger]]",
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntPCodes,
    _IntCodes,
    _LongLongCodes,
]
_DTypeLikeFloat = Union[
    Type[float],
    Type[np.floating],
    "np.dtype[np.floating]",
    "_SupportsDType[np.dtype[np.floating]]",
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
]
_DTypeLikeComplex = Union[
    Type[complex],
    Type[np.complexfloating],
    "np.dtype[np.complexfloating]",
    "_SupportsDType[np.dtype[np.complexfloating]]",
    _Complex64Codes,
    _Complex128Codes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
]
_DTypeLikeDT64 = Union[
    Type[np.timedelta64],
    "np.dtype[np.timedelta64]",
    "_SupportsDType[np.dtype[np.timedelta64]]",
    _TD64Codes,
]
_DTypeLikeTD64 = Union[
    Type[np.datetime64],
    "np.dtype[np.datetime64]",
    "_SupportsDType[np.dtype[np.datetime64]]",
    _DT64Codes,
]
_DTypeLikeStr = Union[
    Type[str],
    Type[np.str_],
    "np.dtype[np.str_]",
    "_SupportsDType[np.dtype[np.str_]]",
    _StrCodes,
]
_DTypeLikeBytes = Union[
    Type[bytes],
    Type[np.bytes_],
    "np.dtype[np.bytes_]",
    "_SupportsDType[np.dtype[np.bytes_]]",
    _BytesCodes,
]
_DTypeLikeVoid = Union[
    Type[np.void],
    "np.dtype[np.void]",
    "_SupportsDType[np.dtype[np.void]]",
    _VoidCodes,
    _VoidDTypeLike,
]
_DTypeLikeObject = Union[
    type,
    "np.dtype[np.object_]",
    "_SupportsDType[np.dtype[np.object_]]",
    _ObjectCodes,
]

_DTypeLikeComplex_co = Union[
    _DTypeLikeBool,
    _DTypeLikeUInt,
    _DTypeLikeInt,
    _DTypeLikeFloat,
    _DTypeLikeComplex,
]
