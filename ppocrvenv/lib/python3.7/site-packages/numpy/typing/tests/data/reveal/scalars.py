import sys
import numpy as np

b: np.bool_
u8: np.uint64
i8: np.int64
f8: np.float64
c8: np.complex64
c16: np.complex128
m: np.timedelta64
U: np.str_
S: np.bytes_

reveal_type(c8.real)  # E: {float32}
reveal_type(c8.imag)  # E: {float32}

reveal_type(c8.real.real)  # E: {float32}
reveal_type(c8.real.imag)  # E: {float32}

reveal_type(c8.itemsize)  # E: int
reveal_type(c8.shape)  # E: Tuple[]
reveal_type(c8.strides)  # E: Tuple[]

reveal_type(c8.ndim)  # E: Literal[0]
reveal_type(c8.size)  # E: Literal[1]

reveal_type(c8.squeeze())  # E: {complex64}
reveal_type(c8.byteswap())  # E: {complex64}
reveal_type(c8.transpose())  # E: {complex64}

reveal_type(c8.dtype)  # E: numpy.dtype[{complex64}]

reveal_type(c8.real)  # E: {float32}
reveal_type(c16.imag)  # E: {float64}

reveal_type(np.unicode_('foo'))  # E: numpy.str_
reveal_type(np.str0('foo'))  # E: numpy.str_

# Aliases
reveal_type(np.unicode_())  # E: numpy.str_
reveal_type(np.str0())  # E: numpy.str_
reveal_type(np.bool8())  # E: numpy.bool_
reveal_type(np.bytes0())  # E: numpy.bytes_
reveal_type(np.string_())  # E: numpy.bytes_
reveal_type(np.object0())  # E: numpy.object_
reveal_type(np.void0(0))  # E: numpy.void

reveal_type(np.byte())  # E: {byte}
reveal_type(np.short())  # E: {short}
reveal_type(np.intc())  # E: {intc}
reveal_type(np.intp())  # E: {intp}
reveal_type(np.int0())  # E: {intp}
reveal_type(np.int_())  # E: {int_}
reveal_type(np.longlong())  # E: {longlong}

reveal_type(np.ubyte())  # E: {ubyte}
reveal_type(np.ushort())  # E: {ushort}
reveal_type(np.uintc())  # E: {uintc}
reveal_type(np.uintp())  # E: {uintp}
reveal_type(np.uint0())  # E: {uintp}
reveal_type(np.uint())  # E: {uint}
reveal_type(np.ulonglong())  # E: {ulonglong}

reveal_type(np.half())  # E: {half}
reveal_type(np.single())  # E: {single}
reveal_type(np.double())  # E: {double}
reveal_type(np.float_())  # E: {double}
reveal_type(np.longdouble())  # E: {longdouble}
reveal_type(np.longfloat())  # E: {longdouble}

reveal_type(np.csingle())  # E: {csingle}
reveal_type(np.singlecomplex())  # E: {csingle}
reveal_type(np.cdouble())  # E: {cdouble}
reveal_type(np.complex_())  # E: {cdouble}
reveal_type(np.cfloat())  # E: {cdouble}
reveal_type(np.clongdouble())  # E: {clongdouble}
reveal_type(np.clongfloat())  # E: {clongdouble}
reveal_type(np.longcomplex())  # E: {clongdouble}

reveal_type(b.item())  # E: bool
reveal_type(i8.item())  # E: int
reveal_type(u8.item())  # E: int
reveal_type(f8.item())  # E: float
reveal_type(c16.item())  # E: complex
reveal_type(U.item())  # E: str
reveal_type(S.item())  # E: bytes

reveal_type(b.tolist())  # E: bool
reveal_type(i8.tolist())  # E: int
reveal_type(u8.tolist())  # E: int
reveal_type(f8.tolist())  # E: float
reveal_type(c16.tolist())  # E: complex
reveal_type(U.tolist())  # E: str
reveal_type(S.tolist())  # E: bytes

reveal_type(b.ravel())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(i8.ravel())  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(u8.ravel())  # E: numpy.ndarray[Any, numpy.dtype[{uint64}]]
reveal_type(f8.ravel())  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(c16.ravel())  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(U.ravel())  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
reveal_type(S.ravel())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(b.flatten())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(i8.flatten())  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(u8.flatten())  # E: numpy.ndarray[Any, numpy.dtype[{uint64}]]
reveal_type(f8.flatten())  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(c16.flatten())  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(U.flatten())  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
reveal_type(S.flatten())  # E: numpy.ndarray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(b.reshape(1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(i8.reshape(1))  # E: numpy.ndarray[Any, numpy.dtype[{int64}]]
reveal_type(u8.reshape(1))  # E: numpy.ndarray[Any, numpy.dtype[{uint64}]]
reveal_type(f8.reshape(1))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(c16.reshape(1))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(U.reshape(1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
reveal_type(S.reshape(1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bytes_]]

reveal_type(f8.as_integer_ratio())  # E: Tuple[builtins.int, builtins.int]
reveal_type(f8.is_integer())  # E: bool
reveal_type(f8.__trunc__())  # E: int
reveal_type(f8.__getformat__("float"))  # E: str
reveal_type(f8.hex())  # E: str
reveal_type(np.float64.fromhex("0x0.0p+0"))  # E: {float64}

reveal_type(f8.__getnewargs__())  # E: Tuple[builtins.float]
reveal_type(c16.__getnewargs__())  # E: Tuple[builtins.float, builtins.float]

reveal_type(i8.numerator)  # E: {int64}
reveal_type(i8.denominator)  # E: Literal[1]
reveal_type(u8.numerator)  # E: {uint64}
reveal_type(u8.denominator)  # E: Literal[1]
reveal_type(m.numerator)  # E: numpy.timedelta64
reveal_type(m.denominator)  # E: Literal[1]

reveal_type(round(i8))  # E: int
reveal_type(round(i8, 3))  # E: {int64}
reveal_type(round(u8))  # E: int
reveal_type(round(u8, 3))  # E: {uint64}
reveal_type(round(f8))  # E: int
reveal_type(round(f8, 3))  # E: {float64}

if sys.version_info >= (3, 9):
    reveal_type(f8.__ceil__())  # E: int
    reveal_type(f8.__floor__())  # E: int

reveal_type(i8.astype(float))  # E: Any
reveal_type(i8.astype(np.float64))  # E: {float64}

reveal_type(i8.view())  # E: {int64}
reveal_type(i8.view(np.float64))  # E: {float64}
reveal_type(i8.view(float))  # E: Any
reveal_type(i8.view(np.float64, np.ndarray))  # E: {float64}

reveal_type(i8.getfield(float))  # E: Any
reveal_type(i8.getfield(np.float64))  # E: {float64}
reveal_type(i8.getfield(np.float64, 8))  # E: {float64}
