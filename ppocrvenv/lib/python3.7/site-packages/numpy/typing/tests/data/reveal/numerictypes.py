import numpy as np

reveal_type(np.issctype(np.generic))  # E: bool
reveal_type(np.issctype("foo"))  # E: bool

reveal_type(np.obj2sctype("S8"))  # E: Union[numpy.generic, None]
reveal_type(np.obj2sctype("S8", default=None))  # E: Union[numpy.generic, None]
reveal_type(
    np.obj2sctype("foo", default=int)  # E: Union[numpy.generic, Type[builtins.int*]]
)

reveal_type(np.issubclass_(np.float64, float))  # E: bool
reveal_type(np.issubclass_(np.float64, (int, float)))  # E: bool

reveal_type(np.sctype2char("S8"))  # E: str
reveal_type(np.sctype2char(list))  # E: str

reveal_type(np.find_common_type([np.int64], [np.int64]))  # E: numpy.dtype

reveal_type(np.cast[int])  # E: _CastFunc
reveal_type(np.cast["i8"])  # E: _CastFunc
reveal_type(np.cast[np.int64])  # E: _CastFunc

reveal_type(np.nbytes[int])  # E: int
reveal_type(np.nbytes["i8"])  # E: int
reveal_type(np.nbytes[np.int64])  # E: int

reveal_type(np.ScalarType)  # E: Tuple
reveal_type(np.ScalarType[0])  # E: Type[builtins.int]
reveal_type(np.ScalarType[4])  # E: Type[builtins.bool]
reveal_type(np.ScalarType[9])  # E: Type[{csingle}]
reveal_type(np.ScalarType[11])  # E: Type[{clongdouble}]

reveal_type(np.typecodes["Character"])  # E: Literal['c']
reveal_type(np.typecodes["Complex"])  # E: Literal['FDG']
reveal_type(np.typecodes["All"])  # E: Literal['?bhilqpBHILQPefdgFDGSUVOMm']
