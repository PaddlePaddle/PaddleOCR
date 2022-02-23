import numpy as np

c16 = np.complex128()
f8 = np.float64()
i8 = np.int64()
u8 = np.uint64()

c8 = np.complex64()
f4 = np.float32()
i4 = np.int32()
u4 = np.uint32()

dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

b_ = np.bool_()

b = bool()
c = complex()
f = float()
i = int()

AR = np.array([0], dtype=np.int64)
AR.setflags(write=False)

SEQ = (0, 1, 2, 3, 4)

# Time structures

reveal_type(dt > dt)  # E: numpy.bool_

reveal_type(td > td)  # E: numpy.bool_
reveal_type(td > i)  # E: numpy.bool_
reveal_type(td > i4)  # E: numpy.bool_
reveal_type(td > i8)  # E: numpy.bool_

reveal_type(td > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(td > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(AR > td)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > td)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

# boolean

reveal_type(b_ > b)  # E: numpy.bool_
reveal_type(b_ > b_)  # E: numpy.bool_
reveal_type(b_ > i)  # E: numpy.bool_
reveal_type(b_ > i8)  # E: numpy.bool_
reveal_type(b_ > i4)  # E: numpy.bool_
reveal_type(b_ > u8)  # E: numpy.bool_
reveal_type(b_ > u4)  # E: numpy.bool_
reveal_type(b_ > f)  # E: numpy.bool_
reveal_type(b_ > f8)  # E: numpy.bool_
reveal_type(b_ > f4)  # E: numpy.bool_
reveal_type(b_ > c)  # E: numpy.bool_
reveal_type(b_ > c16)  # E: numpy.bool_
reveal_type(b_ > c8)  # E: numpy.bool_
reveal_type(b_ > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(b_ > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

# Complex

reveal_type(c16 > c16)  # E: numpy.bool_
reveal_type(c16 > f8)  # E: numpy.bool_
reveal_type(c16 > i8)  # E: numpy.bool_
reveal_type(c16 > c8)  # E: numpy.bool_
reveal_type(c16 > f4)  # E: numpy.bool_
reveal_type(c16 > i4)  # E: numpy.bool_
reveal_type(c16 > b_)  # E: numpy.bool_
reveal_type(c16 > b)  # E: numpy.bool_
reveal_type(c16 > c)  # E: numpy.bool_
reveal_type(c16 > f)  # E: numpy.bool_
reveal_type(c16 > i)  # E: numpy.bool_
reveal_type(c16 > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(c16 > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(c16 > c16)  # E: numpy.bool_
reveal_type(f8 > c16)  # E: numpy.bool_
reveal_type(i8 > c16)  # E: numpy.bool_
reveal_type(c8 > c16)  # E: numpy.bool_
reveal_type(f4 > c16)  # E: numpy.bool_
reveal_type(i4 > c16)  # E: numpy.bool_
reveal_type(b_ > c16)  # E: numpy.bool_
reveal_type(b > c16)  # E: numpy.bool_
reveal_type(c > c16)  # E: numpy.bool_
reveal_type(f > c16)  # E: numpy.bool_
reveal_type(i > c16)  # E: numpy.bool_
reveal_type(AR > c16)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > c16)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(c8 > c16)  # E: numpy.bool_
reveal_type(c8 > f8)  # E: numpy.bool_
reveal_type(c8 > i8)  # E: numpy.bool_
reveal_type(c8 > c8)  # E: numpy.bool_
reveal_type(c8 > f4)  # E: numpy.bool_
reveal_type(c8 > i4)  # E: numpy.bool_
reveal_type(c8 > b_)  # E: numpy.bool_
reveal_type(c8 > b)  # E: numpy.bool_
reveal_type(c8 > c)  # E: numpy.bool_
reveal_type(c8 > f)  # E: numpy.bool_
reveal_type(c8 > i)  # E: numpy.bool_
reveal_type(c8 > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(c8 > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(c16 > c8)  # E: numpy.bool_
reveal_type(f8 > c8)  # E: numpy.bool_
reveal_type(i8 > c8)  # E: numpy.bool_
reveal_type(c8 > c8)  # E: numpy.bool_
reveal_type(f4 > c8)  # E: numpy.bool_
reveal_type(i4 > c8)  # E: numpy.bool_
reveal_type(b_ > c8)  # E: numpy.bool_
reveal_type(b > c8)  # E: numpy.bool_
reveal_type(c > c8)  # E: numpy.bool_
reveal_type(f > c8)  # E: numpy.bool_
reveal_type(i > c8)  # E: numpy.bool_
reveal_type(AR > c8)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > c8)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

# Float

reveal_type(f8 > f8)  # E: numpy.bool_
reveal_type(f8 > i8)  # E: numpy.bool_
reveal_type(f8 > f4)  # E: numpy.bool_
reveal_type(f8 > i4)  # E: numpy.bool_
reveal_type(f8 > b_)  # E: numpy.bool_
reveal_type(f8 > b)  # E: numpy.bool_
reveal_type(f8 > c)  # E: numpy.bool_
reveal_type(f8 > f)  # E: numpy.bool_
reveal_type(f8 > i)  # E: numpy.bool_
reveal_type(f8 > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(f8 > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(f8 > f8)  # E: numpy.bool_
reveal_type(i8 > f8)  # E: numpy.bool_
reveal_type(f4 > f8)  # E: numpy.bool_
reveal_type(i4 > f8)  # E: numpy.bool_
reveal_type(b_ > f8)  # E: numpy.bool_
reveal_type(b > f8)  # E: numpy.bool_
reveal_type(c > f8)  # E: numpy.bool_
reveal_type(f > f8)  # E: numpy.bool_
reveal_type(i > f8)  # E: numpy.bool_
reveal_type(AR > f8)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > f8)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(f4 > f8)  # E: numpy.bool_
reveal_type(f4 > i8)  # E: numpy.bool_
reveal_type(f4 > f4)  # E: numpy.bool_
reveal_type(f4 > i4)  # E: numpy.bool_
reveal_type(f4 > b_)  # E: numpy.bool_
reveal_type(f4 > b)  # E: numpy.bool_
reveal_type(f4 > c)  # E: numpy.bool_
reveal_type(f4 > f)  # E: numpy.bool_
reveal_type(f4 > i)  # E: numpy.bool_
reveal_type(f4 > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(f4 > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(f8 > f4)  # E: numpy.bool_
reveal_type(i8 > f4)  # E: numpy.bool_
reveal_type(f4 > f4)  # E: numpy.bool_
reveal_type(i4 > f4)  # E: numpy.bool_
reveal_type(b_ > f4)  # E: numpy.bool_
reveal_type(b > f4)  # E: numpy.bool_
reveal_type(c > f4)  # E: numpy.bool_
reveal_type(f > f4)  # E: numpy.bool_
reveal_type(i > f4)  # E: numpy.bool_
reveal_type(AR > f4)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > f4)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

# Int

reveal_type(i8 > i8)  # E: numpy.bool_
reveal_type(i8 > u8)  # E: numpy.bool_
reveal_type(i8 > i4)  # E: numpy.bool_
reveal_type(i8 > u4)  # E: numpy.bool_
reveal_type(i8 > b_)  # E: numpy.bool_
reveal_type(i8 > b)  # E: numpy.bool_
reveal_type(i8 > c)  # E: numpy.bool_
reveal_type(i8 > f)  # E: numpy.bool_
reveal_type(i8 > i)  # E: numpy.bool_
reveal_type(i8 > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(i8 > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(u8 > u8)  # E: numpy.bool_
reveal_type(u8 > i4)  # E: numpy.bool_
reveal_type(u8 > u4)  # E: numpy.bool_
reveal_type(u8 > b_)  # E: numpy.bool_
reveal_type(u8 > b)  # E: numpy.bool_
reveal_type(u8 > c)  # E: numpy.bool_
reveal_type(u8 > f)  # E: numpy.bool_
reveal_type(u8 > i)  # E: numpy.bool_
reveal_type(u8 > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(u8 > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(i8 > i8)  # E: numpy.bool_
reveal_type(u8 > i8)  # E: numpy.bool_
reveal_type(i4 > i8)  # E: numpy.bool_
reveal_type(u4 > i8)  # E: numpy.bool_
reveal_type(b_ > i8)  # E: numpy.bool_
reveal_type(b > i8)  # E: numpy.bool_
reveal_type(c > i8)  # E: numpy.bool_
reveal_type(f > i8)  # E: numpy.bool_
reveal_type(i > i8)  # E: numpy.bool_
reveal_type(AR > i8)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > i8)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(u8 > u8)  # E: numpy.bool_
reveal_type(i4 > u8)  # E: numpy.bool_
reveal_type(u4 > u8)  # E: numpy.bool_
reveal_type(b_ > u8)  # E: numpy.bool_
reveal_type(b > u8)  # E: numpy.bool_
reveal_type(c > u8)  # E: numpy.bool_
reveal_type(f > u8)  # E: numpy.bool_
reveal_type(i > u8)  # E: numpy.bool_
reveal_type(AR > u8)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > u8)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(i4 > i8)  # E: numpy.bool_
reveal_type(i4 > i4)  # E: numpy.bool_
reveal_type(i4 > i)  # E: numpy.bool_
reveal_type(i4 > b_)  # E: numpy.bool_
reveal_type(i4 > b)  # E: numpy.bool_
reveal_type(i4 > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(i4 > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(u4 > i8)  # E: numpy.bool_
reveal_type(u4 > i4)  # E: numpy.bool_
reveal_type(u4 > u8)  # E: numpy.bool_
reveal_type(u4 > u4)  # E: numpy.bool_
reveal_type(u4 > i)  # E: numpy.bool_
reveal_type(u4 > b_)  # E: numpy.bool_
reveal_type(u4 > b)  # E: numpy.bool_
reveal_type(u4 > AR)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(u4 > SEQ)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(i8 > i4)  # E: numpy.bool_
reveal_type(i4 > i4)  # E: numpy.bool_
reveal_type(i > i4)  # E: numpy.bool_
reveal_type(b_ > i4)  # E: numpy.bool_
reveal_type(b > i4)  # E: numpy.bool_
reveal_type(AR > i4)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > i4)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

reveal_type(i8 > u4)  # E: numpy.bool_
reveal_type(i4 > u4)  # E: numpy.bool_
reveal_type(u8 > u4)  # E: numpy.bool_
reveal_type(u4 > u4)  # E: numpy.bool_
reveal_type(b_ > u4)  # E: numpy.bool_
reveal_type(b > u4)  # E: numpy.bool_
reveal_type(i > u4)  # E: numpy.bool_
reveal_type(AR > u4)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(SEQ > u4)  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
