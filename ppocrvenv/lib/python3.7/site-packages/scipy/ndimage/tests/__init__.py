
from __future__ import annotations
from typing import List, Type 
import numpy

# list of numarray data types
integer_types: List[Type] = [
    numpy.int8, numpy.uint8, numpy.int16, numpy.uint16,
    numpy.int32, numpy.uint32, numpy.int64, numpy.uint64]

float_types: List[Type] = [numpy.float32, numpy.float64]

complex_types: List[Type] = [numpy.complex64, numpy.complex128]

types: List[Type] = integer_types + float_types
