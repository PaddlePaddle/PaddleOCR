# Copyright (c) 2010-2021 openpyxl

"""
math.prod equivalent for < Python 3.8
"""

import functools
import operator

def product(sequence):
    return functools.reduce(operator.mul, sequence)


try:
    from math import prod
except ImportError:
    prod = product
