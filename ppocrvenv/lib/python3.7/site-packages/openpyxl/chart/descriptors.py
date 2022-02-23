# Copyright (c) 2010-2021 openpyxl



from openpyxl.descriptors.nested import (
    NestedMinMax
    )

from openpyxl.descriptors import Typed

from .data_source import NumFmt

"""
Utility descriptors for the chart module.
For convenience but also clarity.
"""

class NestedGapAmount(NestedMinMax):

    allow_none = True
    min = 0
    max = 500


class NestedOverlap(NestedMinMax):

    allow_none = True
    min = -100
    max = 100


class NumberFormatDescriptor(Typed):
    """
    Allow direct assignment of format code
    """

    expected_type = NumFmt
    allow_none = True

    def __set__(self, instance, value):
        if isinstance(value, str):
            value = NumFmt(value)
        super(NumberFormatDescriptor, self).__set__(instance, value)
