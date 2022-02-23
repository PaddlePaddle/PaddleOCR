"""
Helpers for sharing tests between DataFrame/Series
"""

from pandas import DataFrame


def get_dtype(obj):
    if isinstance(obj, DataFrame):
        # Note: we are assuming only one column
        return obj.dtypes.iat[0]
    else:
        return obj.dtype
