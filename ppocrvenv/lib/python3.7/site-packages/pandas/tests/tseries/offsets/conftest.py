import pytest

from pandas._libs.tslibs.offsets import MonthOffset

import pandas.tseries.offsets as offsets


@pytest.fixture(params=[getattr(offsets, o) for o in offsets.__all__])
def offset_types(request):
    """
    Fixture for all the datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture(
    params=[
        getattr(offsets, o)
        for o in offsets.__all__
        if issubclass(getattr(offsets, o), MonthOffset) and o != "MonthOffset"
    ]
)
def month_classes(request):
    """
    Fixture for month based datetime offsets available for a time series.
    """
    return request.param
