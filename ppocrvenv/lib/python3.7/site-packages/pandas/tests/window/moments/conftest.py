from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    bdate_range,
    notna,
)


@pytest.fixture
def series():
    """Make mocked series as fixture."""
    arr = np.random.randn(100)
    locs = np.arange(20, 40)
    arr[locs] = np.NaN
    series = Series(arr, index=bdate_range(datetime(2009, 1, 1), periods=100))
    return series


@pytest.fixture
def frame():
    """Make mocked frame as fixture."""
    return DataFrame(
        np.random.randn(100, 10),
        index=bdate_range(datetime(2009, 1, 1), periods=100),
        columns=np.arange(10),
    )


# create the data only once as we are not setting it
def _create_consistency_data():
    def create_series():
        return [
            Series(dtype=object),
            Series([np.nan]),
            Series([np.nan, np.nan]),
            Series([3.0]),
            Series([np.nan, 3.0]),
            Series([3.0, np.nan]),
            Series([1.0, 3.0]),
            Series([2.0, 2.0]),
            Series([3.0, 1.0]),
            Series(
                [5.0, 5.0, 5.0, 5.0, np.nan, np.nan, np.nan, 5.0, 5.0, np.nan, np.nan]
            ),
            Series(
                [
                    np.nan,
                    5.0,
                    5.0,
                    5.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    5.0,
                    5.0,
                    np.nan,
                    np.nan,
                ]
            ),
            Series(
                [
                    np.nan,
                    np.nan,
                    5.0,
                    5.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    5.0,
                    5.0,
                    np.nan,
                    np.nan,
                ]
            ),
            Series(
                [
                    np.nan,
                    3.0,
                    np.nan,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    np.nan,
                    np.nan,
                    7.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ]
            ),
            Series(
                [
                    np.nan,
                    5.0,
                    np.nan,
                    2.0,
                    4.0,
                    0.0,
                    9.0,
                    np.nan,
                    np.nan,
                    3.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ]
            ),
            Series(
                [
                    2.0,
                    3.0,
                    np.nan,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    np.nan,
                    np.nan,
                    7.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ]
            ),
            Series(
                [
                    2.0,
                    5.0,
                    np.nan,
                    2.0,
                    4.0,
                    0.0,
                    9.0,
                    np.nan,
                    np.nan,
                    3.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ]
            ),
            Series(range(10)),
            Series(range(20, 0, -2)),
        ]

    def create_dataframes():
        return [
            DataFrame(),
            DataFrame(columns=["a"]),
            DataFrame(columns=["a", "a"]),
            DataFrame(columns=["a", "b"]),
            DataFrame(np.arange(10).reshape((5, 2))),
            DataFrame(np.arange(25).reshape((5, 5))),
            DataFrame(np.arange(25).reshape((5, 5)), columns=["a", "b", 99, "d", "d"]),
        ] + [DataFrame(s) for s in create_series()]

    def is_constant(x):
        values = x.values.ravel("K")
        return len(set(values[notna(values)])) == 1

    def no_nans(x):
        return x.notna().all().all()

    # data is a tuple(object, is_constant, no_nans)
    data = create_series() + create_dataframes()

    return [(x, is_constant(x), no_nans(x)) for x in data]


@pytest.fixture(params=_create_consistency_data())
def consistency_data(request):
    """Create consistency data"""
    return request.param
