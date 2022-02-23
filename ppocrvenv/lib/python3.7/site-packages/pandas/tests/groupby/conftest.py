import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
)
import pandas._testing as tm
from pandas.core.groupby.base import (
    reduction_kernels,
    transformation_kernels,
)


@pytest.fixture(params=[True, False])
def as_index(request):
    return request.param


@pytest.fixture
def mframe():
    index = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    return DataFrame(np.random.randn(10, 3), index=index, columns=["A", "B", "C"])


@pytest.fixture
def df():
    return DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.randn(8),
            "D": np.random.randn(8),
        }
    )


@pytest.fixture
def ts():
    return tm.makeTimeSeries()


@pytest.fixture
def tsd():
    return tm.getTimeSeriesData()


@pytest.fixture
def tsframe(tsd):
    return DataFrame(tsd)


@pytest.fixture
def df_mixed_floats():
    return DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.randn(8),
            "D": np.array(np.random.randn(8), dtype="float32"),
        }
    )


@pytest.fixture
def three_group():
    return DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.randn(11),
            "E": np.random.randn(11),
            "F": np.random.randn(11),
        }
    )


@pytest.fixture(params=sorted(reduction_kernels))
def reduction_func(request):
    """
    yields the string names of all groupby reduction functions, one at a time.
    """
    return request.param


@pytest.fixture(params=sorted(transformation_kernels))
def transformation_func(request):
    """yields the string names of all groupby transformation functions."""
    return request.param


@pytest.fixture(params=sorted(reduction_kernels) + sorted(transformation_kernels))
def groupby_func(request):
    """yields both aggregation and transformation functions."""
    return request.param


@pytest.fixture(params=[True, False])
def parallel(request):
    """parallel keyword argument for numba.jit"""
    return request.param


# Can parameterize nogil & nopython over True | False, but limiting per
# https://github.com/pandas-dev/pandas/pull/41971#issuecomment-860607472


@pytest.fixture(params=[False])
def nogil(request):
    """nogil keyword argument for numba.jit"""
    return request.param


@pytest.fixture(params=[True])
def nopython(request):
    """nopython keyword argument for numba.jit"""
    return request.param
