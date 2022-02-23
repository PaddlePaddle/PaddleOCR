"""
This file is very long and growing, but it was decided to not split it yet, as
it's still manageable (2020-03-17, ~1.1k LoC). See gh-31989

Instead of splitting it was decided to define sections here:
- Configuration / Settings
- Autouse fixtures
- Common arguments
- Missing values & co.
- Classes
- Indices
- Series'
- DataFrames
- Operators & Operations
- Data sets/files
- Time zones
- Dtypes
- Misc
"""

from collections import abc
from datetime import (
    date,
    datetime,
    time,
    timedelta,
    timezone,
)
from decimal import Decimal
import operator
import os

from dateutil.tz import (
    tzlocal,
    tzutc,
)
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
from pytz import (
    FixedOffset,
    utc,
)

import pandas.util._test_decorators as td

from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
)

import pandas as pd
from pandas import (
    DataFrame,
    Interval,
    Period,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
)

# Until https://github.com/numpy/numpy/issues/19078 is sorted out, just suppress
suppress_npdev_promotion_warning = pytest.mark.filterwarnings(
    "ignore:Promotion of numbers and bools:FutureWarning"
)

# ----------------------------------------------------------------
# Configuration / Settings
# ----------------------------------------------------------------
# pytest


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true", help="skip slow tests")
    parser.addoption("--skip-network", action="store_true", help="skip network tests")
    parser.addoption("--skip-db", action="store_true", help="skip db tests")
    parser.addoption(
        "--run-high-memory", action="store_true", help="run high memory tests"
    )
    parser.addoption("--only-slow", action="store_true", help="run only slow tests")
    parser.addoption(
        "--strict-data-files",
        action="store_true",
        help="Fail if a test is skipped for missing data file.",
    )


def pytest_runtest_setup(item):
    if "slow" in item.keywords and item.config.getoption("--skip-slow"):
        pytest.skip("skipping due to --skip-slow")

    if "slow" not in item.keywords and item.config.getoption("--only-slow"):
        pytest.skip("skipping due to --only-slow")

    if "network" in item.keywords and item.config.getoption("--skip-network"):
        pytest.skip("skipping due to --skip-network")

    if "db" in item.keywords and item.config.getoption("--skip-db"):
        pytest.skip("skipping due to --skip-db")

    if "high_memory" in item.keywords and not item.config.getoption(
        "--run-high-memory"
    ):
        pytest.skip("skipping high memory test since --run-high-memory was not set")


def pytest_collection_modifyitems(items):
    for item in items:
        # mark all tests in the pandas/tests/frame directory with "arraymanager"
        if "/frame/" in item.nodeid:
            item.add_marker(pytest.mark.arraymanager)

        item.add_marker(suppress_npdev_promotion_warning)


# Hypothesis
hypothesis.settings.register_profile(
    "ci",
    # Hypothesis timing checks are tuned for scalars by default, so we bump
    # them from 200ms to 500ms per test case as the global default.  If this
    # is too short for a specific test, (a) try to make it faster, and (b)
    # if it really is slow add `@settings(deadline=...)` with a working value,
    # or `deadline=None` to entirely disable timeouts for that test.
    deadline=500,
    suppress_health_check=(hypothesis.HealthCheck.too_slow,),
)
hypothesis.settings.load_profile("ci")

# Registering these strategies makes them globally available via st.from_type,
# which is use for offsets in tests/tseries/offsets/test_offsets_properties.py
for name in "MonthBegin MonthEnd BMonthBegin BMonthEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls, st.builds(cls, n=st.integers(-99, 99), normalize=st.booleans())
    )

for name in "YearBegin YearEnd BYearBegin BYearEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls,
        st.builds(
            cls,
            n=st.integers(-5, 5),
            normalize=st.booleans(),
            month=st.integers(min_value=1, max_value=12),
        ),
    )

for name in "QuarterBegin QuarterEnd BQuarterBegin BQuarterEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls,
        st.builds(
            cls,
            n=st.integers(-24, 24),
            normalize=st.booleans(),
            startingMonth=st.integers(min_value=1, max_value=12),
        ),
    )


# ----------------------------------------------------------------
# Autouse fixtures
# ----------------------------------------------------------------
@pytest.fixture(autouse=True)
def configure_tests():
    """
    Configure settings for all tests and test modules.
    """
    pd.set_option("chained_assignment", "raise")


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    """
    Make `np` and `pd` names available for doctests.
    """
    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd


# ----------------------------------------------------------------
# Common arguments
# ----------------------------------------------------------------
@pytest.fixture(params=[0, 1, "index", "columns"], ids=lambda x: f"axis={repr(x)}")
def axis(request):
    """
    Fixture for returning the axis numbers of a DataFrame.
    """
    return request.param


axis_frame = axis


@pytest.fixture(params=[True, False, None])
def observed(request):
    """
    Pass in the observed keyword to groupby for [True, False]
    This indicates whether categoricals should return values for
    values which are not in the grouper [False / None], or only values which
    appear in the grouper [True]. [None] is supported for future compatibility
    if we decide to change the default (and would need to warn if this
    parameter is not passed).
    """
    return request.param


@pytest.fixture(params=[True, False, None])
def ordered(request):
    """
    Boolean 'ordered' parameter for Categorical.
    """
    return request.param


@pytest.fixture(params=["first", "last", False])
def keep(request):
    """
    Valid values for the 'keep' parameter used in
    .duplicated or .drop_duplicates
    """
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def closed(request):
    """
    Fixture for trying all interval closed parameters.
    """
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def other_closed(request):
    """
    Secondary closed fixture to allow parametrizing over all pairs of closed.
    """
    return request.param


@pytest.fixture(params=[None, "gzip", "bz2", "zip", "xz"])
def compression(request):
    """
    Fixture for trying common compression types in compression tests.
    """
    return request.param


@pytest.fixture(params=["gzip", "bz2", "zip", "xz"])
def compression_only(request):
    """
    Fixture for trying common compression types in compression tests excluding
    uncompressed case.
    """
    return request.param


@pytest.fixture(params=[True, False])
def writable(request):
    """
    Fixture that an array is writable.
    """
    return request.param


@pytest.fixture(params=["inner", "outer", "left", "right"])
def join_type(request):
    """
    Fixture for trying all types of join operations.
    """
    return request.param


@pytest.fixture(params=["nlargest", "nsmallest"])
def nselect_method(request):
    """
    Fixture for trying all nselect methods.
    """
    return request.param


# ----------------------------------------------------------------
# Missing values & co.
# ----------------------------------------------------------------
@pytest.fixture(params=tm.NULL_OBJECTS, ids=lambda x: type(x).__name__)
def nulls_fixture(request):
    """
    Fixture for each null type in pandas.
    """
    return request.param


nulls_fixture2 = nulls_fixture  # Generate cartesian product of nulls_fixture


@pytest.fixture(params=[None, np.nan, pd.NaT])
def unique_nulls_fixture(request):
    """
    Fixture for each null type in pandas, each null type exactly once.
    """
    return request.param


# Generate cartesian product of unique_nulls_fixture:
unique_nulls_fixture2 = unique_nulls_fixture

# ----------------------------------------------------------------
# Classes
# ----------------------------------------------------------------


@pytest.fixture(params=[DataFrame, Series])
def frame_or_series(request):
    """
    Fixture to parametrize over DataFrame and Series.
    """
    return request.param


# error: List item 0 has incompatible type "Type[Index]"; expected "Type[IndexOpsMixin]"
@pytest.fixture(
    params=[Index, Series], ids=["index", "series"]  # type: ignore[list-item]
)
def index_or_series(request):
    """
    Fixture to parametrize over Index and Series, made necessary by a mypy
    bug, giving an error:

    List item 0 has incompatible type "Type[Series]"; expected "Type[PandasObject]"

    See GH#29725
    """
    return request.param


# Generate cartesian product of index_or_series fixture:
index_or_series2 = index_or_series


@pytest.fixture(params=[Index, Series, pd.array], ids=["index", "series", "array"])
def index_or_series_or_array(request):
    """
    Fixture to parametrize over Index, Series, and ExtensionArray
    """
    return request.param


@pytest.fixture
def dict_subclass():
    """
    Fixture for a dictionary subclass.
    """

    class TestSubDict(dict):
        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)

    return TestSubDict


@pytest.fixture
def non_dict_mapping_subclass():
    """
    Fixture for a non-mapping dictionary subclass.
    """

    class TestNonDictMapping(abc.Mapping):
        def __init__(self, underlying_dict):
            self._data = underlying_dict

        def __getitem__(self, key):
            return self._data.__getitem__(key)

        def __iter__(self):
            return self._data.__iter__()

        def __len__(self):
            return self._data.__len__()

    return TestNonDictMapping


# ----------------------------------------------------------------
# Indices
# ----------------------------------------------------------------
@pytest.fixture
def multiindex_year_month_day_dataframe_random_data():
    """
    DataFrame with 3 level MultiIndex (year, month, day) covering
    first 100 business days from 2000-01-01 with random data
    """
    tdf = tm.makeTimeDataFrame(100)
    ymd = tdf.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day]).sum()
    # use Int64Index, to make sure things work
    ymd.index = ymd.index.set_levels([lev.astype("i8") for lev in ymd.index.levels])
    ymd.index.set_names(["year", "month", "day"], inplace=True)
    return ymd


@pytest.fixture
def multiindex_dataframe_random_data():
    """DataFrame with 2 level MultiIndex with random data"""
    index = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    return DataFrame(
        np.random.randn(10, 3), index=index, columns=Index(["A", "B", "C"], name="exp")
    )


def _create_multiindex():
    """
    MultiIndex used to test the general functionality of this object
    """

    # See Also: tests.multi.conftest.idx
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ["first", "second"]
    return MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )


def _create_mi_with_dt64tz_level():
    """
    MultiIndex with a level that is a tzaware DatetimeIndex.
    """
    # GH#8367 round trip with pickle
    return MultiIndex.from_product(
        [[1, 2], ["a", "b"], pd.date_range("20130101", periods=3, tz="US/Eastern")],
        names=["one", "two", "three"],
    )


indices_dict = {
    "unicode": tm.makeUnicodeIndex(100),
    "string": tm.makeStringIndex(100),
    "datetime": tm.makeDateIndex(100),
    "datetime-tz": tm.makeDateIndex(100, tz="US/Pacific"),
    "period": tm.makePeriodIndex(100),
    "timedelta": tm.makeTimedeltaIndex(100),
    "int": tm.makeIntIndex(100),
    "uint": tm.makeUIntIndex(100),
    "range": tm.makeRangeIndex(100),
    "float": tm.makeFloatIndex(100),
    "bool": tm.makeBoolIndex(10),
    "categorical": tm.makeCategoricalIndex(100),
    "interval": tm.makeIntervalIndex(100),
    "empty": Index([]),
    "tuples": MultiIndex.from_tuples(zip(["foo", "bar", "baz"], [1, 2, 3])),
    "mi-with-dt64tz-level": _create_mi_with_dt64tz_level(),
    "multi": _create_multiindex(),
    "repeats": Index([0, 0, 1, 1, 2, 2]),
}


@pytest.fixture(params=indices_dict.keys())
def index(request):
    """
    Fixture for many "simple" kinds of indices.

    These indices are unlikely to cover corner cases, e.g.
        - no names
        - no NaTs/NaNs
        - no values near implementation bounds
        - ...
    """
    # copy to avoid mutation, e.g. setting .name
    return indices_dict[request.param].copy()


# Needed to generate cartesian product of indices
index_fixture2 = index


@pytest.fixture(
    params=[
        key for key in indices_dict if not isinstance(indices_dict[key], MultiIndex)
    ]
)
def index_flat(request):
    """
    index fixture, but excluding MultiIndex cases.
    """
    key = request.param
    return indices_dict[key].copy()


# Alias so we can test with cartesian product of index_flat
index_flat2 = index_flat


@pytest.fixture(
    params=[
        key
        for key in indices_dict
        if key not in ["int", "uint", "range", "empty", "repeats"]
        and not isinstance(indices_dict[key], MultiIndex)
    ]
)
def index_with_missing(request):
    """
    Fixture for indices with missing values.

    Integer-dtype and empty cases are excluded because they cannot hold missing
    values.

    MultiIndex is excluded because isna() is not defined for MultiIndex.
    """

    # GH 35538. Use deep copy to avoid illusive bug on np-dev
    # Azure pipeline that writes into indices_dict despite copy
    ind = indices_dict[request.param].copy(deep=True)
    vals = ind.values
    if request.param in ["tuples", "mi-with-dt64tz-level", "multi"]:
        # For setting missing values in the top level of MultiIndex
        vals = ind.tolist()
        vals[0] = (None,) + vals[0][1:]
        vals[-1] = (None,) + vals[-1][1:]
        return MultiIndex.from_tuples(vals)
    else:
        vals[0] = None
        vals[-1] = None
        return type(ind)(vals)


# ----------------------------------------------------------------
# Series'
# ----------------------------------------------------------------
@pytest.fixture
def empty_series():
    return Series([], index=[], dtype=np.float64)


@pytest.fixture
def string_series():
    """
    Fixture for Series of floats with Index of unique strings
    """
    s = tm.makeStringSeries()
    s.name = "series"
    return s


@pytest.fixture
def object_series():
    """
    Fixture for Series of dtype object with Index of unique strings
    """
    s = tm.makeObjectSeries()
    s.name = "objects"
    return s


@pytest.fixture
def datetime_series():
    """
    Fixture for Series of floats with DatetimeIndex
    """
    s = tm.makeTimeSeries()
    s.name = "ts"
    return s


def _create_series(index):
    """Helper for the _series dict"""
    size = len(index)
    data = np.random.randn(size)
    return Series(data, index=index, name="a")


_series = {
    f"series-with-{index_id}-index": _create_series(index)
    for index_id, index in indices_dict.items()
}


@pytest.fixture
def series_with_simple_index(index):
    """
    Fixture for tests on series with changing types of indices.
    """
    return _create_series(index)


@pytest.fixture
def series_with_multilevel_index():
    """
    Fixture with a Series with a 2-level MultiIndex.
    """
    arrays = [
        ["bar", "bar", "baz", "baz", "qux", "qux", "foo", "foo"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    ]
    tuples = zip(*arrays)
    index = MultiIndex.from_tuples(tuples)
    data = np.random.randn(8)
    ser = Series(data, index=index)
    ser[3] = np.NaN
    return ser


_narrow_dtypes = [
    np.float16,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.uint8,
    np.uint16,
    np.uint32,
]
_narrow_series = {
    f"{dtype.__name__}-series": tm.makeFloatSeries(name="a").astype(dtype)
    for dtype in _narrow_dtypes
}


@pytest.fixture(params=_narrow_series.keys())
def narrow_series(request):
    """
    Fixture for Series with low precision data types
    """
    # copy to avoid mutation, e.g. setting .name
    return _narrow_series[request.param].copy()


_index_or_series_objs = {**indices_dict, **_series, **_narrow_series}


@pytest.fixture(params=_index_or_series_objs.keys())
def index_or_series_obj(request):
    """
    Fixture for tests on indexes, series and series with a narrow dtype
    copy to avoid mutation, e.g. setting .name
    """
    return _index_or_series_objs[request.param].copy(deep=True)


# ----------------------------------------------------------------
# DataFrames
# ----------------------------------------------------------------
@pytest.fixture
def empty_frame():
    return DataFrame()


@pytest.fixture
def int_frame():
    """
    Fixture for DataFrame of ints with index of unique strings

    Columns are ['A', 'B', 'C', 'D']

                A  B  C  D
    vpBeWjM651  1  0  1  0
    5JyxmrP1En -1  0  0  0
    qEDaoD49U2 -1  1  0  0
    m66TkTfsFe  0  0  0  0
    EHPaNzEUFm -1  0 -1  0
    fpRJCevQhi  2  0  0  0
    OlQvnmfi3Q  0  0 -2  0
    ...        .. .. .. ..
    uB1FPlz4uP  0  0  0  1
    EcSe6yNzCU  0  0 -1  0
    L50VudaiI8 -1  1 -2  0
    y3bpw4nwIp  0 -1  0  0
    H0RdLLwrCT  1  1  0  0
    rY82K0vMwm  0  0  0  0
    1OPIUjnkjk  2  0  0  0

    [30 rows x 4 columns]
    """
    return DataFrame(tm.getSeriesData()).astype("int64")


@pytest.fixture
def datetime_frame():
    """
    Fixture for DataFrame of floats with DatetimeIndex

    Columns are ['A', 'B', 'C', 'D']

                       A         B         C         D
    2000-01-03 -1.122153  0.468535  0.122226  1.693711
    2000-01-04  0.189378  0.486100  0.007864 -1.216052
    2000-01-05  0.041401 -0.835752 -0.035279 -0.414357
    2000-01-06  0.430050  0.894352  0.090719  0.036939
    2000-01-07 -0.620982 -0.668211 -0.706153  1.466335
    2000-01-10 -0.752633  0.328434 -0.815325  0.699674
    2000-01-11 -2.236969  0.615737 -0.829076 -1.196106
    ...              ...       ...       ...       ...
    2000-02-03  1.642618 -0.579288  0.046005  1.385249
    2000-02-04 -0.544873 -1.160962 -0.284071 -1.418351
    2000-02-07 -2.656149 -0.601387  1.410148  0.444150
    2000-02-08 -1.201881 -1.289040  0.772992 -1.445300
    2000-02-09  1.377373  0.398619  1.008453 -0.928207
    2000-02-10  0.473194 -0.636677  0.984058  0.511519
    2000-02-11 -0.965556  0.408313 -1.312844 -0.381948

    [30 rows x 4 columns]
    """
    return DataFrame(tm.getTimeSeriesData())


@pytest.fixture
def float_frame():
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].

                       A         B         C         D
    P7GACiRnxd -0.465578 -0.361863  0.886172 -0.053465
    qZKh6afn8n -0.466693 -0.373773  0.266873  1.673901
    tkp0r6Qble  0.148691 -0.059051  0.174817  1.598433
    wP70WOCtv8  0.133045 -0.581994 -0.992240  0.261651
    M2AeYQMnCz -1.207959 -0.185775  0.588206  0.563938
    QEPzyGDYDo -0.381843 -0.758281  0.502575 -0.565053
    r78Jwns6dn -0.653707  0.883127  0.682199  0.206159
    ...              ...       ...       ...       ...
    IHEGx9NO0T -0.277360  0.113021 -1.018314  0.196316
    lPMj8K27FA -1.313667 -0.604776 -1.305618 -0.863999
    qa66YMWQa5  1.110525  0.475310 -0.747865  0.032121
    yOa0ATsmcE -0.431457  0.067094  0.096567 -0.264962
    65znX3uRNG  1.528446  0.160416 -0.109635 -0.032987
    eCOBvKqf3e  0.235281  1.622222  0.781255  0.392871
    xSucinXxuV -1.263557  0.252799 -0.552247  0.400426

    [30 rows x 4 columns]
    """
    return DataFrame(tm.getSeriesData())


@pytest.fixture
def mixed_type_frame():
    """
    Fixture for DataFrame of float/int/string columns with RangeIndex
    Columns are ['a', 'b', 'c', 'float32', 'int32'].
    """
    return DataFrame(
        {
            "a": 1.0,
            "b": 2,
            "c": "foo",
            "float32": np.array([1.0] * 10, dtype="float32"),
            "int32": np.array([1] * 10, dtype="int32"),
        },
        index=np.arange(10),
    )


@pytest.fixture
def rand_series_with_duplicate_datetimeindex():
    """
    Fixture for Series with a DatetimeIndex that has duplicates.
    """
    dates = [
        datetime(2000, 1, 2),
        datetime(2000, 1, 2),
        datetime(2000, 1, 2),
        datetime(2000, 1, 3),
        datetime(2000, 1, 3),
        datetime(2000, 1, 3),
        datetime(2000, 1, 4),
        datetime(2000, 1, 4),
        datetime(2000, 1, 4),
        datetime(2000, 1, 5),
    ]

    return Series(np.random.randn(len(dates)), index=dates)


# ----------------------------------------------------------------
# Scalars
# ----------------------------------------------------------------
@pytest.fixture(
    params=[
        (Interval(left=0, right=5), IntervalDtype("int64", "right")),
        (Interval(left=0.1, right=0.5), IntervalDtype("float64", "right")),
        (Period("2012-01", freq="M"), "period[M]"),
        (Period("2012-02-01", freq="D"), "period[D]"),
        (
            Timestamp("2011-01-01", tz="US/Eastern"),
            DatetimeTZDtype(tz="US/Eastern"),
        ),
        (Timedelta(seconds=500), "timedelta64[ns]"),
    ]
)
def ea_scalar_and_dtype(request):
    return request.param


# ----------------------------------------------------------------
# Operators & Operations
# ----------------------------------------------------------------
_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        ops.radd,
        operator.sub,
        ops.rsub,
        operator.mul,
        ops.rmul,
        operator.truediv,
        ops.rtruediv,
        operator.floordiv,
        ops.rfloordiv,
        operator.mod,
        ops.rmod,
        operator.pow,
        ops.rpow,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.and_,
        ops.rand_,
        operator.xor,
        ops.rxor,
        operator.or_,
        ops.ror_,
    ]
)
def all_binary_operators(request):
    """
    Fixture for operator and roperator arithmetic, comparison, and logical ops.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        ops.radd,
        operator.sub,
        ops.rsub,
        operator.mul,
        ops.rmul,
        operator.truediv,
        ops.rtruediv,
        operator.floordiv,
        ops.rfloordiv,
        operator.mod,
        ops.rmod,
        operator.pow,
        ops.rpow,
    ]
)
def all_arithmetic_functions(request):
    """
    Fixture for operator and roperator arithmetic functions.

    Notes
    -----
    This includes divmod and rdivmod, whereas all_arithmetic_operators
    does not.
    """
    return request.param


_all_numeric_reductions = [
    "sum",
    "max",
    "min",
    "mean",
    "prod",
    "std",
    "var",
    "median",
    "kurt",
    "skew",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.
    """
    return request.param


_all_reductions = _all_numeric_reductions + _all_boolean_reductions


@pytest.fixture(params=_all_reductions)
def all_reductions(request):
    """
    Fixture for all (boolean + numeric) reduction names.
    """
    return request.param


@pytest.fixture(params=["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations

    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    return request.param


@pytest.fixture(params=["__le__", "__lt__", "__ge__", "__gt__"])
def compare_operators_no_eq_ne(request):
    """
    Fixture for dunder names for compare operations except == and !=

    * >=
    * >
    * <
    * <=
    """
    return request.param


@pytest.fixture(
    params=["__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"]
)
def all_logical_operators(request):
    """
    Fixture for dunder names for common logical operations

    * |
    * &
    * ^
    """
    return request.param


# ----------------------------------------------------------------
# Data sets/files
# ----------------------------------------------------------------
@pytest.fixture
def strict_data_files(pytestconfig):
    """
    Returns the configuration for the test setting `--strict-data-files`.
    """
    return pytestconfig.getoption("--strict-data-files")


@pytest.fixture
def datapath(strict_data_files):
    """
    Get the path to a data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``pandas/tests/``

    Returns
    -------
    path including ``pandas/tests``.

    Raises
    ------
    ValueError
        If the path doesn't exist and the --strict-data-files option is set.
    """
    BASE_PATH = os.path.join(os.path.dirname(__file__), "tests")

    def deco(*args):
        path = os.path.join(BASE_PATH, *args)
        if not os.path.exists(path):
            if strict_data_files:
                raise ValueError(
                    f"Could not find file {path} and --strict-data-files is set."
                )
            else:
                pytest.skip(f"Could not find {path}.")
        return path

    return deco


@pytest.fixture
def iris(datapath):
    """
    The iris dataset as a DataFrame.
    """
    return pd.read_csv(datapath("io", "data", "csv", "iris.csv"))


# ----------------------------------------------------------------
# Time zones
# ----------------------------------------------------------------
TIMEZONES = [
    None,
    "UTC",
    "US/Eastern",
    "Asia/Tokyo",
    "dateutil/US/Pacific",
    "dateutil/Asia/Singapore",
    "+01:15",
    "-02:15",
    "UTC+01:15",
    "UTC-02:15",
    tzutc(),
    tzlocal(),
    FixedOffset(300),
    FixedOffset(0),
    FixedOffset(-300),
    timezone.utc,
    timezone(timedelta(hours=1)),
    timezone(timedelta(hours=-1), name="foo"),
]
TIMEZONE_IDS = [repr(i) for i in TIMEZONES]


@td.parametrize_fixture_doc(str(TIMEZONE_IDS))
@pytest.fixture(params=TIMEZONES, ids=TIMEZONE_IDS)
def tz_naive_fixture(request):
    """
    Fixture for trying timezones including default (None): {0}
    """
    return request.param


@td.parametrize_fixture_doc(str(TIMEZONE_IDS[1:]))
@pytest.fixture(params=TIMEZONES[1:], ids=TIMEZONE_IDS[1:])
def tz_aware_fixture(request):
    """
    Fixture for trying explicit timezones: {0}
    """
    return request.param


# Generate cartesian product of tz_aware_fixture:
tz_aware_fixture2 = tz_aware_fixture


@pytest.fixture(params=["utc", "dateutil/UTC", utc, tzutc(), timezone.utc])
def utc_fixture(request):
    """
    Fixture to provide variants of UTC timezone strings and tzinfo objects.
    """
    return request.param


utc_fixture2 = utc_fixture


# ----------------------------------------------------------------
# Dtypes
# ----------------------------------------------------------------
@pytest.fixture(params=tm.STRING_DTYPES)
def string_dtype(request):
    """
    Parametrized fixture for string dtypes.

    * str
    * 'str'
    * 'U'
    """
    return request.param


@pytest.fixture(
    params=[
        "string[python]",
        pytest.param(
            "string[pyarrow]", marks=td.skip_if_no("pyarrow", min_version="1.0.0")
        ),
    ]
)
def nullable_string_dtype(request):
    """
    Parametrized fixture for string dtypes.

    * 'string[python]'
    * 'string[pyarrow]'
    """
    return request.param


@pytest.fixture(
    params=[
        "python",
        pytest.param("pyarrow", marks=td.skip_if_no("pyarrow", min_version="1.0.0")),
    ]
)
def string_storage(request):
    """
    Parametrized fixture for pd.options.mode.string_storage.

    * 'python'
    * 'pyarrow'
    """
    return request.param


# Alias so we can test with cartesian product of string_storage
string_storage2 = string_storage


@pytest.fixture(params=tm.BYTES_DTYPES)
def bytes_dtype(request):
    """
    Parametrized fixture for bytes dtypes.

    * bytes
    * 'bytes'
    """
    return request.param


@pytest.fixture(params=tm.OBJECT_DTYPES)
def object_dtype(request):
    """
    Parametrized fixture for object dtypes.

    * object
    * 'object'
    """
    return request.param


@pytest.fixture(
    params=[
        "object",
        "string[python]",
        pytest.param(
            "string[pyarrow]", marks=td.skip_if_no("pyarrow", min_version="1.0.0")
        ),
    ]
)
def any_string_dtype(request):
    """
    Parametrized fixture for string dtypes.
    * 'object'
    * 'string[python]'
    * 'string[pyarrow]'
    """
    return request.param


@pytest.fixture(params=tm.DATETIME64_DTYPES)
def datetime64_dtype(request):
    """
    Parametrized fixture for datetime64 dtypes.

    * 'datetime64[ns]'
    * 'M8[ns]'
    """
    return request.param


@pytest.fixture(params=tm.TIMEDELTA64_DTYPES)
def timedelta64_dtype(request):
    """
    Parametrized fixture for timedelta64 dtypes.

    * 'timedelta64[ns]'
    * 'm8[ns]'
    """
    return request.param


@pytest.fixture(params=tm.FLOAT_DTYPES)
def float_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    """
    return request.param


@pytest.fixture(params=tm.FLOAT_EA_DTYPES)
def float_ea_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * 'Float32'
    * 'Float64'
    """
    return request.param


@pytest.fixture(params=tm.FLOAT_DTYPES + tm.FLOAT_EA_DTYPES)
def any_float_allowed_nullable_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    * 'Float32'
    * 'Float64'
    """
    return request.param


@pytest.fixture(params=tm.COMPLEX_DTYPES)
def complex_dtype(request):
    """
    Parameterized fixture for complex dtypes.

    * complex
    * 'complex64'
    * 'complex128'
    """
    return request.param


@pytest.fixture(params=tm.SIGNED_INT_DTYPES)
def sint_dtype(request):
    """
    Parameterized fixture for signed integer dtypes.

    * int
    * 'int8'
    * 'int16'
    * 'int32'
    * 'int64'
    """
    return request.param


@pytest.fixture(params=tm.UNSIGNED_INT_DTYPES)
def uint_dtype(request):
    """
    Parameterized fixture for unsigned integer dtypes.

    * 'uint8'
    * 'uint16'
    * 'uint32'
    * 'uint64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_DTYPES)
def any_int_dtype(request):
    """
    Parameterized fixture for any integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_EA_INT_DTYPES)
def any_nullable_int_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_DTYPES + tm.ALL_EA_INT_DTYPES)
def any_int_or_nullable_int_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_EA_INT_DTYPES + tm.FLOAT_EA_DTYPES)
def any_nullable_numeric_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype and
    any float ea dtypes.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    """
    return request.param


@pytest.fixture(params=tm.SIGNED_EA_INT_DTYPES)
def any_signed_nullable_int_dtype(request):
    """
    Parameterized fixture for any signed nullable integer dtype.

    * 'Int8'
    * 'Int16'
    * 'Int32'
    * 'Int64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_DTYPES)
def any_real_dtype(request):
    """
    Parameterized fixture for any (purely) real numeric dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_NUMPY_DTYPES)
def any_numpy_dtype(request):
    """
    Parameterized fixture for all numpy dtypes.

    * bool
    * 'bool'
    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    * complex
    * 'complex64'
    * 'complex128'
    * str
    * 'str'
    * 'U'
    * bytes
    * 'bytes'
    * 'datetime64[ns]'
    * 'M8[ns]'
    * 'timedelta64[ns]'
    * 'm8[ns]'
    * object
    * 'object'
    """
    return request.param


# categoricals are handled separately
_any_skipna_inferred_dtype = [
    ("string", ["a", np.nan, "c"]),
    ("string", ["a", pd.NA, "c"]),
    ("bytes", [b"a", np.nan, b"c"]),
    ("empty", [np.nan, np.nan, np.nan]),
    ("empty", []),
    ("mixed-integer", ["a", np.nan, 2]),
    ("mixed", ["a", np.nan, 2.0]),
    ("floating", [1.0, np.nan, 2.0]),
    ("integer", [1, np.nan, 2]),
    ("mixed-integer-float", [1, np.nan, 2.0]),
    ("decimal", [Decimal(1), np.nan, Decimal(2)]),
    ("boolean", [True, np.nan, False]),
    ("boolean", [True, pd.NA, False]),
    ("datetime64", [np.datetime64("2013-01-01"), np.nan, np.datetime64("2018-01-01")]),
    ("datetime", [Timestamp("20130101"), np.nan, Timestamp("20180101")]),
    ("date", [date(2013, 1, 1), np.nan, date(2018, 1, 1)]),
    # The following two dtypes are commented out due to GH 23554
    # ('complex', [1 + 1j, np.nan, 2 + 2j]),
    # ('timedelta64', [np.timedelta64(1, 'D'),
    #                  np.nan, np.timedelta64(2, 'D')]),
    ("timedelta", [timedelta(1), np.nan, timedelta(2)]),
    ("time", [time(1), np.nan, time(2)]),
    ("period", [Period(2013), pd.NaT, Period(2018)]),
    ("interval", [Interval(0, 1), np.nan, Interval(0, 2)]),
]
ids, _ = zip(*_any_skipna_inferred_dtype)  # use inferred type as fixture-id


@pytest.fixture(params=_any_skipna_inferred_dtype, ids=ids)
def any_skipna_inferred_dtype(request):
    """
    Fixture for all inferred dtypes from _libs.lib.infer_dtype

    The covered (inferred) types are:
    * 'string'
    * 'empty'
    * 'bytes'
    * 'mixed'
    * 'mixed-integer'
    * 'mixed-integer-float'
    * 'floating'
    * 'integer'
    * 'decimal'
    * 'boolean'
    * 'datetime64'
    * 'datetime'
    * 'date'
    * 'timedelta'
    * 'time'
    * 'period'
    * 'interval'

    Returns
    -------
    inferred_dtype : str
        The string for the inferred dtype from _libs.lib.infer_dtype
    values : np.ndarray
        An array of object dtype that will be inferred to have
        `inferred_dtype`

    Examples
    --------
    >>> import pandas._libs.lib as lib
    >>>
    >>> def test_something(any_skipna_inferred_dtype):
    ...     inferred_dtype, values = any_skipna_inferred_dtype
    ...     # will pass
    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype
    """
    inferred_dtype, values = request.param
    values = np.array(values, dtype=object)  # object dtype to avoid casting

    # correctness of inference tested in tests/dtypes/test_inference.py
    return inferred_dtype, values


# ----------------------------------------------------------------
# Misc
# ----------------------------------------------------------------
@pytest.fixture
def ip():
    """
    Get an instance of IPython.InteractiveShell.

    Will raise a skip if IPython is not installed.
    """
    pytest.importorskip("IPython", minversion="6.0.0")
    from IPython.core.interactiveshell import InteractiveShell

    # GH#35711 make sure sqlite history file handle is not leaked
    from traitlets.config import Config  # isort:skip

    c = Config()
    c.HistoryManager.hist_file = ":memory:"

    return InteractiveShell(config=c)


@pytest.fixture(params=["bsr", "coo", "csc", "csr", "dia", "dok", "lil"])
def spmatrix(request):
    """
    Yields scipy sparse matrix classes.
    """
    from scipy import sparse

    return getattr(sparse, request.param + "_matrix")


@pytest.fixture(
    params=[
        getattr(pd.offsets, o)
        for o in pd.offsets.__all__
        if issubclass(getattr(pd.offsets, o), pd.offsets.Tick)
    ]
)
def tick_classes(request):
    """
    Fixture for Tick based datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


@pytest.fixture()
def fsspectest():
    pytest.importorskip("fsspec")
    from fsspec import register_implementation
    from fsspec.implementations.memory import MemoryFileSystem
    from fsspec.registry import _registry as registry

    class TestMemoryFS(MemoryFileSystem):
        protocol = "testmem"
        test = [None]

        def __init__(self, **kwargs):
            self.test[0] = kwargs.pop("test", None)
            super().__init__(**kwargs)

    register_implementation("testmem", TestMemoryFS, clobber=True)
    yield TestMemoryFS()
    registry.pop("testmem", None)
    TestMemoryFS.test[0] = None
    TestMemoryFS.store.clear()


@pytest.fixture(
    params=[
        ("foo", None, None),
        ("Egon", "Venkman", None),
        ("NCC1701D", "NCC1701D", "NCC1701D"),
    ]
)
def names(request):
    """
    A 3-tuple of names, the first two for operands, the last for a result.
    """
    return request.param


@pytest.fixture(params=[tm.setitem, tm.loc, tm.iloc])
def indexer_sli(request):
    """
    Parametrize over __setitem__, loc.__setitem__, iloc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.setitem, tm.iloc])
def indexer_si(request):
    """
    Parametrize over __setitem__, iloc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.setitem, tm.loc])
def indexer_sl(request):
    """
    Parametrize over __setitem__, loc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.at, tm.loc])
def indexer_al(request):
    """
    Parametrize over at.__setitem__, loc.__setitem__
    """
    return request.param


@pytest.fixture
def using_array_manager(request):
    """
    Fixture to check if the array manager is being used.
    """
    return pd.options.mode.data_manager == "array"
