# flake8: noqa

__docformat__ = "restructuredtext"

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "pytz", "dateutil")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies

# numpy compat
from pandas.compat import (
    np_version_under1p18 as _np_version_under1p18,
    is_numpy_dev as _is_numpy_dev,
)

try:
    from pandas._libs import hashtable as _hashtable, lib as _lib, tslib as _tslib
except ImportError as e:  # pragma: no cover
    # hack but overkill to use re
    module = str(e).replace("cannot import name ", "")
    raise ImportError(
        f"C extension: {module} not built. If you want to import "
        "pandas from the source directory, you may need to run "
        "'python setup.py build_ext --force' to build the C extensions first."
    ) from e

from pandas._config import (
    get_option,
    set_option,
    reset_option,
    describe_option,
    option_context,
    options,
)

# let init-time option registration happen
import pandas.core.config_init

from pandas.core.api import (
    # dtype
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
    Float32Dtype,
    Float64Dtype,
    CategoricalDtype,
    PeriodDtype,
    IntervalDtype,
    DatetimeTZDtype,
    StringDtype,
    BooleanDtype,
    # missing
    NA,
    isna,
    isnull,
    notna,
    notnull,
    # indexes
    Index,
    CategoricalIndex,
    Int64Index,
    UInt64Index,
    RangeIndex,
    Float64Index,
    MultiIndex,
    IntervalIndex,
    TimedeltaIndex,
    DatetimeIndex,
    PeriodIndex,
    IndexSlice,
    # tseries
    NaT,
    Period,
    period_range,
    Timedelta,
    timedelta_range,
    Timestamp,
    date_range,
    bdate_range,
    Interval,
    interval_range,
    DateOffset,
    # conversion
    to_numeric,
    to_datetime,
    to_timedelta,
    # misc
    Flags,
    Grouper,
    factorize,
    unique,
    value_counts,
    NamedAgg,
    array,
    Categorical,
    set_eng_float_format,
    Series,
    DataFrame,
)

from pandas.core.arrays.sparse import SparseDtype

from pandas.tseries.api import infer_freq
from pandas.tseries import offsets

from pandas.core.computation.api import eval

from pandas.core.reshape.api import (
    concat,
    lreshape,
    melt,
    wide_to_long,
    merge,
    merge_asof,
    merge_ordered,
    crosstab,
    pivot,
    pivot_table,
    get_dummies,
    cut,
    qcut,
)

import pandas.api
from pandas.util._print_versions import show_versions

from pandas.io.api import (
    # excel
    ExcelFile,
    ExcelWriter,
    read_excel,
    # parsers
    read_csv,
    read_fwf,
    read_table,
    # pickle
    read_pickle,
    to_pickle,
    # pytables
    HDFStore,
    read_hdf,
    # sql
    read_sql,
    read_sql_query,
    read_sql_table,
    # misc
    read_clipboard,
    read_parquet,
    read_orc,
    read_feather,
    read_gbq,
    read_html,
    read_xml,
    read_json,
    read_stata,
    read_sas,
    read_spss,
)

from pandas.io.json import _json_normalize as json_normalize

from pandas.util._tester import test
import pandas.testing
import pandas.arrays

# use the closest tagged version if possible
from pandas._version import get_versions

v = get_versions()
__version__ = v.get("closest-tag", v["version"])
__git_version__ = v.get("full-revisionid")
del get_versions, v


# GH 27101
def __getattr__(name):
    import warnings

    if name == "datetime":
        warnings.warn(
            "The pandas.datetime class is deprecated "
            "and will be removed from pandas in a future version. "
            "Import from datetime module instead.",
            FutureWarning,
            stacklevel=2,
        )

        from datetime import datetime as dt

        return dt

    elif name == "np":

        warnings.warn(
            "The pandas.np module is deprecated "
            "and will be removed from pandas in a future version. "
            "Import numpy directly instead",
            FutureWarning,
            stacklevel=2,
        )
        import numpy as np

        return np

    elif name in {"SparseSeries", "SparseDataFrame"}:
        warnings.warn(
            f"The {name} class is removed from pandas. Accessing it from "
            "the top-level namespace will also be removed in the next version",
            FutureWarning,
            stacklevel=2,
        )

        return type(name, (), {})

    elif name == "SparseArray":

        warnings.warn(
            "The pandas.SparseArray class is deprecated "
            "and will be removed from pandas in a future version. "
            "Use pandas.arrays.SparseArray instead.",
            FutureWarning,
            stacklevel=2,
        )
        from pandas.core.arrays.sparse import SparseArray as _SparseArray

        return _SparseArray

    raise AttributeError(f"module 'pandas' has no attribute '{name}'")


# module level doc-string
__doc__ = """
pandas - a powerful data analysis and manipulation library for Python
=====================================================================

**pandas** is a Python package providing fast, flexible, and expressive data
structures designed to make working with "relational" or "labeled" data both
easy and intuitive. It aims to be the fundamental high-level building block for
doing practical, **real world** data analysis in Python. Additionally, it has
the broader goal of becoming **the most powerful and flexible open source data
analysis / manipulation tool available in any language**. It is already well on
its way toward this goal.

Main Features
-------------
Here are just a few of the things that pandas does well:

  - Easy handling of missing data in floating point as well as non-floating
    point data.
  - Size mutability: columns can be inserted and deleted from DataFrame and
    higher dimensional objects
  - Automatic and explicit data alignment: objects can be explicitly aligned
    to a set of labels, or the user can simply ignore the labels and let
    `Series`, `DataFrame`, etc. automatically align the data for you in
    computations.
  - Powerful, flexible group by functionality to perform split-apply-combine
    operations on data sets, for both aggregating and transforming data.
  - Make it easy to convert ragged, differently-indexed data in other Python
    and NumPy data structures into DataFrame objects.
  - Intelligent label-based slicing, fancy indexing, and subsetting of large
    data sets.
  - Intuitive merging and joining data sets.
  - Flexible reshaping and pivoting of data sets.
  - Hierarchical labeling of axes (possible to have multiple labels per tick).
  - Robust IO tools for loading data from flat files (CSV and delimited),
    Excel files, databases, and saving/loading data from the ultrafast HDF5
    format.
  - Time series-specific functionality: date range generation and frequency
    conversion, moving window statistics, date shifting and lagging.
"""
