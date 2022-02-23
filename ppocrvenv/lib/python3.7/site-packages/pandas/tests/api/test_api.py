from __future__ import annotations

import subprocess
import sys

import pytest

import pandas as pd
from pandas import api
import pandas._testing as tm


class Base:
    def check(self, namespace, expected, ignored=None):
        # see which names are in the namespace, minus optional
        # ignored ones
        # compare vs the expected

        result = sorted(f for f in dir(namespace) if not f.startswith("__"))
        if ignored is not None:
            result = sorted(set(result) - set(ignored))

        expected = sorted(expected)
        tm.assert_almost_equal(result, expected)


class TestPDApi(Base):
    # these are optionally imported based on testing
    # & need to be ignored
    ignored = ["tests", "locale", "conftest"]

    # top-level sub-packages
    lib = [
        "api",
        "arrays",
        "compat",
        "core",
        "errors",
        "pandas",
        "plotting",
        "test",
        "testing",
        "tseries",
        "util",
        "options",
        "io",
    ]

    # these are already deprecated; awaiting removal
    deprecated_modules: list[str] = ["np", "datetime"]

    # misc
    misc = ["IndexSlice", "NaT", "NA"]

    # top-level classes
    classes = [
        "Categorical",
        "CategoricalIndex",
        "DataFrame",
        "DateOffset",
        "DatetimeIndex",
        "ExcelFile",
        "ExcelWriter",
        "Float64Index",
        "Flags",
        "Grouper",
        "HDFStore",
        "Index",
        "Int64Index",
        "MultiIndex",
        "Period",
        "PeriodIndex",
        "RangeIndex",
        "UInt64Index",
        "Series",
        "SparseDtype",
        "StringDtype",
        "Timedelta",
        "TimedeltaIndex",
        "Timestamp",
        "Interval",
        "IntervalIndex",
        "CategoricalDtype",
        "PeriodDtype",
        "IntervalDtype",
        "DatetimeTZDtype",
        "BooleanDtype",
        "Int8Dtype",
        "Int16Dtype",
        "Int32Dtype",
        "Int64Dtype",
        "UInt8Dtype",
        "UInt16Dtype",
        "UInt32Dtype",
        "UInt64Dtype",
        "Float32Dtype",
        "Float64Dtype",
        "NamedAgg",
    ]

    # these are already deprecated; awaiting removal
    deprecated_classes: list[str] = []

    # these should be deprecated in the future
    deprecated_classes_in_future: list[str] = ["SparseArray"]

    # external modules exposed in pandas namespace
    modules: list[str] = []

    # top-level functions
    funcs = [
        "array",
        "bdate_range",
        "concat",
        "crosstab",
        "cut",
        "date_range",
        "interval_range",
        "eval",
        "factorize",
        "get_dummies",
        "infer_freq",
        "isna",
        "isnull",
        "lreshape",
        "melt",
        "notna",
        "notnull",
        "offsets",
        "merge",
        "merge_ordered",
        "merge_asof",
        "period_range",
        "pivot",
        "pivot_table",
        "qcut",
        "show_versions",
        "timedelta_range",
        "unique",
        "value_counts",
        "wide_to_long",
    ]

    # top-level option funcs
    funcs_option = [
        "reset_option",
        "describe_option",
        "get_option",
        "option_context",
        "set_option",
        "set_eng_float_format",
    ]

    # top-level read_* funcs
    funcs_read = [
        "read_clipboard",
        "read_csv",
        "read_excel",
        "read_fwf",
        "read_gbq",
        "read_hdf",
        "read_html",
        "read_xml",
        "read_json",
        "read_pickle",
        "read_sas",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "read_stata",
        "read_table",
        "read_feather",
        "read_parquet",
        "read_orc",
        "read_spss",
    ]

    # top-level json funcs
    funcs_json = ["json_normalize"]

    # top-level to_* funcs
    funcs_to = ["to_datetime", "to_numeric", "to_pickle", "to_timedelta"]

    # top-level to deprecate in the future
    deprecated_funcs_in_future: list[str] = []

    # these are already deprecated; awaiting removal
    deprecated_funcs: list[str] = []

    # private modules in pandas namespace
    private_modules = [
        "_config",
        "_hashtable",
        "_lib",
        "_libs",
        "_np_version_under1p18",
        "_is_numpy_dev",
        "_testing",
        "_tslib",
        "_typing",
        "_version",
    ]

    def test_api(self):

        checkthese = (
            self.lib
            + self.misc
            + self.modules
            + self.classes
            + self.funcs
            + self.funcs_option
            + self.funcs_read
            + self.funcs_json
            + self.funcs_to
            + self.private_modules
        )
        self.check(namespace=pd, expected=checkthese, ignored=self.ignored)

    def test_depr(self):
        deprecated_list = (
            self.deprecated_modules
            + self.deprecated_classes
            + self.deprecated_classes_in_future
            + self.deprecated_funcs
            + self.deprecated_funcs_in_future
        )
        for depr in deprecated_list:
            with tm.assert_produces_warning(FutureWarning):
                _ = getattr(pd, depr)


def test_datetime():
    from datetime import datetime
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert datetime(2015, 1, 2, 0, 0) == datetime(2015, 1, 2, 0, 0)

        assert isinstance(datetime(2015, 1, 2, 0, 0), datetime)


def test_sparsearray():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert isinstance(pd.array([1, 2, 3], dtype="Sparse"), pd.SparseArray)


def test_np():
    import warnings

    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert (pd.np.arange(0, 10) == np.arange(0, 10)).all()


class TestApi(Base):
    allowed = ["types", "extensions", "indexers"]

    def test_api(self):
        self.check(api, self.allowed)


class TestTesting(Base):
    funcs = [
        "assert_frame_equal",
        "assert_series_equal",
        "assert_index_equal",
        "assert_extension_array_equal",
    ]

    def test_testing(self):
        from pandas import testing  # noqa: PDF015

        self.check(testing, self.funcs)

    def test_util_testing_deprecated(self):
        # avoid cache state affecting the test
        sys.modules.pop("pandas.util.testing", None)

        with tm.assert_produces_warning(FutureWarning) as m:
            import pandas.util.testing  # noqa: F401

        assert "pandas.util.testing is deprecated" in str(m[0].message)
        assert "pandas.testing instead" in str(m[0].message)

    def test_util_testing_deprecated_direct(self):
        # avoid cache state affecting the test
        sys.modules.pop("pandas.util.testing", None)
        with tm.assert_produces_warning(FutureWarning) as m:
            from pandas.util.testing import assert_series_equal  # noqa: F401

        assert "pandas.util.testing is deprecated" in str(m[0].message)
        assert "pandas.testing instead" in str(m[0].message)

    def test_util_in_top_level(self):
        # in a subprocess to avoid import caching issues
        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import pandas; pandas.util.testing.assert_series_equal",
            ],
            stderr=subprocess.STDOUT,
        ).decode()
        assert "pandas.util.testing is deprecated" in out

        with pytest.raises(AttributeError, match="foo"):
            pd.util.foo
