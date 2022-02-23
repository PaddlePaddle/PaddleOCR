import builtins

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    isna,
)
import pandas._testing as tm


@pytest.mark.parametrize("agg_func", ["any", "all"])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize(
    "vals",
    [
        ["foo", "bar", "baz"],
        ["foo", "", ""],
        ["", "", ""],
        [1, 2, 3],
        [1, 0, 0],
        [0, 0, 0],
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [True, True, True],
        [True, False, False],
        [False, False, False],
        [np.nan, np.nan, np.nan],
    ],
)
def test_groupby_bool_aggs(agg_func, skipna, vals):
    df = DataFrame({"key": ["a"] * 3 + ["b"] * 3, "val": vals * 2})

    # Figure out expectation using Python builtin
    exp = getattr(builtins, agg_func)(vals)

    # edge case for missing data with skipna and 'any'
    if skipna and all(isna(vals)) and agg_func == "any":
        exp = False

    exp_df = DataFrame([exp] * 2, columns=["val"], index=Index(["a", "b"], name="key"))
    result = getattr(df.groupby("key"), agg_func)(skipna=skipna)
    tm.assert_frame_equal(result, exp_df)


def test_any():
    df = DataFrame(
        [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, "baz"]],
        columns=["A", "B", "C"],
    )
    expected = DataFrame(
        [[True, True], [False, True]], columns=["B", "C"], index=[1, 3]
    )
    expected.index.name = "A"
    result = df.groupby("A").any()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("bool_agg_func", ["any", "all"])
def test_bool_aggs_dup_column_labels(bool_agg_func):
    # 21668
    df = DataFrame([[True, True]], columns=["a", "a"])
    grp_by = df.groupby([0])
    result = getattr(grp_by, bool_agg_func)()

    expected = df
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("bool_agg_func", ["any", "all"])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize(
    "data",
    [
        [False, False, False],
        [True, True, True],
        [pd.NA, pd.NA, pd.NA],
        [False, pd.NA, False],
        [True, pd.NA, True],
        [True, pd.NA, False],
    ],
)
def test_masked_kleene_logic(bool_agg_func, skipna, data):
    # GH#37506
    ser = Series(data, dtype="boolean")

    # The result should match aggregating on the whole series. Correctness
    # there is verified in test_reductions.py::test_any_all_boolean_kleene_logic
    expected_data = getattr(ser, bool_agg_func)(skipna=skipna)
    expected = Series(expected_data, dtype="boolean")

    result = ser.groupby([0, 0, 0]).agg(bool_agg_func, skipna=skipna)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype1,dtype2,exp_col1,exp_col2",
    [
        (
            "float",
            "Float64",
            np.array([True], dtype=bool),
            pd.array([pd.NA], dtype="boolean"),
        ),
        (
            "Int64",
            "float",
            pd.array([pd.NA], dtype="boolean"),
            np.array([True], dtype=bool),
        ),
        (
            "Int64",
            "Int64",
            pd.array([pd.NA], dtype="boolean"),
            pd.array([pd.NA], dtype="boolean"),
        ),
        (
            "Float64",
            "boolean",
            pd.array([pd.NA], dtype="boolean"),
            pd.array([pd.NA], dtype="boolean"),
        ),
    ],
)
def test_masked_mixed_types(dtype1, dtype2, exp_col1, exp_col2):
    # GH#37506
    data = [1.0, np.nan]
    df = DataFrame(
        {"col1": pd.array(data, dtype=dtype1), "col2": pd.array(data, dtype=dtype2)}
    )
    result = df.groupby([1, 1]).agg("all", skipna=False)

    expected = DataFrame({"col1": exp_col1, "col2": exp_col2}, index=[1])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("bool_agg_func", ["any", "all"])
@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
@pytest.mark.parametrize("skipna", [True, False])
def test_masked_bool_aggs_skipna(bool_agg_func, dtype, skipna, frame_or_series):
    # GH#40585
    obj = frame_or_series([pd.NA, 1], dtype=dtype)
    expected_res = True
    if not skipna and bool_agg_func == "all":
        expected_res = pd.NA
    expected = frame_or_series([expected_res], index=[1], dtype="boolean")

    result = obj.groupby([1, 1]).agg(bool_agg_func, skipna=skipna)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "bool_agg_func,data,expected_res",
    [
        ("any", [pd.NA, np.nan], False),
        ("any", [pd.NA, 1, np.nan], True),
        ("all", [pd.NA, pd.NaT], True),
        ("all", [pd.NA, False, pd.NaT], False),
    ],
)
def test_object_type_missing_vals(bool_agg_func, data, expected_res, frame_or_series):
    # GH#37501
    obj = frame_or_series(data, dtype=object)
    result = obj.groupby([1] * len(data)).agg(bool_agg_func)
    expected = frame_or_series([expected_res], index=[1], dtype="bool")
    tm.assert_equal(result, expected)


@pytest.mark.filterwarnings("ignore:Dropping invalid columns:FutureWarning")
@pytest.mark.parametrize("bool_agg_func", ["any", "all"])
def test_object_NA_raises_with_skipna_false(bool_agg_func):
    # GH#37501
    ser = Series([pd.NA], dtype=object)
    with pytest.raises(TypeError, match="boolean value of NA is ambiguous"):
        ser.groupby([1]).agg(bool_agg_func, skipna=False)
