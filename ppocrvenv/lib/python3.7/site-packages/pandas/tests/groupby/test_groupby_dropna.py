import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "dropna, tuples, outputs",
    [
        (
            True,
            [["A", "B"], ["B", "A"]],
            {"c": [13.0, 123.23], "d": [13.0, 123.0], "e": [13.0, 1.0]},
        ),
        (
            False,
            [["A", "B"], ["A", np.nan], ["B", "A"]],
            {
                "c": [13.0, 12.3, 123.23],
                "d": [13.0, 233.0, 123.0],
                "e": [13.0, 12.0, 1.0],
            },
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_nan_in_one_group(
    dropna, tuples, outputs, nulls_fixture
):
    # GH 3729 this is to test that NA is in one group
    df_list = [
        ["A", "B", 12, 12, 12],
        ["A", nulls_fixture, 12.3, 233.0, 12],
        ["B", "A", 123.23, 123, 1],
        ["A", "B", 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d", "e"])
    grouped = df.groupby(["a", "b"], dropna=dropna).sum()

    mi = pd.MultiIndex.from_tuples(tuples, names=list("ab"))

    # Since right now, by default MI will drop NA from levels when we create MI
    # via `from_*`, so we need to add NA for level manually afterwards.
    if not dropna:
        mi = mi.set_levels(["A", "B", np.nan], level="b")
    expected = pd.DataFrame(outputs, index=mi)

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, tuples, outputs",
    [
        (
            True,
            [["A", "B"], ["B", "A"]],
            {"c": [12.0, 123.23], "d": [12.0, 123.0], "e": [12.0, 1.0]},
        ),
        (
            False,
            [["A", "B"], ["A", np.nan], ["B", "A"], [np.nan, "B"]],
            {
                "c": [12.0, 13.3, 123.23, 1.0],
                "d": [12.0, 234.0, 123.0, 1.0],
                "e": [12.0, 13.0, 1.0, 1.0],
            },
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_nan_in_two_groups(
    dropna, tuples, outputs, nulls_fixture, nulls_fixture2
):
    # GH 3729 this is to test that NA in different groups with different representations
    df_list = [
        ["A", "B", 12, 12, 12],
        ["A", nulls_fixture, 12.3, 233.0, 12],
        ["B", "A", 123.23, 123, 1],
        [nulls_fixture2, "B", 1, 1, 1.0],
        ["A", nulls_fixture2, 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d", "e"])
    grouped = df.groupby(["a", "b"], dropna=dropna).sum()

    mi = pd.MultiIndex.from_tuples(tuples, names=list("ab"))

    # Since right now, by default MI will drop NA from levels when we create MI
    # via `from_*`, so we need to add NA for level manually afterwards.
    if not dropna:
        mi = mi.set_levels([["A", "B", np.nan], ["A", "B", np.nan]])
    expected = pd.DataFrame(outputs, index=mi)

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, idx, outputs",
    [
        (True, ["A", "B"], {"b": [123.23, 13.0], "c": [123.0, 13.0], "d": [1.0, 13.0]}),
        (
            False,
            ["A", "B", np.nan],
            {
                "b": [123.23, 13.0, 12.3],
                "c": [123.0, 13.0, 233.0],
                "d": [1.0, 13.0, 12.0],
            },
        ),
    ],
)
def test_groupby_dropna_normal_index_dataframe(dropna, idx, outputs):
    # GH 3729
    df_list = [
        ["B", 12, 12, 12],
        [None, 12.3, 233.0, 12],
        ["A", 123.23, 123, 1],
        ["B", 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d"])
    grouped = df.groupby("a", dropna=dropna).sum()

    expected = pd.DataFrame(outputs, index=pd.Index(idx, dtype="object", name="a"))

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, idx, expected",
    [
        (True, ["a", "a", "b", np.nan], pd.Series([3, 3], index=["a", "b"])),
        (
            False,
            ["a", "a", "b", np.nan],
            pd.Series([3, 3, 3], index=["a", "b", np.nan]),
        ),
    ],
)
def test_groupby_dropna_series_level(dropna, idx, expected):
    ser = pd.Series([1, 2, 3, 3], index=idx)

    result = ser.groupby(level=0, dropna=dropna).sum()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dropna, expected",
    [
        (True, pd.Series([210.0, 350.0], index=["a", "b"], name="Max Speed")),
        (
            False,
            pd.Series([210.0, 350.0, 20.0], index=["a", "b", np.nan], name="Max Speed"),
        ),
    ],
)
def test_groupby_dropna_series_by(dropna, expected):
    ser = pd.Series(
        [390.0, 350.0, 30.0, 20.0],
        index=["Falcon", "Falcon", "Parrot", "Parrot"],
        name="Max Speed",
    )

    result = ser.groupby(["a", "b", "a", np.nan], dropna=dropna).mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dropna", (False, True))
def test_grouper_dropna_propagation(dropna):
    # GH 36604
    df = pd.DataFrame({"A": [0, 0, 1, None], "B": [1, 2, 3, None]})
    gb = df.groupby("A", dropna=dropna)
    assert gb.grouper.dropna == dropna


@pytest.mark.parametrize(
    "dropna,input_index,expected_data,expected_index",
    [
        (True, pd.RangeIndex(0, 4), {"B": [2, 2, 1]}, pd.RangeIndex(0, 3)),
        (True, list("abcd"), {"B": [2, 2, 1]}, list("abc")),
        (
            True,
            pd.MultiIndex.from_tuples(
                [(1, "R"), (1, "B"), (2, "R"), (2, "B")], names=["num", "col"]
            ),
            {"B": [2, 2, 1]},
            pd.MultiIndex.from_tuples(
                [(1, "R"), (1, "B"), (2, "R")], names=["num", "col"]
            ),
        ),
        (False, pd.RangeIndex(0, 4), {"B": [2, 2, 1, 1]}, pd.RangeIndex(0, 4)),
        (False, list("abcd"), {"B": [2, 2, 1, 1]}, list("abcd")),
        (
            False,
            pd.MultiIndex.from_tuples(
                [(1, "R"), (1, "B"), (2, "R"), (2, "B")], names=["num", "col"]
            ),
            {"B": [2, 2, 1, 1]},
            pd.MultiIndex.from_tuples(
                [(1, "R"), (1, "B"), (2, "R"), (2, "B")], names=["num", "col"]
            ),
        ),
    ],
)
def test_groupby_dataframe_slice_then_transform(
    dropna, input_index, expected_data, expected_index
):
    # GH35014 & GH35612

    df = pd.DataFrame({"A": [0, 0, 1, None], "B": [1, 2, 3, None]}, index=input_index)
    gb = df.groupby("A", dropna=dropna)

    result = gb.transform(len)
    expected = pd.DataFrame(expected_data, index=expected_index)
    tm.assert_frame_equal(result, expected)

    result = gb[["B"]].transform(len)
    expected = pd.DataFrame(expected_data, index=expected_index)
    tm.assert_frame_equal(result, expected)

    result = gb["B"].transform(len)
    expected = pd.Series(expected_data["B"], index=expected_index, name="B")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dropna, tuples, outputs",
    [
        (
            True,
            [["A", "B"], ["B", "A"]],
            {"c": [13.0, 123.23], "d": [12.0, 123.0], "e": [1.0, 1.0]},
        ),
        (
            False,
            [["A", "B"], ["A", np.nan], ["B", "A"]],
            {
                "c": [13.0, 12.3, 123.23],
                "d": [12.0, 233.0, 123.0],
                "e": [1.0, 12.0, 1.0],
            },
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_agg(dropna, tuples, outputs):
    # GH 3729
    df_list = [
        ["A", "B", 12, 12, 12],
        ["A", None, 12.3, 233.0, 12],
        ["B", "A", 123.23, 123, 1],
        ["A", "B", 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d", "e"])
    agg_dict = {"c": sum, "d": max, "e": "min"}
    grouped = df.groupby(["a", "b"], dropna=dropna).agg(agg_dict)

    mi = pd.MultiIndex.from_tuples(tuples, names=list("ab"))

    # Since right now, by default MI will drop NA from levels when we create MI
    # via `from_*`, so we need to add NA for level manually afterwards.
    if not dropna:
        mi = mi.set_levels(["A", "B", np.nan], level="b")
    expected = pd.DataFrame(outputs, index=mi)

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.arm_slow
@pytest.mark.parametrize(
    "datetime1, datetime2",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")),
        (pd.Timedelta("-2 days"), pd.Timedelta("-1 days")),
        (pd.Period("2020-01-01"), pd.Period("2020-02-01")),
    ],
)
@pytest.mark.parametrize("dropna, values", [(True, [12, 3]), (False, [12, 3, 6])])
def test_groupby_dropna_datetime_like_data(
    dropna, values, datetime1, datetime2, unique_nulls_fixture, unique_nulls_fixture2
):
    # 3729
    df = pd.DataFrame(
        {
            "values": [1, 2, 3, 4, 5, 6],
            "dt": [
                datetime1,
                unique_nulls_fixture,
                datetime2,
                unique_nulls_fixture2,
                datetime1,
                datetime1,
            ],
        }
    )

    if dropna:
        indexes = [datetime1, datetime2]
    else:
        indexes = [datetime1, datetime2, np.nan]

    grouped = df.groupby("dt", dropna=dropna).agg({"values": sum})
    expected = pd.DataFrame({"values": values}, index=pd.Index(indexes, name="dt"))

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, data, selected_data, levels",
    [
        pytest.param(
            False,
            {"groups": ["a", "a", "b", np.nan], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0, 0]},
            ["a", "b", np.nan],
            id="dropna_false_has_nan",
        ),
        pytest.param(
            True,
            {"groups": ["a", "a", "b", np.nan], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0]},
            None,
            id="dropna_true_has_nan",
        ),
        pytest.param(
            # no nan in "groups"; dropna=True|False should be same.
            False,
            {"groups": ["a", "a", "b", "c"], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0, 0]},
            None,
            id="dropna_false_no_nan",
        ),
        pytest.param(
            # no nan in "groups"; dropna=True|False should be same.
            True,
            {"groups": ["a", "a", "b", "c"], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0, 0]},
            None,
            id="dropna_true_no_nan",
        ),
    ],
)
def test_groupby_apply_with_dropna_for_multi_index(dropna, data, selected_data, levels):
    # GH 35889

    df = pd.DataFrame(data)
    gb = df.groupby("groups", dropna=dropna)
    result = gb.apply(lambda grp: pd.DataFrame({"values": range(len(grp))}))

    mi_tuples = tuple(zip(data["groups"], selected_data["values"]))
    mi = pd.MultiIndex.from_tuples(mi_tuples, names=["groups", None])
    # Since right now, by default MI will drop NA from levels when we create MI
    # via `from_*`, so we need to add NA for level manually afterwards.
    if not dropna and levels:
        mi = mi.set_levels(levels, level="groups")

    expected = pd.DataFrame(selected_data, index=mi)
    tm.assert_frame_equal(result, expected)


def test_groupby_nan_included():
    # GH 35646
    data = {"group": ["g1", np.nan, "g1", "g2", np.nan], "B": [0, 1, 2, 3, 4]}
    df = pd.DataFrame(data)
    grouped = df.groupby("group", dropna=False)
    result = grouped.indices
    dtype = np.intp
    expected = {
        "g1": np.array([0, 2], dtype=dtype),
        "g2": np.array([3], dtype=dtype),
        np.nan: np.array([1, 4], dtype=dtype),
    }
    for result_values, expected_values in zip(result.values(), expected.values()):
        tm.assert_numpy_array_equal(result_values, expected_values)
    assert np.isnan(list(result.keys())[2])
    assert list(result.keys())[0:2] == ["g1", "g2"]
