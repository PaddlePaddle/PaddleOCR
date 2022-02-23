""" test where we are determining what we are grouping, or getting groups """

import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping

# selection
# --------------------------------


class TestSelection:
    def test_select_bad_cols(self):
        df = DataFrame([[1, 2]], columns=["A", "B"])
        g = df.groupby("A")
        with pytest.raises(KeyError, match="\"Columns not found: 'C'\""):
            g[["C"]]

        with pytest.raises(KeyError, match="^[^A]+$"):
            # A should not be referenced as a bad column...
            # will have to rethink regex if you change message!
            g[["A", "C"]]

    def test_groupby_duplicated_column_errormsg(self):
        # GH7511
        df = DataFrame(
            columns=["A", "B", "A", "C"], data=[range(4), range(2, 6), range(0, 8, 2)]
        )

        msg = "Grouper for 'A' not 1-dimensional"
        with pytest.raises(ValueError, match=msg):
            df.groupby("A")
        with pytest.raises(ValueError, match=msg):
            df.groupby(["A", "B"])

        grouped = df.groupby("B")
        c = grouped.count()
        assert c.columns.nlevels == 1
        assert c.columns.size == 3

    def test_column_select_via_attr(self, df):
        result = df.groupby("A").C.sum()
        expected = df.groupby("A")["C"].sum()
        tm.assert_series_equal(result, expected)

        df["mean"] = 1.5
        result = df.groupby("A").mean()
        expected = df.groupby("A").agg(np.mean)
        tm.assert_frame_equal(result, expected)

    def test_getitem_list_of_columns(self):
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.randn(8),
                "D": np.random.randn(8),
                "E": np.random.randn(8),
            }
        )

        result = df.groupby("A")[["C", "D"]].mean()
        result2 = df.groupby("A")[df.columns[2:4]].mean()

        expected = df.loc[:, ["A", "C", "D"]].groupby("A").mean()

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_getitem_numeric_column_names(self):
        # GH #13731
        df = DataFrame(
            {
                0: list("abcd") * 2,
                2: np.random.randn(8),
                4: np.random.randn(8),
                6: np.random.randn(8),
            }
        )
        result = df.groupby(0)[df.columns[1:3]].mean()
        result2 = df.groupby(0)[[2, 4]].mean()

        expected = df.loc[:, [0, 2, 4]].groupby(0).mean()

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # per GH 23566 this should raise a FutureWarning
        with tm.assert_produces_warning(FutureWarning):
            df.groupby(0)[2, 4].mean()

    def test_getitem_single_list_of_columns(self, df):
        # per GH 23566 this should raise a FutureWarning
        with tm.assert_produces_warning(FutureWarning):
            df.groupby("A")["C", "D"].mean()

    def test_getitem_single_column(self):
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.randn(8),
                "D": np.random.randn(8),
                "E": np.random.randn(8),
            }
        )

        result = df.groupby("A")["C"].mean()

        as_frame = df.loc[:, ["A", "C"]].groupby("A").mean()
        as_series = as_frame.iloc[:, 0]
        expected = as_series

        tm.assert_series_equal(result, expected)

    def test_indices_grouped_by_tuple_with_lambda(self):
        # GH 36158
        df = DataFrame(
            {"Tuples": ((x, y) for x in [0, 1] for y in np.random.randint(3, 5, 5))}
        )

        gb = df.groupby("Tuples")
        gb_lambda = df.groupby(lambda x: df.iloc[x, 0])

        expected = gb.indices
        result = gb_lambda.indices

        tm.assert_dict_equal(result, expected)


# grouping
# --------------------------------


class TestGrouping:
    def test_grouper_index_types(self):
        # related GH5375
        # groupby misbehaving when using a Floatlike index
        df = DataFrame(np.arange(10).reshape(5, 2), columns=list("AB"))
        for index in [
            tm.makeFloatIndex,
            tm.makeStringIndex,
            tm.makeUnicodeIndex,
            tm.makeIntIndex,
            tm.makeDateIndex,
            tm.makePeriodIndex,
        ]:

            df.index = index(len(df))
            df.groupby(list("abcde")).apply(lambda x: x)

            df.index = list(reversed(df.index.tolist()))
            df.groupby(list("abcde")).apply(lambda x: x)

    def test_grouper_multilevel_freq(self):

        # GH 7885
        # with level and freq specified in a pd.Grouper
        from datetime import (
            date,
            timedelta,
        )

        d0 = date.today() - timedelta(days=14)
        dates = date_range(d0, date.today())
        date_index = MultiIndex.from_product([dates, dates], names=["foo", "bar"])
        df = DataFrame(np.random.randint(0, 100, 225), index=date_index)

        # Check string level
        expected = (
            df.reset_index()
            .groupby([pd.Grouper(key="foo", freq="W"), pd.Grouper(key="bar", freq="W")])
            .sum()
        )
        # reset index changes columns dtype to object
        expected.columns = Index([0], dtype="int64")

        result = df.groupby(
            [pd.Grouper(level="foo", freq="W"), pd.Grouper(level="bar", freq="W")]
        ).sum()
        tm.assert_frame_equal(result, expected)

        # Check integer level
        result = df.groupby(
            [pd.Grouper(level=0, freq="W"), pd.Grouper(level=1, freq="W")]
        ).sum()
        tm.assert_frame_equal(result, expected)

    def test_grouper_creation_bug(self):

        # GH 8795
        df = DataFrame({"A": [0, 0, 1, 1, 2, 2], "B": [1, 2, 3, 4, 5, 6]})
        g = df.groupby("A")
        expected = g.sum()

        g = df.groupby(pd.Grouper(key="A"))
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        g = df.groupby(pd.Grouper(key="A", axis=0))
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        result = g.apply(lambda x: x.sum())
        expected["A"] = [0, 2, 4]
        expected = expected.loc[:, ["A", "B"]]
        tm.assert_frame_equal(result, expected)

        # GH14334
        # pd.Grouper(key=...) may be passed in a list
        df = DataFrame(
            {"A": [0, 0, 0, 1, 1, 1], "B": [1, 1, 2, 2, 3, 3], "C": [1, 2, 3, 4, 5, 6]}
        )
        # Group by single column
        expected = df.groupby("A").sum()
        g = df.groupby([pd.Grouper(key="A")])
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        # Group by two columns
        # using a combination of strings and Grouper objects
        expected = df.groupby(["A", "B"]).sum()

        # Group with two Grouper objects
        g = df.groupby([pd.Grouper(key="A"), pd.Grouper(key="B")])
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        # Group with a string and a Grouper object
        g = df.groupby(["A", pd.Grouper(key="B")])
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        # Group with a Grouper object and a string
        g = df.groupby([pd.Grouper(key="A"), "B"])
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        # GH8866
        s = Series(
            np.arange(8, dtype="int64"),
            index=MultiIndex.from_product(
                [list("ab"), range(2), date_range("20130101", periods=2)],
                names=["one", "two", "three"],
            ),
        )
        result = s.groupby(pd.Grouper(level="three", freq="M")).sum()
        expected = Series(
            [28],
            index=pd.DatetimeIndex([Timestamp("2013-01-31")], freq="M", name="three"),
        )
        tm.assert_series_equal(result, expected)

        # just specifying a level breaks
        result = s.groupby(pd.Grouper(level="one")).sum()
        expected = s.groupby(level="one").sum()
        tm.assert_series_equal(result, expected)

    def test_grouper_column_and_index(self):
        # GH 14327

        # Grouping a multi-index frame by a column and an index level should
        # be equivalent to resetting the index and grouping by two columns
        idx = MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2), ("b", 3)]
        )
        idx.names = ["outer", "inner"]
        df_multi = DataFrame(
            {"A": np.arange(6), "B": ["one", "one", "two", "two", "one", "one"]},
            index=idx,
        )
        result = df_multi.groupby(["B", pd.Grouper(level="inner")]).mean()
        expected = df_multi.reset_index().groupby(["B", "inner"]).mean()
        tm.assert_frame_equal(result, expected)

        # Test the reverse grouping order
        result = df_multi.groupby([pd.Grouper(level="inner"), "B"]).mean()
        expected = df_multi.reset_index().groupby(["inner", "B"]).mean()
        tm.assert_frame_equal(result, expected)

        # Grouping a single-index frame by a column and the index should
        # be equivalent to resetting the index and grouping by two columns
        df_single = df_multi.reset_index("outer")
        result = df_single.groupby(["B", pd.Grouper(level="inner")]).mean()
        expected = df_single.reset_index().groupby(["B", "inner"]).mean()
        tm.assert_frame_equal(result, expected)

        # Test the reverse grouping order
        result = df_single.groupby([pd.Grouper(level="inner"), "B"]).mean()
        expected = df_single.reset_index().groupby(["inner", "B"]).mean()
        tm.assert_frame_equal(result, expected)

    def test_groupby_levels_and_columns(self):
        # GH9344, GH9049
        idx_names = ["x", "y"]
        idx = MultiIndex.from_tuples([(1, 1), (1, 2), (3, 4), (5, 6)], names=idx_names)
        df = DataFrame(np.arange(12).reshape(-1, 3), index=idx)

        by_levels = df.groupby(level=idx_names).mean()
        # reset_index changes columns dtype to object
        by_columns = df.reset_index().groupby(idx_names).mean()

        # without casting, by_columns.columns is object-dtype
        by_columns.columns = by_columns.columns.astype(np.int64)
        tm.assert_frame_equal(by_levels, by_columns)

    def test_groupby_categorical_index_and_columns(self, observed):
        # GH18432, adapted for GH25871
        columns = ["A", "B", "A", "B"]
        categories = ["B", "A"]
        data = np.array(
            [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]], int
        )
        cat_columns = CategoricalIndex(columns, categories=categories, ordered=True)
        df = DataFrame(data=data, columns=cat_columns)
        result = df.groupby(axis=1, level=0, observed=observed).sum()
        expected_data = np.array([[4, 2], [4, 2], [4, 2], [4, 2], [4, 2]], int)
        expected_columns = CategoricalIndex(
            categories, categories=categories, ordered=True
        )
        expected = DataFrame(data=expected_data, columns=expected_columns)
        tm.assert_frame_equal(result, expected)

        # test transposed version
        df = DataFrame(data.T, index=cat_columns)
        result = df.groupby(axis=0, level=0, observed=observed).sum()
        expected = DataFrame(data=expected_data.T, index=expected_columns)
        tm.assert_frame_equal(result, expected)

    def test_grouper_getting_correct_binner(self):

        # GH 10063
        # using a non-time-based grouper and a time-based grouper
        # and specifying levels
        df = DataFrame(
            {"A": 1},
            index=MultiIndex.from_product(
                [list("ab"), date_range("20130101", periods=80)], names=["one", "two"]
            ),
        )
        result = df.groupby(
            [pd.Grouper(level="one"), pd.Grouper(level="two", freq="M")]
        ).sum()
        expected = DataFrame(
            {"A": [31, 28, 21, 31, 28, 21]},
            index=MultiIndex.from_product(
                [list("ab"), date_range("20130101", freq="M", periods=3)],
                names=["one", "two"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_grouper_iter(self, df):
        assert sorted(df.groupby("A").grouper) == ["bar", "foo"]

    def test_empty_groups(self, df):
        # see gh-1048
        with pytest.raises(ValueError, match="No group keys passed!"):
            df.groupby([])

    def test_groupby_grouper(self, df):
        grouped = df.groupby("A")

        result = df.groupby(grouped.grouper).mean()
        expected = grouped.mean()
        tm.assert_frame_equal(result, expected)

    def test_groupby_dict_mapping(self):
        # GH #679
        from pandas import Series

        s = Series({"T1": 5})
        result = s.groupby({"T1": "T2"}).agg(sum)
        expected = s.groupby(["T2"]).agg(sum)
        tm.assert_series_equal(result, expected)

        s = Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
        mapping = {"a": 0, "b": 0, "c": 1, "d": 1}

        result = s.groupby(mapping).mean()
        result2 = s.groupby(mapping).agg(np.mean)
        expected = s.groupby([0, 0, 1, 1]).mean()
        expected2 = s.groupby([0, 0, 1, 1]).mean()
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result, result2)
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "index",
        [
            [0, 1, 2, 3],
            ["a", "b", "c", "d"],
            [Timestamp(2021, 7, 28 + i) for i in range(4)],
        ],
    )
    def test_groupby_series_named_with_tuple(self, frame_or_series, index):
        # GH 42731
        obj = frame_or_series([1, 2, 3, 4], index=index)
        groups = Series([1, 0, 1, 0], index=index, name=("a", "a"))
        result = obj.groupby(groups).last()
        expected = frame_or_series([4, 3])
        expected.index.name = ("a", "a")
        tm.assert_equal(result, expected)

    def test_groupby_grouper_f_sanity_checked(self):
        dates = date_range("01-Jan-2013", periods=12, freq="MS")
        ts = Series(np.random.randn(12), index=dates)

        # GH3035
        # index.map is used to apply grouper to the index
        # if it fails on the elements, map tries it on the entire index as
        # a sequence. That can yield invalid results that cause trouble
        # down the line.
        # the surprise comes from using key[0:6] rather than str(key)[0:6]
        # when the elements are Timestamp.
        # the result is Index[0:6], very confusing.

        msg = r"Grouper result violates len\(labels\) == len\(data\)"
        with pytest.raises(AssertionError, match=msg):
            ts.groupby(lambda key: key[0:6])

    def test_grouping_error_on_multidim_input(self, df):
        msg = "Grouper for '<class 'pandas.core.frame.DataFrame'>' not 1-dimensional"
        with pytest.raises(ValueError, match=msg):
            Grouping(df.index, df[["A", "A"]])

    def test_multiindex_passthru(self):

        # GH 7997
        # regression from 0.14.1
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        df.columns = MultiIndex.from_tuples([(0, 1), (1, 1), (2, 1)])

        result = df.groupby(axis=1, level=[0, 1]).first()
        tm.assert_frame_equal(result, df)

    def test_multiindex_negative_level(self, mframe):
        # GH 13901
        result = mframe.groupby(level=-1).sum()
        expected = mframe.groupby(level="second").sum()
        tm.assert_frame_equal(result, expected)

        result = mframe.groupby(level=-2).sum()
        expected = mframe.groupby(level="first").sum()
        tm.assert_frame_equal(result, expected)

        result = mframe.groupby(level=[-2, -1]).sum()
        expected = mframe
        tm.assert_frame_equal(result, expected)

        result = mframe.groupby(level=[-1, "first"]).sum()
        expected = mframe.groupby(level=["second", "first"]).sum()
        tm.assert_frame_equal(result, expected)

    def test_multifunc_select_col_integer_cols(self, df):
        df.columns = np.arange(len(df.columns))

        # it works!
        df.groupby(1, as_index=False)[2].agg({"Q": np.mean})

    def test_multiindex_columns_empty_level(self):
        lst = [["count", "values"], ["to filter", ""]]
        midx = MultiIndex.from_tuples(lst)

        df = DataFrame([[1, "A"]], columns=midx)

        grouped = df.groupby("to filter").groups
        assert grouped["A"] == [0]

        grouped = df.groupby([("to filter", "")]).groups
        assert grouped["A"] == [0]

        df = DataFrame([[1, "A"], [2, "B"]], columns=midx)

        expected = df.groupby("to filter").groups
        result = df.groupby([("to filter", "")]).groups
        assert result == expected

        df = DataFrame([[1, "A"], [2, "A"]], columns=midx)

        expected = df.groupby("to filter").groups
        result = df.groupby([("to filter", "")]).groups
        tm.assert_dict_equal(result, expected)

    def test_groupby_multiindex_tuple(self):
        # GH 17979
        df = DataFrame(
            [[1, 2, 3, 4], [3, 4, 5, 6], [1, 4, 2, 3]],
            columns=MultiIndex.from_arrays([["a", "b", "b", "c"], [1, 1, 2, 2]]),
        )
        expected = df.groupby([("b", 1)]).groups
        result = df.groupby(("b", 1)).groups
        tm.assert_dict_equal(expected, result)

        df2 = DataFrame(
            df.values,
            columns=MultiIndex.from_arrays(
                [["a", "b", "b", "c"], ["d", "d", "e", "e"]]
            ),
        )
        expected = df2.groupby([("b", "d")]).groups
        result = df.groupby(("b", 1)).groups
        tm.assert_dict_equal(expected, result)

        df3 = DataFrame(df.values, columns=[("a", "d"), ("b", "d"), ("b", "e"), "c"])
        expected = df3.groupby([("b", "d")]).groups
        result = df.groupby(("b", 1)).groups
        tm.assert_dict_equal(expected, result)

    @pytest.mark.parametrize("sort", [True, False])
    def test_groupby_level(self, sort, mframe, df):
        # GH 17537
        frame = mframe
        deleveled = frame.reset_index()

        result0 = frame.groupby(level=0, sort=sort).sum()
        result1 = frame.groupby(level=1, sort=sort).sum()

        expected0 = frame.groupby(deleveled["first"].values, sort=sort).sum()
        expected1 = frame.groupby(deleveled["second"].values, sort=sort).sum()

        expected0.index.name = "first"
        expected1.index.name = "second"

        assert result0.index.name == "first"
        assert result1.index.name == "second"

        tm.assert_frame_equal(result0, expected0)
        tm.assert_frame_equal(result1, expected1)
        assert result0.index.name == frame.index.names[0]
        assert result1.index.name == frame.index.names[1]

        # groupby level name
        result0 = frame.groupby(level="first", sort=sort).sum()
        result1 = frame.groupby(level="second", sort=sort).sum()
        tm.assert_frame_equal(result0, expected0)
        tm.assert_frame_equal(result1, expected1)

        # axis=1

        result0 = frame.T.groupby(level=0, axis=1, sort=sort).sum()
        result1 = frame.T.groupby(level=1, axis=1, sort=sort).sum()
        tm.assert_frame_equal(result0, expected0.T)
        tm.assert_frame_equal(result1, expected1.T)

        # raise exception for non-MultiIndex
        msg = "level > 0 or level < -1 only valid with MultiIndex"
        with pytest.raises(ValueError, match=msg):
            df.groupby(level=1)

    def test_groupby_level_index_names(self, axis):
        # GH4014 this used to raise ValueError since 'exp'>1 (in py2)
        df = DataFrame({"exp": ["A"] * 3 + ["B"] * 3, "var1": range(6)}).set_index(
            "exp"
        )
        if axis in (1, "columns"):
            df = df.T
        df.groupby(level="exp", axis=axis)
        msg = f"level name foo is not the name of the {df._get_axis_name(axis)}"
        with pytest.raises(ValueError, match=msg):
            df.groupby(level="foo", axis=axis)

    @pytest.mark.parametrize("sort", [True, False])
    def test_groupby_level_with_nas(self, sort):
        # GH 17537
        index = MultiIndex(
            levels=[[1, 0], [0, 1, 2, 3]],
            codes=[[1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]],
        )

        # factorizing doesn't confuse things
        s = Series(np.arange(8.0), index=index)
        result = s.groupby(level=0, sort=sort).sum()
        expected = Series([6.0, 22.0], index=[0, 1])
        tm.assert_series_equal(result, expected)

        index = MultiIndex(
            levels=[[1, 0], [0, 1, 2, 3]],
            codes=[[1, 1, 1, 1, -1, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]],
        )

        # factorizing doesn't confuse things
        s = Series(np.arange(8.0), index=index)
        result = s.groupby(level=0, sort=sort).sum()
        expected = Series([6.0, 18.0], index=[0.0, 1.0])
        tm.assert_series_equal(result, expected)

    def test_groupby_args(self, mframe):
        # PR8618 and issue 8015
        frame = mframe

        msg = "You have to supply one of 'by' and 'level'"
        with pytest.raises(TypeError, match=msg):
            frame.groupby()

        msg = "You have to supply one of 'by' and 'level'"
        with pytest.raises(TypeError, match=msg):
            frame.groupby(by=None, level=None)

    @pytest.mark.parametrize(
        "sort,labels",
        [
            [True, [2, 2, 2, 0, 0, 1, 1, 3, 3, 3]],
            [False, [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]],
        ],
    )
    def test_level_preserve_order(self, sort, labels, mframe):
        # GH 17537
        grouped = mframe.groupby(level=0, sort=sort)
        exp_labels = np.array(labels, np.intp)
        tm.assert_almost_equal(grouped.grouper.codes[0], exp_labels)

    def test_grouping_labels(self, mframe):
        grouped = mframe.groupby(mframe.index.get_level_values(0))
        exp_labels = np.array([2, 2, 2, 0, 0, 1, 1, 3, 3, 3], dtype=np.intp)
        tm.assert_almost_equal(grouped.grouper.codes[0], exp_labels)

    def test_list_grouper_with_nat(self):
        # GH 14715
        df = DataFrame({"date": date_range("1/1/2011", periods=365, freq="D")})
        df.iloc[-1] = pd.NaT
        grouper = pd.Grouper(key="date", freq="AS")

        # Grouper in a list grouping
        result = df.groupby([grouper])
        expected = {Timestamp("2011-01-01"): Index(list(range(364)))}
        tm.assert_dict_equal(result.groups, expected)

        # Test case without a list
        result = df.groupby(grouper)
        expected = {Timestamp("2011-01-01"): 365}
        tm.assert_dict_equal(result.groups, expected)

    @pytest.mark.parametrize(
        "func,expected",
        [
            (
                "transform",
                Series(name=2, dtype=np.float64, index=Index([])),
            ),
            (
                "agg",
                Series(name=2, dtype=np.float64, index=pd.Float64Index([], name=1)),
            ),
            (
                "apply",
                Series(name=2, dtype=np.float64, index=pd.Float64Index([], name=1)),
            ),
        ],
    )
    def test_evaluate_with_empty_groups(self, func, expected):
        # 26208
        # test transform'ing empty groups
        # (not testing other agg fns, because they return
        # different index objects.
        df = DataFrame({1: [], 2: []})
        g = df.groupby(1)
        result = getattr(g[2], func)(lambda x: x)
        tm.assert_series_equal(result, expected)

    def test_groupby_empty(self):
        # https://github.com/pandas-dev/pandas/issues/27190
        s = Series([], name="name", dtype="float64")
        gr = s.groupby([])

        result = gr.mean()
        tm.assert_series_equal(result, s)

        # check group properties
        assert len(gr.grouper.groupings) == 1
        tm.assert_numpy_array_equal(
            gr.grouper.group_info[0], np.array([], dtype=np.dtype(np.intp))
        )

        tm.assert_numpy_array_equal(
            gr.grouper.group_info[1], np.array([], dtype=np.dtype("int"))
        )

        assert gr.grouper.group_info[2] == 0

        # check name
        assert s.groupby(s).grouper.names == ["name"]

    def test_groupby_level_index_value_all_na(self):
        # issue 20519
        df = DataFrame(
            [["x", np.nan, 10], [None, np.nan, 20]], columns=["A", "B", "C"]
        ).set_index(["A", "B"])
        result = df.groupby(level=["A", "B"]).sum()
        expected = DataFrame(
            data=[],
            index=MultiIndex(
                levels=[Index(["x"], dtype="object"), Index([], dtype="float64")],
                codes=[[], []],
                names=["A", "B"],
            ),
            columns=["C"],
            dtype="int64",
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_multiindex_level_empty(self):
        # https://github.com/pandas-dev/pandas/issues/31670
        df = DataFrame(
            [[123, "a", 1.0], [123, "b", 2.0]], columns=["id", "category", "value"]
        )
        df = df.set_index(["id", "category"])
        empty = df[df.value < 0]
        result = empty.groupby("id").sum()
        expected = DataFrame(
            dtype="float64", columns=["value"], index=pd.Int64Index([], name="id")
        )
        tm.assert_frame_equal(result, expected)


# get_group
# --------------------------------


class TestGetGroup:
    def test_get_group(self):
        # GH 5267
        # be datelike friendly
        df = DataFrame(
            {
                "DATE": pd.to_datetime(
                    [
                        "10-Oct-2013",
                        "10-Oct-2013",
                        "10-Oct-2013",
                        "11-Oct-2013",
                        "11-Oct-2013",
                        "11-Oct-2013",
                    ]
                ),
                "label": ["foo", "foo", "bar", "foo", "foo", "bar"],
                "VAL": [1, 2, 3, 4, 5, 6],
            }
        )

        g = df.groupby("DATE")
        key = list(g.groups)[0]
        result1 = g.get_group(key)
        result2 = g.get_group(Timestamp(key).to_pydatetime())
        result3 = g.get_group(str(Timestamp(key)))
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)

        g = df.groupby(["DATE", "label"])

        key = list(g.groups)[0]
        result1 = g.get_group(key)
        result2 = g.get_group((Timestamp(key[0]).to_pydatetime(), key[1]))
        result3 = g.get_group((str(Timestamp(key[0])), key[1]))
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)

        # must pass a same-length tuple with multiple keys
        msg = "must supply a tuple to get_group with multiple grouping keys"
        with pytest.raises(ValueError, match=msg):
            g.get_group("foo")
        with pytest.raises(ValueError, match=msg):
            g.get_group("foo")
        msg = "must supply a same-length tuple to get_group with multiple grouping keys"
        with pytest.raises(ValueError, match=msg):
            g.get_group(("foo", "bar", "baz"))

    def test_get_group_empty_bins(self, observed):

        d = DataFrame([3, 1, 7, 6])
        bins = [0, 5, 10, 15]
        g = d.groupby(pd.cut(d[0], bins), observed=observed)

        # TODO: should prob allow a str of Interval work as well
        # IOW '(0, 5]'
        result = g.get_group(pd.Interval(0, 5))
        expected = DataFrame([3, 1], index=[0, 1])
        tm.assert_frame_equal(result, expected)

        msg = r"Interval\(10, 15, closed='right'\)"
        with pytest.raises(KeyError, match=msg):
            g.get_group(pd.Interval(10, 15))

    def test_get_group_grouped_by_tuple(self):
        # GH 8121
        df = DataFrame([[(1,), (1, 2), (1,), (1, 2)]], index=["ids"]).T
        gr = df.groupby("ids")
        expected = DataFrame({"ids": [(1,), (1,)]}, index=[0, 2])
        result = gr.get_group((1,))
        tm.assert_frame_equal(result, expected)

        dt = pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-01", "2010-01-02"])
        df = DataFrame({"ids": [(x,) for x in dt]})
        gr = df.groupby("ids")
        result = gr.get_group(("2010-01-01",))
        expected = DataFrame({"ids": [(dt[0],), (dt[0],)]}, index=[0, 2])
        tm.assert_frame_equal(result, expected)

    def test_get_group_grouped_by_tuple_with_lambda(self):
        # GH 36158
        df = DataFrame(
            {"Tuples": ((x, y) for x in [0, 1] for y in np.random.randint(3, 5, 5))}
        )

        gb = df.groupby("Tuples")
        gb_lambda = df.groupby(lambda x: df.iloc[x, 0])

        expected = gb.get_group(list(gb.groups.keys())[0])
        result = gb_lambda.get_group(list(gb_lambda.groups.keys())[0])

        tm.assert_frame_equal(result, expected)

    def test_groupby_with_empty(self):
        index = pd.DatetimeIndex(())
        data = ()
        series = Series(data, index, dtype=object)
        grouper = pd.Grouper(freq="D")
        grouped = series.groupby(grouper)
        assert next(iter(grouped), None) is None

    def test_groupby_with_single_column(self):
        df = DataFrame({"a": list("abssbab")})
        tm.assert_frame_equal(df.groupby("a").get_group("a"), df.iloc[[0, 5]])
        # GH 13530
        exp = DataFrame(index=Index(["a", "b", "s"], name="a"))
        tm.assert_frame_equal(df.groupby("a").count(), exp)
        tm.assert_frame_equal(df.groupby("a").sum(), exp)
        tm.assert_frame_equal(df.groupby("a").nth(1), exp)

    def test_gb_key_len_equal_axis_len(self):
        # GH16843
        # test ensures that index and column keys are recognized correctly
        # when number of keys equals axis length of groupby
        df = DataFrame(
            [["foo", "bar", "B", 1], ["foo", "bar", "B", 2], ["foo", "baz", "C", 3]],
            columns=["first", "second", "third", "one"],
        )
        df = df.set_index(["first", "second"])
        df = df.groupby(["first", "second", "third"]).size()
        assert df.loc[("foo", "bar", "B")] == 2
        assert df.loc[("foo", "baz", "C")] == 1


# groups & iteration
# --------------------------------


class TestIteration:
    def test_groups(self, df):
        grouped = df.groupby(["A"])
        groups = grouped.groups
        assert groups is grouped.groups  # caching works

        for k, v in grouped.groups.items():
            assert (df.loc[v]["A"] == k).all()

        grouped = df.groupby(["A", "B"])
        groups = grouped.groups
        assert groups is grouped.groups  # caching works

        for k, v in grouped.groups.items():
            assert (df.loc[v]["A"] == k[0]).all()
            assert (df.loc[v]["B"] == k[1]).all()

    def test_grouping_is_iterable(self, tsframe):
        # this code path isn't used anywhere else
        # not sure it's useful
        grouped = tsframe.groupby([lambda x: x.weekday(), lambda x: x.year])

        # test it works
        for g in grouped.grouper.groupings[0]:
            pass

    def test_multi_iter(self):
        s = Series(np.arange(6))
        k1 = np.array(["a", "a", "a", "b", "b", "b"])
        k2 = np.array(["1", "2", "1", "2", "1", "2"])

        grouped = s.groupby([k1, k2])

        iterated = list(grouped)
        expected = [
            ("a", "1", s[[0, 2]]),
            ("a", "2", s[[1]]),
            ("b", "1", s[[4]]),
            ("b", "2", s[[3, 5]]),
        ]
        for i, ((one, two), three) in enumerate(iterated):
            e1, e2, e3 = expected[i]
            assert e1 == one
            assert e2 == two
            tm.assert_series_equal(three, e3)

    def test_multi_iter_frame(self, three_group):
        k1 = np.array(["b", "b", "b", "a", "a", "a"])
        k2 = np.array(["1", "2", "1", "2", "1", "2"])
        df = DataFrame(
            {"v1": np.random.randn(6), "v2": np.random.randn(6), "k1": k1, "k2": k2},
            index=["one", "two", "three", "four", "five", "six"],
        )

        grouped = df.groupby(["k1", "k2"])

        # things get sorted!
        iterated = list(grouped)
        idx = df.index
        expected = [
            ("a", "1", df.loc[idx[[4]]]),
            ("a", "2", df.loc[idx[[3, 5]]]),
            ("b", "1", df.loc[idx[[0, 2]]]),
            ("b", "2", df.loc[idx[[1]]]),
        ]
        for i, ((one, two), three) in enumerate(iterated):
            e1, e2, e3 = expected[i]
            assert e1 == one
            assert e2 == two
            tm.assert_frame_equal(three, e3)

        # don't iterate through groups with no data
        df["k1"] = np.array(["b", "b", "b", "a", "a", "a"])
        df["k2"] = np.array(["1", "1", "1", "2", "2", "2"])
        grouped = df.groupby(["k1", "k2"])
        groups = {key: gp for key, gp in grouped}
        assert len(groups) == 2

        # axis = 1
        three_levels = three_group.groupby(["A", "B", "C"]).mean()
        grouped = three_levels.T.groupby(axis=1, level=(1, 2))
        for key, group in grouped:
            pass

    def test_dictify(self, df):
        dict(iter(df.groupby("A")))
        dict(iter(df.groupby(["A", "B"])))
        dict(iter(df["C"].groupby(df["A"])))
        dict(iter(df["C"].groupby([df["A"], df["B"]])))
        dict(iter(df.groupby("A")["C"]))
        dict(iter(df.groupby(["A", "B"])["C"]))

    def test_groupby_with_small_elem(self):
        # GH 8542
        # length=2
        df = DataFrame(
            {"event": ["start", "start"], "change": [1234, 5678]},
            index=pd.DatetimeIndex(["2014-09-10", "2013-10-10"]),
        )
        grouped = df.groupby([pd.Grouper(freq="M"), "event"])
        assert len(grouped.groups) == 2
        assert grouped.ngroups == 2
        assert (Timestamp("2014-09-30"), "start") in grouped.groups
        assert (Timestamp("2013-10-31"), "start") in grouped.groups

        res = grouped.get_group((Timestamp("2014-09-30"), "start"))
        tm.assert_frame_equal(res, df.iloc[[0], :])
        res = grouped.get_group((Timestamp("2013-10-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[1], :])

        df = DataFrame(
            {"event": ["start", "start", "start"], "change": [1234, 5678, 9123]},
            index=pd.DatetimeIndex(["2014-09-10", "2013-10-10", "2014-09-15"]),
        )
        grouped = df.groupby([pd.Grouper(freq="M"), "event"])
        assert len(grouped.groups) == 2
        assert grouped.ngroups == 2
        assert (Timestamp("2014-09-30"), "start") in grouped.groups
        assert (Timestamp("2013-10-31"), "start") in grouped.groups

        res = grouped.get_group((Timestamp("2014-09-30"), "start"))
        tm.assert_frame_equal(res, df.iloc[[0, 2], :])
        res = grouped.get_group((Timestamp("2013-10-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[1], :])

        # length=3
        df = DataFrame(
            {"event": ["start", "start", "start"], "change": [1234, 5678, 9123]},
            index=pd.DatetimeIndex(["2014-09-10", "2013-10-10", "2014-08-05"]),
        )
        grouped = df.groupby([pd.Grouper(freq="M"), "event"])
        assert len(grouped.groups) == 3
        assert grouped.ngroups == 3
        assert (Timestamp("2014-09-30"), "start") in grouped.groups
        assert (Timestamp("2013-10-31"), "start") in grouped.groups
        assert (Timestamp("2014-08-31"), "start") in grouped.groups

        res = grouped.get_group((Timestamp("2014-09-30"), "start"))
        tm.assert_frame_equal(res, df.iloc[[0], :])
        res = grouped.get_group((Timestamp("2013-10-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[1], :])
        res = grouped.get_group((Timestamp("2014-08-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[2], :])

    def test_grouping_string_repr(self):
        # GH 13394
        mi = MultiIndex.from_arrays([list("AAB"), list("aba")])
        df = DataFrame([[1, 2, 3]], columns=mi)
        gr = df.groupby(df[("A", "a")])

        result = gr.grouper.groupings[0].__repr__()
        expected = "Grouping(('A', 'a'))"
        assert result == expected
