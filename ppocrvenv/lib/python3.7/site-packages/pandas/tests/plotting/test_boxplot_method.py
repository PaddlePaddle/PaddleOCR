""" Test cases for .boxplot method """

import itertools
import string

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    TestPlotBase,
    _check_plot_works,
)

import pandas.plotting as plotting

pytestmark = pytest.mark.slow


@td.skip_if_no_mpl
class TestDataFramePlots(TestPlotBase):
    def test_boxplot_legacy1(self):
        df = DataFrame(
            np.random.randn(6, 4),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        df["indic"] = ["foo", "bar"] * 3
        df["indic2"] = ["foo", "bar", "foo"] * 2

        _check_plot_works(df.boxplot, return_type="dict")
        _check_plot_works(df.boxplot, column=["one", "two"], return_type="dict")
        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(df.boxplot, column=["one", "two"], by="indic")
        _check_plot_works(df.boxplot, column="one", by=["indic", "indic2"])
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(df.boxplot, by="indic")
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(df.boxplot, by=["indic", "indic2"])
        _check_plot_works(plotting._core.boxplot, data=df["one"], return_type="dict")
        _check_plot_works(df.boxplot, notch=1, return_type="dict")
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(df.boxplot, by="indic", notch=1)

    def test_boxplot_legacy2(self):
        df = DataFrame(np.random.rand(10, 2), columns=["Col1", "Col2"])
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(df.boxplot, by="X")

        # When ax is supplied and required number of axes is 1,
        # passed ax should be used:
        fig, ax = self.plt.subplots()
        axes = df.boxplot("Col1", by="X", ax=ax)
        ax_axes = ax.axes
        assert ax_axes is axes

        fig, ax = self.plt.subplots()
        axes = df.groupby("Y").boxplot(ax=ax, return_type="axes")
        ax_axes = ax.axes
        assert ax_axes is axes["A"]

        # Multiple columns with an ax argument should use same figure
        fig, ax = self.plt.subplots()
        with tm.assert_produces_warning(UserWarning):
            axes = df.boxplot(
                column=["Col1", "Col2"], by="X", ax=ax, return_type="axes"
            )
        assert axes["Col1"].get_figure() is fig

        # When by is None, check that all relevant lines are present in the
        # dict
        fig, ax = self.plt.subplots()
        d = df.boxplot(ax=ax, return_type="dict")
        lines = list(itertools.chain.from_iterable(d.values()))
        assert len(ax.get_lines()) == len(lines)

    def test_boxplot_return_type_none(self):
        # GH 12216; return_type=None & by=None -> axes
        result = self.hist_df.boxplot()
        assert isinstance(result, self.plt.Axes)

    def test_boxplot_return_type_legacy(self):
        # API change in https://github.com/pandas-dev/pandas/pull/7096
        import matplotlib as mpl  # noqa

        df = DataFrame(
            np.random.randn(6, 4),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        msg = "return_type must be {'axes', 'dict', 'both'}"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(return_type="NOT_A_TYPE")

        result = df.boxplot()
        self._check_box_return_type(result, "axes")

        with tm.assert_produces_warning(False):
            result = df.boxplot(return_type="dict")
        self._check_box_return_type(result, "dict")

        with tm.assert_produces_warning(False):
            result = df.boxplot(return_type="axes")
        self._check_box_return_type(result, "axes")

        with tm.assert_produces_warning(False):
            result = df.boxplot(return_type="both")
        self._check_box_return_type(result, "both")

    def test_boxplot_axis_limits(self):
        def _check_ax_limits(col, ax):
            y_min, y_max = ax.get_ylim()
            assert y_min <= col.min()
            assert y_max >= col.max()

        df = self.hist_df.copy()
        df["age"] = np.random.randint(1, 20, df.shape[0])
        # One full row
        height_ax, weight_ax = df.boxplot(["height", "weight"], by="category")
        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        assert weight_ax._sharey == height_ax

        # Two rows, one partial
        p = df.boxplot(["height", "weight", "age"], by="category")
        height_ax, weight_ax, age_ax = p[0, 0], p[0, 1], p[1, 0]
        dummy_ax = p[1, 1]

        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        _check_ax_limits(df["age"], age_ax)
        assert weight_ax._sharey == height_ax
        assert age_ax._sharey == height_ax
        assert dummy_ax._sharey is None

    def test_boxplot_empty_column(self):
        df = DataFrame(np.random.randn(20, 4))
        df.loc[:, 0] = np.nan
        _check_plot_works(df.boxplot, return_type="axes")

    def test_figsize(self):
        df = DataFrame(np.random.rand(10, 5), columns=["A", "B", "C", "D", "E"])
        result = df.boxplot(return_type="axes", figsize=(12, 8))
        assert result.figure.bbox_inches.width == 12
        assert result.figure.bbox_inches.height == 8

    def test_fontsize(self):
        df = DataFrame({"a": [1, 2, 3, 4, 5, 6]})
        self._check_ticks_props(
            df.boxplot("a", fontsize=16), xlabelsize=16, ylabelsize=16
        )

    def test_boxplot_numeric_data(self):
        # GH 22799
        df = DataFrame(
            {
                "a": date_range("2012-01-01", periods=100),
                "b": np.random.randn(100),
                "c": np.random.randn(100) + 2,
                "d": date_range("2012-01-01", periods=100).astype(str),
                "e": date_range("2012-01-01", periods=100, tz="UTC"),
                "f": timedelta_range("1 days", periods=100),
            }
        )
        ax = df.plot(kind="box")
        assert [x.get_text() for x in ax.get_xticklabels()] == ["b", "c"]

    @pytest.mark.parametrize(
        "colors_kwd, expected",
        [
            (
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
            ),
            ({"boxes": "r"}, {"boxes": "r"}),
            ("r", {"boxes": "r", "whiskers": "r", "medians": "r", "caps": "r"}),
        ],
    )
    def test_color_kwd(self, colors_kwd, expected):
        # GH: 26214
        df = DataFrame(np.random.rand(10, 2))
        result = df.boxplot(color=colors_kwd, return_type="dict")
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize(
        "scheme,expected",
        [
            (
                "dark_background",
                {
                    "boxes": "#8dd3c7",
                    "whiskers": "#8dd3c7",
                    "medians": "#bfbbd9",
                    "caps": "#8dd3c7",
                },
            ),
            (
                "default",
                {
                    "boxes": "#1f77b4",
                    "whiskers": "#1f77b4",
                    "medians": "#2ca02c",
                    "caps": "#1f77b4",
                },
            ),
        ],
    )
    def test_colors_in_theme(self, scheme, expected):
        # GH: 40769
        df = DataFrame(np.random.rand(10, 2))
        import matplotlib.pyplot as plt

        plt.style.use(scheme)
        result = df.plot.box(return_type="dict")
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize(
        "dict_colors, msg",
        [({"boxes": "r", "invalid_key": "r"}, "invalid key 'invalid_key'")],
    )
    def test_color_kwd_errors(self, dict_colors, msg):
        # GH: 26214
        df = DataFrame(np.random.rand(10, 2))
        with pytest.raises(ValueError, match=msg):
            df.boxplot(color=dict_colors, return_type="dict")

    @pytest.mark.parametrize(
        "props, expected",
        [
            ("boxprops", "boxes"),
            ("whiskerprops", "whiskers"),
            ("capprops", "caps"),
            ("medianprops", "medians"),
        ],
    )
    def test_specified_props_kwd(self, props, expected):
        # GH 30346
        df = DataFrame({k: np.random.random(100) for k in "ABC"})
        kwd = {props: {"color": "C1"}}
        result = df.boxplot(return_type="dict", **kwd)

        assert result[expected][0].get_color() == "C1"


@td.skip_if_no_mpl
class TestDataFrameGroupByPlots(TestPlotBase):
    def test_boxplot_legacy1(self):
        grouped = self.hist_df.groupby(by="gender")
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(grouped.boxplot, return_type="axes")
        self._check_axes_shape(list(axes.values), axes_num=2, layout=(1, 2))
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type="axes")
        self._check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_boxplot_legacy2(self):
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(np.random.rand(10, 3), index=MultiIndex.from_tuples(tuples))
        grouped = df.groupby(level=1)
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(grouped.boxplot, return_type="axes")
        self._check_axes_shape(list(axes.values), axes_num=10, layout=(4, 3))

        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type="axes")
        self._check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_boxplot_legacy3(self):
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(np.random.rand(10, 3), index=MultiIndex.from_tuples(tuples))
        grouped = df.unstack(level=1).groupby(level=0, axis=1)
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(grouped.boxplot, return_type="axes")
        self._check_axes_shape(list(axes.values), axes_num=3, layout=(2, 2))
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type="axes")
        self._check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_grouped_plot_fignums(self):
        n = 10
        weight = Series(np.random.normal(166, 20, size=n))
        height = Series(np.random.normal(60, 10, size=n))
        with tm.RNGContext(42):
            gender = np.random.choice(["male", "female"], size=n)
        df = DataFrame({"height": height, "weight": weight, "gender": gender})
        gb = df.groupby("gender")

        res = gb.plot()
        assert len(self.plt.get_fignums()) == 2
        assert len(res) == 2
        tm.close()

        res = gb.boxplot(return_type="axes")
        assert len(self.plt.get_fignums()) == 1
        assert len(res) == 2
        tm.close()

        # now works with GH 5610 as gender is excluded
        res = df.groupby("gender").hist()
        tm.close()

    def test_grouped_box_return_type(self):
        df = self.hist_df

        # old style: return_type=None
        result = df.boxplot(by="gender")
        assert isinstance(result, np.ndarray)
        self._check_box_return_type(
            result, None, expected_keys=["height", "weight", "category"]
        )

        # now for groupby
        result = df.groupby("gender").boxplot(return_type="dict")
        self._check_box_return_type(result, "dict", expected_keys=["Male", "Female"])

        columns2 = "X B C D A G Y N Q O".split()
        df2 = DataFrame(np.random.randn(50, 10), columns=columns2)
        categories2 = "A B C D E F G H I J".split()
        df2["category"] = categories2 * 5

        for t in ["dict", "axes", "both"]:
            returned = df.groupby("classroom").boxplot(return_type=t)
            self._check_box_return_type(returned, t, expected_keys=["A", "B", "C"])

            returned = df.boxplot(by="classroom", return_type=t)
            self._check_box_return_type(
                returned, t, expected_keys=["height", "weight", "category"]
            )

            returned = df2.groupby("category").boxplot(return_type=t)
            self._check_box_return_type(returned, t, expected_keys=categories2)

            returned = df2.boxplot(by="category", return_type=t)
            self._check_box_return_type(returned, t, expected_keys=columns2)

    def test_grouped_box_layout(self):
        df = self.hist_df

        msg = "Layout of 1x1 must be larger than required size 2"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=["weight", "height"], by=df.gender, layout=(1, 1))

        msg = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(
                column=["height", "weight", "category"],
                layout=(2, 1),
                return_type="dict",
            )

        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=["weight", "height"], by=df.gender, layout=(-1, -1))

        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning):
            box = _check_plot_works(
                df.groupby("gender").boxplot, column="height", return_type="dict"
            )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=2, layout=(1, 2))

        with tm.assert_produces_warning(UserWarning):
            box = _check_plot_works(
                df.groupby("category").boxplot, column="height", return_type="dict"
            )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=4, layout=(2, 2))

        # GH 6769
        with tm.assert_produces_warning(UserWarning):
            box = _check_plot_works(
                df.groupby("classroom").boxplot, column="height", return_type="dict"
            )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=3, layout=(2, 2))

        # GH 5897
        axes = df.boxplot(
            column=["height", "weight", "category"], by="gender", return_type="axes"
        )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=3, layout=(2, 2))
        for ax in [axes["height"]]:
            self._check_visible(ax.get_xticklabels(), visible=False)
            self._check_visible([ax.xaxis.get_label()], visible=False)
        for ax in [axes["weight"], axes["category"]]:
            self._check_visible(ax.get_xticklabels())
            self._check_visible([ax.xaxis.get_label()])

        box = df.groupby("classroom").boxplot(
            column=["height", "weight", "category"], return_type="dict"
        )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=3, layout=(2, 2))

        with tm.assert_produces_warning(UserWarning):
            box = _check_plot_works(
                df.groupby("category").boxplot,
                column="height",
                layout=(3, 2),
                return_type="dict",
            )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=4, layout=(3, 2))
        with tm.assert_produces_warning(UserWarning):
            box = _check_plot_works(
                df.groupby("category").boxplot,
                column="height",
                layout=(3, -1),
                return_type="dict",
            )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=4, layout=(3, 2))

        box = df.boxplot(
            column=["height", "weight", "category"], by="gender", layout=(4, 1)
        )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=3, layout=(4, 1))

        box = df.boxplot(
            column=["height", "weight", "category"], by="gender", layout=(-1, 1)
        )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=3, layout=(3, 1))

        box = df.groupby("classroom").boxplot(
            column=["height", "weight", "category"], layout=(1, 4), return_type="dict"
        )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=3, layout=(1, 4))

        box = df.groupby("classroom").boxplot(  # noqa
            column=["height", "weight", "category"], layout=(1, -1), return_type="dict"
        )
        self._check_axes_shape(self.plt.gcf().axes, axes_num=3, layout=(1, 3))

    def test_grouped_box_multiple_axes(self):
        # GH 6970, GH 7069
        df = self.hist_df

        # check warning to ignore sharex / sharey
        # this check should be done in the first function which
        # passes multiple axes to plot, hist or boxplot
        # location should be changed if other test is added
        # which has earlier alphabetical order
        with tm.assert_produces_warning(UserWarning):
            fig, axes = self.plt.subplots(2, 2)
            df.groupby("category").boxplot(column="height", return_type="axes", ax=axes)
            self._check_axes_shape(self.plt.gcf().axes, axes_num=4, layout=(2, 2))

        fig, axes = self.plt.subplots(2, 3)
        with tm.assert_produces_warning(UserWarning):
            returned = df.boxplot(
                column=["height", "weight", "category"],
                by="gender",
                return_type="axes",
                ax=axes[0],
            )
        returned = np.array(list(returned.values))
        self._check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[0])
        assert returned[0].figure is fig

        # draw on second row
        with tm.assert_produces_warning(UserWarning):
            returned = df.groupby("classroom").boxplot(
                column=["height", "weight", "category"], return_type="axes", ax=axes[1]
            )
        returned = np.array(list(returned.values))
        self._check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[1])
        assert returned[0].figure is fig

        msg = "The number of passed axes must be 3, the same as the output plot"
        with pytest.raises(ValueError, match=msg):
            fig, axes = self.plt.subplots(2, 3)
            # pass different number of axes from required
            with tm.assert_produces_warning(UserWarning):
                axes = df.groupby("classroom").boxplot(ax=axes)

    def test_fontsize(self):
        df = DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]})
        self._check_ticks_props(
            df.boxplot("a", by="b", fontsize=16), xlabelsize=16, ylabelsize=16
        )

    @pytest.mark.parametrize(
        "col, expected_xticklabel",
        [
            ("v", ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"]),
            (["v"], ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"]),
            ("v1", ["(a, v1)", "(b, v1)", "(c, v1)", "(d, v1)", "(e, v1)"]),
            (
                ["v", "v1"],
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
            (
                None,
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
        ],
    )
    def test_groupby_boxplot_subplots_false(self, col, expected_xticklabel):
        # GH 16748
        df = DataFrame(
            {
                "cat": np.random.choice(list("abcde"), 100),
                "v": np.random.rand(100),
                "v1": np.random.rand(100),
            }
        )
        grouped = df.groupby("cat")

        axes = _check_plot_works(
            grouped.boxplot, subplots=False, column=col, return_type="axes"
        )

        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        assert expected_xticklabel == result_xticklabel

    def test_boxplot_multiindex_column(self):
        # GH 16748
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = list(zip(*arrays))
        index = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = DataFrame(np.random.randn(3, 8), index=["A", "B", "C"], columns=index)

        col = [("bar", "one"), ("bar", "two")]
        axes = _check_plot_works(df.boxplot, column=col, return_type="axes")

        expected_xticklabel = ["(bar, one)", "(bar, two)"]
        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        assert expected_xticklabel == result_xticklabel
