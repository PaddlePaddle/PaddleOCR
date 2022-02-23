""" Test cases for .hist method """
import re

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Index,
    Series,
    to_datetime,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    TestPlotBase,
    _check_plot_works,
)

pytestmark = pytest.mark.slow


@td.skip_if_no_mpl
class TestSeriesPlots(TestPlotBase):
    def setup_method(self, method):
        TestPlotBase.setup_method(self, method)
        import matplotlib as mpl

        mpl.rcdefaults()

        self.ts = tm.makeTimeSeries()
        self.ts.name = "ts"

    def test_hist_legacy(self):
        _check_plot_works(self.ts.hist)
        _check_plot_works(self.ts.hist, grid=False)
        _check_plot_works(self.ts.hist, figsize=(8, 10))
        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(self.ts.hist, by=self.ts.index.month)
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(self.ts.hist, by=self.ts.index.month, bins=5)

        fig, ax = self.plt.subplots(1, 1)
        _check_plot_works(self.ts.hist, ax=ax, default_axes=True)
        _check_plot_works(self.ts.hist, ax=ax, figure=fig, default_axes=True)
        _check_plot_works(self.ts.hist, figure=fig, default_axes=True)
        tm.close()

        fig, (ax1, ax2) = self.plt.subplots(1, 2)
        _check_plot_works(self.ts.hist, figure=fig, ax=ax1, default_axes=True)
        _check_plot_works(self.ts.hist, figure=fig, ax=ax2, default_axes=True)

        msg = (
            "Cannot pass 'figure' when using the 'by' argument, since a new 'Figure' "
            "instance will be created"
        )
        with pytest.raises(ValueError, match=msg):
            self.ts.hist(by=self.ts.index, figure=fig)

    def test_hist_bins_legacy(self):
        df = DataFrame(np.random.randn(10, 2))
        ax = df.hist(bins=2)[0][0]
        assert len(ax.patches) == 2

    def test_hist_layout(self):
        df = self.hist_df
        msg = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=(1, 1))

        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=[1, 1])

    def test_hist_layout_with_by(self):
        df = self.hist_df

        # _check_plot_works adds an `ax` kwarg to the method call
        # so we get a warning about an axis being cleared, even
        # though we don't explicing pass one, see GH #13188
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist, by=df.gender, layout=(2, 1))
        self._check_axes_shape(axes, axes_num=2, layout=(2, 1))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist, by=df.gender, layout=(3, -1))
        self._check_axes_shape(axes, axes_num=2, layout=(3, 1))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist, by=df.category, layout=(4, 1))
        self._check_axes_shape(axes, axes_num=4, layout=(4, 1))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist, by=df.category, layout=(2, -1))
        self._check_axes_shape(axes, axes_num=4, layout=(2, 2))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist, by=df.category, layout=(3, -1))
        self._check_axes_shape(axes, axes_num=4, layout=(3, 2))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist, by=df.category, layout=(-1, 4))
        self._check_axes_shape(axes, axes_num=4, layout=(1, 4))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist, by=df.classroom, layout=(2, 2))
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))

        axes = df.height.hist(by=df.category, layout=(4, 2), figsize=(12, 7))
        self._check_axes_shape(axes, axes_num=4, layout=(4, 2), figsize=(12, 7))

    def test_hist_no_overlap(self):
        from matplotlib.pyplot import (
            gcf,
            subplot,
        )

        x = Series(np.random.randn(2))
        y = Series(np.random.randn(2))
        subplot(121)
        x.hist()
        subplot(122)
        y.hist()
        fig = gcf()
        axes = fig.axes
        assert len(axes) == 2

    def test_hist_by_no_extra_plots(self):
        df = self.hist_df
        axes = df.height.hist(by=df.gender)  # noqa
        assert len(self.plt.get_fignums()) == 1

    def test_plot_fails_when_ax_differs_from_figure(self):
        from pylab import figure

        fig1 = figure()
        fig2 = figure()
        ax1 = fig1.add_subplot(111)
        msg = "passed axis not bound to passed figure"
        with pytest.raises(AssertionError, match=msg):
            self.ts.hist(ax=ax1, figure=fig2)

    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),
            ("barstacked", True),
            ("step", False),
            ("stepfilled", True),
        ],
    )
    def test_histtype_argument(self, histtype, expected):
        # GH23992 Verify functioning of histtype argument
        ser = Series(np.random.randint(1, 10))
        ax = ser.hist(histtype=histtype)
        self._check_patches_all_filled(ax, filled=expected)

    @pytest.mark.parametrize(
        "by, expected_axes_num, expected_layout", [(None, 1, (1, 1)), ("b", 2, (1, 2))]
    )
    def test_hist_with_legend(self, by, expected_axes_num, expected_layout):
        # GH 6279 - Series histogram can have a legend
        index = 15 * ["1"] + 15 * ["2"]
        s = Series(np.random.randn(30), index=index, name="a")
        s.index.name = "b"

        # Use default_axes=True when plotting method generate subplots itself
        axes = _check_plot_works(s.hist, default_axes=True, legend=True, by=by)
        self._check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        self._check_legend_labels(axes, "a")

    @pytest.mark.parametrize("by", [None, "b"])
    def test_hist_with_legend_raises(self, by):
        # GH 6279 - Series histogram with legend and label raises
        index = 15 * ["1"] + 15 * ["2"]
        s = Series(np.random.randn(30), index=index, name="a")
        s.index.name = "b"

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            s.hist(legend=True, by=by, label="c")

    def test_hist_kwargs(self):
        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 5
        self._check_text_labels(ax.yaxis.get_label(), "Frequency")
        tm.close()

        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(orientation="horizontal", ax=ax)
        self._check_text_labels(ax.xaxis.get_label(), "Frequency")
        tm.close()

        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(align="left", stacked=True, ax=ax)
        tm.close()

    @td.skip_if_no_scipy
    def test_hist_kde(self):

        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(logy=True, ax=ax)
        self._check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        # ticks are values, thus ticklabels are blank
        self._check_text_labels(xlabels, [""] * len(xlabels))
        ylabels = ax.get_yticklabels()
        self._check_text_labels(ylabels, [""] * len(ylabels))

        _check_plot_works(self.ts.plot.kde)
        _check_plot_works(self.ts.plot.density)
        _, ax = self.plt.subplots()
        ax = self.ts.plot.kde(logy=True, ax=ax)
        self._check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        self._check_text_labels(xlabels, [""] * len(xlabels))
        ylabels = ax.get_yticklabels()
        self._check_text_labels(ylabels, [""] * len(ylabels))

    @td.skip_if_no_scipy
    def test_hist_kde_color(self):
        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(logy=True, bins=10, color="b", ax=ax)
        self._check_ax_scales(ax, yaxis="log")
        assert len(ax.patches) == 10
        self._check_colors(ax.patches, facecolors=["b"] * 10)

        _, ax = self.plt.subplots()
        ax = self.ts.plot.kde(logy=True, color="r", ax=ax)
        self._check_ax_scales(ax, yaxis="log")
        lines = ax.get_lines()
        assert len(lines) == 1
        self._check_colors(lines, ["r"])


@td.skip_if_no_mpl
class TestDataFramePlots(TestPlotBase):
    def test_hist_df_legacy(self):
        from matplotlib.patches import Rectangle

        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(self.hist_df.hist)

        # make sure layout is handled
        df = DataFrame(np.random.randn(100, 2))
        df[2] = to_datetime(
            np.random.randint(
                self.start_date_to_int64,
                self.end_date_to_int64,
                size=100,
                dtype=np.int64,
            )
        )
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.hist, grid=False)
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))
        assert not axes[1, 1].get_visible()

        _check_plot_works(df[[2]].hist)
        df = DataFrame(np.random.randn(100, 1))
        _check_plot_works(df.hist)

        # make sure layout is handled
        df = DataFrame(np.random.randn(100, 5))
        df[5] = to_datetime(
            np.random.randint(
                self.start_date_to_int64,
                self.end_date_to_int64,
                size=100,
                dtype=np.int64,
            )
        )
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.hist, layout=(4, 2))
        self._check_axes_shape(axes, axes_num=6, layout=(4, 2))

        # make sure sharex, sharey is handled
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(df.hist, sharex=True, sharey=True)

        # handle figsize arg
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(df.hist, figsize=(8, 10))

        # check bins argument
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(df.hist, bins=5)

        # make sure xlabelsize and xrot are handled
        ser = df[0]
        xf, yf = 20, 18
        xrot, yrot = 30, 40
        axes = ser.hist(xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)
        self._check_ticks_props(
            axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot
        )

        xf, yf = 20, 18
        xrot, yrot = 30, 40
        axes = df.hist(xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)
        self._check_ticks_props(
            axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot
        )

        tm.close()

        ax = ser.hist(cumulative=True, bins=4, density=True)
        # height of last bin (index 5) must be 1.0
        rects = [x for x in ax.get_children() if isinstance(x, Rectangle)]
        tm.assert_almost_equal(rects[-1].get_height(), 1.0)

        tm.close()
        ax = ser.hist(log=True)
        # scale of y must be 'log'
        self._check_ax_scales(ax, yaxis="log")

        tm.close()

        # propagate attr exception from matplotlib.Axes.hist
        with tm.external_error_raised(AttributeError):
            ser.hist(foo="bar")

    def test_hist_non_numerical_or_datetime_raises(self):
        # gh-10444, GH32590
        df = DataFrame(
            {
                "a": np.random.rand(10),
                "b": np.random.randint(0, 10, 10),
                "c": to_datetime(
                    np.random.randint(
                        1582800000000000000, 1583500000000000000, 10, dtype=np.int64
                    )
                ),
                "d": to_datetime(
                    np.random.randint(
                        1582800000000000000, 1583500000000000000, 10, dtype=np.int64
                    ),
                    utc=True,
                ),
            }
        )
        df_o = df.astype(object)

        msg = "hist method requires numerical or datetime columns, nothing to plot."
        with pytest.raises(ValueError, match=msg):
            df_o.hist()

    def test_hist_layout(self):
        df = DataFrame(np.random.randn(100, 2))
        df[2] = to_datetime(
            np.random.randint(
                self.start_date_to_int64,
                self.end_date_to_int64,
                size=100,
                dtype=np.int64,
            )
        )

        layout_to_expected_size = (
            {"layout": None, "expected_size": (2, 2)},  # default is 2x2
            {"layout": (2, 2), "expected_size": (2, 2)},
            {"layout": (4, 1), "expected_size": (4, 1)},
            {"layout": (1, 4), "expected_size": (1, 4)},
            {"layout": (3, 3), "expected_size": (3, 3)},
            {"layout": (-1, 4), "expected_size": (1, 4)},
            {"layout": (4, -1), "expected_size": (4, 1)},
            {"layout": (-1, 2), "expected_size": (2, 2)},
            {"layout": (2, -1), "expected_size": (2, 2)},
        )

        for layout_test in layout_to_expected_size:
            axes = df.hist(layout=layout_test["layout"])
            expected = layout_test["expected_size"]
            self._check_axes_shape(axes, axes_num=3, layout=expected)

        # layout too small for all 4 plots
        msg = "Layout of 1x1 must be larger than required size 3"
        with pytest.raises(ValueError, match=msg):
            df.hist(layout=(1, 1))

        # invalid format for layout
        msg = re.escape("Layout must be a tuple of (rows, columns)")
        with pytest.raises(ValueError, match=msg):
            df.hist(layout=(1,))
        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.hist(layout=(-1, -1))

    # GH 9351
    def test_tight_layout(self):
        df = DataFrame(np.random.randn(100, 2))
        df[2] = to_datetime(
            np.random.randint(
                self.start_date_to_int64,
                self.end_date_to_int64,
                size=100,
                dtype=np.int64,
            )
        )
        # Use default_axes=True when plotting method generate subplots itself
        _check_plot_works(df.hist, default_axes=True)
        self.plt.tight_layout()

        tm.close()

    def test_hist_subplot_xrot(self):
        # GH 30288
        df = DataFrame(
            {
                "length": [1.5, 0.5, 1.2, 0.9, 3],
                "animal": ["pig", "rabbit", "pig", "pig", "rabbit"],
            }
        )
        # Use default_axes=True when plotting method generate subplots itself
        axes = _check_plot_works(
            df.hist,
            default_axes=True,
            filterwarnings="always",
            column="length",
            by="animal",
            bins=5,
            xrot=0,
        )
        self._check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize(
        "column, expected",
        [
            (None, ["width", "length", "height"]),
            (["length", "width", "height"], ["length", "width", "height"]),
        ],
    )
    def test_hist_column_order_unchanged(self, column, expected):
        # GH29235

        df = DataFrame(
            {
                "width": [0.7, 0.2, 0.15, 0.2, 1.1],
                "length": [1.5, 0.5, 1.2, 0.9, 3],
                "height": [3, 0.5, 3.4, 2, 1],
            },
            index=["pig", "rabbit", "duck", "chicken", "horse"],
        )

        # Use default_axes=True when plotting method generate subplots itself
        axes = _check_plot_works(
            df.hist,
            default_axes=True,
            column=column,
            layout=(1, 3),
        )
        result = [axes[0, i].get_title() for i in range(3)]
        assert result == expected

    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),
            ("barstacked", True),
            ("step", False),
            ("stepfilled", True),
        ],
    )
    def test_histtype_argument(self, histtype, expected):
        # GH23992 Verify functioning of histtype argument
        df = DataFrame(np.random.randint(1, 10, size=(100, 2)), columns=["a", "b"])
        ax = df.hist(histtype=histtype)
        self._check_patches_all_filled(ax, filled=expected)

    @pytest.mark.parametrize("by", [None, "c"])
    @pytest.mark.parametrize("column", [None, "b"])
    def test_hist_with_legend(self, by, column):
        # GH 6279 - DataFrame histogram can have a legend
        expected_axes_num = 1 if by is None and column is not None else 2
        expected_layout = (1, expected_axes_num)
        expected_labels = column or ["a", "b"]
        if by is not None:
            expected_labels = [expected_labels] * 2

        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(np.random.randn(30, 2), index=index, columns=["a", "b"])

        # Use default_axes=True when plotting method generate subplots itself
        axes = _check_plot_works(
            df.hist,
            default_axes=True,
            legend=True,
            by=by,
            column=column,
        )

        self._check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        if by is None and column is None:
            axes = axes[0]
        for expected_label, ax in zip(expected_labels, axes):
            self._check_legend_labels(ax, expected_label)

    @pytest.mark.parametrize("by", [None, "c"])
    @pytest.mark.parametrize("column", [None, "b"])
    def test_hist_with_legend_raises(self, by, column):
        # GH 6279 - DataFrame histogram with legend and label raises
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(np.random.randn(30, 2), index=index, columns=["a", "b"])

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            df.hist(legend=True, by=by, column=column, label="d")

    def test_hist_df_kwargs(self):
        df = DataFrame(np.random.randn(10, 2))
        _, ax = self.plt.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 10

    def test_hist_df_with_nonnumerics(self):
        # GH 9853
        with tm.RNGContext(1):
            df = DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
        df["E"] = ["x", "y"] * 5
        _, ax = self.plt.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 20

        _, ax = self.plt.subplots()
        ax = df.plot.hist(ax=ax)  # bins=10
        assert len(ax.patches) == 40

    def test_hist_secondary_legend(self):
        # GH 9610
        df = DataFrame(np.random.randn(30, 4), columns=list("abcd"))

        # primary -> secondary
        _, ax = self.plt.subplots()
        ax = df["a"].plot.hist(legend=True, ax=ax)
        df["b"].plot.hist(ax=ax, legend=True, secondary_y=True)
        # both legends are drawn on left ax
        # left and right axis must be visible
        self._check_legend_labels(ax, labels=["a", "b (right)"])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()
        tm.close()

        # secondary -> secondary
        _, ax = self.plt.subplots()
        ax = df["a"].plot.hist(legend=True, secondary_y=True, ax=ax)
        df["b"].plot.hist(ax=ax, legend=True, secondary_y=True)
        # both legends are draw on left ax
        # left axis must be invisible, right axis must be visible
        self._check_legend_labels(ax.left_ax, labels=["a (right)", "b (right)"])
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()

        # secondary -> primary
        _, ax = self.plt.subplots()
        ax = df["a"].plot.hist(legend=True, secondary_y=True, ax=ax)
        # right axes is returned
        df["b"].plot.hist(ax=ax, legend=True)
        # both legends are draw on left ax
        # left and right axis must be visible
        self._check_legend_labels(ax.left_ax, labels=["a (right)", "b"])
        assert ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()


@td.skip_if_no_mpl
class TestDataFrameGroupByPlots(TestPlotBase):
    def test_grouped_hist_legacy(self):
        from matplotlib.patches import Rectangle

        from pandas.plotting._matplotlib.hist import _grouped_hist

        df = DataFrame(np.random.randn(500, 1), columns=["A"])
        df["B"] = to_datetime(
            np.random.randint(
                self.start_date_to_int64,
                self.end_date_to_int64,
                size=500,
                dtype=np.int64,
            )
        )
        df["C"] = np.random.randint(0, 4, 500)
        df["D"] = ["X"] * 500

        axes = _grouped_hist(df.A, by=df.C)
        self._check_axes_shape(axes, axes_num=4, layout=(2, 2))

        tm.close()
        axes = df.hist(by=df.C)
        self._check_axes_shape(axes, axes_num=4, layout=(2, 2))

        tm.close()
        # group by a key with single value
        axes = df.hist(by="D", rot=30)
        self._check_axes_shape(axes, axes_num=1, layout=(1, 1))
        self._check_ticks_props(axes, xrot=30)

        tm.close()
        # make sure kwargs to hist are handled
        xf, yf = 20, 18
        xrot, yrot = 30, 40

        axes = _grouped_hist(
            df.A,
            by=df.C,
            cumulative=True,
            bins=4,
            xlabelsize=xf,
            xrot=xrot,
            ylabelsize=yf,
            yrot=yrot,
            density=True,
        )
        # height of last bin (index 5) must be 1.0
        for ax in axes.ravel():
            rects = [x for x in ax.get_children() if isinstance(x, Rectangle)]
            height = rects[-1].get_height()
            tm.assert_almost_equal(height, 1.0)
        self._check_ticks_props(
            axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot
        )

        tm.close()
        axes = _grouped_hist(df.A, by=df.C, log=True)
        # scale of y must be 'log'
        self._check_ax_scales(axes, yaxis="log")

        tm.close()
        # propagate attr exception from matplotlib.Axes.hist
        with tm.external_error_raised(AttributeError):
            _grouped_hist(df.A, by=df.C, foo="bar")

        msg = "Specify figure size by tuple instead"
        with pytest.raises(ValueError, match=msg):
            df.hist(by="C", figsize="default")

    def test_grouped_hist_legacy2(self):
        n = 10
        weight = Series(np.random.normal(166, 20, size=n))
        height = Series(np.random.normal(60, 10, size=n))
        with tm.RNGContext(42):
            gender_int = np.random.choice([0, 1], size=n)
        df_int = DataFrame({"height": height, "weight": weight, "gender": gender_int})
        gb = df_int.groupby("gender")
        axes = gb.hist()
        assert len(axes) == 2
        assert len(self.plt.get_fignums()) == 2
        tm.close()

    def test_grouped_hist_layout(self):
        df = self.hist_df
        msg = "Layout of 1x1 must be larger than required size 2"
        with pytest.raises(ValueError, match=msg):
            df.hist(column="weight", by=df.gender, layout=(1, 1))

        msg = "Layout of 1x3 must be larger than required size 4"
        with pytest.raises(ValueError, match=msg):
            df.hist(column="height", by=df.category, layout=(1, 3))

        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.hist(column="height", by=df.category, layout=(-1, -1))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(
                df.hist, column="height", by=df.gender, layout=(2, 1)
            )
        self._check_axes_shape(axes, axes_num=2, layout=(2, 1))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(
                df.hist, column="height", by=df.gender, layout=(2, -1)
            )
        self._check_axes_shape(axes, axes_num=2, layout=(2, 1))

        axes = df.hist(column="height", by=df.category, layout=(4, 1))
        self._check_axes_shape(axes, axes_num=4, layout=(4, 1))

        axes = df.hist(column="height", by=df.category, layout=(-1, 1))
        self._check_axes_shape(axes, axes_num=4, layout=(4, 1))

        axes = df.hist(column="height", by=df.category, layout=(4, 2), figsize=(12, 8))
        self._check_axes_shape(axes, axes_num=4, layout=(4, 2), figsize=(12, 8))
        tm.close()

        # GH 6769
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(
                df.hist, column="height", by="classroom", layout=(2, 2)
            )
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))

        # without column
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.hist, by="classroom")
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))

        axes = df.hist(by="gender", layout=(3, 5))
        self._check_axes_shape(axes, axes_num=2, layout=(3, 5))

        axes = df.hist(column=["height", "weight", "category"])
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))

    def test_grouped_hist_multiple_axes(self):
        # GH 6970, GH 7069
        df = self.hist_df

        fig, axes = self.plt.subplots(2, 3)
        returned = df.hist(column=["height", "weight", "category"], ax=axes[0])
        self._check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[0])
        assert returned[0].figure is fig
        returned = df.hist(by="classroom", ax=axes[1])
        self._check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[1])
        assert returned[0].figure is fig

        fig, axes = self.plt.subplots(2, 3)
        # pass different number of axes from required
        msg = "The number of passed axes must be 1, the same as the output plot"
        with pytest.raises(ValueError, match=msg):
            axes = df.hist(column="height", ax=axes)

    def test_axis_share_x(self):
        df = self.hist_df
        # GH4089
        ax1, ax2 = df.hist(column="height", by=df.gender, sharex=True)

        # share x
        assert self.get_x_axis(ax1).joined(ax1, ax2)
        assert self.get_x_axis(ax2).joined(ax1, ax2)

        # don't share y
        assert not self.get_y_axis(ax1).joined(ax1, ax2)
        assert not self.get_y_axis(ax2).joined(ax1, ax2)

    def test_axis_share_y(self):
        df = self.hist_df
        ax1, ax2 = df.hist(column="height", by=df.gender, sharey=True)

        # share y
        assert self.get_y_axis(ax1).joined(ax1, ax2)
        assert self.get_y_axis(ax2).joined(ax1, ax2)

        # don't share x
        assert not self.get_x_axis(ax1).joined(ax1, ax2)
        assert not self.get_x_axis(ax2).joined(ax1, ax2)

    def test_axis_share_xy(self):
        df = self.hist_df
        ax1, ax2 = df.hist(column="height", by=df.gender, sharex=True, sharey=True)

        # share both x and y
        assert self.get_x_axis(ax1).joined(ax1, ax2)
        assert self.get_x_axis(ax2).joined(ax1, ax2)

        assert self.get_y_axis(ax1).joined(ax1, ax2)
        assert self.get_y_axis(ax2).joined(ax1, ax2)

    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),
            ("barstacked", True),
            ("step", False),
            ("stepfilled", True),
        ],
    )
    def test_histtype_argument(self, histtype, expected):
        # GH23992 Verify functioning of histtype argument
        df = DataFrame(np.random.randint(1, 10, size=(100, 2)), columns=["a", "b"])
        ax = df.hist(by="a", histtype=histtype)
        self._check_patches_all_filled(ax, filled=expected)
