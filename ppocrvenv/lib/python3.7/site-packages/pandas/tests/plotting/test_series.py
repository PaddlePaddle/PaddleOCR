""" Test cases for Series.plot """


from datetime import datetime
from itertools import chain

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    TestPlotBase,
    _check_plot_works,
)

import pandas.plotting as plotting

pytestmark = pytest.mark.slow


@td.skip_if_no_mpl
class TestSeriesPlots(TestPlotBase):
    def setup_method(self, method):
        TestPlotBase.setup_method(self, method)
        import matplotlib as mpl

        mpl.rcdefaults()

        self.ts = tm.makeTimeSeries()
        self.ts.name = "ts"

        self.series = tm.makeStringSeries()
        self.series.name = "series"

        self.iseries = tm.makePeriodSeries()
        self.iseries.name = "iseries"

    def test_plot(self):
        _check_plot_works(self.ts.plot, label="foo")
        _check_plot_works(self.ts.plot, use_index=False)
        axes = _check_plot_works(self.ts.plot, rot=0)
        self._check_ticks_props(axes, xrot=0)

        ax = _check_plot_works(self.ts.plot, style=".", logy=True)
        self._check_ax_scales(ax, yaxis="log")

        ax = _check_plot_works(self.ts.plot, style=".", logx=True)
        self._check_ax_scales(ax, xaxis="log")

        ax = _check_plot_works(self.ts.plot, style=".", loglog=True)
        self._check_ax_scales(ax, xaxis="log", yaxis="log")

        _check_plot_works(self.ts[:10].plot.bar)
        _check_plot_works(self.ts.plot.area, stacked=False)
        _check_plot_works(self.iseries.plot)

        for kind in ["line", "bar", "barh", "kde", "hist", "box"]:
            _check_plot_works(self.series[:5].plot, kind=kind)

        _check_plot_works(self.series[:10].plot.barh)
        ax = _check_plot_works(Series(np.random.randn(10)).plot.bar, color="black")
        self._check_colors([ax.patches[0]], facecolors=["black"])

        # GH 6951
        ax = _check_plot_works(self.ts.plot, subplots=True)
        self._check_axes_shape(ax, axes_num=1, layout=(1, 1))

        ax = _check_plot_works(self.ts.plot, subplots=True, layout=(-1, 1))
        self._check_axes_shape(ax, axes_num=1, layout=(1, 1))
        ax = _check_plot_works(self.ts.plot, subplots=True, layout=(1, -1))
        self._check_axes_shape(ax, axes_num=1, layout=(1, 1))

    def test_plot_figsize_and_title(self):
        # figsize and title
        _, ax = self.plt.subplots()
        ax = self.series.plot(title="Test", figsize=(16, 8), ax=ax)
        self._check_text_labels(ax.title, "Test")
        self._check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16, 8))

    def test_dont_modify_rcParams(self):
        # GH 8242
        key = "axes.prop_cycle"
        colors = self.plt.rcParams[key]
        _, ax = self.plt.subplots()
        Series([1, 2, 3]).plot(ax=ax)
        assert colors == self.plt.rcParams[key]

    def test_ts_line_lim(self):
        fig, ax = self.plt.subplots()
        ax = self.ts.plot(ax=ax)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data(orig=False)[0][0]
        assert xmax >= lines[0].get_data(orig=False)[0][-1]
        tm.close()

        ax = self.ts.plot(secondary_y=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data(orig=False)[0][0]
        assert xmax >= lines[0].get_data(orig=False)[0][-1]

    def test_ts_area_lim(self):
        _, ax = self.plt.subplots()
        ax = self.ts.plot.area(stacked=False, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        self._check_ticks_props(ax, xrot=0)
        tm.close()

        # GH 7471
        _, ax = self.plt.subplots()
        ax = self.ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        self._check_ticks_props(ax, xrot=30)
        tm.close()

        tz_ts = self.ts.copy()
        tz_ts.index = tz_ts.tz_localize("GMT").tz_convert("CET")
        _, ax = self.plt.subplots()
        ax = tz_ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        self._check_ticks_props(ax, xrot=0)
        tm.close()

        _, ax = self.plt.subplots()
        ax = tz_ts.plot.area(stacked=False, secondary_y=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        self._check_ticks_props(ax, xrot=0)

    def test_area_sharey_dont_overwrite(self):
        # GH37942
        fig, (ax1, ax2) = self.plt.subplots(1, 2, sharey=True)

        abs(self.ts).plot(ax=ax1, kind="area")
        abs(self.ts).plot(ax=ax2, kind="area")

        assert self.get_y_axis(ax1).joined(ax1, ax2)
        assert self.get_y_axis(ax2).joined(ax1, ax2)

    def test_label(self):
        s = Series([1, 2])
        _, ax = self.plt.subplots()
        ax = s.plot(label="LABEL", legend=True, ax=ax)
        self._check_legend_labels(ax, labels=["LABEL"])
        self.plt.close()
        _, ax = self.plt.subplots()
        ax = s.plot(legend=True, ax=ax)
        self._check_legend_labels(ax, labels=["None"])
        self.plt.close()
        # get name from index
        s.name = "NAME"
        _, ax = self.plt.subplots()
        ax = s.plot(legend=True, ax=ax)
        self._check_legend_labels(ax, labels=["NAME"])
        self.plt.close()
        # override the default
        _, ax = self.plt.subplots()
        ax = s.plot(legend=True, label="LABEL", ax=ax)
        self._check_legend_labels(ax, labels=["LABEL"])
        self.plt.close()
        # Add lebel info, but don't draw
        _, ax = self.plt.subplots()
        ax = s.plot(legend=False, label="LABEL", ax=ax)
        assert ax.get_legend() is None  # Hasn't been drawn
        ax.legend()  # draw it
        self._check_legend_labels(ax, labels=["LABEL"])

    def test_boolean(self):
        # GH 23719
        s = Series([False, False, True])
        _check_plot_works(s.plot, include_bool=True)

        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            _check_plot_works(s.plot)

    def test_line_area_nan_series(self):
        values = [1, 2, np.nan, 3]
        s = Series(values)
        ts = Series(values, index=tm.makeDateIndex(k=4))

        for d in [s, ts]:
            ax = _check_plot_works(d.plot)
            masked = ax.lines[0].get_ydata()
            # remove nan for comparison purpose
            exp = np.array([1, 2, 3], dtype=np.float64)
            tm.assert_numpy_array_equal(np.delete(masked.data, 2), exp)
            tm.assert_numpy_array_equal(
                masked.mask, np.array([False, False, True, False])
            )

            expected = np.array([1, 2, 0, 3], dtype=np.float64)
            ax = _check_plot_works(d.plot, stacked=True)
            tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
            ax = _check_plot_works(d.plot.area)
            tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
            ax = _check_plot_works(d.plot.area, stacked=False)
            tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)

    def test_line_use_index_false(self):
        s = Series([1, 2, 3], index=["a", "b", "c"])
        s.index.name = "The Index"
        _, ax = self.plt.subplots()
        ax = s.plot(use_index=False, ax=ax)
        label = ax.get_xlabel()
        assert label == ""
        _, ax = self.plt.subplots()
        ax2 = s.plot.bar(use_index=False, ax=ax)
        label2 = ax2.get_xlabel()
        assert label2 == ""

    def test_bar_log(self):
        expected = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

        _, ax = self.plt.subplots()
        ax = Series([200, 500]).plot.bar(log=True, ax=ax)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)
        tm.close()

        _, ax = self.plt.subplots()
        ax = Series([200, 500]).plot.barh(log=True, ax=ax)
        tm.assert_numpy_array_equal(ax.xaxis.get_ticklocs(), expected)
        tm.close()

        # GH 9905
        expected = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])

        _, ax = self.plt.subplots()
        ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind="bar", ax=ax)
        ymin = 0.0007943282347242822
        ymax = 0.12589254117941673
        res = ax.get_ylim()
        tm.assert_almost_equal(res[0], ymin)
        tm.assert_almost_equal(res[1], ymax)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)
        tm.close()

        _, ax = self.plt.subplots()
        ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind="barh", ax=ax)
        res = ax.get_xlim()
        tm.assert_almost_equal(res[0], ymin)
        tm.assert_almost_equal(res[1], ymax)
        tm.assert_numpy_array_equal(ax.xaxis.get_ticklocs(), expected)

    def test_bar_ignore_index(self):
        df = Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        _, ax = self.plt.subplots()
        ax = df.plot.bar(use_index=False, ax=ax)
        self._check_text_labels(ax.get_xticklabels(), ["0", "1", "2", "3"])

    def test_bar_user_colors(self):
        s = Series([1, 2, 3, 4])
        ax = s.plot.bar(color=["red", "blue", "blue", "red"])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        assert result == expected

    def test_rotation(self):
        df = DataFrame(np.random.randn(5, 5))
        # Default rot 0
        _, ax = self.plt.subplots()
        axes = df.plot(ax=ax)
        self._check_ticks_props(axes, xrot=0)

        _, ax = self.plt.subplots()
        axes = df.plot(rot=30, ax=ax)
        self._check_ticks_props(axes, xrot=30)

    def test_irregular_datetime(self):
        from pandas.plotting._matplotlib.converter import DatetimeConverter

        rng = date_range("1/1/2000", "3/1/2000")
        rng = rng[[0, 1, 2, 3, 5, 9, 10, 11, 12]]
        ser = Series(np.random.randn(len(rng)), rng)
        _, ax = self.plt.subplots()
        ax = ser.plot(ax=ax)
        xp = DatetimeConverter.convert(datetime(1999, 1, 1), "", ax)
        ax.set_xlim("1/1/1999", "1/1/2001")
        assert xp == ax.get_xlim()[0]
        self._check_ticks_props(ax, xrot=30)

    def test_unsorted_index_xlim(self):
        ser = Series(
            [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0],
        )
        _, ax = self.plt.subplots()
        ax = ser.plot(ax=ax)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data(orig=False)[0])
        assert xmax >= np.nanmax(lines[0].get_data(orig=False)[0])

    def test_pie_series(self):
        # if sum of values is less than 1.0, pie handle them as rate and draw
        # semicircle.
        series = Series(
            np.random.randint(1, 5), index=["a", "b", "c", "d", "e"], name="YLABEL"
        )
        ax = _check_plot_works(series.plot.pie)
        self._check_text_labels(ax.texts, series.index)
        assert ax.get_ylabel() == "YLABEL"

        # without wedge labels
        ax = _check_plot_works(series.plot.pie, labels=None)
        self._check_text_labels(ax.texts, [""] * 5)

        # with less colors than elements
        color_args = ["r", "g", "b"]
        ax = _check_plot_works(series.plot.pie, colors=color_args)

        color_expected = ["r", "g", "b", "r", "g"]
        self._check_colors(ax.patches, facecolors=color_expected)

        # with labels and colors
        labels = ["A", "B", "C", "D", "E"]
        color_args = ["r", "g", "b", "c", "m"]
        ax = _check_plot_works(series.plot.pie, labels=labels, colors=color_args)
        self._check_text_labels(ax.texts, labels)
        self._check_colors(ax.patches, facecolors=color_args)

        # with autopct and fontsize
        ax = _check_plot_works(
            series.plot.pie, colors=color_args, autopct="%.2f", fontsize=7
        )
        pcts = [f"{s*100:.2f}" for s in series.values / series.sum()]
        expected_texts = list(chain.from_iterable(zip(series.index, pcts)))
        self._check_text_labels(ax.texts, expected_texts)
        for t in ax.texts:
            assert t.get_fontsize() == 7

        # includes negative value
        series = Series([1, 2, 0, 4, -1], index=["a", "b", "c", "d", "e"])
        with pytest.raises(ValueError, match="pie plot doesn't allow negative values"):
            series.plot.pie()

        # includes nan
        series = Series([1, 2, np.nan, 4], index=["a", "b", "c", "d"], name="YLABEL")
        ax = _check_plot_works(series.plot.pie)
        self._check_text_labels(ax.texts, ["a", "b", "", "d"])

    def test_pie_nan(self):
        s = Series([1, np.nan, 1, 1])
        _, ax = self.plt.subplots()
        ax = s.plot.pie(legend=True, ax=ax)
        expected = ["0", "", "2", "3"]
        result = [x.get_text() for x in ax.texts]
        assert result == expected

    def test_df_series_secondary_legend(self):
        # GH 9779
        df = DataFrame(np.random.randn(30, 3), columns=list("abc"))
        s = Series(np.random.randn(30), name="x")

        # primary -> secondary (without passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        # both legends are drawn on left ax
        # left and right axis must be visible
        self._check_legend_labels(ax, labels=["a", "b", "c", "x (right)"])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()
        tm.close()

        # primary -> secondary (with passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are drawn on left ax
        # left and right axis must be visible
        self._check_legend_labels(ax, labels=["a", "b", "c", "x (right)"])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()
        tm.close()

        # secondary -> secondary (without passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(secondary_y=True, ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        # both legends are drawn on left ax
        # left axis must be invisible and right axis must be visible
        expected = ["a (right)", "b (right)", "c (right)", "x (right)"]
        self._check_legend_labels(ax.left_ax, labels=expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()

        # secondary -> secondary (with passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(secondary_y=True, ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are drawn on left ax
        # left axis must be invisible and right axis must be visible
        expected = ["a (right)", "b (right)", "c (right)", "x (right)"]
        self._check_legend_labels(ax.left_ax, expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()

        # secondary -> secondary (with passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(secondary_y=True, mark_right=False, ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are drawn on left ax
        # left axis must be invisible and right axis must be visible
        expected = ["a", "b", "c", "x (right)"]
        self._check_legend_labels(ax.left_ax, expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()

    @pytest.mark.parametrize(
        "input_logy, expected_scale", [(True, "log"), ("sym", "symlog")]
    )
    def test_secondary_logy(self, input_logy, expected_scale):
        # GH 25545
        s1 = Series(np.random.randn(30))
        s2 = Series(np.random.randn(30))

        # GH 24980
        ax1 = s1.plot(logy=input_logy)
        ax2 = s2.plot(secondary_y=True, logy=input_logy)

        assert ax1.get_yscale() == expected_scale
        assert ax2.get_yscale() == expected_scale

    def test_plot_fails_with_dupe_color_and_style(self):
        x = Series(np.random.randn(2))
        _, ax = self.plt.subplots()
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' keyword "
            "argument. Please use one or the other or pass 'style' without a color "
            "symbol"
        )
        with pytest.raises(ValueError, match=msg):
            x.plot(style="k--", color="k", ax=ax)

    @td.skip_if_no_scipy
    def test_kde_kwargs(self):
        sample_points = np.linspace(-100, 100, 20)
        _check_plot_works(self.ts.plot.kde, bw_method="scott", ind=20)
        _check_plot_works(self.ts.plot.kde, bw_method=None, ind=20)
        _check_plot_works(self.ts.plot.kde, bw_method=None, ind=np.int_(20))
        _check_plot_works(self.ts.plot.kde, bw_method=0.5, ind=sample_points)
        _check_plot_works(self.ts.plot.density, bw_method=0.5, ind=sample_points)
        _, ax = self.plt.subplots()
        ax = self.ts.plot.kde(logy=True, bw_method=0.5, ind=sample_points, ax=ax)
        self._check_ax_scales(ax, yaxis="log")
        self._check_text_labels(ax.yaxis.get_label(), "Density")

    @td.skip_if_no_scipy
    def test_kde_missing_vals(self):
        s = Series(np.random.uniform(size=50))
        s[0] = np.nan
        axes = _check_plot_works(s.plot.kde)

        # gh-14821: check if the values have any missing values
        assert any(~np.isnan(axes.lines[0].get_xdata()))

    def test_boxplot_series(self):
        _, ax = self.plt.subplots()
        ax = self.ts.plot.box(logy=True, ax=ax)
        self._check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        self._check_text_labels(xlabels, [self.ts.name])
        ylabels = ax.get_yticklabels()
        self._check_text_labels(ylabels, [""] * len(ylabels))

    def test_kind_both_ways(self):
        s = Series(range(3))
        kinds = (
            plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds
        )
        for kind in kinds:
            _, ax = self.plt.subplots()
            s.plot(kind=kind, ax=ax)
            self.plt.close()
            _, ax = self.plt.subplots()
            getattr(s.plot, kind)()
            self.plt.close()

    def test_invalid_plot_data(self):
        s = Series(list("abcd"))
        _, ax = self.plt.subplots()
        for kind in plotting.PlotAccessor._common_kinds:

            msg = "no numeric data to plot"
            with pytest.raises(TypeError, match=msg):
                s.plot(kind=kind, ax=ax)

    def test_valid_object_plot(self):
        s = Series(range(10), dtype=object)
        for kind in plotting.PlotAccessor._common_kinds:
            _check_plot_works(s.plot, kind=kind)

    def test_partially_invalid_plot_data(self):
        s = Series(["a", "b", 1.0, 2])
        _, ax = self.plt.subplots()
        for kind in plotting.PlotAccessor._common_kinds:

            msg = "no numeric data to plot"
            with pytest.raises(TypeError, match=msg):
                s.plot(kind=kind, ax=ax)

    def test_invalid_kind(self):
        s = Series([1, 2])
        with pytest.raises(ValueError, match="invalid_kind is not a valid plot kind"):
            s.plot(kind="invalid_kind")

    def test_dup_datetime_index_plot(self):
        dr1 = date_range("1/1/2009", periods=4)
        dr2 = date_range("1/2/2009", periods=4)
        index = dr1.append(dr2)
        values = np.random.randn(index.size)
        s = Series(values, index=index)
        _check_plot_works(s.plot)

    def test_errorbar_asymmetrical(self):
        # GH9536
        s = Series(np.arange(10), name="x")
        err = np.random.rand(2, 10)

        ax = s.plot(yerr=err, xerr=err)

        result = np.vstack([i.vertices[:, 1] for i in ax.collections[1].get_paths()])
        expected = (err.T * np.array([-1, 1])) + s.to_numpy().reshape(-1, 1)
        tm.assert_numpy_array_equal(result, expected)

        msg = (
            "Asymmetrical error bars should be provided "
            f"with the shape \\(2, {len(s)}\\)"
        )
        with pytest.raises(ValueError, match=msg):
            s.plot(yerr=np.random.rand(2, 11))

        tm.close()

    def test_errorbar_plot(self):

        s = Series(np.arange(10), name="x")
        s_err = np.random.randn(10)
        d_err = DataFrame(np.random.randn(10, 2), index=s.index, columns=["x", "y"])
        # test line and bar plots
        kinds = ["line", "bar"]
        for kind in kinds:
            ax = _check_plot_works(s.plot, yerr=Series(s_err), kind=kind)
            self._check_has_errorbars(ax, xerr=0, yerr=1)
            ax = _check_plot_works(s.plot, yerr=s_err, kind=kind)
            self._check_has_errorbars(ax, xerr=0, yerr=1)
            ax = _check_plot_works(s.plot, yerr=s_err.tolist(), kind=kind)
            self._check_has_errorbars(ax, xerr=0, yerr=1)
            ax = _check_plot_works(s.plot, yerr=d_err, kind=kind)
            self._check_has_errorbars(ax, xerr=0, yerr=1)
            ax = _check_plot_works(s.plot, xerr=0.2, yerr=0.2, kind=kind)
            self._check_has_errorbars(ax, xerr=1, yerr=1)

        ax = _check_plot_works(s.plot, xerr=s_err)
        self._check_has_errorbars(ax, xerr=1, yerr=0)

        # test time series plotting
        ix = date_range("1/1/2000", "1/1/2001", freq="M")
        ts = Series(np.arange(12), index=ix, name="x")
        ts_err = Series(np.random.randn(12), index=ix)
        td_err = DataFrame(np.random.randn(12, 2), index=ix, columns=["x", "y"])

        ax = _check_plot_works(ts.plot, yerr=ts_err)
        self._check_has_errorbars(ax, xerr=0, yerr=1)
        ax = _check_plot_works(ts.plot, yerr=td_err)
        self._check_has_errorbars(ax, xerr=0, yerr=1)

        # check incorrect lengths and types
        with tm.external_error_raised(ValueError):
            s.plot(yerr=np.arange(11))

        s_err = ["zzz"] * 10
        with tm.external_error_raised(TypeError):
            s.plot(yerr=s_err)

    def test_table(self):
        _check_plot_works(self.series.plot, table=True)
        _check_plot_works(self.series.plot, table=self.series)

    def test_series_grid_settings(self):
        # Make sure plot defaults to rcParams['axes.grid'] setting, GH 9792
        self._check_grid_settings(
            Series([1, 2, 3]),
            plotting.PlotAccessor._series_kinds + plotting.PlotAccessor._common_kinds,
        )

    def test_standard_colors(self):
        from pandas.plotting._matplotlib.style import get_standard_colors

        for c in ["r", "red", "green", "#FF0000"]:
            result = get_standard_colors(1, color=c)
            assert result == [c]

            result = get_standard_colors(1, color=[c])
            assert result == [c]

            result = get_standard_colors(3, color=c)
            assert result == [c] * 3

            result = get_standard_colors(3, color=[c])
            assert result == [c] * 3

    def test_standard_colors_all(self):
        import matplotlib.colors as colors

        from pandas.plotting._matplotlib.style import get_standard_colors

        # multiple colors like mediumaquamarine
        for c in colors.cnames:
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

        # single letter colors like k
        for c in colors.ColorConverter.colors:
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

    def test_series_plot_color_kwargs(self):
        # GH1890
        _, ax = self.plt.subplots()
        ax = Series(np.arange(12) + 1).plot(color="green", ax=ax)
        self._check_colors(ax.get_lines(), linecolors=["green"])

    def test_time_series_plot_color_kwargs(self):
        # #1890
        _, ax = self.plt.subplots()
        ax = Series(np.arange(12) + 1, index=date_range("1/1/2000", periods=12)).plot(
            color="green", ax=ax
        )
        self._check_colors(ax.get_lines(), linecolors=["green"])

    def test_time_series_plot_color_with_empty_kwargs(self):
        import matplotlib as mpl

        def_colors = self._unpack_cycler(mpl.rcParams)
        index = date_range("1/1/2000", periods=12)
        s = Series(np.arange(1, 13), index=index)

        ncolors = 3

        _, ax = self.plt.subplots()
        for i in range(ncolors):
            ax = s.plot(ax=ax)
        self._check_colors(ax.get_lines(), linecolors=def_colors[:ncolors])

    def test_xticklabels(self):
        # GH11529
        s = Series(np.arange(10), index=[f"P{i:02d}" for i in range(10)])
        _, ax = self.plt.subplots()
        ax = s.plot(xticks=[0, 3, 5, 9], ax=ax)
        exp = [f"P{i:02d}" for i in [0, 3, 5, 9]]
        self._check_text_labels(ax.get_xticklabels(), exp)

    def test_xtick_barPlot(self):
        # GH28172
        s = Series(range(10), index=[f"P{i:02d}" for i in range(10)])
        ax = s.plot.bar(xticks=range(0, 11, 2))
        exp = np.array(list(range(0, 11, 2)))
        tm.assert_numpy_array_equal(exp, ax.get_xticks())

    def test_custom_business_day_freq(self):
        # GH7222
        from pandas.tseries.offsets import CustomBusinessDay

        s = Series(
            range(100, 121),
            index=pd.bdate_range(
                start="2014-05-01",
                end="2014-06-01",
                freq=CustomBusinessDay(holidays=["2014-05-26"]),
            ),
        )

        _check_plot_works(s.plot)

    @pytest.mark.xfail
    def test_plot_accessor_updates_on_inplace(self):
        s = Series([1, 2, 3, 4])
        _, ax = self.plt.subplots()
        ax = s.plot(ax=ax)
        before = ax.xaxis.get_ticklocs()

        s.drop([0, 1], inplace=True)
        _, ax = self.plt.subplots()
        after = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(before, after)

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_plot_xlim_for_series(self, kind):
        # test if xlim is also correctly plotted in Series for line and area
        # GH 27686
        s = Series([2, 3])
        _, ax = self.plt.subplots()
        s.plot(kind=kind, ax=ax)
        xlims = ax.get_xlim()

        assert xlims[0] < 0
        assert xlims[1] > 1

    def test_plot_no_rows(self):
        # GH 27758
        df = Series(dtype=int)
        assert df.empty
        ax = df.plot()
        assert len(ax.get_lines()) == 1
        line = ax.get_lines()[0]
        assert len(line.get_xdata()) == 0
        assert len(line.get_ydata()) == 0

    def test_plot_no_numeric_data(self):
        df = Series(["a", "b", "c"])
        with pytest.raises(TypeError, match="no numeric data to plot"):
            df.plot()

    @pytest.mark.parametrize(
        "data, index",
        [
            ([1, 2, 3, 4], [3, 2, 1, 0]),
            ([10, 50, 20, 30], [1910, 1920, 1980, 1950]),
        ],
    )
    def test_plot_order(self, data, index):
        # GH38865 Verify plot order of a Series
        ser = Series(data=data, index=index)
        ax = ser.plot(kind="bar")

        expected = ser.tolist()
        result = [
            patch.get_bbox().ymax
            for patch in sorted(ax.patches, key=lambda patch: patch.get_bbox().xmax)
        ]
        assert expected == result

    def test_style_single_ok(self):
        s = Series([1, 2])
        ax = s.plot(style="s", color="C3")
        assert ax.lines[0].get_color() == "C3"

    @pytest.mark.parametrize(
        "index_name, old_label, new_label",
        [(None, "", "new"), ("old", "old", "new"), (None, "", "")],
    )
    @pytest.mark.parametrize("kind", ["line", "area", "bar"])
    def test_xlabel_ylabel_series(self, kind, index_name, old_label, new_label):
        # GH 9093
        ser = Series([1, 2, 3, 4])
        ser.index.name = index_name

        # default is the ylabel is not shown and xlabel is index name
        ax = ser.plot(kind=kind)
        assert ax.get_ylabel() == ""
        assert ax.get_xlabel() == old_label

        # old xlabel will be overridden and assigned ylabel will be used as ylabel
        ax = ser.plot(kind=kind, ylabel=new_label, xlabel=new_label)
        assert ax.get_ylabel() == new_label
        assert ax.get_xlabel() == new_label

    @pytest.mark.parametrize(
        "index",
        [
            pd.timedelta_range(start=0, periods=2, freq="D"),
            [pd.Timedelta(days=1), pd.Timedelta(days=2)],
        ],
    )
    def test_timedelta_index(self, index):
        # GH37454
        xlims = (3, 1)
        ax = Series([1, 2], index=index).plot(xlim=(xlims))
        assert ax.get_xlim() == (3, 1)
