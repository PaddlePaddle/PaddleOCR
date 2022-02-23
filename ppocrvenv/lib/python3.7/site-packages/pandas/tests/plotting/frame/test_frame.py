""" Test cases for DataFrame.plot """
from datetime import (
    date,
    datetime,
)
import itertools
import re
import string
import warnings

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.api import is_list_like

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    PeriodIndex,
    Series,
    bdate_range,
    date_range,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    TestPlotBase,
    _check_plot_works,
)

from pandas.io.formats.printing import pprint_thing
import pandas.plotting as plotting

pytestmark = pytest.mark.slow


@td.skip_if_no_mpl
class TestDataFramePlots(TestPlotBase):
    def setup_method(self, method):
        TestPlotBase.setup_method(self, method)
        import matplotlib as mpl

        mpl.rcdefaults()

        self.tdf = tm.makeTimeDataFrame()
        self.hexbin_df = DataFrame(
            {
                "A": np.random.uniform(size=20),
                "B": np.random.uniform(size=20),
                "C": np.arange(20) + np.random.uniform(size=20),
            }
        )

    def test_plot(self):
        from pandas.plotting._matplotlib.compat import mpl_ge_3_1_0

        df = self.tdf
        _check_plot_works(df.plot, grid=False)

        # _check_plot_works adds an ax so use default_axes=True to avoid warning
        axes = _check_plot_works(df.plot, default_axes=True, subplots=True)
        self._check_axes_shape(axes, axes_num=4, layout=(4, 1))

        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            subplots=True,
            layout=(-1, 2),
        )
        self._check_axes_shape(axes, axes_num=4, layout=(2, 2))

        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            subplots=True,
            use_index=False,
        )
        self._check_ticks_props(axes, xrot=0)
        self._check_axes_shape(axes, axes_num=4, layout=(4, 1))

        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        if mpl_ge_3_1_0():
            msg = "'Line2D' object has no property 'blarg'"
        else:
            msg = "Unknown property blarg"
        with pytest.raises(AttributeError, match=msg):
            df.plot.line(blarg=True)

        df = DataFrame(np.random.rand(10, 3), index=list(string.ascii_letters[:10]))

        ax = _check_plot_works(df.plot, use_index=True)
        self._check_ticks_props(ax, xrot=0)
        _check_plot_works(df.plot, sort_columns=False)
        _check_plot_works(df.plot, yticks=[1, 5, 10])
        _check_plot_works(df.plot, xticks=[1, 5, 10])
        _check_plot_works(df.plot, ylim=(-100, 100), xlim=(-100, 100))

        _check_plot_works(df.plot, default_axes=True, subplots=True, title="blah")

        # We have to redo it here because _check_plot_works does two plots,
        # once without an ax kwarg and once with an ax kwarg and the new sharex
        # behaviour does not remove the visibility of the latter axis (as ax is
        # present).  see: https://github.com/pandas-dev/pandas/issues/9737

        axes = df.plot(subplots=True, title="blah")
        self._check_axes_shape(axes, axes_num=3, layout=(3, 1))
        # axes[0].figure.savefig("test.png")
        for ax in axes[:2]:
            self._check_visible(ax.xaxis)  # xaxis must be visible for grid
            self._check_visible(ax.get_xticklabels(), visible=False)
            self._check_visible(ax.get_xticklabels(minor=True), visible=False)
            self._check_visible([ax.xaxis.get_label()], visible=False)
        for ax in [axes[2]]:
            self._check_visible(ax.xaxis)
            self._check_visible(ax.get_xticklabels())
            self._check_visible([ax.xaxis.get_label()])
            self._check_ticks_props(ax, xrot=0)

        _check_plot_works(df.plot, title="blah")

        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(np.random.rand(10, 3), index=MultiIndex.from_tuples(tuples))
        ax = _check_plot_works(df.plot, use_index=True)
        self._check_ticks_props(ax, xrot=0)

        # unicode
        index = MultiIndex.from_tuples(
            [
                ("\u03b1", 0),
                ("\u03b1", 1),
                ("\u03b2", 2),
                ("\u03b2", 3),
                ("\u03b3", 4),
                ("\u03b3", 5),
                ("\u03b4", 6),
                ("\u03b4", 7),
            ],
            names=["i0", "i1"],
        )
        columns = MultiIndex.from_tuples(
            [("bar", "\u0394"), ("bar", "\u0395")], names=["c0", "c1"]
        )
        df = DataFrame(np.random.randint(0, 10, (8, 2)), columns=columns, index=index)
        _check_plot_works(df.plot, title="\u03A3")

        # GH 6951
        # Test with single column
        df = DataFrame({"x": np.random.rand(10)})
        axes = _check_plot_works(df.plot.bar, subplots=True)
        self._check_axes_shape(axes, axes_num=1, layout=(1, 1))

        axes = _check_plot_works(df.plot.bar, subplots=True, layout=(-1, 1))
        self._check_axes_shape(axes, axes_num=1, layout=(1, 1))
        # When ax is supplied and required number of axes is 1,
        # passed ax should be used:
        fig, ax = self.plt.subplots()
        axes = df.plot.bar(subplots=True, ax=ax)
        assert len(axes) == 1
        result = ax.axes
        assert result is axes[0]

    def test_nullable_int_plot(self):
        # GH 32073
        dates = ["2008", "2009", None, "2011", "2012"]
        df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [1.0, 2.0, 3.0, 4.0, 5.0],
                "C": [7, 5, np.nan, 3, 2],
                "D": pd.to_datetime(dates, format="%Y").view("i8"),
                "E": pd.to_datetime(dates, format="%Y", utc=True).view("i8"),
            },
            dtype=np.int64,
        )

        _check_plot_works(df.plot, x="A", y="B")
        _check_plot_works(df[["A", "B"]].plot, x="A", y="B")
        _check_plot_works(df[["C", "A"]].plot, x="C", y="A")  # nullable value on x-axis
        _check_plot_works(df[["A", "C"]].plot, x="A", y="C")
        _check_plot_works(df[["B", "C"]].plot, x="B", y="C")
        _check_plot_works(df[["A", "D"]].plot, x="A", y="D")
        _check_plot_works(df[["A", "E"]].plot, x="A", y="E")

    def test_integer_array_plot(self):
        # GH 25587
        arr = pd.array([1, 2, 3, 4], dtype="UInt32")

        s = Series(arr)
        _check_plot_works(s.plot.line)
        _check_plot_works(s.plot.bar)
        _check_plot_works(s.plot.hist)
        _check_plot_works(s.plot.pie)

        df = DataFrame({"x": arr, "y": arr})
        _check_plot_works(df.plot.line)
        _check_plot_works(df.plot.bar)
        _check_plot_works(df.plot.hist)
        _check_plot_works(df.plot.pie, y="y")
        _check_plot_works(df.plot.scatter, x="x", y="y")
        _check_plot_works(df.plot.hexbin, x="x", y="y")

    def test_nonnumeric_exclude(self):
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]})
        ax = df.plot()
        assert len(ax.get_lines()) == 1  # B was plotted

    def test_implicit_label(self):
        df = DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        ax = df.plot(x="a", y="b")
        self._check_text_labels(ax.xaxis.get_label(), "a")

    def test_donot_overwrite_index_name(self):
        # GH 8494
        df = DataFrame(np.random.randn(2, 2), columns=["a", "b"])
        df.index.name = "NAME"
        df.plot(y="b", label="LABEL")
        assert df.index.name == "NAME"

    def test_plot_xy(self):
        # columns.inferred_type == 'string'
        df = self.tdf
        self._check_data(df.plot(x=0, y=1), df.set_index("A")["B"].plot())
        self._check_data(df.plot(x=0), df.set_index("A").plot())
        self._check_data(df.plot(y=0), df.B.plot())
        self._check_data(df.plot(x="A", y="B"), df.set_index("A").B.plot())
        self._check_data(df.plot(x="A"), df.set_index("A").plot())
        self._check_data(df.plot(y="B"), df.B.plot())

        # columns.inferred_type == 'integer'
        df.columns = np.arange(1, len(df.columns) + 1)
        self._check_data(df.plot(x=1, y=2), df.set_index(1)[2].plot())
        self._check_data(df.plot(x=1), df.set_index(1).plot())
        self._check_data(df.plot(y=1), df[1].plot())

        # figsize and title
        ax = df.plot(x=1, y=2, title="Test", figsize=(16, 8))
        self._check_text_labels(ax.title, "Test")
        self._check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16.0, 8.0))

        # columns.inferred_type == 'mixed'
        # TODO add MultiIndex test

    @pytest.mark.parametrize(
        "input_log, expected_log", [(True, "log"), ("sym", "symlog")]
    )
    def test_logscales(self, input_log, expected_log):
        df = DataFrame({"a": np.arange(100)}, index=np.arange(100))

        ax = df.plot(logy=input_log)
        self._check_ax_scales(ax, yaxis=expected_log)
        assert ax.get_yscale() == expected_log

        ax = df.plot(logx=input_log)
        self._check_ax_scales(ax, xaxis=expected_log)
        assert ax.get_xscale() == expected_log

        ax = df.plot(loglog=input_log)
        self._check_ax_scales(ax, xaxis=expected_log, yaxis=expected_log)
        assert ax.get_xscale() == expected_log
        assert ax.get_yscale() == expected_log

    @pytest.mark.parametrize("input_param", ["logx", "logy", "loglog"])
    def test_invalid_logscale(self, input_param):
        # GH: 24867
        df = DataFrame({"a": np.arange(100)}, index=np.arange(100))

        msg = "Boolean, None and 'sym' are valid options, 'sm' is given."
        with pytest.raises(ValueError, match=msg):
            df.plot(**{input_param: "sm"})

    def test_xcompat(self):

        df = self.tdf
        ax = df.plot(x_compat=True)
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        self._check_ticks_props(ax, xrot=30)

        tm.close()
        plotting.plot_params["xaxis.compat"] = True
        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        self._check_ticks_props(ax, xrot=30)

        tm.close()
        plotting.plot_params["x_compat"] = False

        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)

        tm.close()
        # useful if you're plotting a bunch together
        with plotting.plot_params.use("x_compat", True):
            ax = df.plot()
            lines = ax.get_lines()
            assert not isinstance(lines[0].get_xdata(), PeriodIndex)
            self._check_ticks_props(ax, xrot=30)

        tm.close()
        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)
        self._check_ticks_props(ax, xrot=0)

    def test_period_compat(self):
        # GH 9012
        # period-array conversions
        df = DataFrame(
            np.random.rand(21, 2),
            index=bdate_range(datetime(2000, 1, 1), datetime(2000, 1, 31)),
            columns=["a", "b"],
        )

        df.plot()
        self.plt.axhline(y=0)
        tm.close()

    def test_unsorted_index(self):
        df = DataFrame(
            {"y": np.arange(100)}, index=np.arange(99, -1, -1), dtype=np.int64
        )
        ax = df.plot()
        lines = ax.get_lines()[0]
        rs = lines.get_xydata()
        rs = Series(rs[:, 1], rs[:, 0], dtype=np.int64, name="y")
        tm.assert_series_equal(rs, df.y, check_index_type=False)
        tm.close()

        df.index = pd.Index(np.arange(99, -1, -1), dtype=np.float64)
        ax = df.plot()
        lines = ax.get_lines()[0]
        rs = lines.get_xydata()
        rs = Series(rs[:, 1], rs[:, 0], dtype=np.int64, name="y")
        tm.assert_series_equal(rs, df.y)

    def test_unsorted_index_lims(self):
        df = DataFrame({"y": [0.0, 1.0, 2.0, 3.0]}, index=[1.0, 0.0, 3.0, 2.0])
        ax = df.plot()
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data()[0])
        assert xmax >= np.nanmax(lines[0].get_data()[0])

        df = DataFrame(
            {"y": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0]},
            index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0],
        )
        ax = df.plot()
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data()[0])
        assert xmax >= np.nanmax(lines[0].get_data()[0])

        df = DataFrame({"y": [0.0, 1.0, 2.0, 3.0], "z": [91.0, 90.0, 93.0, 92.0]})
        ax = df.plot(x="z", y="y")
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data()[0])
        assert xmax >= np.nanmax(lines[0].get_data()[0])

    def test_negative_log(self):
        df = -DataFrame(
            np.random.rand(6, 4),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        msg = "Log-y scales are not supported in area plot"
        with pytest.raises(ValueError, match=msg):
            df.plot.area(logy=True)
        with pytest.raises(ValueError, match=msg):
            df.plot.area(loglog=True)

    def _compare_stacked_y_cood(self, normal_lines, stacked_lines):
        base = np.zeros(len(normal_lines[0].get_data()[1]))
        for nl, sl in zip(normal_lines, stacked_lines):
            base += nl.get_data()[1]  # get y coordinates
            sy = sl.get_data()[1]
            tm.assert_numpy_array_equal(base, sy)

    def test_line_area_stacked(self):
        with tm.RNGContext(42):
            df = DataFrame(np.random.rand(6, 4), columns=["w", "x", "y", "z"])
            neg_df = -df
            # each column has either positive or negative value
            sep_df = DataFrame(
                {
                    "w": np.random.rand(6),
                    "x": np.random.rand(6),
                    "y": -np.random.rand(6),
                    "z": -np.random.rand(6),
                }
            )
            # each column has positive-negative mixed value
            mixed_df = DataFrame(
                np.random.randn(6, 4),
                index=list(string.ascii_letters[:6]),
                columns=["w", "x", "y", "z"],
            )

            for kind in ["line", "area"]:
                ax1 = _check_plot_works(df.plot, kind=kind, stacked=False)
                ax2 = _check_plot_works(df.plot, kind=kind, stacked=True)
                self._compare_stacked_y_cood(ax1.lines, ax2.lines)

                ax1 = _check_plot_works(neg_df.plot, kind=kind, stacked=False)
                ax2 = _check_plot_works(neg_df.plot, kind=kind, stacked=True)
                self._compare_stacked_y_cood(ax1.lines, ax2.lines)

                ax1 = _check_plot_works(sep_df.plot, kind=kind, stacked=False)
                ax2 = _check_plot_works(sep_df.plot, kind=kind, stacked=True)
                self._compare_stacked_y_cood(ax1.lines[:2], ax2.lines[:2])
                self._compare_stacked_y_cood(ax1.lines[2:], ax2.lines[2:])

                _check_plot_works(mixed_df.plot, stacked=False)
                msg = (
                    "When stacked is True, each column must be either all positive or "
                    "all negative. Column 'w' contains both positive and negative "
                    "values"
                )
                with pytest.raises(ValueError, match=msg):
                    mixed_df.plot(stacked=True)

                # Use an index with strictly positive values, preventing
                #  matplotlib from warning about ignoring xlim
                df2 = df.set_index(df.index + 1)
                _check_plot_works(df2.plot, kind=kind, logx=True, stacked=True)

    def test_line_area_nan_df(self):
        values1 = [1, 2, np.nan, 3]
        values2 = [3, np.nan, 2, 1]
        df = DataFrame({"a": values1, "b": values2})
        tdf = DataFrame({"a": values1, "b": values2}, index=tm.makeDateIndex(k=4))

        for d in [df, tdf]:
            ax = _check_plot_works(d.plot)
            masked1 = ax.lines[0].get_ydata()
            masked2 = ax.lines[1].get_ydata()
            # remove nan for comparison purpose

            exp = np.array([1, 2, 3], dtype=np.float64)
            tm.assert_numpy_array_equal(np.delete(masked1.data, 2), exp)

            exp = np.array([3, 2, 1], dtype=np.float64)
            tm.assert_numpy_array_equal(np.delete(masked2.data, 1), exp)
            tm.assert_numpy_array_equal(
                masked1.mask, np.array([False, False, True, False])
            )
            tm.assert_numpy_array_equal(
                masked2.mask, np.array([False, True, False, False])
            )

            expected1 = np.array([1, 2, 0, 3], dtype=np.float64)
            expected2 = np.array([3, 0, 2, 1], dtype=np.float64)

            ax = _check_plot_works(d.plot, stacked=True)
            tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
            tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected1 + expected2)

            ax = _check_plot_works(d.plot.area)
            tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
            tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected1 + expected2)

            ax = _check_plot_works(d.plot.area, stacked=False)
            tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
            tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected2)

    def test_line_lim(self):
        df = DataFrame(np.random.rand(6, 3), columns=["x", "y", "z"])
        ax = df.plot()
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data()[0][0]
        assert xmax >= lines[0].get_data()[0][-1]

        ax = df.plot(secondary_y=True)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data()[0][0]
        assert xmax >= lines[0].get_data()[0][-1]

        axes = df.plot(secondary_y=True, subplots=True)
        self._check_axes_shape(axes, axes_num=3, layout=(3, 1))
        for ax in axes:
            assert hasattr(ax, "left_ax")
            assert not hasattr(ax, "right_ax")
            xmin, xmax = ax.get_xlim()
            lines = ax.get_lines()
            assert xmin <= lines[0].get_data()[0][0]
            assert xmax >= lines[0].get_data()[0][-1]

    @pytest.mark.xfail(
        strict=False,
        reason="2020-12-01 this has been failing periodically on the "
        "ymin==0 assertion for a week or so.",
    )
    @pytest.mark.parametrize("stacked", [True, False])
    def test_area_lim(self, stacked):
        df = DataFrame(np.random.rand(6, 4), columns=["x", "y", "z", "four"])

        neg_df = -df

        ax = _check_plot_works(df.plot.area, stacked=stacked)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data()[0][0]
        assert xmax >= lines[0].get_data()[0][-1]
        assert ymin == 0

        ax = _check_plot_works(neg_df.plot.area, stacked=stacked)
        ymin, ymax = ax.get_ylim()
        assert ymax == 0

    def test_area_sharey_dont_overwrite(self):
        # GH37942
        df = DataFrame(np.random.rand(4, 2), columns=["x", "y"])
        fig, (ax1, ax2) = self.plt.subplots(1, 2, sharey=True)

        df.plot(ax=ax1, kind="area")
        df.plot(ax=ax2, kind="area")

        assert self.get_y_axis(ax1).joined(ax1, ax2)
        assert self.get_y_axis(ax2).joined(ax1, ax2)

    def test_bar_linewidth(self):
        df = DataFrame(np.random.randn(5, 5))

        # regular
        ax = df.plot.bar(linewidth=2)
        for r in ax.patches:
            assert r.get_linewidth() == 2

        # stacked
        ax = df.plot.bar(stacked=True, linewidth=2)
        for r in ax.patches:
            assert r.get_linewidth() == 2

        # subplots
        axes = df.plot.bar(linewidth=2, subplots=True)
        self._check_axes_shape(axes, axes_num=5, layout=(5, 1))
        for ax in axes:
            for r in ax.patches:
                assert r.get_linewidth() == 2

    def test_bar_barwidth(self):
        df = DataFrame(np.random.randn(5, 5))

        width = 0.9

        # regular
        ax = df.plot.bar(width=width)
        for r in ax.patches:
            assert r.get_width() == width / len(df.columns)

        # stacked
        ax = df.plot.bar(stacked=True, width=width)
        for r in ax.patches:
            assert r.get_width() == width

        # horizontal regular
        ax = df.plot.barh(width=width)
        for r in ax.patches:
            assert r.get_height() == width / len(df.columns)

        # horizontal stacked
        ax = df.plot.barh(stacked=True, width=width)
        for r in ax.patches:
            assert r.get_height() == width

        # subplots
        axes = df.plot.bar(width=width, subplots=True)
        for ax in axes:
            for r in ax.patches:
                assert r.get_width() == width

        # horizontal subplots
        axes = df.plot.barh(width=width, subplots=True)
        for ax in axes:
            for r in ax.patches:
                assert r.get_height() == width

    def test_bar_bottom_left(self):
        df = DataFrame(np.random.rand(5, 5))
        ax = df.plot.bar(stacked=False, bottom=1)
        result = [p.get_y() for p in ax.patches]
        assert result == [1] * 25

        ax = df.plot.bar(stacked=True, bottom=[-1, -2, -3, -4, -5])
        result = [p.get_y() for p in ax.patches[:5]]
        assert result == [-1, -2, -3, -4, -5]

        ax = df.plot.barh(stacked=False, left=np.array([1, 1, 1, 1, 1]))
        result = [p.get_x() for p in ax.patches]
        assert result == [1] * 25

        ax = df.plot.barh(stacked=True, left=[1, 2, 3, 4, 5])
        result = [p.get_x() for p in ax.patches[:5]]
        assert result == [1, 2, 3, 4, 5]

        axes = df.plot.bar(subplots=True, bottom=-1)
        for ax in axes:
            result = [p.get_y() for p in ax.patches]
            assert result == [-1] * 5

        axes = df.plot.barh(subplots=True, left=np.array([1, 1, 1, 1, 1]))
        for ax in axes:
            result = [p.get_x() for p in ax.patches]
            assert result == [1] * 5

    def test_bar_nan(self):
        df = DataFrame({"A": [10, np.nan, 20], "B": [5, 10, 20], "C": [1, 2, 3]})
        ax = df.plot.bar()
        expected = [10, 0, 20, 5, 10, 20, 1, 2, 3]
        result = [p.get_height() for p in ax.patches]
        assert result == expected

        ax = df.plot.bar(stacked=True)
        result = [p.get_height() for p in ax.patches]
        assert result == expected

        result = [p.get_y() for p in ax.patches]
        expected = [0.0, 0.0, 0.0, 10.0, 0.0, 20.0, 15.0, 10.0, 40.0]
        assert result == expected

    def test_bar_categorical(self):
        # GH 13019
        df1 = DataFrame(
            np.random.randn(6, 5),
            index=pd.Index(list("ABCDEF")),
            columns=pd.Index(list("abcde")),
        )
        # categorical index must behave the same
        df2 = DataFrame(
            np.random.randn(6, 5),
            index=pd.CategoricalIndex(list("ABCDEF")),
            columns=pd.CategoricalIndex(list("abcde")),
        )

        for df in [df1, df2]:
            ax = df.plot.bar()
            ticks = ax.xaxis.get_ticklocs()
            tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
            assert ax.get_xlim() == (-0.5, 5.5)
            # check left-edge of bars
            assert ax.patches[0].get_x() == -0.25
            assert ax.patches[-1].get_x() == 5.15

            ax = df.plot.bar(stacked=True)
            tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
            assert ax.get_xlim() == (-0.5, 5.5)
            assert ax.patches[0].get_x() == -0.25
            assert ax.patches[-1].get_x() == 4.75

    def test_plot_scatter(self):
        df = DataFrame(
            np.random.randn(6, 4),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )

        _check_plot_works(df.plot.scatter, x="x", y="y")
        _check_plot_works(df.plot.scatter, x=1, y=2)

        msg = re.escape("scatter() missing 1 required positional argument: 'y'")
        with pytest.raises(TypeError, match=msg):
            df.plot.scatter(x="x")
        msg = re.escape("scatter() missing 1 required positional argument: 'x'")
        with pytest.raises(TypeError, match=msg):
            df.plot.scatter(y="y")

        # GH 6951
        axes = df.plot(x="x", y="y", kind="scatter", subplots=True)
        self._check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_raise_error_on_datetime_time_data(self):
        # GH 8113, datetime.time type is not supported by matplotlib in scatter
        df = DataFrame(np.random.randn(10), columns=["a"])
        df["dtime"] = date_range(start="2014-01-01", freq="h", periods=10).time
        msg = "must be a string or a number, not 'datetime.time'"

        with pytest.raises(TypeError, match=msg):
            df.plot(kind="scatter", x="dtime", y="a")

    def test_scatterplot_datetime_data(self):
        # GH 30391
        dates = date_range(start=date(2019, 1, 1), periods=12, freq="W")
        vals = np.random.normal(0, 1, len(dates))
        df = DataFrame({"dates": dates, "vals": vals})

        _check_plot_works(df.plot.scatter, x="dates", y="vals")
        _check_plot_works(df.plot.scatter, x=0, y=1)

    def test_scatterplot_object_data(self):
        # GH 18755
        df = DataFrame({"a": ["A", "B", "C"], "b": [2, 3, 4]})

        _check_plot_works(df.plot.scatter, x="a", y="b")
        _check_plot_works(df.plot.scatter, x=0, y=1)

        df = DataFrame({"a": ["A", "B", "C"], "b": ["a", "b", "c"]})

        _check_plot_works(df.plot.scatter, x="a", y="b")
        _check_plot_works(df.plot.scatter, x=0, y=1)

    @pytest.mark.parametrize("ordered", [True, False])
    @pytest.mark.parametrize(
        "categories",
        (["setosa", "versicolor", "virginica"], ["versicolor", "virginica", "setosa"]),
    )
    def test_scatterplot_color_by_categorical(self, ordered, categories):
        df = DataFrame(
            [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
            columns=["length", "width"],
        )
        df["species"] = pd.Categorical(
            ["setosa", "setosa", "virginica", "virginica", "versicolor"],
            ordered=ordered,
            categories=categories,
        )
        ax = df.plot.scatter(x=0, y=1, c="species")
        (colorbar_collection,) = ax.collections
        colorbar = colorbar_collection.colorbar

        expected_ticks = np.array([0.5, 1.5, 2.5])
        result_ticks = colorbar.get_ticks()
        tm.assert_numpy_array_equal(result_ticks, expected_ticks)

        expected_boundaries = np.array([0.0, 1.0, 2.0, 3.0])
        result_boundaries = colorbar._boundaries
        tm.assert_numpy_array_equal(result_boundaries, expected_boundaries)

        expected_yticklabels = categories
        result_yticklabels = [i.get_text() for i in colorbar.ax.get_ymajorticklabels()]
        assert all(i == j for i, j in zip(result_yticklabels, expected_yticklabels))

    @pytest.mark.parametrize("x, y", [("x", "y"), ("y", "x"), ("y", "y")])
    def test_plot_scatter_with_categorical_data(self, x, y):
        # after fixing GH 18755, should be able to plot categorical data
        df = DataFrame({"x": [1, 2, 3, 4], "y": pd.Categorical(["a", "b", "a", "c"])})

        _check_plot_works(df.plot.scatter, x=x, y=y)

    def test_plot_scatter_with_c(self, request):
        from pandas.plotting._matplotlib.compat import mpl_ge_3_4_0

        df = DataFrame(
            np.random.randn(6, 4),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )

        axes = [df.plot.scatter(x="x", y="y", c="z"), df.plot.scatter(x=0, y=1, c=2)]
        for ax in axes:
            # default to Greys
            assert ax.collections[0].cmap.name == "Greys"

            if mpl_ge_3_4_0():
                assert ax.collections[0].colorbar.ax.get_ylabel() == "z"
            else:
                assert ax.collections[0].colorbar._label == "z"

        cm = "cubehelix"
        ax = df.plot.scatter(x="x", y="y", c="z", colormap=cm)
        assert ax.collections[0].cmap.name == cm

        # verify turning off colorbar works
        ax = df.plot.scatter(x="x", y="y", c="z", colorbar=False)
        assert ax.collections[0].colorbar is None

        # verify that we can still plot a solid color
        ax = df.plot.scatter(x=0, y=1, c="red")
        assert ax.collections[0].colorbar is None
        self._check_colors(ax.collections, facecolors=["r"])

        # Ensure that we can pass an np.array straight through to matplotlib,
        # this functionality was accidentally removed previously.
        # See https://github.com/pandas-dev/pandas/issues/8852 for bug report
        #
        # Exercise colormap path and non-colormap path as they are independent
        #
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        red_rgba = [1.0, 0.0, 0.0, 1.0]
        green_rgba = [0.0, 1.0, 0.0, 1.0]
        rgba_array = np.array([red_rgba, green_rgba])
        ax = df.plot.scatter(x="A", y="B", c=rgba_array)
        # expect the face colors of the points in the non-colormap path to be
        # identical to the values we supplied, normally we'd be on shaky ground
        # comparing floats for equality but here we expect them to be
        # identical.
        tm.assert_numpy_array_equal(ax.collections[0].get_facecolor(), rgba_array)
        # we don't test the colors of the faces in this next plot because they
        # are dependent on the spring colormap, which may change its colors
        # later.
        float_array = np.array([0.0, 1.0])
        df.plot.scatter(x="A", y="B", c=float_array, cmap="spring")

    def test_plot_scatter_with_s(self):
        # this refers to GH 32904
        df = DataFrame(np.random.random((10, 3)) * 100, columns=["a", "b", "c"])

        ax = df.plot.scatter(x="a", y="b", s="c")
        tm.assert_numpy_array_equal(df["c"].values, right=ax.collections[0].get_sizes())

    def test_plot_bar(self):
        df = DataFrame(
            np.random.randn(6, 4),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )

        _check_plot_works(df.plot.bar)
        _check_plot_works(df.plot.bar, legend=False)
        _check_plot_works(df.plot.bar, default_axes=True, subplots=True)
        _check_plot_works(df.plot.bar, stacked=True)

        df = DataFrame(
            np.random.randn(10, 15),
            index=list(string.ascii_letters[:10]),
            columns=range(15),
        )
        _check_plot_works(df.plot.bar)

        df = DataFrame({"a": [0, 1], "b": [1, 0]})
        ax = _check_plot_works(df.plot.bar)
        self._check_ticks_props(ax, xrot=90)

        ax = df.plot.bar(rot=35, fontsize=10)
        self._check_ticks_props(ax, xrot=35, xlabelsize=10, ylabelsize=10)

        ax = _check_plot_works(df.plot.barh)
        self._check_ticks_props(ax, yrot=0)

        ax = df.plot.barh(rot=55, fontsize=11)
        self._check_ticks_props(ax, yrot=55, ylabelsize=11, xlabelsize=11)

    def test_boxplot(self):
        df = self.hist_df
        series = df["height"]
        numeric_cols = df._get_numeric_data().columns
        labels = [pprint_thing(c) for c in numeric_cols]

        ax = _check_plot_works(df.plot.box)
        self._check_text_labels(ax.get_xticklabels(), labels)
        tm.assert_numpy_array_equal(
            ax.xaxis.get_ticklocs(), np.arange(1, len(numeric_cols) + 1)
        )
        assert len(ax.lines) == self.bp_n_objects * len(numeric_cols)
        tm.close()

        axes = series.plot.box(rot=40)
        self._check_ticks_props(axes, xrot=40, yrot=0)
        tm.close()

        ax = _check_plot_works(series.plot.box)

        positions = np.array([1, 6, 7])
        ax = df.plot.box(positions=positions)
        numeric_cols = df._get_numeric_data().columns
        labels = [pprint_thing(c) for c in numeric_cols]
        self._check_text_labels(ax.get_xticklabels(), labels)
        tm.assert_numpy_array_equal(ax.xaxis.get_ticklocs(), positions)
        assert len(ax.lines) == self.bp_n_objects * len(numeric_cols)

    def test_boxplot_vertical(self):
        df = self.hist_df
        numeric_cols = df._get_numeric_data().columns
        labels = [pprint_thing(c) for c in numeric_cols]

        # if horizontal, yticklabels are rotated
        ax = df.plot.box(rot=50, fontsize=8, vert=False)
        self._check_ticks_props(ax, xrot=0, yrot=50, ylabelsize=8)
        self._check_text_labels(ax.get_yticklabels(), labels)
        assert len(ax.lines) == self.bp_n_objects * len(numeric_cols)

        axes = _check_plot_works(
            df.plot.box,
            default_axes=True,
            subplots=True,
            vert=False,
            logx=True,
        )
        self._check_axes_shape(axes, axes_num=3, layout=(1, 3))
        self._check_ax_scales(axes, xaxis="log")
        for ax, label in zip(axes, labels):
            self._check_text_labels(ax.get_yticklabels(), [label])
            assert len(ax.lines) == self.bp_n_objects

        positions = np.array([3, 2, 8])
        ax = df.plot.box(positions=positions, vert=False)
        self._check_text_labels(ax.get_yticklabels(), labels)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), positions)
        assert len(ax.lines) == self.bp_n_objects * len(numeric_cols)

    def test_boxplot_return_type(self):
        df = DataFrame(
            np.random.randn(6, 4),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        msg = "return_type must be {None, 'axes', 'dict', 'both'}"
        with pytest.raises(ValueError, match=msg):
            df.plot.box(return_type="not_a_type")

        result = df.plot.box(return_type="dict")
        self._check_box_return_type(result, "dict")

        result = df.plot.box(return_type="axes")
        self._check_box_return_type(result, "axes")

        result = df.plot.box()  # default axes
        self._check_box_return_type(result, "axes")

        result = df.plot.box(return_type="both")
        self._check_box_return_type(result, "both")

    @td.skip_if_no_scipy
    def test_kde_df(self):
        df = DataFrame(np.random.randn(100, 4))
        ax = _check_plot_works(df.plot, kind="kde")
        expected = [pprint_thing(c) for c in df.columns]
        self._check_legend_labels(ax, labels=expected)
        self._check_ticks_props(ax, xrot=0)

        ax = df.plot(kind="kde", rot=20, fontsize=5)
        self._check_ticks_props(ax, xrot=20, xlabelsize=5, ylabelsize=5)

        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            kind="kde",
            subplots=True,
        )
        self._check_axes_shape(axes, axes_num=4, layout=(4, 1))

        axes = df.plot(kind="kde", logy=True, subplots=True)
        self._check_ax_scales(axes, yaxis="log")

    @td.skip_if_no_scipy
    def test_kde_missing_vals(self):
        df = DataFrame(np.random.uniform(size=(100, 4)))
        df.loc[0, 0] = np.nan
        _check_plot_works(df.plot, kind="kde")

    def test_hist_df(self):
        from matplotlib.patches import Rectangle

        df = DataFrame(np.random.randn(100, 4))
        series = df[0]

        ax = _check_plot_works(df.plot.hist)
        expected = [pprint_thing(c) for c in df.columns]
        self._check_legend_labels(ax, labels=expected)

        axes = _check_plot_works(
            df.plot.hist,
            default_axes=True,
            subplots=True,
            logy=True,
        )
        self._check_axes_shape(axes, axes_num=4, layout=(4, 1))
        self._check_ax_scales(axes, yaxis="log")

        axes = series.plot.hist(rot=40)
        self._check_ticks_props(axes, xrot=40, yrot=0)
        tm.close()

        ax = series.plot.hist(cumulative=True, bins=4, density=True)
        # height of last bin (index 5) must be 1.0
        rects = [x for x in ax.get_children() if isinstance(x, Rectangle)]
        tm.assert_almost_equal(rects[-1].get_height(), 1.0)
        tm.close()

        ax = series.plot.hist(cumulative=True, bins=4)
        rects = [x for x in ax.get_children() if isinstance(x, Rectangle)]

        tm.assert_almost_equal(rects[-2].get_height(), 100.0)
        tm.close()

        # if horizontal, yticklabels are rotated
        axes = df.plot.hist(rot=50, fontsize=8, orientation="horizontal")
        self._check_ticks_props(axes, xrot=0, yrot=50, ylabelsize=8)

    @pytest.mark.parametrize(
        "weights", [0.1 * np.ones(shape=(100,)), 0.1 * np.ones(shape=(100, 2))]
    )
    def test_hist_weights(self, weights):
        # GH 33173
        np.random.seed(0)
        df = DataFrame(dict(zip(["A", "B"], np.random.randn(2, 100))))

        ax1 = _check_plot_works(df.plot, kind="hist", weights=weights)
        ax2 = _check_plot_works(df.plot, kind="hist")

        patch_height_with_weights = [patch.get_height() for patch in ax1.patches]

        # original heights with no weights, and we manually multiply with example
        # weights, so after multiplication, they should be almost same
        expected_patch_height = [0.1 * patch.get_height() for patch in ax2.patches]

        tm.assert_almost_equal(patch_height_with_weights, expected_patch_height)

    def _check_box_coord(
        self,
        patches,
        expected_y=None,
        expected_h=None,
        expected_x=None,
        expected_w=None,
    ):
        result_y = np.array([p.get_y() for p in patches])
        result_height = np.array([p.get_height() for p in patches])
        result_x = np.array([p.get_x() for p in patches])
        result_width = np.array([p.get_width() for p in patches])
        # dtype is depending on above values, no need to check

        if expected_y is not None:
            tm.assert_numpy_array_equal(result_y, expected_y, check_dtype=False)
        if expected_h is not None:
            tm.assert_numpy_array_equal(result_height, expected_h, check_dtype=False)
        if expected_x is not None:
            tm.assert_numpy_array_equal(result_x, expected_x, check_dtype=False)
        if expected_w is not None:
            tm.assert_numpy_array_equal(result_width, expected_w, check_dtype=False)

    def test_hist_df_coord(self):
        normal_df = DataFrame(
            {
                "A": np.repeat(np.array([1, 2, 3, 4, 5]), np.array([10, 9, 8, 7, 6])),
                "B": np.repeat(np.array([1, 2, 3, 4, 5]), np.array([8, 8, 8, 8, 8])),
                "C": np.repeat(np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])),
            },
            columns=["A", "B", "C"],
        )

        nan_df = DataFrame(
            {
                "A": np.repeat(
                    np.array([np.nan, 1, 2, 3, 4, 5]), np.array([3, 10, 9, 8, 7, 6])
                ),
                "B": np.repeat(
                    np.array([1, np.nan, 2, 3, 4, 5]), np.array([8, 3, 8, 8, 8, 8])
                ),
                "C": np.repeat(
                    np.array([1, 2, 3, np.nan, 4, 5]), np.array([6, 7, 8, 3, 9, 10])
                ),
            },
            columns=["A", "B", "C"],
        )

        for df in [normal_df, nan_df]:
            ax = df.plot.hist(bins=5)
            self._check_box_coord(
                ax.patches[:5],
                expected_y=np.array([0, 0, 0, 0, 0]),
                expected_h=np.array([10, 9, 8, 7, 6]),
            )
            self._check_box_coord(
                ax.patches[5:10],
                expected_y=np.array([0, 0, 0, 0, 0]),
                expected_h=np.array([8, 8, 8, 8, 8]),
            )
            self._check_box_coord(
                ax.patches[10:],
                expected_y=np.array([0, 0, 0, 0, 0]),
                expected_h=np.array([6, 7, 8, 9, 10]),
            )

            ax = df.plot.hist(bins=5, stacked=True)
            self._check_box_coord(
                ax.patches[:5],
                expected_y=np.array([0, 0, 0, 0, 0]),
                expected_h=np.array([10, 9, 8, 7, 6]),
            )
            self._check_box_coord(
                ax.patches[5:10],
                expected_y=np.array([10, 9, 8, 7, 6]),
                expected_h=np.array([8, 8, 8, 8, 8]),
            )
            self._check_box_coord(
                ax.patches[10:],
                expected_y=np.array([18, 17, 16, 15, 14]),
                expected_h=np.array([6, 7, 8, 9, 10]),
            )

            axes = df.plot.hist(bins=5, stacked=True, subplots=True)
            self._check_box_coord(
                axes[0].patches,
                expected_y=np.array([0, 0, 0, 0, 0]),
                expected_h=np.array([10, 9, 8, 7, 6]),
            )
            self._check_box_coord(
                axes[1].patches,
                expected_y=np.array([0, 0, 0, 0, 0]),
                expected_h=np.array([8, 8, 8, 8, 8]),
            )
            self._check_box_coord(
                axes[2].patches,
                expected_y=np.array([0, 0, 0, 0, 0]),
                expected_h=np.array([6, 7, 8, 9, 10]),
            )

            # horizontal
            ax = df.plot.hist(bins=5, orientation="horizontal")
            self._check_box_coord(
                ax.patches[:5],
                expected_x=np.array([0, 0, 0, 0, 0]),
                expected_w=np.array([10, 9, 8, 7, 6]),
            )
            self._check_box_coord(
                ax.patches[5:10],
                expected_x=np.array([0, 0, 0, 0, 0]),
                expected_w=np.array([8, 8, 8, 8, 8]),
            )
            self._check_box_coord(
                ax.patches[10:],
                expected_x=np.array([0, 0, 0, 0, 0]),
                expected_w=np.array([6, 7, 8, 9, 10]),
            )

            ax = df.plot.hist(bins=5, stacked=True, orientation="horizontal")
            self._check_box_coord(
                ax.patches[:5],
                expected_x=np.array([0, 0, 0, 0, 0]),
                expected_w=np.array([10, 9, 8, 7, 6]),
            )
            self._check_box_coord(
                ax.patches[5:10],
                expected_x=np.array([10, 9, 8, 7, 6]),
                expected_w=np.array([8, 8, 8, 8, 8]),
            )
            self._check_box_coord(
                ax.patches[10:],
                expected_x=np.array([18, 17, 16, 15, 14]),
                expected_w=np.array([6, 7, 8, 9, 10]),
            )

            axes = df.plot.hist(
                bins=5, stacked=True, subplots=True, orientation="horizontal"
            )
            self._check_box_coord(
                axes[0].patches,
                expected_x=np.array([0, 0, 0, 0, 0]),
                expected_w=np.array([10, 9, 8, 7, 6]),
            )
            self._check_box_coord(
                axes[1].patches,
                expected_x=np.array([0, 0, 0, 0, 0]),
                expected_w=np.array([8, 8, 8, 8, 8]),
            )
            self._check_box_coord(
                axes[2].patches,
                expected_x=np.array([0, 0, 0, 0, 0]),
                expected_w=np.array([6, 7, 8, 9, 10]),
            )

    def test_plot_int_columns(self):
        df = DataFrame(np.random.randn(100, 4)).cumsum()
        _check_plot_works(df.plot, legend=True)

    def test_style_by_column(self):
        import matplotlib.pyplot as plt

        fig = plt.gcf()

        df = DataFrame(np.random.randn(100, 3))
        for markers in [
            {0: "^", 1: "+", 2: "o"},
            {0: "^", 1: "+"},
            ["^", "+", "o"],
            ["^", "+"],
        ]:
            fig.clf()
            fig.add_subplot(111)
            ax = df.plot(style=markers)
            for idx, line in enumerate(ax.get_lines()[: len(markers)]):
                assert line.get_marker() == markers[idx]

    def test_line_label_none(self):
        s = Series([1, 2])
        ax = s.plot()
        assert ax.get_legend() is None

        ax = s.plot(legend=True)
        assert ax.get_legend().get_texts()[0].get_text() == "None"

    @pytest.mark.parametrize(
        "props, expected",
        [
            ("boxprops", "boxes"),
            ("whiskerprops", "whiskers"),
            ("capprops", "caps"),
            ("medianprops", "medians"),
        ],
    )
    def test_specified_props_kwd_plot_box(self, props, expected):
        # GH 30346
        df = DataFrame({k: np.random.random(100) for k in "ABC"})
        kwd = {props: {"color": "C1"}}
        result = df.plot.box(return_type="dict", **kwd)

        assert result[expected][0].get_color() == "C1"

    def test_unordered_ts(self):
        df = DataFrame(
            np.array([3.0, 2.0, 1.0]),
            index=[date(2012, 10, 1), date(2012, 9, 1), date(2012, 8, 1)],
            columns=["test"],
        )
        ax = df.plot()
        xticks = ax.lines[0].get_xdata()
        assert xticks[0] < xticks[1]
        ydata = ax.lines[0].get_ydata()
        tm.assert_numpy_array_equal(ydata, np.array([1.0, 2.0, 3.0]))

    @td.skip_if_no_scipy
    def test_kind_both_ways(self):
        df = DataFrame({"x": [1, 2, 3]})
        for kind in plotting.PlotAccessor._common_kinds:

            df.plot(kind=kind)
            getattr(df.plot, kind)()
        for kind in ["scatter", "hexbin"]:
            df.plot("x", "x", kind=kind)
            getattr(df.plot, kind)("x", "x")

    def test_all_invalid_plot_data(self):
        df = DataFrame(list("abcd"))
        for kind in plotting.PlotAccessor._common_kinds:

            msg = "no numeric data to plot"
            with pytest.raises(TypeError, match=msg):
                df.plot(kind=kind)

    def test_partially_invalid_plot_data(self):
        with tm.RNGContext(42):
            df = DataFrame(np.random.randn(10, 2), dtype=object)
            df[np.random.rand(df.shape[0]) > 0.5] = "a"
            for kind in plotting.PlotAccessor._common_kinds:
                msg = "no numeric data to plot"
                with pytest.raises(TypeError, match=msg):
                    df.plot(kind=kind)

        with tm.RNGContext(42):
            # area plot doesn't support positive/negative mixed data
            df = DataFrame(np.random.rand(10, 2), dtype=object)
            df[np.random.rand(df.shape[0]) > 0.5] = "a"
            with pytest.raises(TypeError, match="no numeric data to plot"):
                df.plot(kind="area")

    def test_invalid_kind(self):
        df = DataFrame(np.random.randn(10, 2))
        msg = "invalid_plot_kind is not a valid plot kind"
        with pytest.raises(ValueError, match=msg):
            df.plot(kind="invalid_plot_kind")

    @pytest.mark.parametrize(
        "x,y,lbl",
        [
            (["B", "C"], "A", "a"),
            (["A"], ["B", "C"], ["b", "c"]),
        ],
    )
    def test_invalid_xy_args(self, x, y, lbl):
        # GH 18671, 19699 allows y to be list-like but not x
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        with pytest.raises(ValueError, match="x must be a label or position"):
            df.plot(x=x, y=y, label=lbl)

    def test_bad_label(self):
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        msg = "label should be list-like and same length as y"
        with pytest.raises(ValueError, match=msg):
            df.plot(x="A", y=["B", "C"], label="bad_label")

    @pytest.mark.parametrize("x,y", [("A", "B"), (["A"], "B")])
    def test_invalid_xy_args_dup_cols(self, x, y):
        # GH 18671, 19699 allows y to be list-like but not x
        df = DataFrame([[1, 3, 5], [2, 4, 6]], columns=list("AAB"))
        with pytest.raises(ValueError, match="x must be a label or position"):
            df.plot(x=x, y=y)

    @pytest.mark.parametrize(
        "x,y,lbl,colors",
        [
            ("A", ["B"], ["b"], ["red"]),
            ("A", ["B", "C"], ["b", "c"], ["red", "blue"]),
            (0, [1, 2], ["bokeh", "cython"], ["green", "yellow"]),
        ],
    )
    def test_y_listlike(self, x, y, lbl, colors):
        # GH 19699: tests list-like y and verifies lbls & colors
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        _check_plot_works(df.plot, x="A", y=y, label=lbl)

        ax = df.plot(x=x, y=y, label=lbl, color=colors)
        assert len(ax.lines) == len(y)
        self._check_colors(ax.get_lines(), linecolors=colors)

    @pytest.mark.parametrize("x,y,colnames", [(0, 1, ["A", "B"]), (1, 0, [0, 1])])
    def test_xy_args_integer(self, x, y, colnames):
        # GH 20056: tests integer args for xy and checks col names
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        df.columns = colnames
        _check_plot_works(df.plot, x=x, y=y)

    def test_hexbin_basic(self):
        df = self.hexbin_df

        ax = df.plot.hexbin(x="A", y="B", gridsize=10)
        # TODO: need better way to test. This just does existence.
        assert len(ax.collections) == 1

        # GH 6951
        axes = df.plot.hexbin(x="A", y="B", subplots=True)
        # hexbin should have 2 axes in the figure, 1 for plotting and another
        # is colorbar
        assert len(axes[0].figure.axes) == 2
        # return value is single axes
        self._check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_hexbin_with_c(self):
        df = self.hexbin_df

        ax = df.plot.hexbin(x="A", y="B", C="C")
        assert len(ax.collections) == 1

        ax = df.plot.hexbin(x="A", y="B", C="C", reduce_C_function=np.std)
        assert len(ax.collections) == 1

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, "BuGn"),  # default cmap
            ({"colormap": "cubehelix"}, "cubehelix"),
            ({"cmap": "YlGn"}, "YlGn"),
        ],
    )
    def test_hexbin_cmap(self, kwargs, expected):
        df = self.hexbin_df
        ax = df.plot.hexbin(x="A", y="B", **kwargs)
        assert ax.collections[0].cmap.name == expected

    def test_pie_df(self):
        df = DataFrame(
            np.random.rand(5, 3),
            columns=["X", "Y", "Z"],
            index=["a", "b", "c", "d", "e"],
        )
        msg = "pie requires either y column or 'subplots=True'"
        with pytest.raises(ValueError, match=msg):
            df.plot.pie()

        ax = _check_plot_works(df.plot.pie, y="Y")
        self._check_text_labels(ax.texts, df.index)

        ax = _check_plot_works(df.plot.pie, y=2)
        self._check_text_labels(ax.texts, df.index)

        axes = _check_plot_works(
            df.plot.pie,
            default_axes=True,
            subplots=True,
        )
        assert len(axes) == len(df.columns)
        for ax in axes:
            self._check_text_labels(ax.texts, df.index)
        for ax, ylabel in zip(axes, df.columns):
            assert ax.get_ylabel() == ylabel

        labels = ["A", "B", "C", "D", "E"]
        color_args = ["r", "g", "b", "c", "m"]
        axes = _check_plot_works(
            df.plot.pie,
            default_axes=True,
            subplots=True,
            labels=labels,
            colors=color_args,
        )
        assert len(axes) == len(df.columns)

        for ax in axes:
            self._check_text_labels(ax.texts, labels)
            self._check_colors(ax.patches, facecolors=color_args)

    def test_pie_df_nan(self):
        import matplotlib as mpl

        df = DataFrame(np.random.rand(4, 4))
        for i in range(4):
            df.iloc[i, i] = np.nan
        fig, axes = self.plt.subplots(ncols=4)

        # GH 37668
        kwargs = {}
        if mpl.__version__ >= "3.3":
            kwargs = {"normalize": True}

        with tm.assert_produces_warning(None):
            df.plot.pie(subplots=True, ax=axes, legend=True, **kwargs)

        base_expected = ["0", "1", "2", "3"]
        for i, ax in enumerate(axes):
            expected = list(base_expected)  # force copy
            expected[i] = ""
            result = [x.get_text() for x in ax.texts]
            assert result == expected

            # legend labels
            # NaN's not included in legend with subplots
            # see https://github.com/pandas-dev/pandas/issues/8390
            result_labels = [x.get_text() for x in ax.get_legend().get_texts()]
            expected_labels = base_expected[:i] + base_expected[i + 1 :]
            assert result_labels == expected_labels

    def test_errorbar_plot(self):
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        d_err = {"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}
        df_err = DataFrame(d_err)

        # check line plots
        ax = _check_plot_works(df.plot, yerr=df_err, logy=True)
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(df.plot, yerr=df_err, logx=True, logy=True)
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(df.plot, yerr=df_err, loglog=True)
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(
            (df + 1).plot, yerr=df_err, xerr=df_err, kind="bar", log=True
        )
        self._check_has_errorbars(ax, xerr=2, yerr=2)

        # yerr is raw error values
        ax = _check_plot_works(df["y"].plot, yerr=np.ones(12) * 0.4)
        self._check_has_errorbars(ax, xerr=0, yerr=1)

        ax = _check_plot_works(df.plot, yerr=np.ones((2, 12)) * 0.4)
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        # yerr is column name
        for yerr in ["yerr", "誤差"]:
            s_df = df.copy()
            s_df[yerr] = np.ones(12) * 0.2

            ax = _check_plot_works(s_df.plot, yerr=yerr)
            self._check_has_errorbars(ax, xerr=0, yerr=2)

            ax = _check_plot_works(s_df.plot, y="y", x="x", yerr=yerr)
            self._check_has_errorbars(ax, xerr=0, yerr=1)

        with tm.external_error_raised(ValueError):
            df.plot(yerr=np.random.randn(11))

        df_err = DataFrame({"x": ["zzz"] * 12, "y": ["zzz"] * 12})
        with tm.external_error_raised(TypeError):
            df.plot(yerr=df_err)

    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    def test_errorbar_plot_different_kinds(self, kind):
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        d_err = {"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}
        df_err = DataFrame(d_err)

        ax = _check_plot_works(df.plot, yerr=df_err["x"], kind=kind)
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(df.plot, yerr=d_err, kind=kind)
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(df.plot, yerr=df_err, xerr=df_err, kind=kind)
        self._check_has_errorbars(ax, xerr=2, yerr=2)

        ax = _check_plot_works(df.plot, yerr=df_err["x"], xerr=df_err["x"], kind=kind)
        self._check_has_errorbars(ax, xerr=2, yerr=2)

        ax = _check_plot_works(df.plot, xerr=0.2, yerr=0.2, kind=kind)
        self._check_has_errorbars(ax, xerr=2, yerr=2)

        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            yerr=df_err,
            xerr=df_err,
            subplots=True,
            kind=kind,
        )
        self._check_has_errorbars(axes, xerr=1, yerr=1)

    @pytest.mark.xfail(reason="Iterator is consumed", raises=ValueError)
    def test_errorbar_plot_iterator(self):
        with warnings.catch_warnings():
            d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
            df = DataFrame(d)

            # yerr is iterator
            ax = _check_plot_works(df.plot, yerr=itertools.repeat(0.1, len(df)))
            self._check_has_errorbars(ax, xerr=0, yerr=2)

    def test_errorbar_with_integer_column_names(self):
        # test with integer column names
        df = DataFrame(np.random.randn(10, 2))
        df_err = DataFrame(np.random.randn(10, 2))
        ax = _check_plot_works(df.plot, yerr=df_err)
        self._check_has_errorbars(ax, xerr=0, yerr=2)
        ax = _check_plot_works(df.plot, y=0, yerr=1)
        self._check_has_errorbars(ax, xerr=0, yerr=1)

    def test_errorbar_with_partial_columns(self):
        df = DataFrame(np.random.randn(10, 3))
        df_err = DataFrame(np.random.randn(10, 2), columns=[0, 2])
        kinds = ["line", "bar"]
        for kind in kinds:
            ax = _check_plot_works(df.plot, yerr=df_err, kind=kind)
            self._check_has_errorbars(ax, xerr=0, yerr=2)

        ix = date_range("1/1/2000", periods=10, freq="M")
        df.set_index(ix, inplace=True)
        df_err.set_index(ix, inplace=True)
        ax = _check_plot_works(df.plot, yerr=df_err, kind="line")
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        d_err = {"x": np.ones(12) * 0.2, "z": np.ones(12) * 0.4}
        df_err = DataFrame(d_err)
        for err in [d_err, df_err]:
            ax = _check_plot_works(df.plot, yerr=err)
            self._check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    def test_errorbar_timeseries(self, kind):
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        d_err = {"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}

        # check time-series plots
        ix = date_range("1/1/2000", "1/1/2001", freq="M")
        tdf = DataFrame(d, index=ix)
        tdf_err = DataFrame(d_err, index=ix)

        ax = _check_plot_works(tdf.plot, yerr=tdf_err, kind=kind)
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(tdf.plot, yerr=d_err, kind=kind)
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(tdf.plot, y="y", yerr=tdf_err["x"], kind=kind)
        self._check_has_errorbars(ax, xerr=0, yerr=1)

        ax = _check_plot_works(tdf.plot, y="y", yerr="x", kind=kind)
        self._check_has_errorbars(ax, xerr=0, yerr=1)

        ax = _check_plot_works(tdf.plot, yerr=tdf_err, kind=kind)
        self._check_has_errorbars(ax, xerr=0, yerr=2)

        axes = _check_plot_works(
            tdf.plot,
            default_axes=True,
            kind=kind,
            yerr=tdf_err,
            subplots=True,
        )
        self._check_has_errorbars(axes, xerr=0, yerr=1)

    def test_errorbar_asymmetrical(self):
        np.random.seed(0)
        err = np.random.rand(3, 2, 5)

        # each column is [0, 1, 2, 3, 4], [3, 4, 5, 6, 7]...
        df = DataFrame(np.arange(15).reshape(3, 5)).T

        ax = df.plot(yerr=err, xerr=err / 2)

        yerr_0_0 = ax.collections[1].get_paths()[0].vertices[:, 1]
        expected_0_0 = err[0, :, 0] * np.array([-1, 1])
        tm.assert_almost_equal(yerr_0_0, expected_0_0)

        msg = re.escape(
            "Asymmetrical error bars should be provided with the shape (3, 2, 5)"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(yerr=err.T)

        tm.close()

    def test_table(self):
        df = DataFrame(np.random.rand(10, 3), index=list(string.ascii_letters[:10]))
        _check_plot_works(df.plot, table=True)
        _check_plot_works(df.plot, table=df)

        # GH 35945 UserWarning
        with tm.assert_produces_warning(None):
            ax = df.plot()
            assert len(ax.tables) == 0
            plotting.table(ax, df.T)
            assert len(ax.tables) == 1

    def test_errorbar_scatter(self):
        df = DataFrame(np.random.randn(5, 2), index=range(5), columns=["x", "y"])
        df_err = DataFrame(
            np.random.randn(5, 2) / 5, index=range(5), columns=["x", "y"]
        )

        ax = _check_plot_works(df.plot.scatter, x="x", y="y")
        self._check_has_errorbars(ax, xerr=0, yerr=0)
        ax = _check_plot_works(df.plot.scatter, x="x", y="y", xerr=df_err)
        self._check_has_errorbars(ax, xerr=1, yerr=0)

        ax = _check_plot_works(df.plot.scatter, x="x", y="y", yerr=df_err)
        self._check_has_errorbars(ax, xerr=0, yerr=1)
        ax = _check_plot_works(df.plot.scatter, x="x", y="y", xerr=df_err, yerr=df_err)
        self._check_has_errorbars(ax, xerr=1, yerr=1)

        def _check_errorbar_color(containers, expected, has_err="has_xerr"):
            lines = []
            errs = [c.lines for c in ax.containers if getattr(c, has_err, False)][0]
            for el in errs:
                if is_list_like(el):
                    lines.extend(el)
                else:
                    lines.append(el)
            err_lines = [x for x in lines if x in ax.collections]
            self._check_colors(
                err_lines, linecolors=np.array([expected] * len(err_lines))
            )

        # GH 8081
        df = DataFrame(np.random.randn(10, 5), columns=["a", "b", "c", "d", "e"])
        ax = df.plot.scatter(x="a", y="b", xerr="d", yerr="e", c="red")
        self._check_has_errorbars(ax, xerr=1, yerr=1)
        _check_errorbar_color(ax.containers, "red", has_err="has_xerr")
        _check_errorbar_color(ax.containers, "red", has_err="has_yerr")

        ax = df.plot.scatter(x="a", y="b", yerr="e", color="green")
        self._check_has_errorbars(ax, xerr=0, yerr=1)
        _check_errorbar_color(ax.containers, "green", has_err="has_yerr")

    def test_sharex_and_ax(self):
        # https://github.com/pandas-dev/pandas/issues/9737 using gridspec,
        # the axis in fig.get_axis() are sorted differently than pandas
        # expected them, so make sure that only the right ones are removed
        import matplotlib.pyplot as plt

        plt.close("all")
        gs, axes = _generate_4_axes_via_gridspec()

        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )

        def _check(axes):
            for ax in axes:
                assert len(ax.lines) == 1
                self._check_visible(ax.get_yticklabels(), visible=True)
            for ax in [axes[0], axes[2]]:
                self._check_visible(ax.get_xticklabels(), visible=False)
                self._check_visible(ax.get_xticklabels(minor=True), visible=False)
            for ax in [axes[1], axes[3]]:
                self._check_visible(ax.get_xticklabels(), visible=True)
                self._check_visible(ax.get_xticklabels(minor=True), visible=True)

        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax, sharex=True)
        gs.tight_layout(plt.gcf())
        _check(axes)
        tm.close()

        gs, axes = _generate_4_axes_via_gridspec()
        with tm.assert_produces_warning(UserWarning):
            axes = df.plot(subplots=True, ax=axes, sharex=True)
        _check(axes)
        tm.close()

        gs, axes = _generate_4_axes_via_gridspec()
        # without sharex, no labels should be touched!
        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax)

        gs.tight_layout(plt.gcf())
        for ax in axes:
            assert len(ax.lines) == 1
            self._check_visible(ax.get_yticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(minor=True), visible=True)
        tm.close()

    def test_sharey_and_ax(self):
        # https://github.com/pandas-dev/pandas/issues/9737 using gridspec,
        # the axis in fig.get_axis() are sorted differently than pandas
        # expected them, so make sure that only the right ones are removed
        import matplotlib.pyplot as plt

        gs, axes = _generate_4_axes_via_gridspec()

        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )

        def _check(axes):
            for ax in axes:
                assert len(ax.lines) == 1
                self._check_visible(ax.get_xticklabels(), visible=True)
                self._check_visible(ax.get_xticklabels(minor=True), visible=True)
            for ax in [axes[0], axes[1]]:
                self._check_visible(ax.get_yticklabels(), visible=True)
            for ax in [axes[2], axes[3]]:
                self._check_visible(ax.get_yticklabels(), visible=False)

        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax, sharey=True)
        gs.tight_layout(plt.gcf())
        _check(axes)
        tm.close()

        gs, axes = _generate_4_axes_via_gridspec()
        with tm.assert_produces_warning(UserWarning):
            axes = df.plot(subplots=True, ax=axes, sharey=True)

        gs.tight_layout(plt.gcf())
        _check(axes)
        tm.close()

        gs, axes = _generate_4_axes_via_gridspec()
        # without sharex, no labels should be touched!
        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax)

        gs.tight_layout(plt.gcf())
        for ax in axes:
            assert len(ax.lines) == 1
            self._check_visible(ax.get_yticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(minor=True), visible=True)

    @td.skip_if_no_scipy
    def test_memory_leak(self):
        """Check that every plot type gets properly collected."""
        import gc
        import weakref

        results = {}
        for kind in plotting.PlotAccessor._all_kinds:

            args = {}
            if kind in ["hexbin", "scatter", "pie"]:
                df = self.hexbin_df
                args = {"x": "A", "y": "B"}
            elif kind == "area":
                df = self.tdf.abs()
            else:
                df = self.tdf

            # Use a weakref so we can see if the object gets collected without
            # also preventing it from being collected
            results[kind] = weakref.proxy(df.plot(kind=kind, **args))

        # have matplotlib delete all the figures
        tm.close()
        # force a garbage collection
        gc.collect()
        msg = "weakly-referenced object no longer exists"
        for key in results:
            # check that every plot was collected
            with pytest.raises(ReferenceError, match=msg):
                # need to actually access something to get an error
                results[key].lines

    def test_df_gridspec_patterns(self):
        # GH 10819
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        ts = Series(np.random.randn(10), index=date_range("1/1/2000", periods=10))

        df = DataFrame(np.random.randn(10, 2), index=ts.index, columns=list("AB"))

        def _get_vertical_grid():
            gs = gridspec.GridSpec(3, 1)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:2, :])
            ax2 = fig.add_subplot(gs[2, :])
            return ax1, ax2

        def _get_horizontal_grid():
            gs = gridspec.GridSpec(1, 3)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:, :2])
            ax2 = fig.add_subplot(gs[:, 2])
            return ax1, ax2

        for ax1, ax2 in [_get_vertical_grid(), _get_horizontal_grid()]:
            ax1 = ts.plot(ax=ax1)
            assert len(ax1.lines) == 1
            ax2 = df.plot(ax=ax2)
            assert len(ax2.lines) == 2
            for ax in [ax1, ax2]:
                self._check_visible(ax.get_yticklabels(), visible=True)
                self._check_visible(ax.get_xticklabels(), visible=True)
                self._check_visible(ax.get_xticklabels(minor=True), visible=True)
            tm.close()

        # subplots=True
        for ax1, ax2 in [_get_vertical_grid(), _get_horizontal_grid()]:
            axes = df.plot(subplots=True, ax=[ax1, ax2])
            assert len(ax1.lines) == 1
            assert len(ax2.lines) == 1
            for ax in axes:
                self._check_visible(ax.get_yticklabels(), visible=True)
                self._check_visible(ax.get_xticklabels(), visible=True)
                self._check_visible(ax.get_xticklabels(minor=True), visible=True)
            tm.close()

        # vertical / subplots / sharex=True / sharey=True
        ax1, ax2 = _get_vertical_grid()
        with tm.assert_produces_warning(UserWarning):
            axes = df.plot(subplots=True, ax=[ax1, ax2], sharex=True, sharey=True)
        assert len(axes[0].lines) == 1
        assert len(axes[1].lines) == 1
        for ax in [ax1, ax2]:
            # yaxis are visible because there is only one column
            self._check_visible(ax.get_yticklabels(), visible=True)
        # xaxis of axes0 (top) are hidden
        self._check_visible(axes[0].get_xticklabels(), visible=False)
        self._check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        self._check_visible(axes[1].get_xticklabels(), visible=True)
        self._check_visible(axes[1].get_xticklabels(minor=True), visible=True)
        tm.close()

        # horizontal / subplots / sharex=True / sharey=True
        ax1, ax2 = _get_horizontal_grid()
        with tm.assert_produces_warning(UserWarning):
            axes = df.plot(subplots=True, ax=[ax1, ax2], sharex=True, sharey=True)
        assert len(axes[0].lines) == 1
        assert len(axes[1].lines) == 1
        self._check_visible(axes[0].get_yticklabels(), visible=True)
        # yaxis of axes1 (right) are hidden
        self._check_visible(axes[1].get_yticklabels(), visible=False)
        for ax in [ax1, ax2]:
            # xaxis are visible because there is only one column
            self._check_visible(ax.get_xticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(minor=True), visible=True)
        tm.close()

        # boxed
        def _get_boxed_grid():
            gs = gridspec.GridSpec(3, 3)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:2, :2])
            ax2 = fig.add_subplot(gs[:2, 2])
            ax3 = fig.add_subplot(gs[2, :2])
            ax4 = fig.add_subplot(gs[2, 2])
            return ax1, ax2, ax3, ax4

        axes = _get_boxed_grid()
        df = DataFrame(np.random.randn(10, 4), index=ts.index, columns=list("ABCD"))
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            # axis are visible because these are not shared
            self._check_visible(ax.get_yticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(minor=True), visible=True)
        tm.close()

        # subplots / sharex=True / sharey=True
        axes = _get_boxed_grid()
        with tm.assert_produces_warning(UserWarning):
            axes = df.plot(subplots=True, ax=axes, sharex=True, sharey=True)
        for ax in axes:
            assert len(ax.lines) == 1
        for ax in [axes[0], axes[2]]:  # left column
            self._check_visible(ax.get_yticklabels(), visible=True)
        for ax in [axes[1], axes[3]]:  # right column
            self._check_visible(ax.get_yticklabels(), visible=False)
        for ax in [axes[0], axes[1]]:  # top row
            self._check_visible(ax.get_xticklabels(), visible=False)
            self._check_visible(ax.get_xticklabels(minor=True), visible=False)
        for ax in [axes[2], axes[3]]:  # bottom row
            self._check_visible(ax.get_xticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(minor=True), visible=True)
        tm.close()

    def test_df_grid_settings(self):
        # Make sure plot defaults to rcParams['axes.grid'] setting, GH 9792
        self._check_grid_settings(
            DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}),
            plotting.PlotAccessor._dataframe_kinds,
            kws={"x": "a", "y": "b"},
        )

    def test_plain_axes(self):

        # supplied ax itself is a SubplotAxes, but figure contains also
        # a plain Axes object (GH11556)
        fig, ax = self.plt.subplots()
        fig.add_axes([0.2, 0.2, 0.2, 0.2])
        Series(np.random.rand(10)).plot(ax=ax)

        # supplied ax itself is a plain Axes, but because the cmap keyword
        # a new ax is created for the colorbar -> also multiples axes (GH11520)
        df = DataFrame({"a": np.random.randn(8), "b": np.random.randn(8)})
        fig = self.plt.figure()
        ax = fig.add_axes((0, 0, 1, 1))
        df.plot(kind="scatter", ax=ax, x="a", y="b", c="a", cmap="hsv")

        # other examples
        fig, ax = self.plt.subplots()
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        Series(np.random.rand(10)).plot(ax=ax)
        Series(np.random.rand(10)).plot(ax=cax)

        fig, ax = self.plt.subplots()
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        iax = inset_axes(ax, width="30%", height=1.0, loc=3)
        Series(np.random.rand(10)).plot(ax=ax)
        Series(np.random.rand(10)).plot(ax=iax)

    @pytest.mark.parametrize("method", ["line", "barh", "bar"])
    def test_secondary_axis_font_size(self, method):
        # GH: 12565
        df = (
            DataFrame(np.random.randn(15, 2), columns=list("AB"))
            .assign(C=lambda df: df.B.cumsum())
            .assign(D=lambda df: df.C * 1.1)
        )

        fontsize = 20
        sy = ["C", "D"]

        kwargs = {"secondary_y": sy, "fontsize": fontsize, "mark_right": True}
        ax = getattr(df.plot, method)(**kwargs)
        self._check_ticks_props(axes=ax.right_ax, ylabelsize=fontsize)

    def test_x_string_values_ticks(self):
        # Test if string plot index have a fixed xtick position
        # GH: 7612, GH: 22334
        df = DataFrame(
            {
                "sales": [3, 2, 3],
                "visits": [20, 42, 28],
                "day": ["Monday", "Tuesday", "Wednesday"],
            }
        )
        ax = df.plot.area(x="day")
        ax.set_xlim(-1, 3)
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        labels_position = dict(zip(xticklabels, ax.get_xticks()))
        # Testing if the label stayed at the right position
        assert labels_position["Monday"] == 0.0
        assert labels_position["Tuesday"] == 1.0
        assert labels_position["Wednesday"] == 2.0

    def test_x_multiindex_values_ticks(self):
        # Test if multiindex plot index have a fixed xtick position
        # GH: 15912
        index = MultiIndex.from_product([[2012, 2013], [1, 2]])
        df = DataFrame(np.random.randn(4, 2), columns=["A", "B"], index=index)
        ax = df.plot()
        ax.set_xlim(-1, 4)
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        labels_position = dict(zip(xticklabels, ax.get_xticks()))
        # Testing if the label stayed at the right position
        assert labels_position["(2012, 1)"] == 0.0
        assert labels_position["(2012, 2)"] == 1.0
        assert labels_position["(2013, 1)"] == 2.0
        assert labels_position["(2013, 2)"] == 3.0

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_xlim_plot_line(self, kind):
        # test if xlim is set correctly in plot.line and plot.area
        # GH 27686
        df = DataFrame([2, 4], index=[1, 2])
        ax = df.plot(kind=kind)
        xlims = ax.get_xlim()
        assert xlims[0] < 1
        assert xlims[1] > 2

    def test_xlim_plot_line_correctly_in_mixed_plot_type(self):
        # test if xlim is set correctly when ax contains multiple different kinds
        # of plots, GH 27686
        fig, ax = self.plt.subplots()

        indexes = ["k1", "k2", "k3", "k4"]
        df = DataFrame(
            {
                "s1": [1000, 2000, 1500, 2000],
                "s2": [900, 1400, 2000, 3000],
                "s3": [1500, 1500, 1600, 1200],
                "secondary_y": [1, 3, 4, 3],
            },
            index=indexes,
        )
        df[["s1", "s2", "s3"]].plot.bar(ax=ax, stacked=False)
        df[["secondary_y"]].plot(ax=ax, secondary_y=True)

        xlims = ax.get_xlim()
        assert xlims[0] < 0
        assert xlims[1] > 3

        # make sure axis labels are plotted correctly as well
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xticklabels == indexes

    def test_plot_no_rows(self):
        # GH 27758
        df = DataFrame(columns=["foo"], dtype=int)
        assert df.empty
        ax = df.plot()
        assert len(ax.get_lines()) == 1
        line = ax.get_lines()[0]
        assert len(line.get_xdata()) == 0
        assert len(line.get_ydata()) == 0

    def test_plot_no_numeric_data(self):
        df = DataFrame(["a", "b", "c"])
        with pytest.raises(TypeError, match="no numeric data to plot"):
            df.plot()

    @pytest.mark.parametrize(
        "index_name, old_label, new_label",
        [
            (None, "", "new"),
            ("old", "old", "new"),
            (None, "", ""),
            (None, "", 1),
            (None, "", [1, 2]),
        ],
    )
    @pytest.mark.parametrize("kind", ["line", "area", "bar"])
    def test_xlabel_ylabel_dataframe_single_plot(
        self, kind, index_name, old_label, new_label
    ):
        # GH 9093
        df = DataFrame([[1, 2], [2, 5]], columns=["Type A", "Type B"])
        df.index.name = index_name

        # default is the ylabel is not shown and xlabel is index name
        ax = df.plot(kind=kind)
        assert ax.get_xlabel() == old_label
        assert ax.get_ylabel() == ""

        # old xlabel will be overridden and assigned ylabel will be used as ylabel
        ax = df.plot(kind=kind, ylabel=new_label, xlabel=new_label)
        assert ax.get_ylabel() == str(new_label)
        assert ax.get_xlabel() == str(new_label)

    @pytest.mark.parametrize(
        "xlabel, ylabel",
        [
            (None, None),
            ("X Label", None),
            (None, "Y Label"),
            ("X Label", "Y Label"),
        ],
    )
    @pytest.mark.parametrize("kind", ["scatter", "hexbin"])
    def test_xlabel_ylabel_dataframe_plane_plot(self, kind, xlabel, ylabel):
        # GH 37001
        xcol = "Type A"
        ycol = "Type B"
        df = DataFrame([[1, 2], [2, 5]], columns=[xcol, ycol])

        # default is the labels are column names
        ax = df.plot(kind=kind, x=xcol, y=ycol, xlabel=xlabel, ylabel=ylabel)
        assert ax.get_xlabel() == (xcol if xlabel is None else xlabel)
        assert ax.get_ylabel() == (ycol if ylabel is None else ylabel)


def _generate_4_axes_via_gridspec():
    import matplotlib as mpl
    import matplotlib.gridspec
    import matplotlib.pyplot as plt

    gs = mpl.gridspec.GridSpec(2, 2)
    ax_tl = plt.subplot(gs[0, 0])
    ax_ll = plt.subplot(gs[1, 0])
    ax_tr = plt.subplot(gs[0, 1])
    ax_lr = plt.subplot(gs[1, 1])

    return gs, [ax_tl, ax_ll, ax_tr, ax_lr]
