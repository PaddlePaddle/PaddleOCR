""" Test cases for DataFrame.plot """

import string
import warnings

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
from pandas.tests.plotting.common import TestPlotBase

from pandas.io.formats.printing import pprint_thing

pytestmark = pytest.mark.slow


@td.skip_if_no_mpl
class TestDataFramePlotsSubplots(TestPlotBase):
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

    def test_subplots(self):
        df = DataFrame(np.random.rand(10, 3), index=list(string.ascii_letters[:10]))

        for kind in ["bar", "barh", "line", "area"]:
            axes = df.plot(kind=kind, subplots=True, sharex=True, legend=True)
            self._check_axes_shape(axes, axes_num=3, layout=(3, 1))
            assert axes.shape == (3,)

            for ax, column in zip(axes, df.columns):
                self._check_legend_labels(ax, labels=[pprint_thing(column)])

            for ax in axes[:-2]:
                self._check_visible(ax.xaxis)  # xaxis must be visible for grid
                self._check_visible(ax.get_xticklabels(), visible=False)
                if not (kind == "bar" and self.mpl_ge_3_1_0):
                    # change https://github.com/pandas-dev/pandas/issues/26714
                    self._check_visible(ax.get_xticklabels(minor=True), visible=False)
                self._check_visible(ax.xaxis.get_label(), visible=False)
                self._check_visible(ax.get_yticklabels())

            self._check_visible(axes[-1].xaxis)
            self._check_visible(axes[-1].get_xticklabels())
            self._check_visible(axes[-1].get_xticklabels(minor=True))
            self._check_visible(axes[-1].xaxis.get_label())
            self._check_visible(axes[-1].get_yticklabels())

            axes = df.plot(kind=kind, subplots=True, sharex=False)
            for ax in axes:
                self._check_visible(ax.xaxis)
                self._check_visible(ax.get_xticklabels())
                self._check_visible(ax.get_xticklabels(minor=True))
                self._check_visible(ax.xaxis.get_label())
                self._check_visible(ax.get_yticklabels())

            axes = df.plot(kind=kind, subplots=True, legend=False)
            for ax in axes:
                assert ax.get_legend() is None

    def test_subplots_timeseries(self):
        idx = date_range(start="2014-07-01", freq="M", periods=10)
        df = DataFrame(np.random.rand(10, 3), index=idx)

        for kind in ["line", "area"]:
            axes = df.plot(kind=kind, subplots=True, sharex=True)
            self._check_axes_shape(axes, axes_num=3, layout=(3, 1))

            for ax in axes[:-2]:
                # GH 7801
                self._check_visible(ax.xaxis)  # xaxis must be visible for grid
                self._check_visible(ax.get_xticklabels(), visible=False)
                self._check_visible(ax.get_xticklabels(minor=True), visible=False)
                self._check_visible(ax.xaxis.get_label(), visible=False)
                self._check_visible(ax.get_yticklabels())

            self._check_visible(axes[-1].xaxis)
            self._check_visible(axes[-1].get_xticklabels())
            self._check_visible(axes[-1].get_xticklabels(minor=True))
            self._check_visible(axes[-1].xaxis.get_label())
            self._check_visible(axes[-1].get_yticklabels())
            self._check_ticks_props(axes, xrot=0)

            axes = df.plot(kind=kind, subplots=True, sharex=False, rot=45, fontsize=7)
            for ax in axes:
                self._check_visible(ax.xaxis)
                self._check_visible(ax.get_xticklabels())
                self._check_visible(ax.get_xticklabels(minor=True))
                self._check_visible(ax.xaxis.get_label())
                self._check_visible(ax.get_yticklabels())
                self._check_ticks_props(ax, xlabelsize=7, xrot=45, ylabelsize=7)

    def test_subplots_timeseries_y_axis(self):
        # GH16953
        data = {
            "numeric": np.array([1, 2, 5]),
            "timedelta": [
                pd.Timedelta(-10, unit="s"),
                pd.Timedelta(10, unit="m"),
                pd.Timedelta(10, unit="h"),
            ],
            "datetime_no_tz": [
                pd.to_datetime("2017-08-01 00:00:00"),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
            "datetime_all_tz": [
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00", utc=True),
                pd.to_datetime("2017-08-02 00:00:00", utc=True),
            ],
            "text": ["This", "should", "fail"],
        }
        testdata = DataFrame(data)

        y_cols = ["numeric", "timedelta", "datetime_no_tz", "datetime_all_tz"]
        for col in y_cols:
            ax = testdata.plot(y=col)
            result = ax.get_lines()[0].get_data()[1]
            expected = testdata[col].values
            assert (result == expected).all()

        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            testdata.plot(y="text")

    @pytest.mark.xfail(reason="not support for period, categorical, datetime_mixed_tz")
    def test_subplots_timeseries_y_axis_not_supported(self):
        """
        This test will fail for:
            period:
                since period isn't yet implemented in ``select_dtypes``
                and because it will need a custom value converter +
                tick formatter (as was done for x-axis plots)

            categorical:
                 because it will need a custom value converter +
                 tick formatter (also doesn't work for x-axis, as of now)

            datetime_mixed_tz:
                because of the way how pandas handles ``Series`` of
                ``datetime`` objects with different timezone,
                generally converting ``datetime`` objects in a tz-aware
                form could help with this problem
        """
        data = {
            "numeric": np.array([1, 2, 5]),
            "period": [
                pd.Period("2017-08-01 00:00:00", freq="H"),
                pd.Period("2017-08-01 02:00", freq="H"),
                pd.Period("2017-08-02 00:00:00", freq="H"),
            ],
            "categorical": pd.Categorical(
                ["c", "b", "a"], categories=["a", "b", "c"], ordered=False
            ),
            "datetime_mixed_tz": [
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
        }
        testdata = DataFrame(data)
        ax_period = testdata.plot(x="numeric", y="period")
        assert (
            ax_period.get_lines()[0].get_data()[1] == testdata["period"].values
        ).all()
        ax_categorical = testdata.plot(x="numeric", y="categorical")
        assert (
            ax_categorical.get_lines()[0].get_data()[1]
            == testdata["categorical"].values
        ).all()
        ax_datetime_mixed_tz = testdata.plot(x="numeric", y="datetime_mixed_tz")
        assert (
            ax_datetime_mixed_tz.get_lines()[0].get_data()[1]
            == testdata["datetime_mixed_tz"].values
        ).all()

    def test_subplots_layout_multi_column(self):
        # GH 6667
        df = DataFrame(np.random.rand(10, 3), index=list(string.ascii_letters[:10]))

        axes = df.plot(subplots=True, layout=(2, 2))
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))
        assert axes.shape == (2, 2)

        axes = df.plot(subplots=True, layout=(-1, 2))
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))
        assert axes.shape == (2, 2)

        axes = df.plot(subplots=True, layout=(2, -1))
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))
        assert axes.shape == (2, 2)

        axes = df.plot(subplots=True, layout=(1, 4))
        self._check_axes_shape(axes, axes_num=3, layout=(1, 4))
        assert axes.shape == (1, 4)

        axes = df.plot(subplots=True, layout=(-1, 4))
        self._check_axes_shape(axes, axes_num=3, layout=(1, 4))
        assert axes.shape == (1, 4)

        axes = df.plot(subplots=True, layout=(4, -1))
        self._check_axes_shape(axes, axes_num=3, layout=(4, 1))
        assert axes.shape == (4, 1)

        msg = "Layout of 1x1 must be larger than required size 3"

        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(1, 1))

        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(-1, -1))

    @pytest.mark.parametrize(
        "kwargs, expected_axes_num, expected_layout, expected_shape",
        [
            ({}, 1, (1, 1), (1,)),
            ({"layout": (3, 3)}, 1, (3, 3), (3, 3)),
        ],
    )
    def test_subplots_layout_single_column(
        self, kwargs, expected_axes_num, expected_layout, expected_shape
    ):

        # GH 6667
        df = DataFrame(np.random.rand(10, 1), index=list(string.ascii_letters[:10]))
        axes = df.plot(subplots=True, **kwargs)
        self._check_axes_shape(
            axes,
            axes_num=expected_axes_num,
            layout=expected_layout,
        )
        assert axes.shape == expected_shape

    def test_subplots_warnings(self):
        # GH 9464
        with tm.assert_produces_warning(None):
            df = DataFrame(np.random.randn(100, 4))
            df.plot(subplots=True, layout=(3, 2))

            df = DataFrame(
                np.random.randn(100, 4), index=date_range("1/1/2000", periods=100)
            )
            df.plot(subplots=True, layout=(3, 2))

    def test_subplots_multiple_axes(self):
        # GH 5353, 6970, GH 7069
        fig, axes = self.plt.subplots(2, 3)
        df = DataFrame(np.random.rand(10, 3), index=list(string.ascii_letters[:10]))

        returned = df.plot(subplots=True, ax=axes[0], sharex=False, sharey=False)
        self._check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        # draw on second row
        returned = df.plot(subplots=True, ax=axes[1], sharex=False, sharey=False)
        self._check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        self._check_axes_shape(axes, axes_num=6, layout=(2, 3))
        tm.close()

        msg = "The number of passed axes must be 3, the same as the output plot"

        with pytest.raises(ValueError, match=msg):
            fig, axes = self.plt.subplots(2, 3)
            # pass different number of axes from required
            df.plot(subplots=True, ax=axes)

        # pass 2-dim axes and invalid layout
        # invalid lauout should not affect to input and return value
        # (show warning is tested in
        # TestDataFrameGroupByPlots.test_grouped_box_multiple_axes
        fig, axes = self.plt.subplots(2, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = DataFrame(np.random.rand(10, 4), index=list(string.ascii_letters[:10]))

            returned = df.plot(
                subplots=True, ax=axes, layout=(2, 1), sharex=False, sharey=False
            )
            self._check_axes_shape(returned, axes_num=4, layout=(2, 2))
            assert returned.shape == (4,)

            returned = df.plot(
                subplots=True, ax=axes, layout=(2, -1), sharex=False, sharey=False
            )
            self._check_axes_shape(returned, axes_num=4, layout=(2, 2))
            assert returned.shape == (4,)

            returned = df.plot(
                subplots=True, ax=axes, layout=(-1, 2), sharex=False, sharey=False
            )
        self._check_axes_shape(returned, axes_num=4, layout=(2, 2))
        assert returned.shape == (4,)

        # single column
        fig, axes = self.plt.subplots(1, 1)
        df = DataFrame(np.random.rand(10, 1), index=list(string.ascii_letters[:10]))

        axes = df.plot(subplots=True, ax=[axes], sharex=False, sharey=False)
        self._check_axes_shape(axes, axes_num=1, layout=(1, 1))
        assert axes.shape == (1,)

    def test_subplots_ts_share_axes(self):
        # GH 3964
        fig, axes = self.plt.subplots(3, 3, sharex=True, sharey=True)
        self.plt.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
        df = DataFrame(
            np.random.randn(10, 9),
            index=date_range(start="2014-07-01", freq="M", periods=10),
        )
        for i, ax in enumerate(axes.ravel()):
            df[i].plot(ax=ax, fontsize=5)

        # Rows other than bottom should not be visible
        for ax in axes[0:-1].ravel():
            self._check_visible(ax.get_xticklabels(), visible=False)

        # Bottom row should be visible
        for ax in axes[-1].ravel():
            self._check_visible(ax.get_xticklabels(), visible=True)

        # First column should be visible
        for ax in axes[[0, 1, 2], [0]].ravel():
            self._check_visible(ax.get_yticklabels(), visible=True)

        # Other columns should not be visible
        for ax in axes[[0, 1, 2], [1]].ravel():
            self._check_visible(ax.get_yticklabels(), visible=False)
        for ax in axes[[0, 1, 2], [2]].ravel():
            self._check_visible(ax.get_yticklabels(), visible=False)

    def test_subplots_sharex_axes_existing_axes(self):
        # GH 9158
        d = {"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 1.0], "C": [5, 1, 3, 4]}
        df = DataFrame(d, index=date_range("2014 10 11", "2014 10 14"))

        axes = df[["A", "B"]].plot(subplots=True)
        df["C"].plot(ax=axes[0], secondary_y=True)

        self._check_visible(axes[0].get_xticklabels(), visible=False)
        self._check_visible(axes[1].get_xticklabels(), visible=True)
        for ax in axes.ravel():
            self._check_visible(ax.get_yticklabels(), visible=True)

    def test_subplots_dup_columns(self):
        # GH 10962
        df = DataFrame(np.random.rand(5, 5), columns=list("aaaaa"))
        axes = df.plot(subplots=True)
        for ax in axes:
            self._check_legend_labels(ax, labels=["a"])
            assert len(ax.lines) == 1
        tm.close()

        axes = df.plot(subplots=True, secondary_y="a")
        for ax in axes:
            # (right) is only attached when subplots=False
            self._check_legend_labels(ax, labels=["a"])
            assert len(ax.lines) == 1
        tm.close()

        ax = df.plot(secondary_y="a")
        self._check_legend_labels(ax, labels=["a (right)"] * 5)
        assert len(ax.lines) == 0
        assert len(ax.right_ax.lines) == 5

    def test_bar_log_no_subplots(self):
        # GH3254, GH3298 matplotlib/matplotlib#1882, #1892
        # regressions in 1.2.1
        expected = np.array([0.1, 1.0, 10.0, 100])

        # no subplots
        df = DataFrame({"A": [3] * 5, "B": list(range(1, 6))}, index=range(5))
        ax = df.plot.bar(grid=True, log=True)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)

    def test_bar_log_subplots(self):
        expected = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 1e4])

        ax = DataFrame([Series([200, 300]), Series([300, 500])]).plot.bar(
            log=True, subplots=True
        )

        tm.assert_numpy_array_equal(ax[0].yaxis.get_ticklocs(), expected)
        tm.assert_numpy_array_equal(ax[1].yaxis.get_ticklocs(), expected)

    def test_boxplot_subplots_return_type(self):
        df = self.hist_df

        # normal style: return_type=None
        result = df.plot.box(subplots=True)
        assert isinstance(result, Series)
        self._check_box_return_type(
            result, None, expected_keys=["height", "weight", "category"]
        )

        for t in ["dict", "axes", "both"]:
            returned = df.plot.box(return_type=t, subplots=True)
            self._check_box_return_type(
                returned,
                t,
                expected_keys=["height", "weight", "category"],
                check_ax_title=False,
            )

    def test_df_subplots_patterns_minorticks(self):
        # GH 10657
        import matplotlib.pyplot as plt

        df = DataFrame(
            np.random.randn(10, 2),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )

        # shared subplots
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            self._check_visible(ax.get_yticklabels(), visible=True)
        # xaxis of 1st ax must be hidden
        self._check_visible(axes[0].get_xticklabels(), visible=False)
        self._check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        self._check_visible(axes[1].get_xticklabels(), visible=True)
        self._check_visible(axes[1].get_xticklabels(minor=True), visible=True)
        tm.close()

        fig, axes = plt.subplots(2, 1)
        with tm.assert_produces_warning(UserWarning):
            axes = df.plot(subplots=True, ax=axes, sharex=True)
        for ax in axes:
            assert len(ax.lines) == 1
            self._check_visible(ax.get_yticklabels(), visible=True)
        # xaxis of 1st ax must be hidden
        self._check_visible(axes[0].get_xticklabels(), visible=False)
        self._check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        self._check_visible(axes[1].get_xticklabels(), visible=True)
        self._check_visible(axes[1].get_xticklabels(minor=True), visible=True)
        tm.close()

        # not shared
        fig, axes = plt.subplots(2, 1)
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            self._check_visible(ax.get_yticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(), visible=True)
            self._check_visible(ax.get_xticklabels(minor=True), visible=True)
        tm.close()

    def test_subplots_sharex_false(self):
        # test when sharex is set to False, two plots should have different
        # labels, GH 25160
        df = DataFrame(np.random.rand(10, 2))
        df.iloc[5:, 1] = np.nan
        df.iloc[:5, 0] = np.nan

        figs, axs = self.plt.subplots(2, 1)
        df.plot.line(ax=axs, subplots=True, sharex=False)

        expected_ax1 = np.arange(4.5, 10, 0.5)
        expected_ax2 = np.arange(-0.5, 5, 0.5)

        tm.assert_numpy_array_equal(axs[0].get_xticks(), expected_ax1)
        tm.assert_numpy_array_equal(axs[1].get_xticks(), expected_ax2)

    def test_subplots_constrained_layout(self):
        # GH 25261
        idx = date_range(start="now", periods=10)
        df = DataFrame(np.random.rand(10, 3), index=idx)
        kwargs = {}
        if hasattr(self.plt.Figure, "get_constrained_layout"):
            kwargs["constrained_layout"] = True
        fig, axes = self.plt.subplots(2, **kwargs)
        with tm.assert_produces_warning(None):
            df.plot(ax=axes[0])
            with tm.ensure_clean(return_filelike=True) as path:
                self.plt.savefig(path)

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
    def test_xlabel_ylabel_dataframe_subplots(
        self, kind, index_name, old_label, new_label
    ):
        # GH 9093
        df = DataFrame([[1, 2], [2, 5]], columns=["Type A", "Type B"])
        df.index.name = index_name

        # default is the ylabel is not shown and xlabel is index name
        axes = df.plot(kind=kind, subplots=True)
        assert all(ax.get_ylabel() == "" for ax in axes)
        assert all(ax.get_xlabel() == old_label for ax in axes)

        # old xlabel will be overridden and assigned ylabel will be used as ylabel
        axes = df.plot(kind=kind, ylabel=new_label, xlabel=new_label, subplots=True)
        assert all(ax.get_ylabel() == str(new_label) for ax in axes)
        assert all(ax.get_xlabel() == str(new_label) for ax in axes)

    @pytest.mark.parametrize(
        "kwargs",
        [
            # stacked center
            {"kind": "bar", "stacked": True},
            {"kind": "bar", "stacked": True, "width": 0.9},
            {"kind": "barh", "stacked": True},
            {"kind": "barh", "stacked": True, "width": 0.9},
            # center
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": False, "width": 0.9},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": False, "width": 0.9},
            # subplots center
            {"kind": "bar", "subplots": True},
            {"kind": "bar", "subplots": True, "width": 0.9},
            {"kind": "barh", "subplots": True},
            {"kind": "barh", "subplots": True, "width": 0.9},
            # align edge
            {"kind": "bar", "stacked": True, "align": "edge"},
            {"kind": "bar", "stacked": True, "width": 0.9, "align": "edge"},
            {"kind": "barh", "stacked": True, "align": "edge"},
            {"kind": "barh", "stacked": True, "width": 0.9, "align": "edge"},
            {"kind": "bar", "stacked": False, "align": "edge"},
            {"kind": "bar", "stacked": False, "width": 0.9, "align": "edge"},
            {"kind": "barh", "stacked": False, "align": "edge"},
            {"kind": "barh", "stacked": False, "width": 0.9, "align": "edge"},
            {"kind": "bar", "subplots": True, "align": "edge"},
            {"kind": "bar", "subplots": True, "width": 0.9, "align": "edge"},
            {"kind": "barh", "subplots": True, "align": "edge"},
            {"kind": "barh", "subplots": True, "width": 0.9, "align": "edge"},
        ],
    )
    def test_bar_align_multiple_columns(self, kwargs):
        # GH2157
        df = DataFrame({"A": [3] * 5, "B": list(range(5))}, index=range(5))
        self._check_bar_alignment(df, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": True},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": True},
            {"kind": "bar", "subplots": True},
            {"kind": "barh", "subplots": True},
        ],
    )
    def test_bar_align_single_column(self, kwargs):
        df = DataFrame(np.random.randn(5))
        self._check_bar_alignment(df, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": True},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": True},
            {"kind": "bar", "subplots": True},
            {"kind": "barh", "subplots": True},
        ],
    )
    def test_bar_barwidth_position(self, kwargs):
        df = DataFrame(np.random.randn(5, 5))
        self._check_bar_alignment(df, width=0.9, position=0.2, **kwargs)

    def test_bar_barwidth_position_int(self):
        # GH 12979
        df = DataFrame(np.random.randn(5, 5))

        for w in [1, 1.0]:
            ax = df.plot.bar(stacked=True, width=w)
            ticks = ax.xaxis.get_ticklocs()
            tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4]))
            assert ax.get_xlim() == (-0.75, 4.75)
            # check left-edge of bars
            assert ax.patches[0].get_x() == -0.5
            assert ax.patches[-1].get_x() == 3.5

        self._check_bar_alignment(df, kind="bar", stacked=True, width=1)
        self._check_bar_alignment(df, kind="barh", stacked=False, width=1)
        self._check_bar_alignment(df, kind="barh", stacked=True, width=1)
        self._check_bar_alignment(df, kind="bar", subplots=True, width=1)
        self._check_bar_alignment(df, kind="barh", subplots=True, width=1)

    def _check_bar_alignment(
        self,
        df,
        kind="bar",
        stacked=False,
        subplots=False,
        align="center",
        width=0.5,
        position=0.5,
    ):

        axes = df.plot(
            kind=kind,
            stacked=stacked,
            subplots=subplots,
            align=align,
            width=width,
            position=position,
            grid=True,
        )

        axes = self._flatten_visible(axes)

        for ax in axes:
            if kind == "bar":
                axis = ax.xaxis
                ax_min, ax_max = ax.get_xlim()
                min_edge = min(p.get_x() for p in ax.patches)
                max_edge = max(p.get_x() + p.get_width() for p in ax.patches)
            elif kind == "barh":
                axis = ax.yaxis
                ax_min, ax_max = ax.get_ylim()
                min_edge = min(p.get_y() for p in ax.patches)
                max_edge = max(p.get_y() + p.get_height() for p in ax.patches)
            else:
                raise ValueError

            # GH 7498
            # compare margins between lim and bar edges
            tm.assert_almost_equal(ax_min, min_edge - 0.25)
            tm.assert_almost_equal(ax_max, max_edge + 0.25)

            p = ax.patches[0]
            if kind == "bar" and (stacked is True or subplots is True):
                edge = p.get_x()
                center = edge + p.get_width() * position
            elif kind == "bar" and stacked is False:
                center = p.get_x() + p.get_width() * len(df.columns) * position
                edge = p.get_x()
            elif kind == "barh" and (stacked is True or subplots is True):
                center = p.get_y() + p.get_height() * position
                edge = p.get_y()
            elif kind == "barh" and stacked is False:
                center = p.get_y() + p.get_height() * len(df.columns) * position
                edge = p.get_y()
            else:
                raise ValueError

            # Check the ticks locates on integer
            assert (axis.get_ticklocs() == np.arange(len(df))).all()

            if align == "center":
                # Check whether the bar locates on center
                tm.assert_almost_equal(axis.get_ticklocs()[0], center)
            elif align == "edge":
                # Check whether the bar's edge starts from the tick
                tm.assert_almost_equal(axis.get_ticklocs()[0], edge)
            else:
                raise ValueError

        return axes
