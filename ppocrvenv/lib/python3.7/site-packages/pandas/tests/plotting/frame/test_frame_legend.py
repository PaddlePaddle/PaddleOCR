import numpy as np
import pytest

from pandas import (
    DataFrame,
    date_range,
)
from pandas.tests.plotting.common import TestPlotBase

pytestmark = pytest.mark.slow


class TestFrameLegend(TestPlotBase):
    @pytest.mark.xfail(
        reason=(
            "Open bug in matplotlib "
            "https://github.com/matplotlib/matplotlib/issues/11357"
        )
    )
    def test_mixed_yerr(self):
        # https://github.com/pandas-dev/pandas/issues/39522
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        df = DataFrame([{"x": 1, "a": 1, "b": 1}, {"x": 2, "a": 2, "b": 3}])

        ax = df.plot("x", "a", c="orange", yerr=0.1, label="orange")
        df.plot("x", "b", c="blue", yerr=None, ax=ax, label="blue")

        legend = ax.get_legend()
        result_handles = legend.legendHandles

        assert isinstance(result_handles[0], LineCollection)
        assert isinstance(result_handles[1], Line2D)

    def test_legend_false(self):
        # https://github.com/pandas-dev/pandas/issues/40044
        df = DataFrame({"a": [1, 1], "b": [2, 3]})
        df2 = DataFrame({"d": [2.5, 2.5]})

        ax = df.plot(legend=True, color={"a": "blue", "b": "green"}, secondary_y="b")
        df2.plot(legend=True, color={"d": "red"}, ax=ax)
        legend = ax.get_legend()
        result = [handle.get_color() for handle in legend.legendHandles]
        expected = ["blue", "green", "red"]
        assert result == expected

    def test_df_legend_labels(self):
        kinds = ["line", "bar", "barh", "kde", "area", "hist"]
        df = DataFrame(np.random.rand(3, 3), columns=["a", "b", "c"])
        df2 = DataFrame(np.random.rand(3, 3), columns=["d", "e", "f"])
        df3 = DataFrame(np.random.rand(3, 3), columns=["g", "h", "i"])
        df4 = DataFrame(np.random.rand(3, 3), columns=["j", "k", "l"])

        for kind in kinds:

            ax = df.plot(kind=kind, legend=True)
            self._check_legend_labels(ax, labels=df.columns)

            ax = df2.plot(kind=kind, legend=False, ax=ax)
            self._check_legend_labels(ax, labels=df.columns)

            ax = df3.plot(kind=kind, legend=True, ax=ax)
            self._check_legend_labels(ax, labels=df.columns.union(df3.columns))

            ax = df4.plot(kind=kind, legend="reverse", ax=ax)
            expected = list(df.columns.union(df3.columns)) + list(reversed(df4.columns))
            self._check_legend_labels(ax, labels=expected)

        # Secondary Y
        ax = df.plot(legend=True, secondary_y="b")
        self._check_legend_labels(ax, labels=["a", "b (right)", "c"])
        ax = df2.plot(legend=False, ax=ax)
        self._check_legend_labels(ax, labels=["a", "b (right)", "c"])
        ax = df3.plot(kind="bar", legend=True, secondary_y="h", ax=ax)
        self._check_legend_labels(
            ax, labels=["a", "b (right)", "c", "g", "h (right)", "i"]
        )

        # Time Series
        ind = date_range("1/1/2014", periods=3)
        df = DataFrame(np.random.randn(3, 3), columns=["a", "b", "c"], index=ind)
        df2 = DataFrame(np.random.randn(3, 3), columns=["d", "e", "f"], index=ind)
        df3 = DataFrame(np.random.randn(3, 3), columns=["g", "h", "i"], index=ind)
        ax = df.plot(legend=True, secondary_y="b")
        self._check_legend_labels(ax, labels=["a", "b (right)", "c"])
        ax = df2.plot(legend=False, ax=ax)
        self._check_legend_labels(ax, labels=["a", "b (right)", "c"])
        ax = df3.plot(legend=True, ax=ax)
        self._check_legend_labels(ax, labels=["a", "b (right)", "c", "g", "h", "i"])

        # scatter
        ax = df.plot.scatter(x="a", y="b", label="data1")
        self._check_legend_labels(ax, labels=["data1"])
        ax = df2.plot.scatter(x="d", y="e", legend=False, label="data2", ax=ax)
        self._check_legend_labels(ax, labels=["data1"])
        ax = df3.plot.scatter(x="g", y="h", label="data3", ax=ax)
        self._check_legend_labels(ax, labels=["data1", "data3"])

        # ensure label args pass through and
        # index name does not mutate
        # column names don't mutate
        df5 = df.set_index("a")
        ax = df5.plot(y="b")
        self._check_legend_labels(ax, labels=["b"])
        ax = df5.plot(y="b", label="LABEL_b")
        self._check_legend_labels(ax, labels=["LABEL_b"])
        self._check_text_labels(ax.xaxis.get_label(), "a")
        ax = df5.plot(y="c", label="LABEL_c", ax=ax)
        self._check_legend_labels(ax, labels=["LABEL_b", "LABEL_c"])
        assert df5.columns.tolist() == ["b", "c"]

    def test_missing_marker_multi_plots_on_same_ax(self):
        # GH 18222
        df = DataFrame(data=[[1, 1, 1, 1], [2, 2, 4, 8]], columns=["x", "r", "g", "b"])
        fig, ax = self.plt.subplots(nrows=1, ncols=3)
        # Left plot
        df.plot(x="x", y="r", linewidth=0, marker="o", color="r", ax=ax[0])
        df.plot(x="x", y="g", linewidth=1, marker="x", color="g", ax=ax[0])
        df.plot(x="x", y="b", linewidth=1, marker="o", color="b", ax=ax[0])
        self._check_legend_labels(ax[0], labels=["r", "g", "b"])
        self._check_legend_marker(ax[0], expected_markers=["o", "x", "o"])
        # Center plot
        df.plot(x="x", y="b", linewidth=1, marker="o", color="b", ax=ax[1])
        df.plot(x="x", y="r", linewidth=0, marker="o", color="r", ax=ax[1])
        df.plot(x="x", y="g", linewidth=1, marker="x", color="g", ax=ax[1])
        self._check_legend_labels(ax[1], labels=["b", "r", "g"])
        self._check_legend_marker(ax[1], expected_markers=["o", "o", "x"])
        # Right plot
        df.plot(x="x", y="g", linewidth=1, marker="x", color="g", ax=ax[2])
        df.plot(x="x", y="b", linewidth=1, marker="o", color="b", ax=ax[2])
        df.plot(x="x", y="r", linewidth=0, marker="o", color="r", ax=ax[2])
        self._check_legend_labels(ax[2], labels=["g", "b", "r"])
        self._check_legend_marker(ax[2], expected_markers=["x", "o", "o"])

    def test_legend_name(self):
        multi = DataFrame(
            np.random.randn(4, 4),
            columns=[np.array(["a", "a", "b", "b"]), np.array(["x", "y", "x", "y"])],
        )
        multi.columns.names = ["group", "individual"]

        ax = multi.plot()
        leg_title = ax.legend_.get_title()
        self._check_text_labels(leg_title, "group,individual")

        df = DataFrame(np.random.randn(5, 5))
        ax = df.plot(legend=True, ax=ax)
        leg_title = ax.legend_.get_title()
        self._check_text_labels(leg_title, "group,individual")

        df.columns.name = "new"
        ax = df.plot(legend=False, ax=ax)
        leg_title = ax.legend_.get_title()
        self._check_text_labels(leg_title, "group,individual")

        ax = df.plot(legend=True, ax=ax)
        leg_title = ax.legend_.get_title()
        self._check_text_labels(leg_title, "new")

    def test_no_legend(self):
        kinds = ["line", "bar", "barh", "kde", "area", "hist"]
        df = DataFrame(np.random.rand(3, 3), columns=["a", "b", "c"])

        for kind in kinds:
            ax = df.plot(kind=kind, legend=False)
            self._check_legend_labels(ax, visible=False)

    def test_missing_markers_legend(self):
        # 14958
        df = DataFrame(np.random.randn(8, 3), columns=["A", "B", "C"])
        ax = df.plot(y=["A"], marker="x", linestyle="solid")
        df.plot(y=["B"], marker="o", linestyle="dotted", ax=ax)
        df.plot(y=["C"], marker="<", linestyle="dotted", ax=ax)

        self._check_legend_labels(ax, labels=["A", "B", "C"])
        self._check_legend_marker(ax, expected_markers=["x", "o", "<"])

    def test_missing_markers_legend_using_style(self):
        # 14563
        df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6],
                "B": [2, 4, 1, 3, 2, 4],
                "C": [3, 3, 2, 6, 4, 2],
                "X": [1, 2, 3, 4, 5, 6],
            }
        )

        fig, ax = self.plt.subplots()
        for kind in "ABC":
            df.plot("X", kind, label=kind, ax=ax, style=".")

        self._check_legend_labels(ax, labels=["A", "B", "C"])
        self._check_legend_marker(ax, expected_markers=[".", ".", "."])
