""" Test cases for DataFrame.plot """
import re
import warnings

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
    TestPlotBase,
    _check_plot_works,
)

pytestmark = pytest.mark.slow


@td.skip_if_no_mpl
class TestDataFrameColor(TestPlotBase):
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

    def test_mpl2_color_cycle_str(self):
        # GH 15516
        df = DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", "MatplotlibDeprecationWarning")

            for color in colors:
                _check_plot_works(df.plot, color=color)

            # if warning is raised, check that it is the exact problematic one
            # GH 36972
            if w:
                match = "Support for uppercase single-letter colors is deprecated"
                warning_message = str(w[0].message)
                msg = "MatplotlibDeprecationWarning related to CN colors was raised"
                assert match not in warning_message, msg

    def test_color_single_series_list(self):
        # GH 3486
        df = DataFrame({"A": [1, 2, 3]})
        _check_plot_works(df.plot, color=["red"])

    @pytest.mark.parametrize("color", [(1, 0, 0), (1, 0, 0, 0.5)])
    def test_rgb_tuple_color(self, color):
        # GH 16695
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        _check_plot_works(df.plot, x="x", y="y", color=color)

    def test_color_empty_string(self):
        df = DataFrame(np.random.randn(10, 2))
        with pytest.raises(ValueError, match="Invalid color argument:"):
            df.plot(color="")

    def test_color_and_style_arguments(self):
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        # passing both 'color' and 'style' arguments should be allowed
        # if there is no color symbol in the style strings:
        ax = df.plot(color=["red", "black"], style=["-", "--"])
        # check that the linestyles are correctly set:
        linestyle = [line.get_linestyle() for line in ax.lines]
        assert linestyle == ["-", "--"]
        # check that the colors are correctly set:
        color = [line.get_color() for line in ax.lines]
        assert color == ["red", "black"]
        # passing both 'color' and 'style' arguments should not be allowed
        # if there is a color symbol in the style strings:
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' keyword "
            "argument. Please use one or the other or pass 'style' without a color "
            "symbol"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(color=["red", "black"], style=["k-", "r--"])

    @pytest.mark.parametrize(
        "color, expected",
        [
            ("green", ["green"] * 4),
            (["yellow", "red", "green", "blue"], ["yellow", "red", "green", "blue"]),
        ],
    )
    def test_color_and_marker(self, color, expected):
        # GH 21003
        df = DataFrame(np.random.random((7, 4)))
        ax = df.plot(color=color, style="d--")
        # check colors
        result = [i.get_color() for i in ax.lines]
        assert result == expected
        # check markers and linestyles
        assert all(i.get_linestyle() == "--" for i in ax.lines)
        assert all(i.get_marker() == "d" for i in ax.lines)

    def test_bar_colors(self):
        import matplotlib.pyplot as plt

        default_colors = self._unpack_cycler(plt.rcParams)

        df = DataFrame(np.random.randn(5, 5))
        ax = df.plot.bar()
        self._check_colors(ax.patches[::5], facecolors=default_colors[:5])
        tm.close()

        custom_colors = "rgcby"
        ax = df.plot.bar(color=custom_colors)
        self._check_colors(ax.patches[::5], facecolors=custom_colors)
        tm.close()

        from matplotlib import cm

        # Test str -> colormap functionality
        ax = df.plot.bar(colormap="jet")
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        self._check_colors(ax.patches[::5], facecolors=rgba_colors)
        tm.close()

        # Test colormap functionality
        ax = df.plot.bar(colormap=cm.jet)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        self._check_colors(ax.patches[::5], facecolors=rgba_colors)
        tm.close()

        ax = df.loc[:, [0]].plot.bar(color="DodgerBlue")
        self._check_colors([ax.patches[0]], facecolors=["DodgerBlue"])
        tm.close()

        ax = df.plot(kind="bar", color="green")
        self._check_colors(ax.patches[::5], facecolors=["green"] * 5)
        tm.close()

    def test_bar_user_colors(self):
        df = DataFrame(
            {"A": range(4), "B": range(1, 5), "color": ["red", "blue", "blue", "red"]}
        )
        # This should *only* work when `y` is specified, else
        # we use one color per column
        ax = df.plot.bar(y="A", color=df["color"])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        assert result == expected

    def test_if_scatterplot_colorbar_affects_xaxis_visibility(self):
        # addressing issue #10611, to ensure colobar does not
        # interfere with x-axis label and ticklabels with
        # ipython inline backend.
        random_array = np.random.random((1000, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        ax1 = df.plot.scatter(x="A label", y="B label")
        ax2 = df.plot.scatter(x="A label", y="B label", c="C label")

        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_minorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_minorticklabels()]
        assert vis1 == vis2

        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_majorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_majorticklabels()]
        assert vis1 == vis2

        assert (
            ax1.xaxis.get_label().get_visible() == ax2.xaxis.get_label().get_visible()
        )

    def test_if_hexbin_xaxis_label_is_visible(self):
        # addressing issue #10678, to ensure colobar does not
        # interfere with x-axis label and ticklabels with
        # ipython inline backend.
        random_array = np.random.random((1000, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        ax = df.plot.hexbin("A label", "B label", gridsize=12)
        assert all(vis.get_visible() for vis in ax.xaxis.get_minorticklabels())
        assert all(vis.get_visible() for vis in ax.xaxis.get_majorticklabels())
        assert ax.xaxis.get_label().get_visible()

    def test_if_scatterplot_colorbars_are_next_to_parent_axes(self):
        import matplotlib.pyplot as plt

        random_array = np.random.random((1000, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        fig, axes = plt.subplots(1, 2)
        df.plot.scatter("A label", "B label", c="C label", ax=axes[0])
        df.plot.scatter("A label", "B label", c="C label", ax=axes[1])
        plt.tight_layout()

        points = np.array([ax.get_position().get_points() for ax in fig.axes])
        axes_x_coords = points[:, :, 0]
        parent_distance = axes_x_coords[1, :] - axes_x_coords[0, :]
        colorbar_distance = axes_x_coords[3, :] - axes_x_coords[2, :]
        assert np.isclose(parent_distance, colorbar_distance, atol=1e-7).all()

    @pytest.mark.parametrize("cmap", [None, "Greys"])
    def test_scatter_with_c_column_name_with_colors(self, cmap):
        # https://github.com/pandas-dev/pandas/issues/34316
        df = DataFrame(
            [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
            columns=["length", "width"],
        )
        df["species"] = ["r", "r", "g", "g", "b"]
        ax = df.plot.scatter(x=0, y=1, c="species", cmap=cmap)
        assert ax.collections[0].colorbar is None

    def test_scatter_colors(self):
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        with pytest.raises(TypeError, match="Specify exactly one of `c` and `color`"):
            df.plot.scatter(x="a", y="b", c="c", color="green")

        default_colors = self._unpack_cycler(self.plt.rcParams)

        ax = df.plot.scatter(x="a", y="b", c="c")
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array(self.colorconverter.to_rgba(default_colors[0])),
        )

        ax = df.plot.scatter(x="a", y="b", color="white")
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array([1, 1, 1, 1], dtype=np.float64),
        )

    def test_scatter_colorbar_different_cmap(self):
        # GH 33389
        import matplotlib.pyplot as plt

        df = DataFrame({"x": [1, 2, 3], "y": [1, 3, 2], "c": [1, 2, 3]})
        df["x2"] = df["x"] + 1

        fig, ax = plt.subplots()
        df.plot("x", "y", c="c", kind="scatter", cmap="cividis", ax=ax)
        df.plot("x2", "y", c="c", kind="scatter", cmap="magma", ax=ax)

        assert ax.collections[0].cmap.name == "cividis"
        assert ax.collections[1].cmap.name == "magma"

    def test_line_colors(self):
        from matplotlib import cm

        custom_colors = "rgcby"
        df = DataFrame(np.random.randn(5, 5))

        ax = df.plot(color=custom_colors)
        self._check_colors(ax.get_lines(), linecolors=custom_colors)

        tm.close()

        ax2 = df.plot(color=custom_colors)
        lines2 = ax2.get_lines()

        for l1, l2 in zip(ax.get_lines(), lines2):
            assert l1.get_color() == l2.get_color()

        tm.close()

        ax = df.plot(colormap="jet")
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        self._check_colors(ax.get_lines(), linecolors=rgba_colors)
        tm.close()

        ax = df.plot(colormap=cm.jet)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        self._check_colors(ax.get_lines(), linecolors=rgba_colors)
        tm.close()

        # make color a list if plotting one column frame
        # handles cases like df.plot(color='DodgerBlue')
        ax = df.loc[:, [0]].plot(color="DodgerBlue")
        self._check_colors(ax.lines, linecolors=["DodgerBlue"])

        ax = df.plot(color="red")
        self._check_colors(ax.get_lines(), linecolors=["red"] * 5)
        tm.close()

        # GH 10299
        custom_colors = ["#FF0000", "#0000FF", "#FFFF00", "#000000", "#FFFFFF"]
        ax = df.plot(color=custom_colors)
        self._check_colors(ax.get_lines(), linecolors=custom_colors)
        tm.close()

    def test_dont_modify_colors(self):
        colors = ["r", "g", "b"]
        DataFrame(np.random.rand(10, 2)).plot(color=colors)
        assert len(colors) == 3

    def test_line_colors_and_styles_subplots(self):
        # GH 9894
        from matplotlib import cm

        default_colors = self._unpack_cycler(self.plt.rcParams)

        df = DataFrame(np.random.randn(5, 5))

        axes = df.plot(subplots=True)
        for ax, c in zip(axes, list(default_colors)):
            self._check_colors(ax.get_lines(), linecolors=[c])
        tm.close()

        # single color char
        axes = df.plot(subplots=True, color="k")
        for ax in axes:
            self._check_colors(ax.get_lines(), linecolors=["k"])
        tm.close()

        # single color str
        axes = df.plot(subplots=True, color="green")
        for ax in axes:
            self._check_colors(ax.get_lines(), linecolors=["green"])
        tm.close()

        custom_colors = "rgcby"
        axes = df.plot(color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors)):
            self._check_colors(ax.get_lines(), linecolors=[c])
        tm.close()

        axes = df.plot(color=list(custom_colors), subplots=True)
        for ax, c in zip(axes, list(custom_colors)):
            self._check_colors(ax.get_lines(), linecolors=[c])
        tm.close()

        # GH 10299
        custom_colors = ["#FF0000", "#0000FF", "#FFFF00", "#000000", "#FFFFFF"]
        axes = df.plot(color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors)):
            self._check_colors(ax.get_lines(), linecolors=[c])
        tm.close()

        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        for cmap in ["jet", cm.jet]:
            axes = df.plot(colormap=cmap, subplots=True)
            for ax, c in zip(axes, rgba_colors):
                self._check_colors(ax.get_lines(), linecolors=[c])
            tm.close()

        # make color a list if plotting one column frame
        # handles cases like df.plot(color='DodgerBlue')
        axes = df.loc[:, [0]].plot(color="DodgerBlue", subplots=True)
        self._check_colors(axes[0].lines, linecolors=["DodgerBlue"])

        # single character style
        axes = df.plot(style="r", subplots=True)
        for ax in axes:
            self._check_colors(ax.get_lines(), linecolors=["r"])
        tm.close()

        # list of styles
        styles = list("rgcby")
        axes = df.plot(style=styles, subplots=True)
        for ax, c in zip(axes, styles):
            self._check_colors(ax.get_lines(), linecolors=[c])
        tm.close()

    def test_area_colors(self):
        from matplotlib import cm
        from matplotlib.collections import PolyCollection

        custom_colors = "rgcby"
        df = DataFrame(np.random.rand(5, 5))

        ax = df.plot.area(color=custom_colors)
        self._check_colors(ax.get_lines(), linecolors=custom_colors)
        poly = [o for o in ax.get_children() if isinstance(o, PolyCollection)]
        self._check_colors(poly, facecolors=custom_colors)

        handles, labels = ax.get_legend_handles_labels()
        self._check_colors(handles, facecolors=custom_colors)

        for h in handles:
            assert h.get_alpha() is None
        tm.close()

        ax = df.plot.area(colormap="jet")
        jet_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        self._check_colors(ax.get_lines(), linecolors=jet_colors)
        poly = [o for o in ax.get_children() if isinstance(o, PolyCollection)]
        self._check_colors(poly, facecolors=jet_colors)

        handles, labels = ax.get_legend_handles_labels()
        self._check_colors(handles, facecolors=jet_colors)
        for h in handles:
            assert h.get_alpha() is None
        tm.close()

        # When stacked=False, alpha is set to 0.5
        ax = df.plot.area(colormap=cm.jet, stacked=False)
        self._check_colors(ax.get_lines(), linecolors=jet_colors)
        poly = [o for o in ax.get_children() if isinstance(o, PolyCollection)]
        jet_with_alpha = [(c[0], c[1], c[2], 0.5) for c in jet_colors]
        self._check_colors(poly, facecolors=jet_with_alpha)

        handles, labels = ax.get_legend_handles_labels()
        linecolors = jet_with_alpha
        self._check_colors(handles[: len(jet_colors)], linecolors=linecolors)
        for h in handles:
            assert h.get_alpha() == 0.5

    def test_hist_colors(self):
        default_colors = self._unpack_cycler(self.plt.rcParams)

        df = DataFrame(np.random.randn(5, 5))
        ax = df.plot.hist()
        self._check_colors(ax.patches[::10], facecolors=default_colors[:5])
        tm.close()

        custom_colors = "rgcby"
        ax = df.plot.hist(color=custom_colors)
        self._check_colors(ax.patches[::10], facecolors=custom_colors)
        tm.close()

        from matplotlib import cm

        # Test str -> colormap functionality
        ax = df.plot.hist(colormap="jet")
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        self._check_colors(ax.patches[::10], facecolors=rgba_colors)
        tm.close()

        # Test colormap functionality
        ax = df.plot.hist(colormap=cm.jet)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        self._check_colors(ax.patches[::10], facecolors=rgba_colors)
        tm.close()

        ax = df.loc[:, [0]].plot.hist(color="DodgerBlue")
        self._check_colors([ax.patches[0]], facecolors=["DodgerBlue"])

        ax = df.plot(kind="hist", color="green")
        self._check_colors(ax.patches[::10], facecolors=["green"] * 5)
        tm.close()

    @td.skip_if_no_scipy
    def test_kde_colors(self):
        from matplotlib import cm

        custom_colors = "rgcby"
        df = DataFrame(np.random.rand(5, 5))

        ax = df.plot.kde(color=custom_colors)
        self._check_colors(ax.get_lines(), linecolors=custom_colors)
        tm.close()

        ax = df.plot.kde(colormap="jet")
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        self._check_colors(ax.get_lines(), linecolors=rgba_colors)
        tm.close()

        ax = df.plot.kde(colormap=cm.jet)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        self._check_colors(ax.get_lines(), linecolors=rgba_colors)

    @td.skip_if_no_scipy
    def test_kde_colors_and_styles_subplots(self):
        from matplotlib import cm

        default_colors = self._unpack_cycler(self.plt.rcParams)

        df = DataFrame(np.random.randn(5, 5))

        axes = df.plot(kind="kde", subplots=True)
        for ax, c in zip(axes, list(default_colors)):
            self._check_colors(ax.get_lines(), linecolors=[c])
        tm.close()

        # single color char
        axes = df.plot(kind="kde", color="k", subplots=True)
        for ax in axes:
            self._check_colors(ax.get_lines(), linecolors=["k"])
        tm.close()

        # single color str
        axes = df.plot(kind="kde", color="red", subplots=True)
        for ax in axes:
            self._check_colors(ax.get_lines(), linecolors=["red"])
        tm.close()

        custom_colors = "rgcby"
        axes = df.plot(kind="kde", color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors)):
            self._check_colors(ax.get_lines(), linecolors=[c])
        tm.close()

        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        for cmap in ["jet", cm.jet]:
            axes = df.plot(kind="kde", colormap=cmap, subplots=True)
            for ax, c in zip(axes, rgba_colors):
                self._check_colors(ax.get_lines(), linecolors=[c])
            tm.close()

        # make color a list if plotting one column frame
        # handles cases like df.plot(color='DodgerBlue')
        axes = df.loc[:, [0]].plot(kind="kde", color="DodgerBlue", subplots=True)
        self._check_colors(axes[0].lines, linecolors=["DodgerBlue"])

        # single character style
        axes = df.plot(kind="kde", style="r", subplots=True)
        for ax in axes:
            self._check_colors(ax.get_lines(), linecolors=["r"])
        tm.close()

        # list of styles
        styles = list("rgcby")
        axes = df.plot(kind="kde", style=styles, subplots=True)
        for ax, c in zip(axes, styles):
            self._check_colors(ax.get_lines(), linecolors=[c])
        tm.close()

    def test_boxplot_colors(self):
        def _check_colors(bp, box_c, whiskers_c, medians_c, caps_c="k", fliers_c=None):
            # TODO: outside this func?
            if fliers_c is None:
                fliers_c = "k"
            self._check_colors(bp["boxes"], linecolors=[box_c] * len(bp["boxes"]))
            self._check_colors(
                bp["whiskers"], linecolors=[whiskers_c] * len(bp["whiskers"])
            )
            self._check_colors(
                bp["medians"], linecolors=[medians_c] * len(bp["medians"])
            )
            self._check_colors(bp["fliers"], linecolors=[fliers_c] * len(bp["fliers"]))
            self._check_colors(bp["caps"], linecolors=[caps_c] * len(bp["caps"]))

        default_colors = self._unpack_cycler(self.plt.rcParams)

        df = DataFrame(np.random.randn(5, 5))
        bp = df.plot.box(return_type="dict")
        _check_colors(
            bp,
            default_colors[0],
            default_colors[0],
            default_colors[2],
            default_colors[0],
        )
        tm.close()

        dict_colors = {
            "boxes": "#572923",
            "whiskers": "#982042",
            "medians": "#804823",
            "caps": "#123456",
        }
        bp = df.plot.box(color=dict_colors, sym="r+", return_type="dict")
        _check_colors(
            bp,
            dict_colors["boxes"],
            dict_colors["whiskers"],
            dict_colors["medians"],
            dict_colors["caps"],
            "r",
        )
        tm.close()

        # partial colors
        dict_colors = {"whiskers": "c", "medians": "m"}
        bp = df.plot.box(color=dict_colors, return_type="dict")
        _check_colors(bp, default_colors[0], "c", "m", default_colors[0])
        tm.close()

        from matplotlib import cm

        # Test str -> colormap functionality
        bp = df.plot.box(colormap="jet", return_type="dict")
        jet_colors = [cm.jet(n) for n in np.linspace(0, 1, 3)]
        _check_colors(bp, jet_colors[0], jet_colors[0], jet_colors[2], jet_colors[0])
        tm.close()

        # Test colormap functionality
        bp = df.plot.box(colormap=cm.jet, return_type="dict")
        _check_colors(bp, jet_colors[0], jet_colors[0], jet_colors[2], jet_colors[0])
        tm.close()

        # string color is applied to all artists except fliers
        bp = df.plot.box(color="DodgerBlue", return_type="dict")
        _check_colors(bp, "DodgerBlue", "DodgerBlue", "DodgerBlue", "DodgerBlue")

        # tuple is also applied to all artists except fliers
        bp = df.plot.box(color=(0, 1, 0), sym="#123456", return_type="dict")
        _check_colors(bp, (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), "#123456")

        msg = re.escape(
            "color dict contains invalid key 'xxxx'. The key must be either "
            "['boxes', 'whiskers', 'medians', 'caps']"
        )
        with pytest.raises(ValueError, match=msg):
            # Color contains invalid key results in ValueError
            df.plot.box(color={"boxes": "red", "xxxx": "blue"})

    def test_default_color_cycle(self):
        import cycler
        import matplotlib.pyplot as plt

        colors = list("rgbk")
        plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", colors)

        df = DataFrame(np.random.randn(5, 3))
        ax = df.plot()

        expected = self._unpack_cycler(plt.rcParams)[:3]
        self._check_colors(ax.get_lines(), linecolors=expected)

    def test_no_color_bar(self):
        df = self.hexbin_df
        ax = df.plot.hexbin(x="A", y="B", colorbar=None)
        assert ax.collections[0].colorbar is None

    def test_mixing_cmap_and_colormap_raises(self):
        df = self.hexbin_df
        msg = "Only specify one of `cmap` and `colormap`"
        with pytest.raises(TypeError, match=msg):
            df.plot.hexbin(x="A", y="B", cmap="YlGn", colormap="BuGn")

    def test_passed_bar_colors(self):
        import matplotlib as mpl

        color_tuples = [(0.9, 0, 0, 1), (0, 0.9, 0, 1), (0, 0, 0.9, 1)]
        colormap = mpl.colors.ListedColormap(color_tuples)
        barplot = DataFrame([[1, 2, 3]]).plot(kind="bar", cmap=colormap)
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    def test_rcParams_bar_colors(self):
        import matplotlib as mpl

        color_tuples = [(0.9, 0, 0, 1), (0, 0.9, 0, 1), (0, 0, 0.9, 1)]
        with mpl.rc_context(rc={"axes.prop_cycle": mpl.cycler("color", color_tuples)}):
            barplot = DataFrame([[1, 2, 3]]).plot(kind="bar")
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    def test_colors_of_columns_with_same_name(self):
        # ISSUE 11136 -> https://github.com/pandas-dev/pandas/issues/11136
        # Creating a DataFrame with duplicate column labels and testing colors of them.
        df = DataFrame({"b": [0, 1, 0], "a": [1, 2, 3]})
        df1 = DataFrame({"a": [2, 4, 6]})
        df_concat = pd.concat([df, df1], axis=1)
        result = df_concat.plot()
        for legend, line in zip(result.get_legend().legendHandles, result.lines):
            assert legend.get_color() == line.get_color()

    def test_invalid_colormap(self):
        df = DataFrame(np.random.randn(3, 2), columns=["A", "B"])
        msg = (
            "'invalid_colormap' is not a valid value for name; supported values are "
            "'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', "
            "'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', "
            "'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', "
            "'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', "
            "'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', "
            "'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', "
            "'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', "
            "'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', "
            "'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', "
            "'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', "
            "'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', "
            "'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', "
            "'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', "
            "'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', "
            "'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', "
            "'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', "
            "'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', "
            "'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', "
            "'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', "
            "'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', "
            "'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', "
            "'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', "
            "'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', "
            "'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', "
            "'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', "
            "'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', "
            "'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', "
            "'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(colormap="invalid_colormap")
