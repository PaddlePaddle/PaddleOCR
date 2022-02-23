""" Test cases for misc plot functions """

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    TestPlotBase,
    _check_plot_works,
)

import pandas.plotting as plotting

pytestmark = pytest.mark.slow


@td.skip_if_mpl
def test_import_error_message():
    # GH-19810
    df = DataFrame({"A": [1, 2]})

    with pytest.raises(ImportError, match="matplotlib is required for plotting"):
        df.plot()


def test_get_accessor_args():
    func = plotting._core.PlotAccessor._get_call_args

    msg = "Called plot accessor for type list, expected Series or DataFrame"
    with pytest.raises(TypeError, match=msg):
        func(backend_name="", data=[], args=[], kwargs={})

    msg = "should not be called with positional arguments"
    with pytest.raises(TypeError, match=msg):
        func(backend_name="", data=Series(dtype=object), args=["line", None], kwargs={})

    x, y, kind, kwargs = func(
        backend_name="",
        data=DataFrame(),
        args=["x"],
        kwargs={"y": "y", "kind": "bar", "grid": False},
    )
    assert x == "x"
    assert y == "y"
    assert kind == "bar"
    assert kwargs == {"grid": False}

    x, y, kind, kwargs = func(
        backend_name="pandas.plotting._matplotlib",
        data=Series(dtype=object),
        args=[],
        kwargs={},
    )
    assert x is None
    assert y is None
    assert kind == "line"
    assert len(kwargs) == 24


@td.skip_if_no_mpl
class TestSeriesPlots(TestPlotBase):
    def setup_method(self, method):
        TestPlotBase.setup_method(self, method)
        import matplotlib as mpl

        mpl.rcdefaults()

        self.ts = tm.makeTimeSeries()
        self.ts.name = "ts"

    def test_autocorrelation_plot(self):
        from pandas.plotting import autocorrelation_plot

        # Ensure no UserWarning when making plot
        with tm.assert_produces_warning(None):
            _check_plot_works(autocorrelation_plot, series=self.ts)
            _check_plot_works(autocorrelation_plot, series=self.ts.values)

            ax = autocorrelation_plot(self.ts, label="Test")
        self._check_legend_labels(ax, labels=["Test"])

    def test_lag_plot(self):
        from pandas.plotting import lag_plot

        _check_plot_works(lag_plot, series=self.ts)
        _check_plot_works(lag_plot, series=self.ts, lag=5)

    def test_bootstrap_plot(self):
        from pandas.plotting import bootstrap_plot

        _check_plot_works(bootstrap_plot, series=self.ts, size=10)


@td.skip_if_no_mpl
class TestDataFramePlots(TestPlotBase):
    @td.skip_if_no_scipy
    @pytest.mark.parametrize("pass_axis", [False, True])
    def test_scatter_matrix_axis(self, pass_axis):
        from pandas.plotting._matplotlib.compat import mpl_ge_3_0_0

        scatter_matrix = plotting.scatter_matrix

        ax = None
        if pass_axis:
            _, ax = self.plt.subplots(3, 3)

        with tm.RNGContext(42):
            df = DataFrame(np.random.randn(100, 3))

        # we are plotting multiples on a sub-plot
        with tm.assert_produces_warning(
            UserWarning, raise_on_extra_warnings=mpl_ge_3_0_0()
        ):
            axes = _check_plot_works(
                scatter_matrix,
                filterwarnings="always",
                frame=df,
                range_padding=0.1,
                ax=ax,
            )
        axes0_labels = axes[0][0].yaxis.get_majorticklabels()

        # GH 5662
        expected = ["-2", "0", "2"]
        self._check_text_labels(axes0_labels, expected)
        self._check_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)

        df[0] = (df[0] - 2) / 3

        # we are plotting multiples on a sub-plot
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(
                scatter_matrix,
                filterwarnings="always",
                frame=df,
                range_padding=0.1,
                ax=ax,
            )
        axes0_labels = axes[0][0].yaxis.get_majorticklabels()
        expected = ["-1.0", "-0.5", "0.0"]
        self._check_text_labels(axes0_labels, expected)
        self._check_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)

    def test_andrews_curves(self, iris):
        from matplotlib import cm

        from pandas.plotting import andrews_curves

        df = iris
        # Ensure no UserWarning when making plot
        with tm.assert_produces_warning(None):
            _check_plot_works(andrews_curves, frame=df, class_column="Name")

        rgba = ("#556270", "#4ECDC4", "#C7F464")
        ax = _check_plot_works(
            andrews_curves, frame=df, class_column="Name", color=rgba
        )
        self._check_colors(
            ax.get_lines()[:10], linecolors=rgba, mapping=df["Name"][:10]
        )

        cnames = ["dodgerblue", "aquamarine", "seagreen"]
        ax = _check_plot_works(
            andrews_curves, frame=df, class_column="Name", color=cnames
        )
        self._check_colors(
            ax.get_lines()[:10], linecolors=cnames, mapping=df["Name"][:10]
        )

        ax = _check_plot_works(
            andrews_curves, frame=df, class_column="Name", colormap=cm.jet
        )
        cmaps = [cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        self._check_colors(
            ax.get_lines()[:10], linecolors=cmaps, mapping=df["Name"][:10]
        )

        length = 10
        df = DataFrame(
            {
                "A": np.random.rand(length),
                "B": np.random.rand(length),
                "C": np.random.rand(length),
                "Name": ["A"] * length,
            }
        )

        _check_plot_works(andrews_curves, frame=df, class_column="Name")

        rgba = ("#556270", "#4ECDC4", "#C7F464")
        ax = _check_plot_works(
            andrews_curves, frame=df, class_column="Name", color=rgba
        )
        self._check_colors(
            ax.get_lines()[:10], linecolors=rgba, mapping=df["Name"][:10]
        )

        cnames = ["dodgerblue", "aquamarine", "seagreen"]
        ax = _check_plot_works(
            andrews_curves, frame=df, class_column="Name", color=cnames
        )
        self._check_colors(
            ax.get_lines()[:10], linecolors=cnames, mapping=df["Name"][:10]
        )

        ax = _check_plot_works(
            andrews_curves, frame=df, class_column="Name", colormap=cm.jet
        )
        cmaps = [cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        self._check_colors(
            ax.get_lines()[:10], linecolors=cmaps, mapping=df["Name"][:10]
        )

        colors = ["b", "g", "r"]
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3], "Name": colors})
        ax = andrews_curves(df, "Name", color=colors)
        handles, labels = ax.get_legend_handles_labels()
        self._check_colors(handles, linecolors=colors)

    def test_parallel_coordinates(self, iris):
        from matplotlib import cm

        from pandas.plotting import parallel_coordinates

        df = iris

        ax = _check_plot_works(parallel_coordinates, frame=df, class_column="Name")
        nlines = len(ax.get_lines())
        nxticks = len(ax.xaxis.get_ticklabels())

        rgba = ("#556270", "#4ECDC4", "#C7F464")
        ax = _check_plot_works(
            parallel_coordinates, frame=df, class_column="Name", color=rgba
        )
        self._check_colors(
            ax.get_lines()[:10], linecolors=rgba, mapping=df["Name"][:10]
        )

        cnames = ["dodgerblue", "aquamarine", "seagreen"]
        ax = _check_plot_works(
            parallel_coordinates, frame=df, class_column="Name", color=cnames
        )
        self._check_colors(
            ax.get_lines()[:10], linecolors=cnames, mapping=df["Name"][:10]
        )

        ax = _check_plot_works(
            parallel_coordinates, frame=df, class_column="Name", colormap=cm.jet
        )
        cmaps = [cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        self._check_colors(
            ax.get_lines()[:10], linecolors=cmaps, mapping=df["Name"][:10]
        )

        ax = _check_plot_works(
            parallel_coordinates, frame=df, class_column="Name", axvlines=False
        )
        assert len(ax.get_lines()) == (nlines - nxticks)

        colors = ["b", "g", "r"]
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3], "Name": colors})
        ax = parallel_coordinates(df, "Name", color=colors)
        handles, labels = ax.get_legend_handles_labels()
        self._check_colors(handles, linecolors=colors)

    # not sure if this is indicative of a problem
    @pytest.mark.filterwarnings("ignore:Attempting to set:UserWarning")
    def test_parallel_coordinates_with_sorted_labels(self):
        """For #15908"""
        from pandas.plotting import parallel_coordinates

        df = DataFrame(
            {
                "feat": list(range(30)),
                "class": [2 for _ in range(10)]
                + [3 for _ in range(10)]
                + [1 for _ in range(10)],
            }
        )
        ax = parallel_coordinates(df, "class", sort_labels=True)
        polylines, labels = ax.get_legend_handles_labels()
        color_label_tuples = zip(
            [polyline.get_color() for polyline in polylines], labels
        )
        ordered_color_label_tuples = sorted(color_label_tuples, key=lambda x: x[1])
        prev_next_tupels = zip(
            list(ordered_color_label_tuples[0:-1]), list(ordered_color_label_tuples[1:])
        )
        for prev, nxt in prev_next_tupels:
            # labels and colors are ordered strictly increasing
            assert prev[1] < nxt[1] and prev[0] < nxt[0]

    def test_radviz(self, iris):
        from matplotlib import cm

        from pandas.plotting import radviz

        df = iris
        # Ensure no UserWarning when making plot
        with tm.assert_produces_warning(None):
            _check_plot_works(radviz, frame=df, class_column="Name")

        rgba = ("#556270", "#4ECDC4", "#C7F464")
        ax = _check_plot_works(radviz, frame=df, class_column="Name", color=rgba)
        # skip Circle drawn as ticks
        patches = [p for p in ax.patches[:20] if p.get_label() != ""]
        self._check_colors(patches[:10], facecolors=rgba, mapping=df["Name"][:10])

        cnames = ["dodgerblue", "aquamarine", "seagreen"]
        _check_plot_works(radviz, frame=df, class_column="Name", color=cnames)
        patches = [p for p in ax.patches[:20] if p.get_label() != ""]
        self._check_colors(patches, facecolors=cnames, mapping=df["Name"][:10])

        _check_plot_works(radviz, frame=df, class_column="Name", colormap=cm.jet)
        cmaps = [cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        patches = [p for p in ax.patches[:20] if p.get_label() != ""]
        self._check_colors(patches, facecolors=cmaps, mapping=df["Name"][:10])

        colors = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
        df = DataFrame(
            {"A": [1, 2, 3], "B": [2, 1, 3], "C": [3, 2, 1], "Name": ["b", "g", "r"]}
        )
        ax = radviz(df, "Name", color=colors)
        handles, labels = ax.get_legend_handles_labels()
        self._check_colors(handles, facecolors=colors)

    def test_subplot_titles(self, iris):
        df = iris.drop("Name", axis=1).head()
        # Use the column names as the subplot titles
        title = list(df.columns)

        # Case len(title) == len(df)
        plot = df.plot(subplots=True, title=title)
        assert [p.get_title() for p in plot] == title

        # Case len(title) > len(df)
        msg = (
            "The length of `title` must equal the number of columns if "
            "using `title` of type `list` and `subplots=True`"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, title=title + ["kittens > puppies"])

        # Case len(title) < len(df)
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, title=title[:2])

        # Case subplots=False and title is of type list
        msg = (
            "Using `title` of type `list` is not supported unless "
            "`subplots=True` is passed"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=False, title=title)

        # Case df with 3 numeric columns but layout of (2,2)
        plot = df.drop("SepalWidth", axis=1).plot(
            subplots=True, layout=(2, 2), title=title[:-1]
        )
        title_list = [ax.get_title() for sublist in plot for ax in sublist]
        assert title_list == title[:3] + [""]

    def test_get_standard_colors_random_seed(self):
        # GH17525
        df = DataFrame(np.zeros((10, 10)))

        # Make sure that the np.random.seed isn't reset by get_standard_colors
        plotting.parallel_coordinates(df, 0)
        rand1 = np.random.random()
        plotting.parallel_coordinates(df, 0)
        rand2 = np.random.random()
        assert rand1 != rand2

        # Make sure it produces the same colors every time it's called
        from pandas.plotting._matplotlib.style import get_standard_colors

        color1 = get_standard_colors(1, color_type="random")
        color2 = get_standard_colors(1, color_type="random")
        assert color1 == color2

    def test_get_standard_colors_default_num_colors(self):
        from pandas.plotting._matplotlib.style import get_standard_colors

        # Make sure the default color_types returns the specified amount
        color1 = get_standard_colors(1, color_type="default")
        color2 = get_standard_colors(9, color_type="default")
        color3 = get_standard_colors(20, color_type="default")
        assert len(color1) == 1
        assert len(color2) == 9
        assert len(color3) == 20

    def test_plot_single_color(self):
        # Example from #20585. All 3 bars should have the same color
        df = DataFrame(
            {
                "account-start": ["2017-02-03", "2017-03-03", "2017-01-01"],
                "client": ["Alice Anders", "Bob Baker", "Charlie Chaplin"],
                "balance": [-1432.32, 10.43, 30000.00],
                "db-id": [1234, 2424, 251],
                "proxy-id": [525, 1525, 2542],
                "rank": [52, 525, 32],
            }
        )
        ax = df.client.value_counts().plot.bar()
        colors = [rect.get_facecolor() for rect in ax.get_children()[0:3]]
        assert all(color == colors[0] for color in colors)

    def test_get_standard_colors_no_appending(self):
        # GH20726

        # Make sure not to add more colors so that matplotlib can cycle
        # correctly.
        from matplotlib import cm

        from pandas.plotting._matplotlib.style import get_standard_colors

        color_before = cm.gnuplot(range(5))
        color_after = get_standard_colors(1, color=color_before)
        assert len(color_after) == len(color_before)

        df = DataFrame(np.random.randn(48, 4), columns=list("ABCD"))

        color_list = cm.gnuplot(np.linspace(0, 1, 16))
        p = df.A.plot.bar(figsize=(16, 7), color=color_list)
        assert p.patches[1].get_facecolor() == p.patches[17].get_facecolor()

    def test_dictionary_color(self):
        # issue-8193
        # Test plot color dictionary format
        data_files = ["a", "b"]

        expected = [(0.5, 0.24, 0.6), (0.3, 0.7, 0.7)]

        df1 = DataFrame(np.random.rand(2, 2), columns=data_files)
        dic_color = {"b": (0.3, 0.7, 0.7), "a": (0.5, 0.24, 0.6)}

        # Bar color test
        ax = df1.plot(kind="bar", color=dic_color)
        colors = [rect.get_facecolor()[0:-1] for rect in ax.get_children()[0:3:2]]
        assert all(color == expected[index] for index, color in enumerate(colors))

        # Line color test
        ax = df1.plot(kind="line", color=dic_color)
        colors = [rect.get_color() for rect in ax.get_lines()[0:2]]
        assert all(color == expected[index] for index, color in enumerate(colors))

    def test_has_externally_shared_axis_x_axis(self):
        # GH33819
        # Test _has_externally_shared_axis() works for x-axis
        func = plotting._matplotlib.tools._has_externally_shared_axis

        fig = self.plt.figure()
        plots = fig.subplots(2, 4)

        # Create *externally* shared axes for first and third columns
        plots[0][0] = fig.add_subplot(231, sharex=plots[1][0])
        plots[0][2] = fig.add_subplot(233, sharex=plots[1][2])

        # Create *internally* shared axes for second and third columns
        plots[0][1].twinx()
        plots[0][2].twinx()

        # First  column is only externally shared
        # Second column is only internally shared
        # Third  column is both
        # Fourth column is neither
        assert func(plots[0][0], "x")
        assert not func(plots[0][1], "x")
        assert func(plots[0][2], "x")
        assert not func(plots[0][3], "x")

    def test_has_externally_shared_axis_y_axis(self):
        # GH33819
        # Test _has_externally_shared_axis() works for y-axis
        func = plotting._matplotlib.tools._has_externally_shared_axis

        fig = self.plt.figure()
        plots = fig.subplots(4, 2)

        # Create *externally* shared axes for first and third rows
        plots[0][0] = fig.add_subplot(321, sharey=plots[0][1])
        plots[2][0] = fig.add_subplot(325, sharey=plots[2][1])

        # Create *internally* shared axes for second and third rows
        plots[1][0].twiny()
        plots[2][0].twiny()

        # First  row is only externally shared
        # Second row is only internally shared
        # Third  row is both
        # Fourth row is neither
        assert func(plots[0][0], "y")
        assert not func(plots[1][0], "y")
        assert func(plots[2][0], "y")
        assert not func(plots[3][0], "y")

    def test_has_externally_shared_axis_invalid_compare_axis(self):
        # GH33819
        # Test _has_externally_shared_axis() raises an exception when
        # passed an invalid value as compare_axis parameter
        func = plotting._matplotlib.tools._has_externally_shared_axis

        fig = self.plt.figure()
        plots = fig.subplots(4, 2)

        # Create arbitrary axes
        plots[0][0] = fig.add_subplot(321, sharey=plots[0][1])

        # Check that an invalid compare_axis value triggers the expected exception
        msg = "needs 'x' or 'y' as a second parameter"
        with pytest.raises(ValueError, match=msg):
            func(plots[0][0], "z")

    def test_externally_shared_axes(self):
        # Example from GH33819
        # Create data
        df = DataFrame({"a": np.random.randn(1000), "b": np.random.randn(1000)})

        # Create figure
        fig = self.plt.figure()
        plots = fig.subplots(2, 3)

        # Create *externally* shared axes
        plots[0][0] = fig.add_subplot(231, sharex=plots[1][0])
        # note: no plots[0][1] that's the twin only case
        plots[0][2] = fig.add_subplot(233, sharex=plots[1][2])

        # Create *internally* shared axes
        # note: no plots[0][0] that's the external only case
        twin_ax1 = plots[0][1].twinx()
        twin_ax2 = plots[0][2].twinx()

        # Plot data to primary axes
        df["a"].plot(ax=plots[0][0], title="External share only").set_xlabel(
            "this label should never be visible"
        )
        df["a"].plot(ax=plots[1][0])

        df["a"].plot(ax=plots[0][1], title="Internal share (twin) only").set_xlabel(
            "this label should always be visible"
        )
        df["a"].plot(ax=plots[1][1])

        df["a"].plot(ax=plots[0][2], title="Both").set_xlabel(
            "this label should never be visible"
        )
        df["a"].plot(ax=plots[1][2])

        # Plot data to twinned axes
        df["b"].plot(ax=twin_ax1, color="green")
        df["b"].plot(ax=twin_ax2, color="yellow")

        assert not plots[0][0].xaxis.get_label().get_visible()
        assert plots[0][1].xaxis.get_label().get_visible()
        assert not plots[0][2].xaxis.get_label().get_visible()
