import collections
import platform
from unittest import mock

import numpy as np
import pytest

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import rc_context
from matplotlib.font_manager import FontProperties


def test_legend_ordereddict():
    # smoketest that ordereddict inputs work...

    X = np.random.randn(10)
    Y = np.random.randn(10)
    labels = ['a'] * 5 + ['b'] * 5
    colors = ['r'] * 5 + ['g'] * 5

    fig, ax = plt.subplots()
    for x, y, label, color in zip(X, Y, labels, colors):
        ax.scatter(x, y, label=label, c=color)

    handles, labels = ax.get_legend_handles_labels()
    legend = collections.OrderedDict(zip(labels, handles))
    ax.legend(legend.values(), legend.keys(),
              loc='center left', bbox_to_anchor=(1, .5))


@image_comparison(['legend_auto1'], remove_text=True)
def test_legend_auto1():
    """Test automatic legend placement"""
    fig, ax = plt.subplots()
    x = np.arange(100)
    ax.plot(x, 50 - x, 'o', label='y=1')
    ax.plot(x, x - 50, 'o', label='y=-1')
    ax.legend(loc='best')


@image_comparison(['legend_auto2'], remove_text=True)
def test_legend_auto2():
    """Test automatic legend placement"""
    fig, ax = plt.subplots()
    x = np.arange(100)
    b1 = ax.bar(x, x, align='edge', color='m')
    b2 = ax.bar(x, x[::-1], align='edge', color='g')
    ax.legend([b1[0], b2[0]], ['up', 'down'], loc='best')


@image_comparison(['legend_auto3'])
def test_legend_auto3():
    """Test automatic legend placement"""
    fig, ax = plt.subplots()
    x = [0.9, 0.1, 0.1, 0.9, 0.9, 0.5]
    y = [0.95, 0.95, 0.05, 0.05, 0.5, 0.5]
    ax.plot(x, y, 'o-', label='line')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='best')


@image_comparison(['legend_various_labels'], remove_text=True)
def test_various_labels():
    # tests all sorts of label types
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(np.arange(4), 'o', label=1)
    ax.plot(np.linspace(4, 4.1), 'o', label='Développés')
    ax.plot(np.arange(4, 1, -1), 'o', label='__nolegend__')
    ax.legend(numpoints=1, loc='best')


@image_comparison(['legend_labels_first.png'], remove_text=True)
def test_labels_first():
    # test labels to left of markers
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), '-o', label=1)
    ax.plot(np.ones(10)*5, ':x', label="x")
    ax.plot(np.arange(20, 10, -1), 'd', label="diamond")
    ax.legend(loc='best', markerfirst=False)


@image_comparison(['legend_multiple_keys.png'], remove_text=True)
def test_multiple_keys():
    # test legend entries with multiple keys
    fig, ax = plt.subplots()
    p1, = ax.plot([1, 2, 3], '-o')
    p2, = ax.plot([2, 3, 4], '-x')
    p3, = ax.plot([3, 4, 5], '-d')
    ax.legend([(p1, p2), (p2, p1), p3], ['two keys', 'pad=0', 'one key'],
              numpoints=1,
              handler_map={(p1, p2): HandlerTuple(ndivide=None),
                           (p2, p1): HandlerTuple(ndivide=None, pad=0)})


@image_comparison(['rgba_alpha.png'], remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.01)
def test_alpha_rgba():
    fig, ax = plt.subplots()
    ax.plot(range(10), lw=5)
    leg = plt.legend(['Longlabel that will go away'], loc='center')
    leg.legendPatch.set_facecolor([1, 0, 0, 0.5])


@image_comparison(['rcparam_alpha.png'], remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.01)
def test_alpha_rcparam():
    fig, ax = plt.subplots()
    ax.plot(range(10), lw=5)
    with mpl.rc_context(rc={'legend.framealpha': .75}):
        leg = plt.legend(['Longlabel that will go away'], loc='center')
        # this alpha is going to be over-ridden by the rcparam with
        # sets the alpha of the patch to be non-None which causes the alpha
        # value of the face color to be discarded.  This behavior may not be
        # ideal, but it is what it is and we should keep track of it changing
        leg.legendPatch.set_facecolor([1, 0, 0, 0.5])


@image_comparison(['fancy'], remove_text=True)
def test_fancy():
    # using subplot triggers some offsetbox functionality untested elsewhere
    plt.subplot(121)
    plt.plot([5] * 10, 'o--', label='XX')
    plt.scatter(np.arange(10), np.arange(10, 0, -1), label='XX\nXX')
    plt.errorbar(np.arange(10), np.arange(10), xerr=0.5,
                 yerr=0.5, label='XX')
    plt.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
               ncol=2, shadow=True, title="My legend", numpoints=1)


@image_comparison(['framealpha'], remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.02)
def test_framealpha():
    x = np.linspace(1, 100, 100)
    y = x
    plt.plot(x, y, label='mylabel', lw=10)
    plt.legend(framealpha=0.5)


@image_comparison(['scatter_rc3', 'scatter_rc1'], remove_text=True)
def test_rc():
    # using subplot triggers some offsetbox functionality untested elsewhere
    plt.figure()
    ax = plt.subplot(121)
    ax.scatter(np.arange(10), np.arange(10, 0, -1), label='three')
    ax.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
              title="My legend")

    mpl.rcParams['legend.scatterpoints'] = 1
    plt.figure()
    ax = plt.subplot(121)
    ax.scatter(np.arange(10), np.arange(10, 0, -1), label='one')
    ax.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
              title="My legend")


@image_comparison(['legend_expand'], remove_text=True)
def test_legend_expand():
    """Test expand mode"""
    legend_modes = [None, "expand"]
    fig, axs = plt.subplots(len(legend_modes), 1)
    x = np.arange(100)
    for ax, mode in zip(axs, legend_modes):
        ax.plot(x, 50 - x, 'o', label='y=1')
        l1 = ax.legend(loc='upper left', mode=mode)
        ax.add_artist(l1)
        ax.plot(x, x - 50, 'o', label='y=-1')
        l2 = ax.legend(loc='right', mode=mode)
        ax.add_artist(l2)
        ax.legend(loc='lower left', mode=mode, ncol=2)


@image_comparison(['hatching'], remove_text=True, style='default')
def test_hatching():
    # Remove this line when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6

    fig, ax = plt.subplots()

    # Patches
    patch = plt.Rectangle((0, 0), 0.3, 0.3, hatch='xx',
                          label='Patch\ndefault color\nfilled')
    ax.add_patch(patch)
    patch = plt.Rectangle((0.33, 0), 0.3, 0.3, hatch='||', edgecolor='C1',
                          label='Patch\nexplicit color\nfilled')
    ax.add_patch(patch)
    patch = plt.Rectangle((0, 0.4), 0.3, 0.3, hatch='xx', fill=False,
                          label='Patch\ndefault color\nunfilled')
    ax.add_patch(patch)
    patch = plt.Rectangle((0.33, 0.4), 0.3, 0.3, hatch='||', fill=False,
                          edgecolor='C1',
                          label='Patch\nexplicit color\nunfilled')
    ax.add_patch(patch)

    # Paths
    ax.fill_between([0, .15, .3], [.8, .8, .8], [.9, 1.0, .9],
                    hatch='+', label='Path\ndefault color')
    ax.fill_between([.33, .48, .63], [.8, .8, .8], [.9, 1.0, .9],
                    hatch='+', edgecolor='C2', label='Path\nexplicit color')

    ax.set_xlim(-0.01, 1.1)
    ax.set_ylim(-0.01, 1.1)
    ax.legend(handlelength=4, handleheight=4)


def test_legend_remove():
    fig, ax = plt.subplots()
    lines = ax.plot(range(10))
    leg = fig.legend(lines, "test")
    leg.remove()
    assert fig.legends == []
    leg = ax.legend("test")
    leg.remove()
    assert ax.get_legend() is None


class TestLegendFunction:
    # Tests the legend function on the Axes and pyplot.
    def test_legend_no_args(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend()
        Legend.assert_called_with(plt.gca(), lines, ['hello world'])

    def test_legend_positional_handles_labels(self):
        lines = plt.plot(range(10))
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend(lines, ['hello world'])
        Legend.assert_called_with(plt.gca(), lines, ['hello world'])

    def test_legend_positional_handles_only(self):
        lines = plt.plot(range(10))
        with pytest.raises(TypeError, match='but found an Artist'):
            # a single arg is interpreted as labels
            # it's a common error to just pass handles
            plt.legend(lines)

    def test_legend_positional_labels_only(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend(['foobar'])
        Legend.assert_called_with(plt.gca(), lines, ['foobar'])

    def test_legend_three_args(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend(lines, ['foobar'], loc='right')
        Legend.assert_called_with(plt.gca(), lines, ['foobar'], loc='right')

    def test_legend_handler_map(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.'
                        '_get_legend_handles_labels') as handles_labels:
            handles_labels.return_value = lines, ['hello world']
            plt.legend(handler_map={'1': 2})
        handles_labels.assert_called_with([plt.gca()], {'1': 2})

    def test_legend_kwargs_handles_only(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, 11)
        ln1, = ax.plot(x, x, label='x')
        ln2, = ax.plot(x, 2*x, label='2x')
        ln3, = ax.plot(x, 3*x, label='3x')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            ax.legend(handles=[ln3, ln2])  # reversed and not ln1
        Legend.assert_called_with(ax, [ln3, ln2], ['3x', '2x'])

    def test_legend_kwargs_labels_only(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, 11)
        ln1, = ax.plot(x, x)
        ln2, = ax.plot(x, 2*x)
        with mock.patch('matplotlib.legend.Legend') as Legend:
            ax.legend(labels=['x', '2x'])
        Legend.assert_called_with(ax, [ln1, ln2], ['x', '2x'])

    def test_legend_kwargs_handles_labels(self):
        fig, ax = plt.subplots()
        th = np.linspace(0, 2*np.pi, 1024)
        lns, = ax.plot(th, np.sin(th), label='sin')
        lnc, = ax.plot(th, np.cos(th), label='cos')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # labels of lns, lnc are overwritten with explicit ('a', 'b')
            ax.legend(labels=('a', 'b'), handles=(lnc, lns))
        Legend.assert_called_with(ax, (lnc, lns), ('a', 'b'))

    def test_warn_mixed_args_and_kwargs(self):
        fig, ax = plt.subplots()
        th = np.linspace(0, 2*np.pi, 1024)
        lns, = ax.plot(th, np.sin(th), label='sin')
        lnc, = ax.plot(th, np.cos(th), label='cos')
        with pytest.warns(UserWarning) as record:
            ax.legend((lnc, lns), labels=('a', 'b'))
        assert len(record) == 1
        assert str(record[0].message) == (
            "You have mixed positional and keyword arguments, some input may "
            "be discarded.")

    def test_parasite(self):
        from mpl_toolkits.axes_grid1 import host_subplot

        host = host_subplot(111)
        par = host.twinx()

        p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
        p2, = par.plot([0, 1, 2], [0, 3, 2], label="Temperature")

        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend()
        Legend.assert_called_with(host, [p1, p2], ['Density', 'Temperature'])


class TestLegendFigureFunction:
    # Tests the legend function for figure
    def test_legend_handle_label(self):
        fig, ax = plt.subplots()
        lines = ax.plot(range(10))
        with mock.patch('matplotlib.legend.Legend') as Legend:
            fig.legend(lines, ['hello world'])
        Legend.assert_called_with(fig, lines, ['hello world'],
                                  bbox_transform=fig.transFigure)

    def test_legend_no_args(self):
        fig, ax = plt.subplots()
        lines = ax.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            fig.legend()
        Legend.assert_called_with(fig, lines, ['hello world'],
                                  bbox_transform=fig.transFigure)

    def test_legend_label_arg(self):
        fig, ax = plt.subplots()
        lines = ax.plot(range(10))
        with mock.patch('matplotlib.legend.Legend') as Legend:
            fig.legend(['foobar'])
        Legend.assert_called_with(fig, lines, ['foobar'],
                                  bbox_transform=fig.transFigure)

    def test_legend_label_three_args(self):
        fig, ax = plt.subplots()
        lines = ax.plot(range(10))
        with mock.patch('matplotlib.legend.Legend') as Legend:
            fig.legend(lines, ['foobar'], 'right')
        Legend.assert_called_with(fig, lines, ['foobar'], 'right',
                                  bbox_transform=fig.transFigure)

    def test_legend_label_three_args_pluskw(self):
        # test that third argument and loc=  called together give
        # Exception
        fig, ax = plt.subplots()
        lines = ax.plot(range(10))
        with pytest.raises(Exception):
            fig.legend(lines, ['foobar'], 'right', loc='left')

    def test_legend_kw_args(self):
        fig, axs = plt.subplots(1, 2)
        lines = axs[0].plot(range(10))
        lines2 = axs[1].plot(np.arange(10) * 2.)
        with mock.patch('matplotlib.legend.Legend') as Legend:
            fig.legend(loc='right', labels=('a', 'b'), handles=(lines, lines2))
        Legend.assert_called_with(
            fig, (lines, lines2), ('a', 'b'), loc='right',
            bbox_transform=fig.transFigure)

    def test_warn_args_kwargs(self):
        fig, axs = plt.subplots(1, 2)
        lines = axs[0].plot(range(10))
        lines2 = axs[1].plot(np.arange(10) * 2.)
        with pytest.warns(UserWarning) as record:
            fig.legend((lines, lines2), labels=('a', 'b'))
        assert len(record) == 1
        assert str(record[0].message) == (
            "You have mixed positional and keyword arguments, some input may "
            "be discarded.")


@image_comparison(['legend_stackplot.png'])
def test_legend_stackplot():
    """Test legend for PolyCollection using stackplot."""
    # related to #1341, #1943, and PR #3303
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 10)
    y1 = 1.0 * x
    y2 = 2.0 * x + 1
    y3 = 3.0 * x + 2
    ax.stackplot(x, y1, y2, y3, labels=['y1', 'y2', 'y3'])
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 70))
    ax.legend(loc='best')


def test_cross_figure_patch_legend():
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    brs = ax.bar(range(3), range(3))
    fig2.legend(brs, 'foo')


def test_nanscatter():
    fig, ax = plt.subplots()

    h = ax.scatter([np.nan], [np.nan], marker="o",
                   facecolor="r", edgecolor="r", s=3)

    ax.legend([h], ["scatter"])

    fig, ax = plt.subplots()
    for color in ['red', 'green', 'blue']:
        n = 750
        x, y = np.random.rand(2, n)
        scale = 200.0 * np.random.rand(n)
        ax.scatter(x, y, c=color, s=scale, label=color,
                   alpha=0.3, edgecolors='none')

    ax.legend()
    ax.grid(True)


def test_legend_repeatcheckok():
    fig, ax = plt.subplots()
    ax.scatter(0.0, 1.0, color='k', marker='o', label='test')
    ax.scatter(0.5, 0.0, color='r', marker='v', label='test')
    ax.legend()
    hand, lab = mlegend._get_legend_handles_labels([ax])
    assert len(lab) == 2
    fig, ax = plt.subplots()
    ax.scatter(0.0, 1.0, color='k', marker='o', label='test')
    ax.scatter(0.5, 0.0, color='k', marker='v', label='test')
    ax.legend()
    hand, lab = mlegend._get_legend_handles_labels([ax])
    assert len(lab) == 2


@image_comparison(['not_covering_scatter.png'])
def test_not_covering_scatter():
    colors = ['b', 'g', 'r']

    for n in range(3):
        plt.scatter([n], [n], color=colors[n])

    plt.legend(['foo', 'foo', 'foo'], loc='best')
    plt.gca().set_xlim(-0.5, 2.2)
    plt.gca().set_ylim(-0.5, 2.2)


@image_comparison(['not_covering_scatter_transform.png'])
def test_not_covering_scatter_transform():
    # Offsets point to top left, the default auto position
    offset = mtransforms.Affine2D().translate(-20, 20)
    x = np.linspace(0, 30, 1000)
    plt.plot(x, x)

    plt.scatter([20], [10], transform=offset + plt.gca().transData)

    plt.legend(['foo', 'bar'], loc='best')


def test_linecollection_scaled_dashes():
    lines1 = [[(0, .5), (.5, 1)], [(.3, .6), (.2, .2)]]
    lines2 = [[[0.7, .2], [.8, .4]], [[.5, .7], [.6, .1]]]
    lines3 = [[[0.6, .2], [.8, .4]], [[.5, .7], [.1, .1]]]
    lc1 = mcollections.LineCollection(lines1, linestyles="--", lw=3)
    lc2 = mcollections.LineCollection(lines2, linestyles="-.")
    lc3 = mcollections.LineCollection(lines3, linestyles=":", lw=.5)

    fig, ax = plt.subplots()
    ax.add_collection(lc1)
    ax.add_collection(lc2)
    ax.add_collection(lc3)

    leg = ax.legend([lc1, lc2, lc3], ["line1", "line2", 'line 3'])
    h1, h2, h3 = leg.legendHandles

    for oh, lh in zip((lc1, lc2, lc3), (h1, h2, h3)):
        assert oh.get_linestyles()[0][1] == lh._dashSeq
        assert oh.get_linestyles()[0][0] == lh._dashOffset


def test_handler_numpoints():
    """Test legend handler with numpoints <= 1."""
    # related to #6921 and PR #8478
    fig, ax = plt.subplots()
    ax.plot(range(5), label='test')
    ax.legend(numpoints=0.5)


def test_empty_bar_chart_with_legend():
    """Test legend when bar chart is empty with a label."""
    # related to issue #13003. Calling plt.legend() should not
    # raise an IndexError.
    plt.bar([], [], label='test')
    plt.legend()


def test_shadow_framealpha():
    # Test if framealpha is activated when shadow is True
    # and framealpha is not explicitly passed'''
    fig, ax = plt.subplots()
    ax.plot(range(100), label="test")
    leg = ax.legend(shadow=True, facecolor='w')
    assert leg.get_frame().get_alpha() == 1


def test_legend_title_empty():
    # test that if we don't set the legend title, that
    # it comes back as an empty string, and that it is not
    # visible:
    fig, ax = plt.subplots()
    ax.plot(range(10))
    leg = ax.legend()
    assert leg.get_title().get_text() == ""
    assert not leg.get_title().get_visible()


def test_legend_proper_window_extent():
    # test that legend returns the expected extent under various dpi...
    fig, ax = plt.subplots(dpi=100)
    ax.plot(range(10), label='Aardvark')
    leg = ax.legend()
    x01 = leg.get_window_extent(fig.canvas.get_renderer()).x0

    fig, ax = plt.subplots(dpi=200)
    ax.plot(range(10), label='Aardvark')
    leg = ax.legend()
    x02 = leg.get_window_extent(fig.canvas.get_renderer()).x0
    assert pytest.approx(x01*2, 0.1) == x02


def test_window_extent_cached_renderer():
    fig, ax = plt.subplots(dpi=100)
    ax.plot(range(10), label='Aardvark')
    leg = ax.legend()
    leg2 = fig.legend()
    fig.canvas.draw()
    # check that get_window_extent will use the cached renderer
    leg.get_window_extent()
    leg2.get_window_extent()


def test_legend_title_fontprop_fontsize():
    # test the title_fontsize kwarg
    plt.plot(range(10))
    with pytest.raises(ValueError):
        plt.legend(title='Aardvark', title_fontsize=22,
                   title_fontproperties={'family': 'serif', 'size': 22})

    leg = plt.legend(title='Aardvark', title_fontproperties=FontProperties(
                                       family='serif', size=22))
    assert leg.get_title().get_size() == 22

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flat
    axes[0].plot(range(10))
    leg0 = axes[0].legend(title='Aardvark', title_fontsize=22)
    assert leg0.get_title().get_fontsize() == 22
    axes[1].plot(range(10))
    leg1 = axes[1].legend(title='Aardvark',
                          title_fontproperties={'family': 'serif', 'size': 22})
    assert leg1.get_title().get_fontsize() == 22
    axes[2].plot(range(10))
    mpl.rcParams['legend.title_fontsize'] = None
    leg2 = axes[2].legend(title='Aardvark',
                          title_fontproperties={'family': 'serif'})
    assert leg2.get_title().get_fontsize() == mpl.rcParams['font.size']
    axes[3].plot(range(10))
    leg3 = axes[3].legend(title='Aardvark')
    assert leg3.get_title().get_fontsize() == mpl.rcParams['font.size']
    axes[4].plot(range(10))
    mpl.rcParams['legend.title_fontsize'] = 20
    leg4 = axes[4].legend(title='Aardvark',
                          title_fontproperties={'family': 'serif'})
    assert leg4.get_title().get_fontsize() == 20
    axes[5].plot(range(10))
    leg5 = axes[5].legend(title='Aardvark')
    assert leg5.get_title().get_fontsize() == 20


@pytest.mark.parametrize('color', ('red', 'none', (.5, .5, .5)))
def test_legend_labelcolor_single(color):
    # test labelcolor for a single color
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3')

    leg = ax.legend(labelcolor=color)
    for text in leg.get_texts():
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_list():
    # test labelcolor for a list of colors
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3')

    leg = ax.legend(labelcolor=['r', 'g', 'b'])
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_linecolor():
    # test the labelcolor for labelcolor='linecolor'
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', color='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', color='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', color='b')

    leg = ax.legend(labelcolor='linecolor')
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_markeredgecolor():
    # test the labelcolor for labelcolor='markeredgecolor'
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markeredgecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markeredgecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markeredgecolor='b')

    leg = ax.legend(labelcolor='markeredgecolor')
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_markerfacecolor():
    # test the labelcolor for labelcolor='markerfacecolor'
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markerfacecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markerfacecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markerfacecolor='b')

    leg = ax.legend(labelcolor='markerfacecolor')
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


@pytest.mark.parametrize('color', ('red', 'none', (.5, .5, .5)))
def test_legend_labelcolor_rcparam_single(color):
    # test the rcParams legend.labelcolor for a single color
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3')

    mpl.rcParams['legend.labelcolor'] = color
    leg = ax.legend()
    for text in leg.get_texts():
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_rcparam_linecolor():
    # test the rcParams legend.labelcolor for a linecolor
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', color='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', color='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', color='b')

    mpl.rcParams['legend.labelcolor'] = 'linecolor'
    leg = ax.legend()
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_rcparam_markeredgecolor():
    # test the labelcolor for labelcolor='markeredgecolor'
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markeredgecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markeredgecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markeredgecolor='b')

    mpl.rcParams['legend.labelcolor'] = 'markeredgecolor'
    leg = ax.legend()
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_rcparam_markeredgecolor_short():
    # test the labelcolor for labelcolor='markeredgecolor'
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markeredgecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markeredgecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markeredgecolor='b')

    mpl.rcParams['legend.labelcolor'] = 'mec'
    leg = ax.legend()
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_rcparam_markerfacecolor():
    # test the labelcolor for labelcolor='markeredgecolor'
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markerfacecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markerfacecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markerfacecolor='b')

    mpl.rcParams['legend.labelcolor'] = 'markerfacecolor'
    leg = ax.legend()
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_rcparam_markerfacecolor_short():
    # test the labelcolor for labelcolor='markeredgecolor'
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markerfacecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markerfacecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markerfacecolor='b')

    mpl.rcParams['legend.labelcolor'] = 'mfc'
    leg = ax.legend()
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_get_set_draggable():
    legend = plt.legend()
    assert not legend.get_draggable()
    legend.set_draggable(True)
    assert legend.get_draggable()
    legend.set_draggable(False)
    assert not legend.get_draggable()


def test_alpha_handles():
    x, n, hh = plt.hist([1, 2, 3], alpha=0.25, label='data', color='red')
    legend = plt.legend()
    for lh in legend.legendHandles:
        lh.set_alpha(1.0)
    assert lh.get_facecolor()[:-1] == hh[1].get_facecolor()[:-1]
    assert lh.get_edgecolor()[:-1] == hh[1].get_edgecolor()[:-1]


@pytest.mark.skipif(
    not mpl.checkdep_usetex(True),
    reason="This test needs a TeX installation")
def test_usetex_no_warn(caplog):
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern'
    mpl.rcParams['text.usetex'] = True

    fig, ax = plt.subplots()
    ax.plot(0, 0, label='input')
    ax.legend(title="My legend")

    fig.canvas.draw()
    assert "Font family ['serif'] not found." not in caplog.text


def test_warn_big_data_best_loc():
    fig, ax = plt.subplots()
    fig.canvas.draw()  # So that we can call draw_artist later.
    for idx in range(1000):
        ax.plot(np.arange(5000), label=idx)
    with rc_context({'legend.loc': 'best'}):
        legend = ax.legend()
    with pytest.warns(UserWarning) as records:
        fig.draw_artist(legend)  # Don't bother drawing the lines -- it's slow.
    # The _find_best_position method of Legend is called twice, duplicating
    # the warning message.
    assert len(records) == 2
    for record in records:
        assert str(record.message) == (
            'Creating legend with loc="best" can be slow with large '
            'amounts of data.')


def test_no_warn_big_data_when_loc_specified():
    fig, ax = plt.subplots()
    fig.canvas.draw()
    for idx in range(1000):
        ax.plot(np.arange(5000), label=idx)
    legend = ax.legend('best')
    fig.draw_artist(legend)  # Check that no warning is emitted.


@pytest.mark.parametrize('label_array', [['low', 'high'],
                                         ('low', 'high'),
                                         np.array(['low', 'high'])])
def test_plot_multiple_input_multiple_label(label_array):
    # test ax.plot() with multidimensional input
    # and multiple labels
    x = [1, 2, 3]
    y = [[1, 2],
         [2, 5],
         [4, 9]]

    fig, ax = plt.subplots()
    ax.plot(x, y, label=label_array)
    leg = ax.legend()
    legend_texts = [entry.get_text() for entry in leg.get_texts()]
    assert legend_texts == ['low', 'high']


@pytest.mark.parametrize('label', ['one', 1, int])
def test_plot_multiple_input_single_label(label):
    # test ax.plot() with multidimensional input
    # and single label
    x = [1, 2, 3]
    y = [[1, 2],
         [2, 5],
         [4, 9]]

    fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    leg = ax.legend()
    legend_texts = [entry.get_text() for entry in leg.get_texts()]
    assert legend_texts == [str(label)] * 2


@pytest.mark.parametrize('label_array', [['low', 'high'],
                                         ('low', 'high'),
                                         np.array(['low', 'high'])])
def test_plot_single_input_multiple_label(label_array):
    # test ax.plot() with 1D array like input
    # and iterable label
    x = [1, 2, 3]
    y = [2, 5, 6]
    fig, ax = plt.subplots()
    ax.plot(x, y, label=label_array)
    leg = ax.legend()
    assert len(leg.get_texts()) == 1
    assert leg.get_texts()[0].get_text() == str(label_array)


def test_plot_multiple_label_incorrect_length_exception():
    # check that excepton is raised if multiple labels
    # are given, but number of on labels != number of lines
    with pytest.raises(ValueError):
        x = [1, 2, 3]
        y = [[1, 2],
             [2, 5],
             [4, 9]]
        label = ['high', 'low', 'medium']
        fig, ax = plt.subplots()
        ax.plot(x, y, label=label)


def test_legend_face_edgecolor():
    # Smoke test for PolyCollection legend handler with 'face' edgecolor.
    fig, ax = plt.subplots()
    ax.fill_between([0, 1, 2], [1, 2, 3], [2, 3, 4],
                    facecolor='r', edgecolor='face', label='Fill')
    ax.legend()


def test_legend_text_axes():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4], label='line')
    leg = ax.legend()

    assert leg.axes is ax
    assert leg.get_texts()[0].axes is ax


def test_handlerline2d():
    # Test marker consistency for monolithic Line2D legend handler (#11357).
    fig, ax = plt.subplots()
    ax.scatter([0, 1], [0, 1], marker="v")
    handles = [mlines.Line2D([0], [0], marker="v")]
    leg = ax.legend(handles, ["Aardvark"], numpoints=1)
    assert handles[0].get_marker() == leg.legendHandles[0].get_marker()


def test_subfigure_legend():
    # Test that legend can be added to subfigure (#20723)
    subfig = plt.figure().subfigures()
    ax = subfig.subplots()
    ax.plot([0, 1], [0, 1], label="line")
    leg = subfig.legend()
    assert leg.figure is subfig
