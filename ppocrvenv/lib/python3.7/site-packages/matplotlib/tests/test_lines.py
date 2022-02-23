"""
Tests specific to the lines module.
"""

import itertools
import timeit
from types import SimpleNamespace

from cycler import cycler
import numpy as np
from numpy.testing import assert_array_equal
import pytest

import matplotlib
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import image_comparison, check_figures_equal


def test_segment_hits():
    """Test a problematic case."""
    cx, cy = 553, 902
    x, y = np.array([553., 553.]), np.array([95., 947.])
    radius = 6.94
    assert_array_equal(mlines.segment_hits(cx, cy, x, y, radius), [0])


# Runtimes on a loaded system are inherently flaky. Not so much that a rerun
# won't help, hopefully.
@pytest.mark.flaky(reruns=3)
def test_invisible_Line_rendering():
    """
    GitHub issue #1256 identified a bug in Line.draw method

    Despite visibility attribute set to False, the draw method was not
    returning early enough and some pre-rendering code was executed
    though not necessary.

    Consequence was an excessive draw time for invisible Line instances
    holding a large number of points (Npts> 10**6)
    """
    # Creates big x and y data:
    N = 10**7
    x = np.linspace(0, 1, N)
    y = np.random.normal(size=N)

    # Create a plot figure:
    fig = plt.figure()
    ax = plt.subplot()

    # Create a "big" Line instance:
    l = mlines.Line2D(x, y)
    l.set_visible(False)
    # but don't add it to the Axis instance `ax`

    # [here Interactive panning and zooming is pretty responsive]
    # Time the canvas drawing:
    t_no_line = min(timeit.repeat(fig.canvas.draw, number=1, repeat=3))
    # (gives about 25 ms)

    # Add the big invisible Line:
    ax.add_line(l)

    # [Now interactive panning and zooming is very slow]
    # Time the canvas drawing:
    t_invisible_line = min(timeit.repeat(fig.canvas.draw, number=1, repeat=3))
    # gives about 290 ms for N = 10**7 pts

    slowdown_factor = t_invisible_line / t_no_line
    slowdown_threshold = 2  # trying to avoid false positive failures
    assert slowdown_factor < slowdown_threshold


def test_set_line_coll_dash():
    fig, ax = plt.subplots()
    np.random.seed(0)
    # Testing setting linestyles for line collections.
    # This should not produce an error.
    ax.contour(np.random.randn(20, 30), linestyles=[(0, (3, 3))])


@image_comparison(['line_dashes'], remove_text=True)
def test_line_dashes():
    fig, ax = plt.subplots()

    ax.plot(range(10), linestyle=(0, (3, 3)), lw=5)


def test_line_colors():
    fig, ax = plt.subplots()
    ax.plot(range(10), color='none')
    ax.plot(range(10), color='r')
    ax.plot(range(10), color='.3')
    ax.plot(range(10), color=(1, 0, 0, 1))
    ax.plot(range(10), color=(1, 0, 0))
    fig.canvas.draw()


def test_valid_colors():
    line = mlines.Line2D([], [])
    with pytest.raises(ValueError):
        line.set_color("foobar")


def test_linestyle_variants():
    fig, ax = plt.subplots()
    for ls in ["-", "solid", "--", "dashed",
               "-.", "dashdot", ":", "dotted"]:
        ax.plot(range(10), linestyle=ls)
    fig.canvas.draw()


def test_valid_linestyles():
    line = mlines.Line2D([], [])
    with pytest.raises(ValueError):
        line.set_linestyle('aardvark')


@image_comparison(['drawstyle_variants.png'], remove_text=True)
def test_drawstyle_variants():
    fig, axs = plt.subplots(6)
    dss = ["default", "steps-mid", "steps-pre", "steps-post", "steps", None]
    # We want to check that drawstyles are properly handled even for very long
    # lines (for which the subslice optimization is on); however, we need
    # to zoom in so that the difference between the drawstyles is actually
    # visible.
    for ax, ds in zip(axs.flat, dss):
        ax.plot(range(2000), drawstyle=ds)
        ax.set(xlim=(0, 2), ylim=(0, 2))


@check_figures_equal(extensions=('png',))
def test_no_subslice_with_transform(fig_ref, fig_test):
    ax = fig_ref.add_subplot()
    x = np.arange(2000)
    ax.plot(x + 2000, x)

    ax = fig_test.add_subplot()
    t = mtransforms.Affine2D().translate(2000.0, 0.0)
    ax.plot(x, x, transform=t+ax.transData)


def test_valid_drawstyles():
    line = mlines.Line2D([], [])
    with pytest.raises(ValueError):
        line.set_drawstyle('foobar')


def test_set_drawstyle():
    x = np.linspace(0, 2*np.pi, 10)
    y = np.sin(x)

    fig, ax = plt.subplots()
    line, = ax.plot(x, y)
    line.set_drawstyle("steps-pre")
    assert len(line.get_path().vertices) == 2*len(x)-1

    line.set_drawstyle("default")
    assert len(line.get_path().vertices) == len(x)


@image_comparison(['line_collection_dashes'], remove_text=True, style='mpl20')
def test_set_line_coll_dash_image():
    fig, ax = plt.subplots()
    np.random.seed(0)
    ax.contour(np.random.randn(20, 30), linestyles=[(0, (3, 3))])


@image_comparison(['marker_fill_styles.png'], remove_text=True)
def test_marker_fill_styles():
    colors = itertools.cycle([[0, 0, 1], 'g', '#ff0000', 'c', 'm', 'y',
                              np.array([0, 0, 0])])
    altcolor = 'lightgreen'

    y = np.array([1, 1])
    x = np.array([0, 9])
    fig, ax = plt.subplots()

    for j, marker in enumerate(mlines.Line2D.filled_markers):
        for i, fs in enumerate(mlines.Line2D.fillStyles):
            color = next(colors)
            ax.plot(j * 10 + x, y + i + .5 * (j % 2),
                    marker=marker,
                    markersize=20,
                    markerfacecoloralt=altcolor,
                    fillstyle=fs,
                    label=fs,
                    linewidth=5,
                    color=color,
                    markeredgecolor=color,
                    markeredgewidth=2)

    ax.set_ylim([0, 7.5])
    ax.set_xlim([-5, 155])


def test_markerfacecolor_fillstyle():
    """Test that markerfacecolor does not override fillstyle='none'."""
    l, = plt.plot([1, 3, 2], marker=MarkerStyle('o', fillstyle='none'),
                  markerfacecolor='red')
    assert l.get_fillstyle() == 'none'
    assert l.get_markerfacecolor() == 'none'


@image_comparison(['scaled_lines'], style='default')
def test_lw_scaling():
    th = np.linspace(0, 32)
    fig, ax = plt.subplots()
    lins_styles = ['dashed', 'dotted', 'dashdot']
    cy = cycler(matplotlib.rcParams['axes.prop_cycle'])
    for j, (ls, sty) in enumerate(zip(lins_styles, cy)):
        for lw in np.linspace(.5, 10, 10):
            ax.plot(th, j*np.ones(50) + .1 * lw, linestyle=ls, lw=lw, **sty)


def test_nan_is_sorted():
    line = mlines.Line2D([], [])
    assert line._is_sorted(np.array([1, 2, 3]))
    assert line._is_sorted(np.array([1, np.nan, 3]))
    assert not line._is_sorted([3, 5] + [np.nan] * 100 + [0, 2])


@check_figures_equal()
def test_step_markers(fig_test, fig_ref):
    fig_test.subplots().step([0, 1], "-o")
    fig_ref.subplots().plot([0, 0, 1], [0, 1, 1], "-o", markevery=[0, 2])


@pytest.mark.parametrize("parent", ["figure", "axes"])
@check_figures_equal(extensions=('png',))
def test_markevery(fig_test, fig_ref, parent):
    np.random.seed(42)
    x = np.linspace(0, 1, 14)
    y = np.random.rand(len(x))

    cases_test = [None, 4, (2, 5), [1, 5, 11],
                  [0, -1], slice(5, 10, 2),
                  np.arange(len(x))[y > 0.5],
                  0.3, (0.3, 0.4)]
    cases_ref = ["11111111111111", "10001000100010", "00100001000010",
                 "01000100000100", "10000000000001", "00000101010000",
                 "01110001110110", "11011011011110", "01010011011101"]

    if parent == "figure":
        # float markevery ("relative to axes size") is not supported.
        cases_test = cases_test[:-2]
        cases_ref = cases_ref[:-2]

        def add_test(x, y, *, markevery):
            fig_test.add_artist(
                mlines.Line2D(x, y, marker="o", markevery=markevery))

        def add_ref(x, y, *, markevery):
            fig_ref.add_artist(
                mlines.Line2D(x, y, marker="o", markevery=markevery))

    elif parent == "axes":
        axs_test = iter(fig_test.subplots(3, 3).flat)
        axs_ref = iter(fig_ref.subplots(3, 3).flat)

        def add_test(x, y, *, markevery):
            next(axs_test).plot(x, y, "-gD", markevery=markevery)

        def add_ref(x, y, *, markevery):
            next(axs_ref).plot(x, y, "-gD", markevery=markevery)

    for case in cases_test:
        add_test(x, y, markevery=case)

    for case in cases_ref:
        me = np.array(list(case)).astype(int).astype(bool)
        add_ref(x, y, markevery=me)


def test_markevery_figure_line_unsupported_relsize():
    fig = plt.figure()
    fig.add_artist(mlines.Line2D([0, 1], [0, 1], marker="o", markevery=.5))
    with pytest.raises(ValueError):
        fig.canvas.draw()


def test_marker_as_markerstyle():
    fig, ax = plt.subplots()
    line, = ax.plot([2, 4, 3], marker=MarkerStyle("D"))
    fig.canvas.draw()
    assert line.get_marker() == "D"

    # continue with smoke tests:
    line.set_marker("s")
    fig.canvas.draw()
    line.set_marker(MarkerStyle("o"))
    fig.canvas.draw()
    # test Path roundtrip
    triangle1 = Path([[-1., -1.], [1., -1.], [0., 2.], [0., 0.]], closed=True)
    line2, = ax.plot([1, 3, 2], marker=MarkerStyle(triangle1), ms=22)
    line3, = ax.plot([0, 2, 1], marker=triangle1, ms=22)

    assert_array_equal(line2.get_marker().vertices, triangle1.vertices)
    assert_array_equal(line3.get_marker().vertices, triangle1.vertices)


@check_figures_equal()
def test_odd_dashes(fig_test, fig_ref):
    fig_test.add_subplot().plot([1, 2], dashes=[1, 2, 3])
    fig_ref.add_subplot().plot([1, 2], dashes=[1, 2, 3, 1, 2, 3])


def test_picking():
    fig, ax = plt.subplots()
    mouse_event = SimpleNamespace(x=fig.bbox.width // 2,
                                  y=fig.bbox.height // 2 + 15)

    # Default pickradius is 5, so event should not pick this line.
    l0, = ax.plot([0, 1], [0, 1], picker=True)
    found, indices = l0.contains(mouse_event)
    assert not found

    # But with a larger pickradius, this should be picked.
    l1, = ax.plot([0, 1], [0, 1], picker=True, pickradius=20)
    found, indices = l1.contains(mouse_event)
    assert found
    assert_array_equal(indices['ind'], [0])

    # And if we modify the pickradius after creation, it should work as well.
    l2, = ax.plot([0, 1], [0, 1], picker=True)
    found, indices = l2.contains(mouse_event)
    assert not found
    l2.set_pickradius(20)
    found, indices = l2.contains(mouse_event)
    assert found
    assert_array_equal(indices['ind'], [0])
