from collections import namedtuple
import io

import numpy as np
from numpy.testing import assert_allclose
import pytest

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton

from matplotlib.offsetbox import (
    AnchoredOffsetbox, AnnotationBbox, AnchoredText, DrawingArea, OffsetBox,
    OffsetImage, TextArea, _get_packed_offsets)


@image_comparison(['offsetbox_clipping'], remove_text=True)
def test_offsetbox_clipping():
    # - create a plot
    # - put an AnchoredOffsetbox with a child DrawingArea
    #   at the center of the axes
    # - give the DrawingArea a gray background
    # - put a black line across the bounds of the DrawingArea
    # - see that the black line is clipped to the edges of
    #   the DrawingArea.
    fig, ax = plt.subplots()
    size = 100
    da = DrawingArea(size, size, clip=True)
    bg = mpatches.Rectangle((0, 0), size, size,
                            facecolor='#CCCCCC',
                            edgecolor='None',
                            linewidth=0)
    line = mlines.Line2D([-size*.5, size*1.5], [size/2, size/2],
                         color='black',
                         linewidth=10)
    anchored_box = AnchoredOffsetbox(
        loc='center',
        child=da,
        pad=0.,
        frameon=False,
        bbox_to_anchor=(.5, .5),
        bbox_transform=ax.transAxes,
        borderpad=0.)

    da.add_artist(bg)
    da.add_artist(line)
    ax.add_artist(anchored_box)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))


def test_offsetbox_clip_children():
    # - create a plot
    # - put an AnchoredOffsetbox with a child DrawingArea
    #   at the center of the axes
    # - give the DrawingArea a gray background
    # - put a black line across the bounds of the DrawingArea
    # - see that the black line is clipped to the edges of
    #   the DrawingArea.
    fig, ax = plt.subplots()
    size = 100
    da = DrawingArea(size, size, clip=True)
    bg = mpatches.Rectangle((0, 0), size, size,
                            facecolor='#CCCCCC',
                            edgecolor='None',
                            linewidth=0)
    line = mlines.Line2D([-size*.5, size*1.5], [size/2, size/2],
                         color='black',
                         linewidth=10)
    anchored_box = AnchoredOffsetbox(
        loc='center',
        child=da,
        pad=0.,
        frameon=False,
        bbox_to_anchor=(.5, .5),
        bbox_transform=ax.transAxes,
        borderpad=0.)

    da.add_artist(bg)
    da.add_artist(line)
    ax.add_artist(anchored_box)

    fig.canvas.draw()
    assert not fig.stale
    da.clip_children = True
    assert fig.stale


def test_offsetbox_loc_codes():
    # Check that valid string location codes all work with an AnchoredOffsetbox
    codes = {'upper right': 1,
             'upper left': 2,
             'lower left': 3,
             'lower right': 4,
             'right': 5,
             'center left': 6,
             'center right': 7,
             'lower center': 8,
             'upper center': 9,
             'center': 10,
             }
    fig, ax = plt.subplots()
    da = DrawingArea(100, 100)
    for code in codes:
        anchored_box = AnchoredOffsetbox(loc=code, child=da)
        ax.add_artist(anchored_box)
    fig.canvas.draw()


def test_expand_with_tight_layout():
    # Check issue reported in #10476, and updated due to #10784
    fig, ax = plt.subplots()

    d1 = [1, 2]
    d2 = [2, 1]
    ax.plot(d1, label='series 1')
    ax.plot(d2, label='series 2')
    ax.legend(ncol=2, mode='expand')

    fig.tight_layout()  # where the crash used to happen


@pytest.mark.parametrize('wd_list',
                         ([(150, 1)], [(150, 1)]*3, [(0.1, 1)], [(0.1, 1)]*2))
@pytest.mark.parametrize('total', (250, 100, 0, -1, None))
@pytest.mark.parametrize('sep', (250, 1, 0, -1))
@pytest.mark.parametrize('mode', ("expand", "fixed", "equal"))
def test_get_packed_offsets(wd_list, total, sep, mode):
    # Check a (rather arbitrary) set of parameters due to successive similar
    # issue tickets (at least #10476 and #10784) related to corner cases
    # triggered inside this function when calling higher-level functions
    # (e.g. `Axes.legend`).
    # These are just some additional smoke tests. The output is untested.
    _get_packed_offsets(wd_list, total, sep, mode=mode)


_Params = namedtuple('_params', 'wd_list, total, sep, expected')


@pytest.mark.parametrize('wd_list, total, sep, expected', [
    _Params(  # total=None
        [(3, 0), (1, 0), (2, 0)], total=None, sep=1, expected=(8, [0, 4, 6])),
    _Params(  # total larger than required
        [(3, 0), (1, 0), (2, 0)], total=10, sep=1, expected=(10, [0, 4, 6])),
    _Params(  # total smaller than required
        [(3, 0), (1, 0), (2, 0)], total=5, sep=1, expected=(5, [0, 4, 6])),
])
def test_get_packed_offsets_fixed(wd_list, total, sep, expected):
    result = _get_packed_offsets(wd_list, total, sep, mode='fixed')
    assert result[0] == expected[0]
    assert_allclose(result[1], expected[1])


@pytest.mark.parametrize('wd_list, total, sep, expected', [
    _Params(  # total=None (implicit 1)
        [(.1, 0)] * 3, total=None, sep=None, expected=(1, [0, .45, .9])),
    _Params(  # total larger than sum of widths
        [(3, 0), (1, 0), (2, 0)], total=10, sep=1, expected=(10, [0, 5, 8])),
    _Params(  # total smaller sum of widths: overlapping boxes
        [(3, 0), (1, 0), (2, 0)], total=5, sep=1, expected=(5, [0, 2.5, 3])),
])
def test_get_packed_offsets_expand(wd_list, total, sep, expected):
    result = _get_packed_offsets(wd_list, total, sep, mode='expand')
    assert result[0] == expected[0]
    assert_allclose(result[1], expected[1])


@pytest.mark.parametrize('wd_list, total, sep, expected', [
    _Params(  # total larger than required
        [(3, 0), (2, 0), (1, 0)], total=6, sep=None, expected=(6, [0, 2, 4])),
    _Params(  # total smaller sum of widths: overlapping boxes
        [(3, 0), (2, 0), (1, 0), (.5, 0)], total=2, sep=None,
        expected=(2, [0, 0.5, 1, 1.5])),
    _Params(  # total larger than required
        [(.5, 0), (1, 0), (.2, 0)], total=None, sep=1,
        expected=(6, [0, 2, 4])),
    # the case total=None, sep=None is tested separately below
])
def test_get_packed_offsets_equal(wd_list, total, sep, expected):
    result = _get_packed_offsets(wd_list, total, sep, mode='equal')
    assert result[0] == expected[0]
    assert_allclose(result[1], expected[1])


def test_get_packed_offsets_equal_total_none_sep_none():
    with pytest.raises(ValueError):
        _get_packed_offsets([(1, 0)] * 3, total=None, sep=None, mode='equal')


@pytest.mark.parametrize('child_type', ['draw', 'image', 'text'])
@pytest.mark.parametrize('boxcoords',
                         ['axes fraction', 'axes pixels', 'axes points',
                          'data'])
def test_picking(child_type, boxcoords):
    # These all take up approximately the same area.
    if child_type == 'draw':
        picking_child = DrawingArea(5, 5)
        picking_child.add_artist(mpatches.Rectangle((0, 0), 5, 5, linewidth=0))
    elif child_type == 'image':
        im = np.ones((5, 5))
        im[2, 2] = 0
        picking_child = OffsetImage(im)
    elif child_type == 'text':
        picking_child = TextArea('\N{Black Square}', textprops={'fontsize': 5})
    else:
        assert False, f'Unknown picking child type {child_type}'

    fig, ax = plt.subplots()
    ab = AnnotationBbox(picking_child, (0.5, 0.5), boxcoords=boxcoords)
    ab.set_picker(True)
    ax.add_artist(ab)

    calls = []
    fig.canvas.mpl_connect('pick_event', lambda event: calls.append(event))

    # Annotation should be picked by an event occurring at its center.
    if boxcoords == 'axes points':
        x, y = ax.transAxes.transform_point((0, 0))
        x += 0.5 * fig.dpi / 72
        y += 0.5 * fig.dpi / 72
    elif boxcoords == 'axes pixels':
        x, y = ax.transAxes.transform_point((0, 0))
        x += 0.5
        y += 0.5
    else:
        x, y = ax.transAxes.transform_point((0.5, 0.5))
    fig.canvas.draw()
    calls.clear()
    fig.canvas.button_press_event(x, y, MouseButton.LEFT)
    assert len(calls) == 1 and calls[0].artist == ab

    # Annotation should *not* be picked by an event at its original center
    # point when the limits have changed enough to hide the *xy* point.
    ax.set_xlim(-1, 0)
    ax.set_ylim(-1, 0)
    fig.canvas.draw()
    calls.clear()
    fig.canvas.button_press_event(x, y, MouseButton.LEFT)
    assert len(calls) == 0


@image_comparison(['anchoredtext_align.png'], remove_text=True, style='mpl20')
def test_anchoredtext_horizontal_alignment():
    fig, ax = plt.subplots()

    text0 = AnchoredText("test\ntest long text", loc="center left",
                         pad=0.2, prop={"ha": "left"})
    ax.add_artist(text0)
    text1 = AnchoredText("test\ntest long text", loc="center",
                         pad=0.2, prop={"ha": "center"})
    ax.add_artist(text1)
    text2 = AnchoredText("test\ntest long text", loc="center right",
                         pad=0.2, prop={"ha": "right"})
    ax.add_artist(text2)


def test_annotationbbox_extents():
    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

    ax.axis([0, 1, 0, 1])

    an1 = ax.annotate("Annotation", xy=(.9, .9), xytext=(1.1, 1.1),
                      arrowprops=dict(arrowstyle="->"), clip_on=False,
                      va="baseline", ha="left")

    da = DrawingArea(20, 20, 0, 0, clip=True)
    p = mpatches.Circle((-10, 30), 32)
    da.add_artist(p)

    ab3 = AnnotationBbox(da, [.5, .5], xybox=(-0.2, 0.5), xycoords='data',
                         boxcoords="axes fraction", box_alignment=(0., .5),
                         arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab3)

    im = OffsetImage(np.random.rand(10, 10), zoom=3)
    im.image.axes = ax
    ab6 = AnnotationBbox(im, (0.5, -.3), xybox=(0, 75),
                         xycoords='axes fraction',
                         boxcoords="offset points", pad=0.3,
                         arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab6)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Test Annotation
    bb1w = an1.get_window_extent(renderer)
    bb1e = an1.get_tightbbox(renderer)

    target1 = [332.9, 242.8, 467.0, 298.9]
    assert_allclose(bb1w.extents, target1, atol=2)
    assert_allclose(bb1e.extents, target1, atol=2)

    # Test AnnotationBbox
    bb3w = ab3.get_window_extent(renderer)
    bb3e = ab3.get_tightbbox(renderer)

    target3 = [-17.6, 129.0, 200.7, 167.9]
    assert_allclose(bb3w.extents, target3, atol=2)
    assert_allclose(bb3e.extents, target3, atol=2)

    bb6w = ab6.get_window_extent(renderer)
    bb6e = ab6.get_tightbbox(renderer)

    target6 = [180.0, -32.0, 230.0, 92.9]
    assert_allclose(bb6w.extents, target6, atol=2)
    assert_allclose(bb6e.extents, target6, atol=2)

    # Test bbox_inches='tight'
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    shape = plt.imread(buf).shape
    targetshape = (350, 504, 4)
    assert_allclose(shape, targetshape, atol=2)

    # Simple smoke test for tight_layout, to make sure it does not error out.
    fig.canvas.draw()
    fig.tight_layout()
    fig.canvas.draw()


def test_zorder():
    assert OffsetBox(zorder=42).zorder == 42
