from matplotlib._api.deprecation import MatplotlibDeprecationWarning
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing.widgets import do_event, get_ax, mock_event

from numpy.testing import assert_allclose

import pytest


def check_rectangle(**kwargs):
    ax = get_ax()

    def onselect(epress, erelease):
        ax._got_onselect = True
        assert epress.xdata == 100
        assert epress.ydata == 100
        assert erelease.xdata == 199
        assert erelease.ydata == 199

    tool = widgets.RectangleSelector(ax, onselect, **kwargs)
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    do_event(tool, 'onmove', xdata=199, ydata=199, button=1)

    # purposely drag outside of axis for release
    do_event(tool, 'release', xdata=250, ydata=250, button=1)

    if kwargs.get('drawtype', None) not in ['line', 'none']:
        assert_allclose(tool.geometry,
                        [[100., 100, 199, 199, 100],
                         [100, 199, 199, 100, 100]],
                        err_msg=tool.geometry)

    assert ax._got_onselect


def test_rectangle_selector():
    check_rectangle()

    with pytest.warns(
        MatplotlibDeprecationWarning,
            match="Support for drawtype='line' is deprecated"):
        check_rectangle(drawtype='line', useblit=False)

    check_rectangle(useblit=True, button=1)

    with pytest.warns(
        MatplotlibDeprecationWarning,
            match="Support for drawtype='none' is deprecated"):
        check_rectangle(drawtype='none', minspanx=10, minspany=10)

    check_rectangle(minspanx=10, minspany=10, spancoords='pixels')
    check_rectangle(props=dict(fill=True))


def _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new,
                      use_key=None):
    do_event(tool, 'press', xdata=xdata, ydata=ydata, button=1)
    if use_key is not None:
        do_event(tool, 'on_key_press', key=use_key)
    do_event(tool, 'onmove', xdata=xdata_new, ydata=ydata_new, button=1)
    if use_key is not None:
        do_event(tool, 'on_key_release', key=use_key)
    do_event(tool, 'release', xdata=xdata_new, ydata=ydata_new, button=1)

    return tool


@pytest.mark.parametrize('drag_from_anywhere, new_center',
                         [[True, (60, 75)],
                          [False, (30, 20)]])
def test_rectangle_drag(drag_from_anywhere, new_center):
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.RectangleSelector(ax, onselect, interactive=True,
                                     drag_from_anywhere=drag_from_anywhere)
    # Create rectangle
    do_event(tool, 'press', xdata=0, ydata=10, button=1)
    do_event(tool, 'onmove', xdata=100, ydata=120, button=1)
    do_event(tool, 'release', xdata=100, ydata=120, button=1)
    assert tool.center == (50, 65)
    # Drag inside rectangle, but away from centre handle
    #
    # If drag_from_anywhere == True, this will move the rectangle by (10, 10),
    # giving it a new center of (60, 75)
    #
    # If drag_from_anywhere == False, this will create a new rectangle with
    # center (30, 20)
    do_event(tool, 'press', xdata=25, ydata=15, button=1)
    do_event(tool, 'onmove', xdata=35, ydata=25, button=1)
    do_event(tool, 'release', xdata=35, ydata=25, button=1)
    assert tool.center == new_center
    # Check that in both cases, dragging outside the rectangle draws a new
    # rectangle
    do_event(tool, 'press', xdata=175, ydata=185, button=1)
    do_event(tool, 'onmove', xdata=185, ydata=195, button=1)
    do_event(tool, 'release', xdata=185, ydata=195, button=1)
    assert tool.center == (180, 190)


def test_rectangle_selector_set_props_handle_props():
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.RectangleSelector(ax, onselect, interactive=True,
                                     props=dict(facecolor='b', alpha=0.2),
                                     handle_props=dict(alpha=0.5))
    # Create rectangle
    do_event(tool, 'press', xdata=0, ydata=10, button=1)
    do_event(tool, 'onmove', xdata=100, ydata=120, button=1)
    do_event(tool, 'release', xdata=100, ydata=120, button=1)

    artist = tool._selection_artist
    assert artist.get_facecolor() == mcolors.to_rgba('b', alpha=0.2)
    tool.set_props(facecolor='r', alpha=0.3)
    assert artist.get_facecolor() == mcolors.to_rgba('r', alpha=0.3)

    for artist in tool._handles_artists:
        assert artist.get_markeredgecolor() == 'black'
        assert artist.get_alpha() == 0.5
    tool.set_handle_props(markeredgecolor='r', alpha=0.3)
    for artist in tool._handles_artists:
        assert artist.get_markeredgecolor() == 'r'
        assert artist.get_alpha() == 0.3


def test_rectangle_resize():
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.RectangleSelector(ax, onselect, interactive=True)
    # Create rectangle
    _resize_rectangle(tool, 0, 10, 100, 120)
    assert tool.extents == (0.0, 100.0, 10.0, 120.0)

    # resize NE handle
    extents = tool.extents
    xdata, ydata = extents[1], extents[3]
    xdata_new, ydata_new = xdata + 10, ydata + 5
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (extents[0], xdata_new, extents[2], ydata_new)

    # resize E handle
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdata_new, ydata_new = xdata + 10, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (extents[0], xdata_new, extents[2], extents[3])

    # resize W handle
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdata_new, ydata_new = xdata + 15, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (xdata_new, extents[1], extents[2], extents[3])

    # resize SW handle
    extents = tool.extents
    xdata, ydata = extents[0], extents[2]
    xdata_new, ydata_new = xdata + 20, ydata + 25
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (xdata_new, extents[1], ydata_new, extents[3])


@pytest.mark.parametrize('use_default_state', [True, False])
def test_rectangle_resize_center(use_default_state):
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.RectangleSelector(ax, onselect, interactive=True)
    # Create rectangle
    _resize_rectangle(tool, 70, 65, 125, 130)
    assert tool.extents == (70.0, 125.0, 65.0, 130.0)

    if use_default_state:
        tool._default_state.add('center')
        use_key = None
    else:
        use_key = 'control'

    # resize NE handle
    extents = tool.extents
    xdata, ydata = extents[1], extents[3]
    xdiff, ydiff = 10, 5
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (extents[0] - xdiff, xdata_new,
                            extents[2] - ydiff, ydata_new)

    # resize E handle
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 10
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (extents[0] - xdiff, xdata_new,
                            extents[2], extents[3])

    # resize E handle negative diff
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -20
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (extents[0] - xdiff, xdata_new,
                            extents[2], extents[3])

    # resize W handle
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 15
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (xdata_new, extents[1] - xdiff,
                            extents[2], extents[3])

    # resize W handle negative diff
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -25
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (xdata_new, extents[1] - xdiff,
                            extents[2], extents[3])

    # resize SW handle
    extents = tool.extents
    xdata, ydata = extents[0], extents[2]
    xdiff, ydiff = 20, 25
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (xdata_new, extents[1] - xdiff,
                            ydata_new, extents[3] - ydiff)


@pytest.mark.parametrize('use_default_state', [True, False])
def test_rectangle_resize_square(use_default_state):
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.RectangleSelector(ax, onselect, interactive=True)
    # Create rectangle
    _resize_rectangle(tool, 70, 65, 120, 115)
    assert tool.extents == (70.0, 120.0, 65.0, 115.0)

    if use_default_state:
        tool._default_state.add('square')
        use_key = None
    else:
        use_key = 'shift'

    # resize NE handle
    extents = tool.extents
    xdata, ydata = extents[1], extents[3]
    xdiff, ydiff = 10, 5
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (extents[0], xdata_new,
                            extents[2], extents[3] + xdiff)

    # resize E handle
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 10
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (extents[0], xdata_new,
                            extents[2], extents[3] + xdiff)

    # resize E handle negative diff
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -20
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (extents[0], xdata_new,
                            extents[2], extents[3] + xdiff)

    # resize W handle
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 15
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (xdata_new, extents[1],
                            extents[2], extents[3] - xdiff)

    # resize W handle negative diff
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -25
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (xdata_new, extents[1],
                            extents[2], extents[3] - xdiff)

    # resize SW handle
    extents = tool.extents
    xdata, ydata = extents[0], extents[2]
    xdiff, ydiff = 20, 25
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new, use_key)
    assert tool.extents == (extents[0] + ydiff, extents[1],
                            ydata_new, extents[3])


def test_rectangle_resize_square_center():
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.RectangleSelector(ax, onselect, interactive=True)
    # Create rectangle
    _resize_rectangle(tool, 70, 65, 120, 115)
    tool._default_state.add('square')
    tool._default_state.add('center')
    assert tool.extents == (70.0, 120.0, 65.0, 115.0)

    # resize NE handle
    extents = tool.extents
    xdata, ydata = extents[1], extents[3]
    xdiff, ydiff = 10, 5
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (extents[0] - xdiff, xdata_new,
                            extents[2] - xdiff, extents[3] + xdiff)

    # resize E handle
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 10
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (extents[0] - xdiff, xdata_new,
                            extents[2] - xdiff, extents[3] + xdiff)

    # resize E handle negative diff
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -20
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (extents[0] - xdiff, xdata_new,
                            extents[2] - xdiff, extents[3] + xdiff)

    # resize W handle
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 5
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (xdata_new, extents[1] - xdiff,
                            extents[2] + xdiff, extents[3] - xdiff)

    # resize W handle negative diff
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -25
    xdata_new, ydata_new = xdata + xdiff, ydata
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (xdata_new, extents[1] - xdiff,
                            extents[2] + xdiff, extents[3] - xdiff)

    # resize SW handle
    extents = tool.extents
    xdata, ydata = extents[0], extents[2]
    xdiff, ydiff = 20, 25
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    _resize_rectangle(tool, xdata, ydata, xdata_new, ydata_new)
    assert tool.extents == (extents[0] + ydiff, extents[1] - ydiff,
                            ydata_new, extents[3] - ydiff)


def test_ellipse():
    """For ellipse, test out the key modifiers"""
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.EllipseSelector(ax, onselect=onselect,
                                   grab_range=10, interactive=True)
    tool.extents = (100, 150, 100, 150)

    # drag the rectangle
    do_event(tool, 'press', xdata=10, ydata=10, button=1,
             key=' ')

    do_event(tool, 'onmove', xdata=30, ydata=30, button=1)
    do_event(tool, 'release', xdata=30, ydata=30, button=1)
    assert tool.extents == (120, 170, 120, 170)

    # create from center
    do_event(tool, 'on_key_press', xdata=100, ydata=100, button=1,
             key='control')
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    do_event(tool, 'onmove', xdata=125, ydata=125, button=1)
    do_event(tool, 'release', xdata=125, ydata=125, button=1)
    do_event(tool, 'on_key_release', xdata=100, ydata=100, button=1,
             key='control')
    assert tool.extents == (75, 125, 75, 125)

    # create a square
    do_event(tool, 'on_key_press', xdata=10, ydata=10, button=1,
             key='shift')
    do_event(tool, 'press', xdata=10, ydata=10, button=1)
    do_event(tool, 'onmove', xdata=35, ydata=30, button=1)
    do_event(tool, 'release', xdata=35, ydata=30, button=1)
    do_event(tool, 'on_key_release', xdata=10, ydata=10, button=1,
             key='shift')
    extents = [int(e) for e in tool.extents]
    assert extents == [10, 35, 10, 34]

    # create a square from center
    do_event(tool, 'on_key_press', xdata=100, ydata=100, button=1,
             key='ctrl+shift')
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    do_event(tool, 'onmove', xdata=125, ydata=130, button=1)
    do_event(tool, 'release', xdata=125, ydata=130, button=1)
    do_event(tool, 'on_key_release', xdata=100, ydata=100, button=1,
             key='ctrl+shift')
    extents = [int(e) for e in tool.extents]
    assert extents == [70, 129, 70, 130]

    assert tool.geometry.shape == (2, 73)
    assert_allclose(tool.geometry[:, 0], [70., 100])


def test_rectangle_handles():
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.RectangleSelector(ax, onselect=onselect,
                                     grab_range=10,
                                     interactive=True,
                                     handle_props={'markerfacecolor': 'r',
                                                   'markeredgecolor': 'b'})
    tool.extents = (100, 150, 100, 150)

    assert tool.corners == (
        (100, 150, 150, 100), (100, 100, 150, 150))
    assert tool.extents == (100, 150, 100, 150)
    assert tool.edge_centers == (
        (100, 125.0, 150, 125.0), (125.0, 100, 125.0, 150))
    assert tool.extents == (100, 150, 100, 150)

    # grab a corner and move it
    do_event(tool, 'press', xdata=100, ydata=100)
    do_event(tool, 'onmove', xdata=120, ydata=120)
    do_event(tool, 'release', xdata=120, ydata=120)
    assert tool.extents == (120, 150, 120, 150)

    # grab the center and move it
    do_event(tool, 'press', xdata=132, ydata=132)
    do_event(tool, 'onmove', xdata=120, ydata=120)
    do_event(tool, 'release', xdata=120, ydata=120)
    assert tool.extents == (108, 138, 108, 138)

    # create a new rectangle
    do_event(tool, 'press', xdata=10, ydata=10)
    do_event(tool, 'onmove', xdata=100, ydata=100)
    do_event(tool, 'release', xdata=100, ydata=100)
    assert tool.extents == (10, 100, 10, 100)

    # Check that marker_props worked.
    assert mcolors.same_color(
        tool._corner_handles.artists[0].get_markerfacecolor(), 'r')
    assert mcolors.same_color(
        tool._corner_handles.artists[0].get_markeredgecolor(), 'b')


@pytest.mark.parametrize('interactive', [True, False])
def test_rectangle_selector_onselect(interactive):
    # check when press and release events take place at the same position
    ax = get_ax()

    def onselect(vmin, vmax):
        ax._got_onselect = True

    tool = widgets.RectangleSelector(ax, onselect, interactive=interactive)
    do_event(tool, 'press', xdata=100, ydata=110, button=1)
    # move outside of axis
    do_event(tool, 'onmove', xdata=150, ydata=120, button=1)
    do_event(tool, 'release', xdata=150, ydata=120, button=1)

    assert tool.ax._got_onselect
    assert tool.extents == (100.0, 150.0, 110.0, 120.0)

    # Reset tool.ax._got_onselect
    tool.ax._got_onselect = False

    do_event(tool, 'press', xdata=10, ydata=100, button=1)
    do_event(tool, 'release', xdata=10, ydata=100, button=1)

    assert tool.ax._got_onselect


@pytest.mark.parametrize('ignore_event_outside', [True, False])
def test_rectangle_selector_ignore_outside(ignore_event_outside):
    ax = get_ax()
    def onselect(vmin, vmax):
        ax._got_onselect = True

    tool = widgets.RectangleSelector(ax, onselect,
                                     ignore_event_outside=ignore_event_outside)
    do_event(tool, 'press', xdata=100, ydata=110, button=1)
    do_event(tool, 'onmove', xdata=150, ydata=120, button=1)
    do_event(tool, 'release', xdata=150, ydata=120, button=1)

    assert tool.ax._got_onselect
    assert tool.extents == (100.0, 150.0, 110.0, 120.0)

    # Reset
    ax._got_onselect = False
    # Trigger event outside of span
    do_event(tool, 'press', xdata=150, ydata=150, button=1)
    do_event(tool, 'onmove', xdata=160, ydata=160, button=1)
    do_event(tool, 'release', xdata=160, ydata=160, button=1)
    if ignore_event_outside:
        # event have been ignored and span haven't changed.
        assert not ax._got_onselect
        assert tool.extents == (100.0, 150.0, 110.0, 120.0)
    else:
        # A new shape is created
        assert ax._got_onselect
        assert tool.extents == (150.0, 160.0, 150.0, 160.0)


def check_span(*args, **kwargs):
    ax = get_ax()

    def onselect(vmin, vmax):
        ax._got_onselect = True
        assert vmin == 100
        assert vmax == 199

    def onmove(vmin, vmax):
        assert vmin == 100
        assert vmax == 199
        ax._got_on_move = True

    if 'onmove_callback' in kwargs:
        kwargs['onmove_callback'] = onmove

    tool = widgets.SpanSelector(ax, onselect, *args, **kwargs)
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    # move outside of axis
    do_event(tool, 'onmove', xdata=199, ydata=199, button=1)
    do_event(tool, 'release', xdata=250, ydata=250, button=1)

    assert ax._got_onselect

    if 'onmove_callback' in kwargs:
        assert ax._got_on_move


def test_span_selector():
    check_span('horizontal', minspan=10, useblit=True)
    check_span('vertical', onmove_callback=True, button=1)
    check_span('horizontal', props=dict(fill=True))
    check_span('horizontal', interactive=True)


@pytest.mark.parametrize('interactive', [True, False])
def test_span_selector_onselect(interactive):
    # check when press and release events take place at the same position
    ax = get_ax()

    def onselect(vmin, vmax):
        ax._got_onselect = True

    tool = widgets.SpanSelector(ax, onselect, 'horizontal',
                                interactive=interactive)
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    # move outside of axis
    do_event(tool, 'onmove', xdata=150, ydata=100, button=1)
    do_event(tool, 'release', xdata=150, ydata=100, button=1)

    assert tool.ax._got_onselect
    assert tool.extents == (100, 150)

    # Reset tool.ax._got_onselect
    tool.ax._got_onselect = False

    do_event(tool, 'press', xdata=10, ydata=100, button=1)
    do_event(tool, 'release', xdata=10, ydata=100, button=1)

    assert tool.ax._got_onselect


@pytest.mark.parametrize('ignore_event_outside', [True, False])
def test_span_selector_ignore_outside(ignore_event_outside):
    ax = get_ax()
    def onselect(vmin, vmax):
        ax._got_onselect = True

    def onmove(vmin, vmax):
        ax._got_on_move = True

    tool = widgets.SpanSelector(ax, onselect, 'horizontal',
                                onmove_callback=onmove,
                                ignore_event_outside=ignore_event_outside)
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    do_event(tool, 'onmove', xdata=125, ydata=125, button=1)
    do_event(tool, 'release', xdata=125, ydata=125, button=1)
    assert ax._got_onselect
    assert ax._got_on_move
    assert tool.extents == (100, 125)

    # Reset
    ax._got_onselect = False
    ax._got_on_move = False
    # Trigger event outside of span
    do_event(tool, 'press', xdata=150, ydata=150, button=1)
    do_event(tool, 'onmove', xdata=160, ydata=160, button=1)
    do_event(tool, 'release', xdata=160, ydata=160, button=1)
    if ignore_event_outside:
        # event have been ignored and span haven't changed.
        assert not ax._got_onselect
        assert not ax._got_on_move
        assert tool.extents == (100, 125)
    else:
        # A new shape is created
        assert ax._got_onselect
        assert ax._got_on_move
        assert tool.extents == (150, 160)


@pytest.mark.parametrize('drag_from_anywhere', [True, False])
def test_span_selector_drag(drag_from_anywhere):
    ax = get_ax()

    def onselect(*args):
        pass

    # Create span
    tool = widgets.SpanSelector(ax, onselect, 'horizontal', interactive=True,
                                drag_from_anywhere=drag_from_anywhere)
    do_event(tool, 'press', xdata=10, ydata=10, button=1)
    do_event(tool, 'onmove', xdata=100, ydata=120, button=1)
    do_event(tool, 'release', xdata=100, ydata=120, button=1)
    assert tool.extents == (10, 100)
    # Drag inside span
    #
    # If drag_from_anywhere == True, this will move the span by 10,
    # giving new value extents = 20, 110
    #
    # If drag_from_anywhere == False, this will create a new span with
    # value extents = 25, 35
    do_event(tool, 'press', xdata=25, ydata=15, button=1)
    do_event(tool, 'onmove', xdata=35, ydata=25, button=1)
    do_event(tool, 'release', xdata=35, ydata=25, button=1)
    if drag_from_anywhere:
        assert tool.extents == (20, 110)
    else:
        assert tool.extents == (25, 35)

    # Check that in both cases, dragging outside the span draws a new span
    do_event(tool, 'press', xdata=175, ydata=185, button=1)
    do_event(tool, 'onmove', xdata=185, ydata=195, button=1)
    do_event(tool, 'release', xdata=185, ydata=195, button=1)
    assert tool.extents == (175, 185)


def test_span_selector_direction():
    ax = get_ax()

    def onselect(*args):
        pass

    tool = widgets.SpanSelector(ax, onselect, 'horizontal', interactive=True)
    assert tool.direction == 'horizontal'
    assert tool._edge_handles.direction == 'horizontal'

    with pytest.raises(ValueError):
        tool = widgets.SpanSelector(ax, onselect, 'invalid_direction')

    tool.direction = 'vertical'
    assert tool.direction == 'vertical'
    assert tool._edge_handles.direction == 'vertical'

    with pytest.raises(ValueError):
        tool.direction = 'invalid_string'


def test_span_selector_set_props_handle_props():
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.SpanSelector(ax, onselect, 'horizontal', interactive=True,
                                props=dict(facecolor='b', alpha=0.2),
                                handle_props=dict(alpha=0.5))
    # Create rectangle
    do_event(tool, 'press', xdata=0, ydata=10, button=1)
    do_event(tool, 'onmove', xdata=100, ydata=120, button=1)
    do_event(tool, 'release', xdata=100, ydata=120, button=1)

    artist = tool._selection_artist
    assert artist.get_facecolor() == mcolors.to_rgba('b', alpha=0.2)
    tool.set_props(facecolor='r', alpha=0.3)
    assert artist.get_facecolor() == mcolors.to_rgba('r', alpha=0.3)

    for artist in tool._handles_artists:
        assert artist.get_color() == 'b'
        assert artist.get_alpha() == 0.5
    tool.set_handle_props(color='r', alpha=0.3)
    for artist in tool._handles_artists:
        assert artist.get_color() == 'r'
        assert artist.get_alpha() == 0.3


@pytest.mark.parametrize('selector', ['span', 'rectangle'])
def test_selector_clear(selector):
    ax = get_ax()

    def onselect(*args):
        pass

    kwargs = dict(ax=ax, onselect=onselect, interactive=True)
    if selector == 'span':
        Selector = widgets.SpanSelector
        kwargs['direction'] = 'horizontal'
    else:
        Selector = widgets.RectangleSelector

    tool = Selector(**kwargs)
    do_event(tool, 'press', xdata=10, ydata=10, button=1)
    do_event(tool, 'onmove', xdata=100, ydata=120, button=1)
    do_event(tool, 'release', xdata=100, ydata=120, button=1)

    # press-release event outside the selector to clear the selector
    do_event(tool, 'press', xdata=130, ydata=130, button=1)
    do_event(tool, 'release', xdata=130, ydata=130, button=1)
    assert not tool._selection_completed

    ax = get_ax()
    kwargs['ignore_event_outside'] = True
    tool = Selector(**kwargs)
    assert tool.ignore_event_outside
    do_event(tool, 'press', xdata=10, ydata=10, button=1)
    do_event(tool, 'onmove', xdata=100, ydata=120, button=1)
    do_event(tool, 'release', xdata=100, ydata=120, button=1)

    # press-release event outside the selector ignored
    do_event(tool, 'press', xdata=130, ydata=130, button=1)
    do_event(tool, 'release', xdata=130, ydata=130, button=1)
    assert tool._selection_completed

    do_event(tool, 'on_key_press', key='escape')
    assert not tool._selection_completed


@pytest.mark.parametrize('selector', ['span', 'rectangle'])
def test_selector_clear_method(selector):
    ax = get_ax()

    def onselect(*args):
        pass

    if selector == 'span':
        tool = widgets.SpanSelector(ax, onselect, 'horizontal',
                                    interactive=True,
                                    ignore_event_outside=True)
    else:
        tool = widgets.RectangleSelector(ax, onselect, interactive=True)
    do_event(tool, 'press', xdata=10, ydata=10, button=1)
    do_event(tool, 'onmove', xdata=100, ydata=120, button=1)
    do_event(tool, 'release', xdata=100, ydata=120, button=1)
    assert tool._selection_completed
    assert tool.visible
    if selector == 'span':
        assert tool.extents == (10, 100)

    tool.clear()
    assert not tool._selection_completed
    assert not tool.visible

    # Do another cycle of events to make sure we can
    do_event(tool, 'press', xdata=10, ydata=10, button=1)
    do_event(tool, 'onmove', xdata=50, ydata=120, button=1)
    do_event(tool, 'release', xdata=50, ydata=120, button=1)
    assert tool._selection_completed
    assert tool.visible
    if selector == 'span':
        assert tool.extents == (10, 50)


def test_tool_line_handle():
    ax = get_ax()

    positions = [20, 30, 50]

    tool_line_handle = widgets.ToolLineHandles(ax, positions, 'horizontal',
                                               useblit=False)

    for artist in tool_line_handle.artists:
        assert not artist.get_animated()
        assert not artist.get_visible()

    tool_line_handle.set_visible(True)
    tool_line_handle.set_animated(True)

    for artist in tool_line_handle.artists:
        assert artist.get_animated()
        assert artist.get_visible()

    assert tool_line_handle.positions == positions


@pytest.mark.parametrize('direction', ("horizontal", "vertical"))
def test_span_selector_bound(direction):
    fig, ax = plt.subplots(1, 1)
    ax.plot([10, 20], [10, 30])
    ax.figure.canvas.draw()
    x_bound = ax.get_xbound()
    y_bound = ax.get_ybound()

    tool = widgets.SpanSelector(ax, print, direction, interactive=True)
    assert ax.get_xbound() == x_bound
    assert ax.get_ybound() == y_bound

    bound = x_bound if direction == 'horizontal' else y_bound
    assert tool._edge_handles.positions == list(bound)

    press_data = [10.5, 11.5]
    move_data = [11, 13]  # Updating selector is done in onmove
    release_data = move_data
    do_event(tool, 'press', xdata=press_data[0], ydata=press_data[1], button=1)
    do_event(tool, 'onmove', xdata=move_data[0], ydata=move_data[1], button=1)

    assert ax.get_xbound() == x_bound
    assert ax.get_ybound() == y_bound

    index = 0 if direction == 'horizontal' else 1
    handle_positions = [press_data[index], release_data[index]]
    assert tool._edge_handles.positions == handle_positions


def check_lasso_selector(**kwargs):
    ax = get_ax()

    def onselect(verts):
        ax._got_onselect = True
        assert verts == [(100, 100), (125, 125), (150, 150)]

    tool = widgets.LassoSelector(ax, onselect, **kwargs)
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    do_event(tool, 'onmove', xdata=125, ydata=125, button=1)
    do_event(tool, 'release', xdata=150, ydata=150, button=1)

    assert ax._got_onselect


def test_lasso_selector():
    check_lasso_selector()
    check_lasso_selector(useblit=False, props=dict(color='red'))
    check_lasso_selector(useblit=True, button=1)


def test_CheckButtons():
    ax = get_ax()
    check = widgets.CheckButtons(ax, ('a', 'b', 'c'), (True, False, True))
    assert check.get_status() == [True, False, True]
    check.set_active(0)
    assert check.get_status() == [False, False, True]

    cid = check.on_clicked(lambda: None)
    check.disconnect(cid)


@pytest.mark.parametrize("toolbar", ["none", "toolbar2", "toolmanager"])
def test_TextBox(toolbar):
    # Avoid "toolmanager is provisional" warning.
    dict.__setitem__(plt.rcParams, "toolbar", toolbar)

    from unittest.mock import Mock
    submit_event = Mock()
    text_change_event = Mock()
    ax = get_ax()
    tool = widgets.TextBox(ax, '')
    tool.on_submit(submit_event)
    tool.on_text_change(text_change_event)

    assert tool.text == ''

    do_event(tool, '_click')

    tool.set_val('x**2')

    assert tool.text == 'x**2'
    assert text_change_event.call_count == 1

    tool.begin_typing(tool.text)
    tool.stop_typing()

    assert submit_event.call_count == 2

    do_event(tool, '_click')
    do_event(tool, '_keypress', key='+')
    do_event(tool, '_keypress', key='5')

    assert text_change_event.call_count == 3


@image_comparison(['check_radio_buttons.png'], style='mpl20', remove_text=True)
def test_check_radio_buttons_image():
    # Remove this line when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6

    get_ax()
    plt.subplots_adjust(left=0.3)
    rax1 = plt.axes([0.05, 0.7, 0.15, 0.15])
    rax2 = plt.axes([0.05, 0.2, 0.15, 0.15])
    widgets.RadioButtons(rax1, ('Radio 1', 'Radio 2', 'Radio 3'))
    widgets.CheckButtons(rax2, ('Check 1', 'Check 2', 'Check 3'),
                         (False, True, True))


@image_comparison(['check_bunch_of_radio_buttons.png'],
                  style='mpl20', remove_text=True)
def test_check_bunch_of_radio_buttons():
    rax = plt.axes([0.05, 0.1, 0.15, 0.7])
    widgets.RadioButtons(rax, ('B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                               'B7', 'B8', 'B9', 'B10', 'B11', 'B12',
                               'B13', 'B14', 'B15'))


def test_slider_slidermin_slidermax_invalid():
    fig, ax = plt.subplots()
    # test min/max with floats
    with pytest.raises(ValueError):
        widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                       slidermin=10.0)
    with pytest.raises(ValueError):
        widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                       slidermax=10.0)


def test_slider_slidermin_slidermax():
    fig, ax = plt.subplots()
    slider_ = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                             valinit=5.0)

    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=1.0, slidermin=slider_)
    assert slider.val == slider_.val

    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=10.0, slidermax=slider_)
    assert slider.val == slider_.val


def test_slider_valmin_valmax():
    fig, ax = plt.subplots()
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=-10.0)
    assert slider.val == slider.valmin

    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=25.0)
    assert slider.val == slider.valmax


def test_slider_valstep_snapping():
    fig, ax = plt.subplots()
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=11.4, valstep=1)
    assert slider.val == 11

    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=11.4, valstep=[0, 1, 5.5, 19.7])
    assert slider.val == 5.5


def test_slider_horizontal_vertical():
    fig, ax = plt.subplots()
    slider = widgets.Slider(ax=ax, label='', valmin=0, valmax=24,
                            valinit=12, orientation='horizontal')
    slider.set_val(10)
    assert slider.val == 10
    # check the dimension of the slider patch in axes units
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.bounds, [0, .25, 10/24, .5])

    fig, ax = plt.subplots()
    slider = widgets.Slider(ax=ax, label='', valmin=0, valmax=24,
                            valinit=12, orientation='vertical')
    slider.set_val(10)
    assert slider.val == 10
    # check the dimension of the slider patch in axes units
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.bounds, [.25, 0, .5, 10/24])


def test_slider_reset():
    fig, ax = plt.subplots()
    slider = widgets.Slider(ax=ax, label='', valmin=0, valmax=1, valinit=.5)
    slider.set_val(0.75)
    slider.reset()
    assert slider.val == 0.5


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_range_slider(orientation):
    if orientation == "vertical":
        idx = [1, 0, 3, 2]
    else:
        idx = [0, 1, 2, 3]

    fig, ax = plt.subplots()

    slider = widgets.RangeSlider(
        ax=ax, label="", valmin=0.0, valmax=1.0, orientation=orientation,
        valinit=[0.1, 0.34]
    )
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.get_points().flatten()[idx], [0.1, 0.25, 0.34, 0.75])

    # Check initial value is set correctly
    assert_allclose(slider.val, (0.1, 0.34))

    slider.set_val((0.2, 0.6))
    assert_allclose(slider.val, (0.2, 0.6))
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.get_points().flatten()[idx], [0.2, .25, 0.6, .75])

    slider.set_val((0.2, 0.1))
    assert_allclose(slider.val, (0.1, 0.2))

    slider.set_val((-1, 10))
    assert_allclose(slider.val, (0, 1))

    slider.reset()
    assert_allclose(slider.val, [0.1, 0.34])


def check_polygon_selector(event_sequence, expected_result, selections_count):
    """
    Helper function to test Polygon Selector.

    Parameters
    ----------
    event_sequence : list of tuples (etype, dict())
        A sequence of events to perform. The sequence is a list of tuples
        where the first element of the tuple is an etype (e.g., 'onmove',
        'press', etc.), and the second element of the tuple is a dictionary of
         the arguments for the event (e.g., xdata=5, key='shift', etc.).
    expected_result : list of vertices (xdata, ydata)
        The list of vertices that are expected to result from the event
        sequence.
    selections_count : int
        Wait for the tool to call its `onselect` function `selections_count`
        times, before comparing the result to the `expected_result`
    """
    ax = get_ax()

    ax._selections_count = 0

    def onselect(vertices):
        ax._selections_count += 1
        ax._current_result = vertices

    tool = widgets.PolygonSelector(ax, onselect)

    for (etype, event_args) in event_sequence:
        do_event(tool, etype, **event_args)

    assert ax._selections_count == selections_count
    assert ax._current_result == expected_result


def polygon_place_vertex(xdata, ydata):
    return [('onmove', dict(xdata=xdata, ydata=ydata)),
            ('press', dict(xdata=xdata, ydata=ydata)),
            ('release', dict(xdata=xdata, ydata=ydata))]


def polygon_remove_vertex(xdata, ydata):
    return [('onmove', dict(xdata=xdata, ydata=ydata)),
            ('press', dict(xdata=xdata, ydata=ydata, button=3)),
            ('release', dict(xdata=xdata, ydata=ydata, button=3))]


def test_polygon_selector():
    # Simple polygon
    expected_result = [(50, 50), (150, 50), (50, 150)]
    event_sequence = (polygon_place_vertex(50, 50)
                      + polygon_place_vertex(150, 50)
                      + polygon_place_vertex(50, 150)
                      + polygon_place_vertex(50, 50))
    check_polygon_selector(event_sequence, expected_result, 1)

    # Move first vertex before completing the polygon.
    expected_result = [(75, 50), (150, 50), (50, 150)]
    event_sequence = (polygon_place_vertex(50, 50)
                      + polygon_place_vertex(150, 50)
                      + [('on_key_press', dict(key='control')),
                         ('onmove', dict(xdata=50, ydata=50)),
                         ('press', dict(xdata=50, ydata=50)),
                         ('onmove', dict(xdata=75, ydata=50)),
                         ('release', dict(xdata=75, ydata=50)),
                         ('on_key_release', dict(key='control'))]
                      + polygon_place_vertex(50, 150)
                      + polygon_place_vertex(75, 50))
    check_polygon_selector(event_sequence, expected_result, 1)

    # Move first two vertices at once before completing the polygon.
    expected_result = [(50, 75), (150, 75), (50, 150)]
    event_sequence = (polygon_place_vertex(50, 50)
                      + polygon_place_vertex(150, 50)
                      + [('on_key_press', dict(key='shift')),
                         ('onmove', dict(xdata=100, ydata=100)),
                         ('press', dict(xdata=100, ydata=100)),
                         ('onmove', dict(xdata=100, ydata=125)),
                         ('release', dict(xdata=100, ydata=125)),
                         ('on_key_release', dict(key='shift'))]
                      + polygon_place_vertex(50, 150)
                      + polygon_place_vertex(50, 75))
    check_polygon_selector(event_sequence, expected_result, 1)

    # Move first vertex after completing the polygon.
    expected_result = [(75, 50), (150, 50), (50, 150)]
    event_sequence = (polygon_place_vertex(50, 50)
                      + polygon_place_vertex(150, 50)
                      + polygon_place_vertex(50, 150)
                      + polygon_place_vertex(50, 50)
                      + [('onmove', dict(xdata=50, ydata=50)),
                         ('press', dict(xdata=50, ydata=50)),
                         ('onmove', dict(xdata=75, ydata=50)),
                         ('release', dict(xdata=75, ydata=50))])
    check_polygon_selector(event_sequence, expected_result, 2)

    # Move all vertices after completing the polygon.
    expected_result = [(75, 75), (175, 75), (75, 175)]
    event_sequence = (polygon_place_vertex(50, 50)
                      + polygon_place_vertex(150, 50)
                      + polygon_place_vertex(50, 150)
                      + polygon_place_vertex(50, 50)
                      + [('on_key_press', dict(key='shift')),
                         ('onmove', dict(xdata=100, ydata=100)),
                         ('press', dict(xdata=100, ydata=100)),
                         ('onmove', dict(xdata=125, ydata=125)),
                         ('release', dict(xdata=125, ydata=125)),
                         ('on_key_release', dict(key='shift'))])
    check_polygon_selector(event_sequence, expected_result, 2)

    # Try to move a vertex and move all before placing any vertices.
    expected_result = [(50, 50), (150, 50), (50, 150)]
    event_sequence = ([('on_key_press', dict(key='control')),
                       ('onmove', dict(xdata=100, ydata=100)),
                       ('press', dict(xdata=100, ydata=100)),
                       ('onmove', dict(xdata=125, ydata=125)),
                       ('release', dict(xdata=125, ydata=125)),
                       ('on_key_release', dict(key='control')),
                       ('on_key_press', dict(key='shift')),
                       ('onmove', dict(xdata=100, ydata=100)),
                       ('press', dict(xdata=100, ydata=100)),
                       ('onmove', dict(xdata=125, ydata=125)),
                       ('release', dict(xdata=125, ydata=125)),
                       ('on_key_release', dict(key='shift'))]
                      + polygon_place_vertex(50, 50)
                      + polygon_place_vertex(150, 50)
                      + polygon_place_vertex(50, 150)
                      + polygon_place_vertex(50, 50))
    check_polygon_selector(event_sequence, expected_result, 1)

    # Try to place vertex out-of-bounds, then reset, and start a new polygon.
    expected_result = [(50, 50), (150, 50), (50, 150)]
    event_sequence = (polygon_place_vertex(50, 50)
                      + polygon_place_vertex(250, 50)
                      + [('on_key_press', dict(key='escape')),
                         ('on_key_release', dict(key='escape'))]
                      + polygon_place_vertex(50, 50)
                      + polygon_place_vertex(150, 50)
                      + polygon_place_vertex(50, 150)
                      + polygon_place_vertex(50, 50))
    check_polygon_selector(event_sequence, expected_result, 1)


def test_polygon_selector_set_props_handle_props():
    ax = get_ax()

    ax._selections_count = 0

    def onselect(vertices):
        ax._selections_count += 1
        ax._current_result = vertices

    tool = widgets.PolygonSelector(ax, onselect,
                                   props=dict(color='b', alpha=0.2),
                                   handle_props=dict(alpha=0.5))

    event_sequence = (polygon_place_vertex(50, 50)
                      + polygon_place_vertex(150, 50)
                      + polygon_place_vertex(50, 150)
                      + polygon_place_vertex(50, 50))

    for (etype, event_args) in event_sequence:
        do_event(tool, etype, **event_args)

    artist = tool._selection_artist
    assert artist.get_color() == 'b'
    assert artist.get_alpha() == 0.2
    tool.set_props(color='r', alpha=0.3)
    assert artist.get_color() == 'r'
    assert artist.get_alpha() == 0.3

    for artist in tool._handles_artists:
        assert artist.get_color() == 'b'
        assert artist.get_alpha() == 0.5
    tool.set_handle_props(color='r', alpha=0.3)
    for artist in tool._handles_artists:
        assert artist.get_color() == 'r'
        assert artist.get_alpha() == 0.3


@pytest.mark.parametrize(
    "horizOn, vertOn",
    [(True, True), (True, False), (False, True)],
)
def test_MultiCursor(horizOn, vertOn):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    # useblit=false to avoid having to draw the figure to cache the renderer
    multi = widgets.MultiCursor(
        fig.canvas, (ax1, ax2), useblit=False, horizOn=horizOn, vertOn=vertOn
    )

    # Only two of the axes should have a line drawn on them.
    if vertOn:
        assert len(multi.vlines) == 2
    if horizOn:
        assert len(multi.hlines) == 2

    # mock a motion_notify_event
    # Can't use `do_event` as that helper requires the widget
    # to have a single .ax attribute.
    event = mock_event(ax1, xdata=.5, ydata=.25)
    multi.onmove(event)

    # the lines in the first two ax should both move
    for l in multi.vlines:
        assert l.get_xdata() == (.5, .5)
    for l in multi.hlines:
        assert l.get_ydata() == (.25, .25)

    # test a move event in an axes not part of the MultiCursor
    # the lines in ax1 and ax2 should not have moved.
    event = mock_event(ax3, xdata=.75, ydata=.75)
    multi.onmove(event)
    for l in multi.vlines:
        assert l.get_xdata() == (.5, .5)
    for l in multi.hlines:
        assert l.get_ydata() == (.25, .25)


@check_figures_equal()
def test_rect_visibility(fig_test, fig_ref):
    # Check that requesting an invisible selector makes it invisible
    ax_test = fig_test.subplots()
    _ = fig_ref.subplots()

    def onselect(verts):
        pass

    tool = widgets.RectangleSelector(ax_test, onselect,
                                     props={'visible': False})
    tool.extents = (0.2, 0.8, 0.3, 0.7)


# Change the order that the extra point is inserted in
@pytest.mark.parametrize('idx', [1, 2, 3])
def test_polygon_selector_remove(idx):
    verts = [(50, 50), (150, 50), (50, 150)]
    event_sequence = [polygon_place_vertex(*verts[0]),
                      polygon_place_vertex(*verts[1]),
                      polygon_place_vertex(*verts[2]),
                      # Finish the polygon
                      polygon_place_vertex(*verts[0])]
    # Add an extra point
    event_sequence.insert(idx, polygon_place_vertex(200, 200))
    # Remove the extra point
    event_sequence.append(polygon_remove_vertex(200, 200))
    # Flatten list of lists
    event_sequence = sum(event_sequence, [])
    check_polygon_selector(event_sequence, verts, 2)


def test_polygon_selector_remove_first_point():
    verts = [(50, 50), (150, 50), (50, 150)]
    event_sequence = (polygon_place_vertex(*verts[0]) +
                      polygon_place_vertex(*verts[1]) +
                      polygon_place_vertex(*verts[2]) +
                      polygon_place_vertex(*verts[0]) +
                      polygon_remove_vertex(*verts[0]))
    check_polygon_selector(event_sequence, verts[1:], 2)


def test_polygon_selector_redraw():
    verts = [(50, 50), (150, 50), (50, 150)]
    event_sequence = (polygon_place_vertex(*verts[0]) +
                      polygon_place_vertex(*verts[1]) +
                      polygon_place_vertex(*verts[2]) +
                      polygon_place_vertex(*verts[0]) +
                      # Polygon completed, now remove first two verts
                      polygon_remove_vertex(*verts[1]) +
                      polygon_remove_vertex(*verts[2]) +
                      # At this point the tool should be reset so we can add
                      # more vertices
                      polygon_place_vertex(*verts[1]))

    ax = get_ax()

    def onselect(vertices):
        pass

    tool = widgets.PolygonSelector(ax, onselect)
    for (etype, event_args) in event_sequence:
        do_event(tool, etype, **event_args)
    # After removing two verts, only one remains, and the
    # selector should be automatically resete
    assert tool.verts == verts[0:2]
