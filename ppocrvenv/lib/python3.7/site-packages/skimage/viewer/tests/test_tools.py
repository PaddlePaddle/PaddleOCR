from collections import namedtuple
import numpy as np
from numpy.testing import assert_equal
import pytest

pytest.importorskip("matplotlib")

from skimage import data
from skimage.viewer import ImageViewer, has_qt
from skimage.viewer.canvastools import (
    LineTool, ThickLineTool, RectangleTool, PaintTool)
from skimage.viewer.canvastools.base import CanvasToolBase

try:
    from matplotlib.testing.decorators import cleanup
except ImportError:
    def cleanup(func):
        return func


def get_end_points(image):
    h, w = image.shape[0:2]
    x = [w / 3, 2 * w / 3]
    y = [h / 2] * 2
    return np.transpose([x, y])


def do_event(viewer, etype, button=1, xdata=0, ydata=0, key=None):
    """
     *name*
        the event name

    *canvas*
        the FigureCanvas instance generating the event

    *guiEvent*
        the GUI event that triggered the matplotlib event

    *x*
        x position - pixels from left of canvas

    *y*
        y position - pixels from bottom of canvas

    *inaxes*
        the :class:`~matplotlib.axes.Axes` instance if mouse is over axes

    *xdata*
        x coord of mouse in data coords

    *ydata*
        y coord of mouse in data coords

     *button*
        button pressed None, 1, 2, 3, 'up', 'down' (up and down are used
        for scroll events)

    *key*
        the key depressed when the mouse event triggered (see
        :class:`KeyEvent`)

    *step*
        number of scroll steps (positive for 'up', negative for 'down')
    """
    ax = viewer.ax
    event = namedtuple('Event',
                       ('name canvas guiEvent x y inaxes xdata ydata '
                        'button key step'))
    event.button = button
    event.x, event.y = ax.transData.transform((xdata, ydata))
    event.xdata, event.ydata = xdata, ydata
    event.inaxes = ax
    event.canvas = ax.figure.canvas
    event.key = key
    event.step = 1
    event.guiEvent = None
    event.name = 'Custom'

    func = getattr(viewer._event_manager, 'on_%s' % etype)
    func(event)


@cleanup
@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_line_tool():
    img = data.camera()
    viewer = ImageViewer(img)

    tool = LineTool(viewer, maxdist=10, line_props=dict(linewidth=3),
                    handle_props=dict(markersize=5))
    tool.end_points = get_end_points(img)
    assert_equal(tool.end_points, np.array([[170, 256], [341, 256]]))

    # grab a handle and move it
    do_event(viewer, 'mouse_press', xdata=170, ydata=256)
    do_event(viewer, 'move', xdata=180, ydata=260)
    do_event(viewer, 'mouse_release')

    assert_equal(tool.geometry, np.array([[180, 260], [341, 256]]))

    # create a new line
    do_event(viewer, 'mouse_press', xdata=10, ydata=10)
    do_event(viewer, 'move', xdata=100, ydata=100)
    do_event(viewer, 'mouse_release')

    assert_equal(tool.geometry, np.array([[100, 100], [10, 10]]))


@cleanup
@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_thick_line_tool():
    img = data.camera()
    viewer = ImageViewer(img)

    tool = ThickLineTool(viewer, maxdist=10, line_props=dict(color='red'),
                         handle_props=dict(markersize=5))
    tool.end_points = get_end_points(img)

    do_event(viewer, 'scroll', button='up')
    assert_equal(tool.linewidth, 2)

    do_event(viewer, 'scroll', button='down')

    assert_equal(tool.linewidth, 1)

    do_event(viewer, 'key_press',  key='+')
    assert_equal(tool.linewidth, 2)

    do_event(viewer, 'key_press', key='-')
    assert_equal(tool.linewidth, 1)


@cleanup
@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_rect_tool():
    img = data.camera()
    viewer = ImageViewer(img)

    tool = RectangleTool(viewer, maxdist=10)
    tool.extents = (100, 150, 100, 150)

    assert_equal(tool.corners,
                 ((100, 150, 150, 100), (100, 100, 150, 150)))
    assert_equal(tool.extents, (100, 150, 100, 150))
    assert_equal(tool.edge_centers,
                 ((100, 125.0, 150, 125.0), (125.0, 100, 125.0, 150)))
    assert_equal(tool.geometry, (100, 150, 100, 150))

    # grab a corner and move it
    do_event(viewer, 'mouse_press', xdata=100, ydata=100)
    do_event(viewer, 'move', xdata=120, ydata=120)
    do_event(viewer, 'mouse_release')
    # assert_equal(tool.geometry, [120, 150, 120, 150])

    # create a new line
    do_event(viewer, 'mouse_press', xdata=10, ydata=10)
    do_event(viewer, 'move', xdata=100, ydata=100)
    do_event(viewer, 'mouse_release')
    assert_equal(tool.geometry, [10, 100,  10, 100])


@cleanup
@pytest.mark.skipif(not has_qt, reason="Qt not installed")
@pytest.mark.parametrize('img', [data.moon(), data.astronaut()])
def test_paint_tool(img):
    viewer = ImageViewer(img)

    tool = PaintTool(viewer, img.shape)

    tool.radius = 10
    assert_equal(tool.radius, 10)
    tool.label = 2
    assert_equal(tool.label, 2)
    assert_equal(tool.shape, img.shape[:2])

    do_event(viewer, 'mouse_press', xdata=100, ydata=100)
    do_event(viewer, 'move', xdata=110, ydata=110)
    do_event(viewer, 'mouse_release')

    assert_equal(tool.overlay[tool.overlay == 2].size, 761)

    tool.label = 5
    do_event(viewer, 'mouse_press', xdata=20, ydata=20)
    do_event(viewer, 'move', xdata=40, ydata=40)
    do_event(viewer, 'mouse_release')

    assert_equal(tool.overlay[tool.overlay == 5].size, 881)
    assert_equal(tool.overlay[tool.overlay == 2].size, 761)

    do_event(viewer, 'key_press', key='enter')

    tool.overlay = tool.overlay * 0
    assert_equal(tool.overlay.sum(), 0)
    assert_equal(tool.cmap.N, tool._overlay_plot.norm.vmax)


@cleanup
@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_base_tool():
    img = data.moon()
    viewer = ImageViewer(img)

    tool = CanvasToolBase(viewer)
    tool.set_visible(False)
    tool.set_visible(True)

    do_event(viewer, 'key_press', key='enter')

    tool.redraw()
    tool.remove()

    tool = CanvasToolBase(viewer, useblit=False)
    tool.redraw()
