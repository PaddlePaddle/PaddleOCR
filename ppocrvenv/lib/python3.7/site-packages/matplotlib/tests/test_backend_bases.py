import re

from matplotlib.testing import _check_for_pgf
from matplotlib.backend_bases import (
    FigureCanvasBase, LocationEvent, MouseButton, MouseEvent,
    NavigationToolbar2, RendererBase)
from matplotlib.backend_tools import (ToolZoom, ToolPan, RubberbandBase,
                                      ToolViewsPositions, _views_positions)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.path as path

import numpy as np
import pytest

needs_xelatex = pytest.mark.skipif(not _check_for_pgf('xelatex'),
                                   reason='xelatex + pgf is required')


def test_uses_per_path():
    id = transforms.Affine2D()
    paths = [path.Path.unit_regular_polygon(i) for i in range(3, 7)]
    tforms_matrices = [id.rotate(i).get_matrix().copy() for i in range(1, 5)]
    offsets = np.arange(20).reshape((10, 2))
    facecolors = ['red', 'green']
    edgecolors = ['red', 'green']

    def check(master_transform, paths, all_transforms,
              offsets, facecolors, edgecolors):
        rb = RendererBase()
        raw_paths = list(rb._iter_collection_raw_paths(
            master_transform, paths, all_transforms))
        gc = rb.new_gc()
        ids = [path_id for xo, yo, path_id, gc0, rgbFace in
               rb._iter_collection(
                   gc, master_transform, all_transforms,
                   range(len(raw_paths)), offsets,
                   transforms.AffineDeltaTransform(master_transform),
                   facecolors, edgecolors, [], [], [False],
                   [], 'screen')]
        uses = rb._iter_collection_uses_per_path(
            paths, all_transforms, offsets, facecolors, edgecolors)
        if raw_paths:
            seen = np.bincount(ids, minlength=len(raw_paths))
            assert set(seen).issubset([uses - 1, uses])

    check(id, paths, tforms_matrices, offsets, facecolors, edgecolors)
    check(id, paths[0:1], tforms_matrices, offsets, facecolors, edgecolors)
    check(id, [], tforms_matrices, offsets, facecolors, edgecolors)
    check(id, paths, tforms_matrices[0:1], offsets, facecolors, edgecolors)
    check(id, paths, [], offsets, facecolors, edgecolors)
    for n in range(0, offsets.shape[0]):
        check(id, paths, tforms_matrices, offsets[0:n, :],
              facecolors, edgecolors)
    check(id, paths, tforms_matrices, offsets, [], edgecolors)
    check(id, paths, tforms_matrices, offsets, facecolors, [])
    check(id, paths, tforms_matrices, offsets, [], [])
    check(id, paths, tforms_matrices, offsets, facecolors[0:1], edgecolors)


def test_canvas_ctor():
    assert isinstance(FigureCanvasBase().figure, Figure)


def test_get_default_filename():
    assert plt.figure().canvas.get_default_filename() == 'image.png'


def test_canvas_change():
    fig = plt.figure()
    # Replaces fig.canvas
    canvas = FigureCanvasBase(fig)
    # Should still work.
    plt.close(fig)
    assert not plt.fignum_exists(fig.number)


@pytest.mark.backend('pdf')
def test_non_gui_warning(monkeypatch):
    plt.subplots()

    monkeypatch.setenv("DISPLAY", ":999")

    with pytest.warns(UserWarning) as rec:
        plt.show()
        assert len(rec) == 1
        assert ('Matplotlib is currently using pdf, which is a non-GUI backend'
                in str(rec[0].message))

    with pytest.warns(UserWarning) as rec:
        plt.gcf().show()
        assert len(rec) == 1
        assert ('Matplotlib is currently using pdf, which is a non-GUI backend'
                in str(rec[0].message))


@pytest.mark.parametrize(
    "x, y", [(42, 24), (None, 42), (None, None), (200, 100.01), (205.75, 2.0)])
def test_location_event_position(x, y):
    # LocationEvent should cast its x and y arguments to int unless it is None.
    fig, ax = plt.subplots()
    canvas = FigureCanvasBase(fig)
    event = LocationEvent("test_event", canvas, x, y)
    if x is None:
        assert event.x is None
    else:
        assert event.x == int(x)
        assert isinstance(event.x, int)
    if y is None:
        assert event.y is None
    else:
        assert event.y == int(y)
        assert isinstance(event.y, int)
    if x is not None and y is not None:
        assert re.match(
            "x={} +y={}".format(ax.format_xdata(x), ax.format_ydata(y)),
            ax.format_coord(x, y))
        ax.fmt_xdata = ax.fmt_ydata = lambda x: "foo"
        assert re.match("x=foo +y=foo", ax.format_coord(x, y))


def test_pick():
    fig = plt.figure()
    fig.text(.5, .5, "hello", ha="center", va="center", picker=True)
    fig.canvas.draw()
    picks = []
    fig.canvas.mpl_connect("pick_event", lambda event: picks.append(event))
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *fig.transFigure.transform((.5, .5)),
        MouseButton.LEFT)
    fig.canvas.callbacks.process(start_event.name, start_event)
    assert len(picks) == 1


def test_interactive_zoom():
    fig, ax = plt.subplots()
    ax.set(xscale="logit")
    assert ax.get_navigate_mode() is None

    tb = NavigationToolbar2(fig.canvas)
    tb.zoom()
    assert ax.get_navigate_mode() == 'ZOOM'

    xlim0 = ax.get_xlim()
    ylim0 = ax.get_ylim()

    # Zoom from x=1e-6, y=0.1 to x=1-1e-5, 0.8 (data coordinates, "d").
    d0 = (1e-6, 0.1)
    d1 = (1-1e-5, 0.8)
    # Convert to screen coordinates ("s").  Events are defined only with pixel
    # precision, so round the pixel values, and below, check against the
    # corresponding xdata/ydata, which are close but not equal to d0/d1.
    s0 = ax.transData.transform(d0).astype(int)
    s1 = ax.transData.transform(d1).astype(int)

    # Zoom in.
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *s0, MouseButton.LEFT)
    fig.canvas.callbacks.process(start_event.name, start_event)
    stop_event = MouseEvent(
        "button_release_event", fig.canvas, *s1, MouseButton.LEFT)
    fig.canvas.callbacks.process(stop_event.name, stop_event)
    assert ax.get_xlim() == (start_event.xdata, stop_event.xdata)
    assert ax.get_ylim() == (start_event.ydata, stop_event.ydata)

    # Zoom out.
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *s1, MouseButton.RIGHT)
    fig.canvas.callbacks.process(start_event.name, start_event)
    stop_event = MouseEvent(
        "button_release_event", fig.canvas, *s0, MouseButton.RIGHT)
    fig.canvas.callbacks.process(stop_event.name, stop_event)
    # Absolute tolerance much less than original xmin (1e-7).
    assert ax.get_xlim() == pytest.approx(xlim0, rel=0, abs=1e-10)
    assert ax.get_ylim() == pytest.approx(ylim0, rel=0, abs=1e-10)

    tb.zoom()
    assert ax.get_navigate_mode() is None

    assert not ax.get_autoscalex_on() and not ax.get_autoscaley_on()


@pytest.mark.parametrize("plot_func", ["imshow", "contourf"])
@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("tool,button,expected",
                         [("zoom", MouseButton.LEFT, (4, 6)),  # zoom in
                          ("zoom", MouseButton.RIGHT, (-20, 30)),  # zoom out
                          ("pan", MouseButton.LEFT, (-2, 8))])
def test_interactive_colorbar(plot_func, orientation, tool, button, expected):
    fig, ax = plt.subplots()
    data = np.arange(12).reshape((4, 3))
    vmin0, vmax0 = 0, 10
    coll = getattr(ax, plot_func)(data, vmin=vmin0, vmax=vmax0)

    cb = fig.colorbar(coll, ax=ax, orientation=orientation)
    if plot_func == "contourf":
        # Just determine we can't navigate and exit out of the test
        assert not cb.ax.get_navigate()
        return

    assert cb.ax.get_navigate()

    # Mouse from 4 to 6 (data coordinates, "d").
    vmin, vmax = 4, 6
    # The y coordinate doesn't matter, it just needs to be between 0 and 1
    # However, we will set d0/d1 to the same y coordinate to test that small
    # pixel changes in that coordinate doesn't cancel the zoom like a normal
    # axes would.
    d0 = (vmin, 0.5)
    d1 = (vmax, 0.5)
    # Swap them if the orientation is vertical
    if orientation == "vertical":
        d0 = d0[::-1]
        d1 = d1[::-1]
    # Convert to screen coordinates ("s").  Events are defined only with pixel
    # precision, so round the pixel values, and below, check against the
    # corresponding xdata/ydata, which are close but not equal to d0/d1.
    s0 = cb.ax.transData.transform(d0).astype(int)
    s1 = cb.ax.transData.transform(d1).astype(int)

    # Set up the mouse movements
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *s0, button)
    stop_event = MouseEvent(
        "button_release_event", fig.canvas, *s1, button)

    tb = NavigationToolbar2(fig.canvas)
    if tool == "zoom":
        tb.zoom()
        tb.press_zoom(start_event)
        tb.drag_zoom(stop_event)
        tb.release_zoom(stop_event)
    else:
        tb.pan()
        tb.press_pan(start_event)
        tb.drag_pan(stop_event)
        tb.release_pan(stop_event)

    # Should be close, but won't be exact due to screen integer resolution
    assert (cb.vmin, cb.vmax) == pytest.approx(expected, abs=0.15)


def test_toolbar_zoompan():
    expected_warning_regex = (
        r"Treat the new Tool classes introduced in "
        r"v[0-9]*.[0-9]* as experimental for now; "
        "the API and rcParam may change in future versions.")
    with pytest.warns(UserWarning, match=expected_warning_regex):
        plt.rcParams['toolbar'] = 'toolmanager'
    ax = plt.gca()
    assert ax.get_navigate_mode() is None
    ax.figure.canvas.manager.toolmanager.add_tool(name="zoom",
                                                  tool=ToolZoom)
    ax.figure.canvas.manager.toolmanager.add_tool(name="pan",
                                                  tool=ToolPan)
    ax.figure.canvas.manager.toolmanager.add_tool(name=_views_positions,
                                                  tool=ToolViewsPositions)
    ax.figure.canvas.manager.toolmanager.add_tool(name='rubberband',
                                                  tool=RubberbandBase)
    ax.figure.canvas.manager.toolmanager.trigger_tool('zoom')
    assert ax.get_navigate_mode() == "ZOOM"
    ax.figure.canvas.manager.toolmanager.trigger_tool('pan')
    assert ax.get_navigate_mode() == "PAN"


@pytest.mark.parametrize(
    "backend", ['svg', 'ps', 'pdf', pytest.param('pgf', marks=needs_xelatex)]
)
def test_draw(backend):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvas
    test_backend = pytest.importorskip(
        f'matplotlib.backends.backend_{backend}'
    )
    TestCanvas = test_backend.FigureCanvas
    fig_test = Figure(constrained_layout=True)
    TestCanvas(fig_test)
    axes_test = fig_test.subplots(2, 2)

    # defaults to FigureCanvasBase
    fig_agg = Figure(constrained_layout=True)
    # put a backends.backend_agg.FigureCanvas on it
    FigureCanvas(fig_agg)
    axes_agg = fig_agg.subplots(2, 2)

    init_pos = [ax.get_position() for ax in axes_test.ravel()]

    fig_test.canvas.draw()
    fig_agg.canvas.draw()

    layed_out_pos_test = [ax.get_position() for ax in axes_test.ravel()]
    layed_out_pos_agg = [ax.get_position() for ax in axes_agg.ravel()]

    for init, placed in zip(init_pos, layed_out_pos_test):
        assert not np.allclose(init, placed, atol=0.005)

    for ref, test in zip(layed_out_pos_agg, layed_out_pos_test):
        np.testing.assert_allclose(ref, test, atol=0.005)
