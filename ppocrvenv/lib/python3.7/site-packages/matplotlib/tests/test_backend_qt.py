import copy
import importlib
import inspect
import os
import signal
import subprocess
import sys

from datetime import date, datetime
from unittest import mock

import pytest

import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib import _c_internal_utils


try:
    from matplotlib.backends.qt_compat import QtGui, QtWidgets
    from matplotlib.backends.qt_editor import _formlayout
except ImportError:
    pytestmark = pytest.mark.skip('No usable Qt bindings')


_test_timeout = 60  # A reasonably safe value for slower architectures.


@pytest.fixture
def qt_core(request):
    backend, = request.node.get_closest_marker('backend').args
    qt_compat = pytest.importorskip('matplotlib.backends.qt_compat')
    QtCore = qt_compat.QtCore

    return QtCore


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_fig_close():

    # save the state of Gcf.figs
    init_figs = copy.copy(Gcf.figs)

    # make a figure using pyplot interface
    fig = plt.figure()

    # simulate user clicking the close button by reaching in
    # and calling close on the underlying Qt object
    fig.canvas.manager.window.close()

    # assert that we have removed the reference to the FigureManager
    # that got added by plt.figure()
    assert init_figs == Gcf.figs


class WaitForStringPopen(subprocess.Popen):
    """
    A Popen that passes flags that allow triggering KeyboardInterrupt.
    """

    def __init__(self, *args, **kwargs):
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        super().__init__(
            *args, **kwargs,
            # Force Agg so that each test can switch to its desired Qt backend.
            env={**os.environ, "MPLBACKEND": "Agg", "SOURCE_DATE_EPOCH": "0"},
            stdout=subprocess.PIPE, universal_newlines=True)

    def wait_for(self, terminator):
        """Read until the terminator is reached."""
        buf = ''
        while True:
            c = self.stdout.read(1)
            if not c:
                raise RuntimeError(
                    f'Subprocess died before emitting expected {terminator!r}')
            buf += c
            if buf.endswith(terminator):
                return


def _test_sigint_impl(backend, target_name, kwargs):
    import sys
    import matplotlib.pyplot as plt
    import os
    import threading

    plt.switch_backend(backend)
    from matplotlib.backends.qt_compat import QtCore

    def interupter():
        if sys.platform == 'win32':
            import win32api
            win32api.GenerateConsoleCtrlEvent(0, 0)
        else:
            import signal
            os.kill(os.getpid(), signal.SIGINT)

    target = getattr(plt, target_name)
    timer = threading.Timer(1, interupter)
    fig = plt.figure()
    fig.canvas.mpl_connect(
        'draw_event',
        lambda *args: print('DRAW', flush=True)
    )
    fig.canvas.mpl_connect(
        'draw_event',
        lambda *args: timer.start()
    )
    try:
        target(**kwargs)
    except KeyboardInterrupt:
        print('SUCCESS', flush=True)


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
@pytest.mark.parametrize("target, kwargs", [
    ('show', {'block': True}),
    ('pause', {'interval': 10})
])
def test_sigint(target, kwargs):
    backend = plt.get_backend()
    proc = WaitForStringPopen(
        [sys.executable, "-c",
         inspect.getsource(_test_sigint_impl) +
         f"\n_test_sigint_impl({backend!r}, {target!r}, {kwargs!r})"])
    try:
        proc.wait_for('DRAW')
        stdout, _ = proc.communicate(timeout=_test_timeout)
    except:
        proc.kill()
        stdout, _ = proc.communicate()
        raise
    print(stdout)
    assert 'SUCCESS' in stdout


def _test_other_signal_before_sigint_impl(backend, target_name, kwargs):
    import signal
    import sys
    import matplotlib.pyplot as plt
    plt.switch_backend(backend)
    from matplotlib.backends.qt_compat import QtCore

    target = getattr(plt, target_name)

    fig = plt.figure()
    fig.canvas.mpl_connect('draw_event',
                           lambda *args: print('DRAW', flush=True))

    timer = fig.canvas.new_timer(interval=1)
    timer.single_shot = True
    timer.add_callback(print, 'SIGUSR1', flush=True)

    def custom_signal_handler(signum, frame):
        timer.start()
    signal.signal(signal.SIGUSR1, custom_signal_handler)

    try:
        target(**kwargs)
    except KeyboardInterrupt:
        print('SUCCESS', flush=True)


@pytest.mark.skipif(sys.platform == 'win32',
                    reason='No other signal available to send on Windows')
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
@pytest.mark.parametrize("target, kwargs", [
    ('show', {'block': True}),
    ('pause', {'interval': 10})
])
def test_other_signal_before_sigint(target, kwargs):
    backend = plt.get_backend()
    proc = WaitForStringPopen(
        [sys.executable, "-c",
         inspect.getsource(_test_other_signal_before_sigint_impl) +
         "\n_test_other_signal_before_sigint_impl("
            f"{backend!r}, {target!r}, {kwargs!r})"])
    try:
        proc.wait_for('DRAW')
        os.kill(proc.pid, signal.SIGUSR1)
        proc.wait_for('SIGUSR1')
        os.kill(proc.pid, signal.SIGINT)
        stdout, _ = proc.communicate(timeout=_test_timeout)
    except:
        proc.kill()
        stdout, _ = proc.communicate()
        raise
    print(stdout)
    assert 'SUCCESS' in stdout
    plt.figure()


@pytest.mark.backend('Qt5Agg')
def test_fig_sigint_override(qt_core):
    from matplotlib.backends.backend_qt5 import _BackendQT5
    # Create a figure
    plt.figure()

    # Variable to access the handler from the inside of the event loop
    event_loop_handler = None

    # Callback to fire during event loop: save SIGINT handler, then exit
    def fire_signal_and_quit():
        # Save event loop signal
        nonlocal event_loop_handler
        event_loop_handler = signal.getsignal(signal.SIGINT)

        # Request event loop exit
        qt_core.QCoreApplication.exit()

    # Timer to exit event loop
    qt_core.QTimer.singleShot(0, fire_signal_and_quit)

    # Save original SIGINT handler
    original_handler = signal.getsignal(signal.SIGINT)

    # Use our own SIGINT handler to be 100% sure this is working
    def custom_handler(signum, frame):
        pass

    signal.signal(signal.SIGINT, custom_handler)

    try:
        # mainloop() sets SIGINT, starts Qt event loop (which triggers timer
        # and exits) and then mainloop() resets SIGINT
        matplotlib.backends.backend_qt._BackendQT.mainloop()

        # Assert: signal handler during loop execution is changed
        # (can't test equality with func)
        assert event_loop_handler != custom_handler

        # Assert: current signal handler is the same as the one we set before
        assert signal.getsignal(signal.SIGINT) == custom_handler

        # Repeat again to test that SIG_DFL and SIG_IGN will not be overridden
        for custom_handler in (signal.SIG_DFL, signal.SIG_IGN):
            qt_core.QTimer.singleShot(0, fire_signal_and_quit)
            signal.signal(signal.SIGINT, custom_handler)

            _BackendQT5.mainloop()

            assert event_loop_handler == custom_handler
            assert signal.getsignal(signal.SIGINT) == custom_handler

    finally:
        # Reset SIGINT handler to what it was before the test
        signal.signal(signal.SIGINT, original_handler)


@pytest.mark.parametrize(
    "qt_key, qt_mods, answer",
    [
        ("Key_A", ["ShiftModifier"], "A"),
        ("Key_A", [], "a"),
        ("Key_A", ["ControlModifier"], ("ctrl+a")),
        (
            "Key_Aacute",
            ["ShiftModifier"],
            "\N{LATIN CAPITAL LETTER A WITH ACUTE}",
        ),
        ("Key_Aacute", [], "\N{LATIN SMALL LETTER A WITH ACUTE}"),
        ("Key_Control", ["AltModifier"], ("alt+control")),
        ("Key_Alt", ["ControlModifier"], "ctrl+alt"),
        (
            "Key_Aacute",
            ["ControlModifier", "AltModifier", "MetaModifier"],
            ("ctrl+alt+meta+\N{LATIN SMALL LETTER A WITH ACUTE}"),
        ),
        # We do not currently map the media keys, this may change in the
        # future.  This means the callback will never fire
        ("Key_Play", [], None),
        ("Key_Backspace", [], "backspace"),
        (
            "Key_Backspace",
            ["ControlModifier"],
            "ctrl+backspace",
        ),
    ],
    ids=[
        'shift',
        'lower',
        'control',
        'unicode_upper',
        'unicode_lower',
        'alt_control',
        'control_alt',
        'modifier_order',
        'non_unicode_key',
        'backspace',
        'backspace_mod',
    ]
)
@pytest.mark.parametrize('backend', [
    # Note: the value is irrelevant; the important part is the marker.
    pytest.param(
        'Qt5Agg',
        marks=pytest.mark.backend('Qt5Agg', skip_on_importerror=True)),
    pytest.param(
        'QtAgg',
        marks=pytest.mark.backend('QtAgg', skip_on_importerror=True)),
])
def test_correct_key(backend, qt_core, qt_key, qt_mods, answer):
    """
    Make a figure.
    Send a key_press_event event (using non-public, qtX backend specific api).
    Catch the event.
    Assert sent and caught keys are the same.
    """
    from matplotlib.backends.qt_compat import _enum, _to_int

    if sys.platform == "darwin" and answer is not None:
        answer = answer.replace("ctrl", "cmd")
        answer = answer.replace("control", "cmd")
        answer = answer.replace("meta", "ctrl")
    result = None
    qt_mod = _enum("QtCore.Qt.KeyboardModifier").NoModifier
    for mod in qt_mods:
        qt_mod |= getattr(_enum("QtCore.Qt.KeyboardModifier"), mod)

    class _Event:
        def isAutoRepeat(self): return False
        def key(self): return _to_int(getattr(_enum("QtCore.Qt.Key"), qt_key))
        def modifiers(self): return qt_mod

    def on_key_press(event):
        nonlocal result
        result = event.key

    qt_canvas = plt.figure().canvas
    qt_canvas.mpl_connect('key_press_event', on_key_press)
    qt_canvas.keyPressEvent(_Event())
    assert result == answer


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_device_pixel_ratio_change():
    """
    Make sure that if the pixel ratio changes, the figure dpi changes but the
    widget remains the same logical size.
    """

    prop = 'matplotlib.backends.backend_qt.FigureCanvasQT.devicePixelRatioF'
    with mock.patch(prop) as p:
        p.return_value = 3

        fig = plt.figure(figsize=(5, 2), dpi=120)
        qt_canvas = fig.canvas
        qt_canvas.show()

        def set_device_pixel_ratio(ratio):
            p.return_value = ratio

            # The value here doesn't matter, as we can't mock the C++ QScreen
            # object, but can override the functional wrapper around it.
            # Emitting this event is simply to trigger the DPI change handler
            # in Matplotlib in the same manner that it would occur normally.
            screen.logicalDotsPerInchChanged.emit(96)

            qt_canvas.draw()
            qt_canvas.flush_events()

            # Make sure the mocking worked
            assert qt_canvas.device_pixel_ratio == ratio

        qt_canvas.manager.show()
        size = qt_canvas.size()
        screen = qt_canvas.window().windowHandle().screen()
        set_device_pixel_ratio(3)

        # The DPI and the renderer width/height change
        assert fig.dpi == 360
        assert qt_canvas.renderer.width == 1800
        assert qt_canvas.renderer.height == 720

        # The actual widget size and figure logical size don't change.
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()

        set_device_pixel_ratio(2)

        # The DPI and the renderer width/height change
        assert fig.dpi == 240
        assert qt_canvas.renderer.width == 1200
        assert qt_canvas.renderer.height == 480

        # The actual widget size and figure logical size don't change.
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()

        set_device_pixel_ratio(1.5)

        # The DPI and the renderer width/height change
        assert fig.dpi == 180
        assert qt_canvas.renderer.width == 900
        assert qt_canvas.renderer.height == 360

        # The actual widget size and figure logical size don't change.
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_subplottool():
    fig, ax = plt.subplots()
    with mock.patch("matplotlib.backends.qt_compat._exec", lambda obj: None):
        fig.canvas.manager.toolbar.configure_subplots()


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_figureoptions():
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    ax.imshow([[1]])
    ax.scatter(range(3), range(3), c=range(3))
    with mock.patch("matplotlib.backends.qt_compat._exec", lambda obj: None):
        fig.canvas.manager.toolbar.edit_parameters()


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_figureoptions_with_datetime_axes():
    fig, ax = plt.subplots()
    xydata = [
        datetime(year=2021, month=1, day=1),
        datetime(year=2021, month=2, day=1)
    ]
    ax.plot(xydata, xydata)
    with mock.patch("matplotlib.backends.qt_compat._exec", lambda obj: None):
        fig.canvas.manager.toolbar.edit_parameters()


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_double_resize():
    # Check that resizing a figure twice keeps the same window size
    fig, ax = plt.subplots()
    fig.canvas.draw()
    window = fig.canvas.manager.window

    w, h = 3, 2
    fig.set_size_inches(w, h)
    assert fig.canvas.width() == w * matplotlib.rcParams['figure.dpi']
    assert fig.canvas.height() == h * matplotlib.rcParams['figure.dpi']

    old_width = window.width()
    old_height = window.height()

    fig.set_size_inches(w, h)
    assert window.width() == old_width
    assert window.height() == old_height


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_canvas_reinit():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    called = False

    def crashing_callback(fig, stale):
        nonlocal called
        fig.canvas.draw_idle()
        called = True

    fig, ax = plt.subplots()
    fig.stale_callback = crashing_callback
    # this should not raise
    canvas = FigureCanvasQTAgg(fig)
    fig.stale = True
    assert called


@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
def test_form_widget_get_with_datetime_and_date_fields():
    from matplotlib.backends.backend_qt import _create_qApp
    _create_qApp()

    form = [
        ("Datetime field", datetime(year=2021, month=3, day=11)),
        ("Date field", date(year=2021, month=3, day=11))
    ]
    widget = _formlayout.FormWidget(form)
    widget.setup()
    values = widget.get()
    assert values == [
        datetime(year=2021, month=3, day=11),
        date(year=2021, month=3, day=11)
    ]


# The source of this function gets extracted and run in another process, so it
# must be fully self-contained.
def _test_enums_impl():
    import sys

    from matplotlib.backends.qt_compat import _enum, _to_int, QtCore
    from matplotlib.backend_bases import cursors, MouseButton

    _enum("QtGui.QDoubleValidator.State").Acceptable

    _enum("QtWidgets.QDialogButtonBox.StandardButton").Ok
    _enum("QtWidgets.QDialogButtonBox.StandardButton").Cancel
    _enum("QtWidgets.QDialogButtonBox.StandardButton").Apply
    for btn_type in ["Ok", "Cancel"]:
        getattr(_enum("QtWidgets.QDialogButtonBox.StandardButton"), btn_type)

    _enum("QtGui.QImage.Format").Format_ARGB32_Premultiplied
    _enum("QtGui.QImage.Format").Format_ARGB32_Premultiplied
    # SPECIAL_KEYS are Qt::Key that do *not* return their unicode name instead
    # they have manually specified names.
    SPECIAL_KEYS = {
        _to_int(getattr(_enum("QtCore.Qt.Key"), k)): v
        for k, v in [
            ("Key_Escape", "escape"),
            ("Key_Tab", "tab"),
            ("Key_Backspace", "backspace"),
            ("Key_Return", "enter"),
            ("Key_Enter", "enter"),
            ("Key_Insert", "insert"),
            ("Key_Delete", "delete"),
            ("Key_Pause", "pause"),
            ("Key_SysReq", "sysreq"),
            ("Key_Clear", "clear"),
            ("Key_Home", "home"),
            ("Key_End", "end"),
            ("Key_Left", "left"),
            ("Key_Up", "up"),
            ("Key_Right", "right"),
            ("Key_Down", "down"),
            ("Key_PageUp", "pageup"),
            ("Key_PageDown", "pagedown"),
            ("Key_Shift", "shift"),
            # In OSX, the control and super (aka cmd/apple) keys are switched.
            ("Key_Control", "control" if sys.platform != "darwin" else "cmd"),
            ("Key_Meta", "meta" if sys.platform != "darwin" else "control"),
            ("Key_Alt", "alt"),
            ("Key_CapsLock", "caps_lock"),
            ("Key_F1", "f1"),
            ("Key_F2", "f2"),
            ("Key_F3", "f3"),
            ("Key_F4", "f4"),
            ("Key_F5", "f5"),
            ("Key_F6", "f6"),
            ("Key_F7", "f7"),
            ("Key_F8", "f8"),
            ("Key_F9", "f9"),
            ("Key_F10", "f10"),
            ("Key_F10", "f11"),
            ("Key_F12", "f12"),
            ("Key_Super_L", "super"),
            ("Key_Super_R", "super"),
        ]
    }
    # Define which modifier keys are collected on keyboard events.  Elements
    # are (Qt::KeyboardModifiers, Qt::Key) tuples.  Order determines the
    # modifier order (ctrl+alt+...) reported by Matplotlib.
    _MODIFIER_KEYS = [
        (
            _to_int(getattr(_enum("QtCore.Qt.KeyboardModifier"), mod)),
            _to_int(getattr(_enum("QtCore.Qt.Key"), key)),
        )
        for mod, key in [
            ("ControlModifier", "Key_Control"),
            ("AltModifier", "Key_Alt"),
            ("ShiftModifier", "Key_Shift"),
            ("MetaModifier", "Key_Meta"),
        ]
    ]
    cursord = {
        k: getattr(_enum("QtCore.Qt.CursorShape"), v)
        for k, v in [
            (cursors.MOVE, "SizeAllCursor"),
            (cursors.HAND, "PointingHandCursor"),
            (cursors.POINTER, "ArrowCursor"),
            (cursors.SELECT_REGION, "CrossCursor"),
            (cursors.WAIT, "WaitCursor"),
        ]
    }

    buttond = {
        getattr(_enum("QtCore.Qt.MouseButton"), k): v
        for k, v in [
            ("LeftButton", MouseButton.LEFT),
            ("RightButton", MouseButton.RIGHT),
            ("MiddleButton", MouseButton.MIDDLE),
            ("XButton1", MouseButton.BACK),
            ("XButton2", MouseButton.FORWARD),
        ]
    }

    _enum("QtCore.Qt.WidgetAttribute").WA_OpaquePaintEvent
    _enum("QtCore.Qt.FocusPolicy").StrongFocus
    _enum("QtCore.Qt.ToolBarArea").TopToolBarArea
    _enum("QtCore.Qt.ToolBarArea").TopToolBarArea
    _enum("QtCore.Qt.AlignmentFlag").AlignRight
    _enum("QtCore.Qt.AlignmentFlag").AlignVCenter
    _enum("QtWidgets.QSizePolicy.Policy").Expanding
    _enum("QtWidgets.QSizePolicy.Policy").Ignored
    _enum("QtCore.Qt.MaskMode").MaskOutColor
    _enum("QtCore.Qt.ToolBarArea").TopToolBarArea
    _enum("QtCore.Qt.ToolBarArea").TopToolBarArea
    _enum("QtCore.Qt.AlignmentFlag").AlignRight
    _enum("QtCore.Qt.AlignmentFlag").AlignVCenter
    _enum("QtWidgets.QSizePolicy.Policy").Expanding
    _enum("QtWidgets.QSizePolicy.Policy").Ignored


def _get_testable_qt_backends():
    envs = []
    for deps, env in [
            ([qt_api], {"MPLBACKEND": "qtagg", "QT_API": qt_api})
            for qt_api in ["PyQt6", "PySide6", "PyQt5", "PySide2"]
    ]:
        reason = None
        missing = [dep for dep in deps if not importlib.util.find_spec(dep)]
        if (sys.platform == "linux" and
                not _c_internal_utils.display_is_valid()):
            reason = "$DISPLAY and $WAYLAND_DISPLAY are unset"
        elif missing:
            reason = "{} cannot be imported".format(", ".join(missing))
        elif env["MPLBACKEND"] == 'macosx' and os.environ.get('TF_BUILD'):
            reason = "macosx backend fails on Azure"
        marks = []
        if reason:
            marks.append(pytest.mark.skip(
                reason=f"Skipping {env} because {reason}"))
        envs.append(pytest.param(env, marks=marks, id=str(env)))
    return envs


@pytest.mark.parametrize("env", _get_testable_qt_backends())
def test_enums_available(env):
    proc = subprocess.run(
        [sys.executable, "-c",
         inspect.getsource(_test_enums_impl) + "\n_test_enums_impl()"],
        env={**os.environ, "SOURCE_DATE_EPOCH": "0", **env},
        timeout=_test_timeout, check=True,
        stdout=subprocess.PIPE, universal_newlines=True)
