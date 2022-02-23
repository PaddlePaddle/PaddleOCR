import functools
import inspect
import os
import platform
import re
import subprocess
import sys

import pytest

_test_timeout = 60  # A reasonably safe value for slower architectures.


def _isolated_tk_test(success_count, func=None):
    """
    A decorator to run *func* in a subprocess and assert that it prints
    "success" *success_count* times and nothing on stderr.

    TkAgg tests seem to have interactions between tests, so isolate each test
    in a subprocess. See GH#18261

    The decorated function must be fully self-contained, and thus perform
    all the imports it needs.  Because its source is extracted and run by
    itself, coverage will consider it as not being run, so it should be marked
    with ``# pragma: no cover``
    """

    if func is None:
        return functools.partial(_isolated_tk_test, success_count)

    # Remove decorators.
    source = re.search(r"(?ms)^def .*", inspect.getsource(func)).group(0)

    @functools.wraps(func)
    def test_func():
        try:
            proc = subprocess.run(
                [sys.executable, "-c", f"{source}\n{func.__name__}()"],
                env={**os.environ, "MPLBACKEND": "TkAgg"},
                timeout=_test_timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                universal_newlines=True,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Subprocess timed out")
        except subprocess.CalledProcessError:
            pytest.fail("Subprocess failed to test intended behavior")
        else:
            # macOS may actually emit irrelevant errors about Accelerated
            # OpenGL vs. software OpenGL, so suppress them.
            # Asserting stderr first (and printing it on failure) should be
            # more helpful for debugging that printing a failed success count.
            assert not [line for line in proc.stderr.splitlines()
                        if "OpenGL" not in line]
            assert proc.stdout.count("success") == success_count

    return test_func


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
@_isolated_tk_test(success_count=6)  # len(bad_boxes)
def test_blit():  # pragma: no cover
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends import _tkagg

    fig, ax = plt.subplots()
    photoimage = fig.canvas._tkphoto
    data = np.ones((4, 4, 4))
    height, width = data.shape[:2]
    dataptr = (height, width, data.ctypes.data)
    # Test out of bounds blitting.
    bad_boxes = ((-1, 2, 0, 2),
                 (2, 0, 0, 2),
                 (1, 6, 0, 2),
                 (0, 2, -1, 2),
                 (0, 2, 2, 0),
                 (0, 2, 1, 6))
    for bad_box in bad_boxes:
        try:
            _tkagg.blit(
                photoimage.tk.interpaddr(), str(photoimage), dataptr, 0,
                (0, 1, 2, 3), bad_box)
        except ValueError:
            print("success")


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
@_isolated_tk_test(success_count=1)
def test_figuremanager_preserves_host_mainloop():  # pragma: no cover
    import tkinter
    import matplotlib.pyplot as plt
    success = []

    def do_plot():
        plt.figure()
        plt.plot([1, 2], [3, 5])
        plt.close()
        root.after(0, legitimate_quit)

    def legitimate_quit():
        root.quit()
        success.append(True)

    root = tkinter.Tk()
    root.after(0, do_plot)
    root.mainloop()

    if success:
        print("success")


@pytest.mark.skipif(platform.python_implementation() != 'CPython',
                    reason='PyPy does not support Tkinter threading: '
                           'https://foss.heptapod.net/pypy/pypy/-/issues/1929')
@pytest.mark.backend('TkAgg', skip_on_importerror=True)
@pytest.mark.flaky(reruns=3)
@_isolated_tk_test(success_count=1)
def test_figuremanager_cleans_own_mainloop():  # pragma: no cover
    import tkinter
    import time
    import matplotlib.pyplot as plt
    import threading
    from matplotlib.cbook import _get_running_interactive_framework

    root = tkinter.Tk()
    plt.plot([1, 2, 3], [1, 2, 5])

    def target():
        while not 'tk' == _get_running_interactive_framework():
            time.sleep(.01)
        plt.close()
        if show_finished_event.wait():
            print('success')

    show_finished_event = threading.Event()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    plt.show(block=True)  # Testing if this function hangs.
    show_finished_event.set()
    thread.join()


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
@pytest.mark.flaky(reruns=3)
@_isolated_tk_test(success_count=0)
def test_never_update():  # pragma: no cover
    import tkinter
    del tkinter.Misc.update
    del tkinter.Misc.update_idletasks

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.show(block=False)

    plt.draw()  # Test FigureCanvasTkAgg.
    fig.canvas.toolbar.configure_subplots()  # Test NavigationToolbar2Tk.

    # Check for update() or update_idletasks() in the event queue, functionally
    # equivalent to tkinter.Misc.update.
    # Must pause >= 1 ms to process tcl idle events plus extra time to avoid
    # flaky tests on slow systems.
    plt.pause(0.1)

    plt.close(fig)  # Test FigureCanvasTk filter_destroy callback

    # Note that exceptions would be printed to stderr; _isolated_tk_test
    # checks them.


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
@_isolated_tk_test(success_count=2)
def test_missing_back_button():  # pragma: no cover
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

    class Toolbar(NavigationToolbar2Tk):
        # Only display the buttons we need.
        toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                     t[0] in ('Home', 'Pan', 'Zoom')]

    fig = plt.figure()
    print("success")
    Toolbar(fig.canvas, fig.canvas.manager.window)  # This should not raise.
    print("success")
