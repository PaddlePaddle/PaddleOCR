"""
========================
Widget testing utilities
========================

See also :mod:`matplotlib.tests.test_widgets`.
"""

import matplotlib.pyplot as plt
from unittest import mock


def get_ax():
    """Create a plot and return its axes."""
    fig, ax = plt.subplots(1, 1)
    ax.plot([0, 200], [0, 200])
    ax.set_aspect(1.0)
    ax.figure.canvas.draw()
    return ax


def mock_event(ax, button=1, xdata=0, ydata=0, key=None, step=1):
    r"""
    Create a mock event that can stand in for `.Event` and its subclasses.

    This event is intended to be used in tests where it can be passed into
    event handling functions.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The axes the event will be in.
    xdata : int
        x coord of mouse in data coords.
    ydata : int
        y coord of mouse in data coords.
    button : None or `MouseButton` or {'up', 'down'}
        The mouse button pressed in this event (see also `.MouseEvent`).
    key : None or str
        The key pressed when the mouse event triggered (see also `.KeyEvent`).
    step : int
        Number of scroll steps (positive for 'up', negative for 'down').

    Returns
    -------
    event
        A `.Event`\-like Mock instance.
    """
    event = mock.Mock()
    event.button = button
    event.x, event.y = ax.transData.transform([(xdata, ydata),
                                               (xdata, ydata)])[0]
    event.xdata, event.ydata = xdata, ydata
    event.inaxes = ax
    event.canvas = ax.figure.canvas
    event.key = key
    event.step = step
    event.guiEvent = None
    event.name = 'Custom'
    return event


def do_event(tool, etype, button=1, xdata=0, ydata=0, key=None, step=1):
    """
    Trigger an event on the given tool.

    Parameters
    ----------
    tool : matplotlib.widgets.RectangleSelector
    etype : str
        The event to trigger.
    xdata : int
        x coord of mouse in data coords.
    ydata : int
        y coord of mouse in data coords.
    button : None or `MouseButton` or {'up', 'down'}
        The mouse button pressed in this event (see also `.MouseEvent`).
    key : None or str
        The key pressed when the mouse event triggered (see also `.KeyEvent`).
    step : int
        Number of scroll steps (positive for 'up', negative for 'down').
    """
    event = mock_event(tool.ax, button, xdata, ydata, key, step)
    func = getattr(tool, etype)
    func(event)
