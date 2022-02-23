"""
Classes used for blocking interaction with figure windows:

`BlockingInput`
    Creates a callable object to retrieve events in a blocking way for
    interactive sessions.  Base class of the other classes listed here.

`BlockingKeyMouseInput`
    Creates a callable object to retrieve key or mouse clicks in a blocking
    way for interactive sessions.  Used by `~.Figure.waitforbuttonpress`.

`BlockingMouseInput`
    Creates a callable object to retrieve mouse clicks in a blocking way for
    interactive sessions.  Used by `~.Figure.ginput`.

`BlockingContourLabeler`
    Creates a callable object to retrieve mouse clicks in a blocking way that
    will then be used to place labels on a `.ContourSet`.  Used by
    `~.Axes.clabel`.
"""

import logging
from numbers import Integral

from matplotlib import _api
from matplotlib.backend_bases import MouseButton
import matplotlib.lines as mlines

_api.warn_deprecated("3.5", name=__name__, obj_type="module")
_log = logging.getLogger(__name__)


class BlockingInput:
    """Callable for retrieving events in a blocking way."""

    def __init__(self, fig, eventslist=()):
        self.fig = fig
        self.eventslist = eventslist

    def on_event(self, event):
        """
        Event handler; will be passed to the current figure to retrieve events.
        """
        # Add a new event to list - using a separate function is overkill for
        # the base class, but this is consistent with subclasses.
        self.add_event(event)
        _log.info("Event %i", len(self.events))

        # This will extract info from events.
        self.post_event()

        # Check if we have enough events already.
        if len(self.events) >= self.n > 0:
            self.fig.canvas.stop_event_loop()

    def post_event(self):
        """For baseclass, do nothing but collect events."""

    def cleanup(self):
        """Disconnect all callbacks."""
        for cb in self.callbacks:
            self.fig.canvas.mpl_disconnect(cb)
        self.callbacks = []

    def add_event(self, event):
        """For base class, this just appends an event to events."""
        self.events.append(event)

    def pop_event(self, index=-1):
        """
        Remove an event from the event list -- by default, the last.

        Note that this does not check that there are events, much like the
        normal pop method.  If no events exist, this will throw an exception.
        """
        self.events.pop(index)

    pop = pop_event

    def __call__(self, n=1, timeout=30):
        """Blocking call to retrieve *n* events."""
        _api.check_isinstance(Integral, n=n)
        self.n = n
        self.events = []

        if self.fig.canvas.manager:
            # Ensure that the figure is shown, if we are managing it.
            self.fig.show()
        # Connect the events to the on_event function call.
        self.callbacks = [self.fig.canvas.mpl_connect(name, self.on_event)
                          for name in self.eventslist]
        try:
            # Start event loop.
            self.fig.canvas.start_event_loop(timeout=timeout)
        finally:  # Run even on exception like ctrl-c.
            # Disconnect the callbacks.
            self.cleanup()
        # Return the events in this case.
        return self.events


class BlockingMouseInput(BlockingInput):
    """
    Callable for retrieving mouse clicks in a blocking way.

    This class will also retrieve keypresses and map them to mouse clicks:
    delete and backspace are a right click, enter is like a middle click,
    and all others are like a left click.
    """

    button_add = MouseButton.LEFT
    button_pop = MouseButton.RIGHT
    button_stop = MouseButton.MIDDLE

    def __init__(self, fig,
                 mouse_add=MouseButton.LEFT,
                 mouse_pop=MouseButton.RIGHT,
                 mouse_stop=MouseButton.MIDDLE):
        super().__init__(fig=fig,
                         eventslist=('button_press_event', 'key_press_event'))
        self.button_add = mouse_add
        self.button_pop = mouse_pop
        self.button_stop = mouse_stop

    def post_event(self):
        """Process an event."""
        if len(self.events) == 0:
            _log.warning("No events yet")
        elif self.events[-1].name == 'key_press_event':
            self.key_event()
        else:
            self.mouse_event()

    def mouse_event(self):
        """Process a mouse click event."""
        event = self.events[-1]
        button = event.button
        if button == self.button_pop:
            self.mouse_event_pop(event)
        elif button == self.button_stop:
            self.mouse_event_stop(event)
        elif button == self.button_add:
            self.mouse_event_add(event)

    def key_event(self):
        """
        Process a key press event, mapping keys to appropriate mouse clicks.
        """
        event = self.events[-1]
        if event.key is None:
            # At least in OSX gtk backend some keys return None.
            return
        if event.key in ['backspace', 'delete']:
            self.mouse_event_pop(event)
        elif event.key in ['escape', 'enter']:
            self.mouse_event_stop(event)
        else:
            self.mouse_event_add(event)

    def mouse_event_add(self, event):
        """
        Process an button-1 event (add a click if inside axes).

        Parameters
        ----------
        event : `~.backend_bases.MouseEvent`
        """
        if event.inaxes:
            self.add_click(event)
        else:  # If not a valid click, remove from event list.
            BlockingInput.pop(self)

    def mouse_event_stop(self, event):
        """
        Process an button-2 event (end blocking input).

        Parameters
        ----------
        event : `~.backend_bases.MouseEvent`
        """
        # Remove last event just for cleanliness.
        BlockingInput.pop(self)
        # This will exit even if not in infinite mode.  This is consistent with
        # MATLAB and sometimes quite useful, but will require the user to test
        # how many points were actually returned before using data.
        self.fig.canvas.stop_event_loop()

    def mouse_event_pop(self, event):
        """
        Process an button-3 event (remove the last click).

        Parameters
        ----------
        event : `~.backend_bases.MouseEvent`
        """
        # Remove this last event.
        BlockingInput.pop(self)
        # Now remove any existing clicks if possible.
        if self.events:
            self.pop(event)

    def add_click(self, event):
        """
        Add the coordinates of an event to the list of clicks.

        Parameters
        ----------
        event : `~.backend_bases.MouseEvent`
        """
        self.clicks.append((event.xdata, event.ydata))
        _log.info("input %i: %f, %f",
                  len(self.clicks), event.xdata, event.ydata)
        # If desired, plot up click.
        if self.show_clicks:
            line = mlines.Line2D([event.xdata], [event.ydata],
                                 marker='+', color='r')
            event.inaxes.add_line(line)
            self.marks.append(line)
            self.fig.canvas.draw()

    def pop_click(self, event, index=-1):
        """
        Remove a click (by default, the last) from the list of clicks.

        Parameters
        ----------
        event : `~.backend_bases.MouseEvent`
        """
        self.clicks.pop(index)
        if self.show_clicks:
            self.marks.pop(index).remove()
            self.fig.canvas.draw()

    def pop(self, event, index=-1):
        """
        Remove a click and the associated event from the list of clicks.

        Defaults to the last click.
        """
        self.pop_click(event, index)
        super().pop(index)

    def cleanup(self, event=None):
        """
        Parameters
        ----------
        event : `~.backend_bases.MouseEvent`, optional
            Not used
        """
        # Clean the figure.
        if self.show_clicks:
            for mark in self.marks:
                mark.remove()
            self.marks = []
            self.fig.canvas.draw()
        # Call base class to remove callbacks.
        super().cleanup()

    def __call__(self, n=1, timeout=30, show_clicks=True):
        """
        Blocking call to retrieve *n* coordinate pairs through mouse clicks.
        """
        self.show_clicks = show_clicks
        self.clicks = []
        self.marks = []
        super().__call__(n=n, timeout=timeout)
        return self.clicks


class BlockingContourLabeler(BlockingMouseInput):
    """
    Callable for retrieving mouse clicks and key presses in a blocking way.

    Used to place contour labels.
    """

    def __init__(self, cs):
        self.cs = cs
        super().__init__(fig=cs.axes.figure)

    def add_click(self, event):
        self.button1(event)

    def pop_click(self, event, index=-1):
        self.button3(event)

    def button1(self, event):
        """
        Process an button-1 event (add a label to a contour).

        Parameters
        ----------
        event : `~.backend_bases.MouseEvent`
        """
        # Shorthand
        if event.inaxes == self.cs.ax:
            self.cs.add_label_near(event.x, event.y, self.inline,
                                   inline_spacing=self.inline_spacing,
                                   transform=False)
            self.fig.canvas.draw()
        else:  # Remove event if not valid
            BlockingInput.pop(self)

    def button3(self, event):
        """
        Process an button-3 event (remove a label if not in inline mode).

        Unfortunately, if one is doing inline labels, then there is currently
        no way to fix the broken contour - once humpty-dumpty is broken, he
        can't be put back together.  In inline mode, this does nothing.

        Parameters
        ----------
        event : `~.backend_bases.MouseEvent`
        """
        if self.inline:
            pass
        else:
            self.cs.pop_label()
            self.cs.ax.figure.canvas.draw()

    def __call__(self, inline, inline_spacing=5, n=-1, timeout=-1):
        self.inline = inline
        self.inline_spacing = inline_spacing
        super().__call__(n=n, timeout=timeout, show_clicks=False)


class BlockingKeyMouseInput(BlockingInput):
    """
    Callable for retrieving mouse clicks and key presses in a blocking way.
    """

    def __init__(self, fig):
        super().__init__(fig=fig,
                         eventslist=('button_press_event', 'key_press_event'))

    def post_event(self):
        """Determine if it is a key event."""
        if self.events:
            self.keyormouse = self.events[-1].name == 'key_press_event'
        else:
            _log.warning("No events yet.")

    def __call__(self, timeout=30):
        """
        Blocking call to retrieve a single mouse click or key press.

        Returns ``True`` if key press, ``False`` if mouse click, or ``None`` if
        timed out.
        """
        self.keyormouse = None
        super().__call__(n=1, timeout=timeout)

        return self.keyormouse
