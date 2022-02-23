"""
Widgets for interacting with ImageViewer.

These widgets should be added to a Plugin subclass using its `add_widget`
method or calling::

    plugin += Widget(...)

on a Plugin instance. The Plugin will delegate action based on the widget's
parameter type specified by its `ptype` attribute, which can be::

    'arg' : positional argument passed to Plugin's `filter_image` method.
    'kwarg' : keyword argument passed to Plugin's `filter_image` method.
    'plugin' : attribute of Plugin. You'll probably need to add a class
        property of the same name that updates the display.

"""

from .core import *
from .history import *
