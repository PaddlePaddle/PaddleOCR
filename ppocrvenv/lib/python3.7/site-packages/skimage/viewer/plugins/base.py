"""
Base class for Plugins that interact with ImageViewer.
"""
from warnings import warn

import numpy as np
from ..qt import QtWidgets, QtCore, Signal
from ..utils import RequiredAttr, init_qtapp


class Plugin(QtWidgets.QDialog):
    """Base class for plugins that interact with an ImageViewer.

    A plugin connects an image filter (or another function) to an image viewer.
    Note that a Plugin is initialized *without* an image viewer and attached in
    a later step. See example below for details.

    Parameters
    ----------
    image_viewer : ImageViewer
        Window containing image used in measurement/manipulation.
    image_filter : function
        Function that gets called to update image in image viewer. This value
        can be `None` if, for example, you have a plugin that extracts
        information from an image and doesn't manipulate it. Alternatively,
        this function can be defined as a method in a Plugin subclass.
    height, width : int
        Size of plugin window in pixels. Note that Qt will automatically resize
        a window to fit components. So if you're adding rows of components, you
        can leave `height = 0` and just let Qt determine the final height.
    useblit : bool
        If True, use blitting to speed up animation. Only available on some
        Matplotlib backends. If None, set to True when using Agg backend.
        This only has an effect if you draw on top of an image viewer.

    Attributes
    ----------
    image_viewer : ImageViewer
        Window containing image used in measurement.
    name : str
        Name of plugin. This is displayed as the window title.
    artist : list
        List of Matplotlib artists and canvastools. Any artists created by the
        plugin should be added to this list so that it gets cleaned up on
        close.

    Examples
    --------
    >>> from skimage.viewer import ImageViewer
    >>> from skimage.viewer.widgets import Slider
    >>> from skimage import data
    >>>
    >>> plugin = Plugin(image_filter=lambda img,
    ...                 threshold: img > threshold) # doctest: +SKIP
    >>> plugin += Slider('threshold', 0, 255)       # doctest: +SKIP
    >>>
    >>> image = data.coins()
    >>> viewer = ImageViewer(image)       # doctest: +SKIP
    >>> viewer += plugin                  # doctest: +SKIP
    >>> thresholded = viewer.show()[0][0] # doctest: +SKIP

    The plugin will automatically delegate parameters to `image_filter` based
    on its parameter type, i.e., `ptype` (widgets for required arguments must
    be added in the order they appear in the function). The image attached
    to the viewer is **automatically passed as the first argument** to the
    filter function.

    #TODO: Add flag so image is not passed to filter function by default.

    `ptype = 'kwarg'` is the default for most widgets so it's unnecessary here.

    """
    name = 'Plugin'
    image_viewer = RequiredAttr("%s is not attached to ImageViewer" % name)

    # Signals used when viewers are linked to the Plugin output.
    image_changed = Signal(np.ndarray)
    _started = Signal(int)

    def __init__(self, image_filter=None, height=0, width=400, useblit=True,
                 dock='bottom'):
        init_qtapp()
        super(Plugin, self).__init__()

        self.dock = dock

        self.image_viewer = None
        # If subclass defines `image_filter` method ignore input.
        if not hasattr(self, 'image_filter'):
            self.image_filter = image_filter
        elif image_filter is not None:
            warn("If the Plugin class defines an `image_filter` method, "
                 "then the `image_filter` argument is ignored.")

        self.setWindowTitle(self.name)
        self.layout = QtWidgets.QGridLayout(self)
        self.resize(width, height)
        self.row = 0

        self.arguments = []
        self.keyword_arguments = {}

        self.useblit = useblit
        self.cids = []
        self.artists = []

    def attach(self, image_viewer):
        """Attach the plugin to an ImageViewer.

        Note that the ImageViewer will automatically call this method when the
        plugin is added to the ImageViewer. For example::

            viewer += Plugin(...)

        Also note that `attach` automatically calls the filter function so that
        the image matches the filtered value specified by attached widgets.
        """
        self.setParent(image_viewer)
        self.setWindowFlags(QtCore.Qt.Dialog)

        self.image_viewer = image_viewer
        self.image_viewer.plugins.append(self)
        #TODO: Always passing image as first argument may be bad assumption.
        self.arguments = [self.image_viewer.original_image]

        # Call filter so that filtered image matches widget values
        self.filter_image()

    def add_widget(self, widget):
        """Add widget to plugin.

        Alternatively, Plugin's `__add__` method is overloaded to add widgets::

            plugin += Widget(...)

        Widgets can adjust required or optional arguments of filter function or
        parameters for the plugin. This is specified by the Widget's `ptype`.
        """
        if widget.ptype == 'kwarg':
            name = widget.name.replace(' ', '_')
            self.keyword_arguments[name] = widget
            widget.callback = self.filter_image
        elif widget.ptype == 'arg':
            self.arguments.append(widget)
            widget.callback = self.filter_image
        elif widget.ptype == 'plugin':
            widget.callback = self.update_plugin
        widget.plugin = self
        self.layout.addWidget(widget, self.row, 0)
        self.row += 1

    def __add__(self, widget):
        self.add_widget(widget)
        return self

    def filter_image(self, *widget_arg):
        """Call `image_filter` with widget args and kwargs

        Note: `display_filtered_image` is automatically called.
        """
        # `widget_arg` is passed by the active widget but is unused since all
        # filter arguments are pulled directly from attached the widgets.

        if self.image_filter is None:
            return
        arguments = [self._get_value(a) for a in self.arguments]
        kwargs = {name: self._get_value(a)
                  for name, a in self.keyword_arguments.items()}
        filtered = self.image_filter(*arguments, **kwargs)

        self.display_filtered_image(filtered)
        self.image_changed.emit(filtered)

    def _get_value(self, param):
        # If param is a widget, return its `val` attribute.
        return param if not hasattr(param, 'val') else param.val

    def _update_original_image(self, image):
        """Update the original image argument passed to the filter function.

        This method is called by the viewer when the original image is updated.
        """
        self.arguments[0] = image
        self._on_new_image(image)
        self.filter_image()

    def _on_new_image(self, image):
        """Override this method to update your plugin for new images."""
        pass

    @property
    def filtered_image(self):
        """Return filtered image."""
        return self.image_viewer.image

    def display_filtered_image(self, image):
        """Display the filtered image on image viewer.

        If you don't want to simply replace the displayed image with the
        filtered image (e.g., you want to display a transparent overlay),
        you can override this method.
        """
        self.image_viewer.image = image

    def update_plugin(self, name, value):
        """Update keyword parameters of the plugin itself.

        These parameters will typically be implemented as class properties so
        that they update the image or some other component.
        """
        setattr(self, name, value)

    def show(self, main_window=True):
        """Show plugin."""
        super(Plugin, self).show()
        self.activateWindow()
        self.raise_()

        # Emit signal with x-hint so new windows can be displayed w/o overlap.
        size = self.frameGeometry()
        x_hint = size.x() + size.width()
        self._started.emit(x_hint)

    def closeEvent(self, event):
        """On close disconnect all artists and events from ImageViewer.

        Note that artists must be appended to `self.artists`.
        """
        self.clean_up()
        self.close()

    def clean_up(self):
        self.remove_image_artists()
        if self in self.image_viewer.plugins:
            self.image_viewer.plugins.remove(self)
        self.image_viewer.reset_image()
        self.image_viewer.redraw()

    def remove_image_artists(self):
        """Remove artists that are connected to the image viewer."""
        for a in self.artists:
            a.remove()

    def output(self):
        """Return the plugin's representation and data.

        Returns
        -------
        image : array, same shape as ``self.image_viewer.image``, or None
            The filtered image.
        data : None
            Any data associated with the plugin.

        Notes
        -----
        Derived classes should override this method to return a tuple
        containing an *overlay* of the same shape of the image, and a
        *data* object. Either of these is optional: return ``None`` if
        you don't want to return a value.
        """
        return (self.image_viewer.image, None)
