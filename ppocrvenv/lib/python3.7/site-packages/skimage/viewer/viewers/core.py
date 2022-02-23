"""
ImageViewer class for viewing and interacting with images.
"""

import numpy as np
from ... import io
from ...util.dtype import dtype_range, img_as_float
from ...exposure import rescale_intensity
from ..qt import QtWidgets, QtGui, Qt, Signal
from ..widgets import Slider
from ..utils import (dialogs, init_qtapp, figimage, start_qtapp,
                     update_axes_image)
from ..utils.canvas import BlitManager, EventManager
from ..plugins.base import Plugin

from warnings import warn

__all__ = ['ImageViewer', 'CollectionViewer']


def mpl_image_to_rgba(mpl_image):
    """Return RGB image from the given matplotlib image object.

    Each image in a matplotlib figure has its own colormap and normalization
    function. Return RGBA (RGB + alpha channel) image with float dtype.

    Parameters
    ----------
    mpl_image : matplotlib.image.AxesImage object
        The image being converted.

    Returns
    -------
    img : array of float, shape (M, N, 4)
        An image of float values in [0, 1].
    """
    image = mpl_image.get_array()
    if image.ndim == 2:
        input_range = (mpl_image.norm.vmin, mpl_image.norm.vmax)
        image = rescale_intensity(image, in_range=input_range)
        # cmap complains on bool arrays
        image = mpl_image.cmap(img_as_float(image))
    elif image.ndim == 3 and image.shape[2] == 3:
        # add alpha channel if it's missing
        image = np.dstack((image, np.ones_like(image)))
    return img_as_float(image)


class ImageViewer(QtWidgets.QMainWindow):
    """Viewer for displaying images.

    This viewer is a simple container object that holds a Matplotlib axes
    for showing images. `ImageViewer` doesn't subclass the Matplotlib axes (or
    figure) because of the high probability of name collisions.

    Subclasses and plugins will likely extend the `update_image` method to add
    custom overlays or filter the displayed image.

    Parameters
    ----------
    image : array
        Image being viewed.

    Attributes
    ----------
    canvas, fig, ax : Matplotlib canvas, figure, and axes
        Matplotlib canvas, figure, and axes used to display image.
    image : array
        Image being viewed. Setting this value will update the displayed frame.
    original_image : array
        Plugins typically operate on (but don't change) the *original* image.
    plugins : list
        List of attached plugins.

    Examples
    --------
    >>> from skimage import data
    >>> image = data.coins()
    >>> viewer = ImageViewer(image) # doctest: +SKIP
    >>> viewer.show()               # doctest: +SKIP

    """

    dock_areas = {'top': Qt.TopDockWidgetArea,
                  'bottom': Qt.BottomDockWidgetArea,
                  'left': Qt.LeftDockWidgetArea,
                  'right': Qt.RightDockWidgetArea}

    # Signal that the original image has been changed
    original_image_changed = Signal(np.ndarray)

    def __init__(self, image, useblit=True):

        warn('`viewer` is deprecated and will be removed in 0.20. '
             'For alternatives, refer to '
             'https://scikit-image.org/docs/stable/user_guide/visualization.html',
             FutureWarning, stacklevel=2)

        # Start main loop
        init_qtapp()
        super(ImageViewer, self).__init__()

        # TODO: Add ImageViewer to skimage.io window manager

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Image Viewer")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('Open file', self.open_file,
                                 Qt.CTRL + Qt.Key_O)
        self.file_menu.addAction('Save to file', self.save_to_file,
                                 Qt.CTRL + Qt.Key_S)
        self.file_menu.addAction('Quit', self.close,
                                 Qt.CTRL + Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        if isinstance(image, Plugin):
            plugin = image
            image = plugin.filtered_image
            plugin.image_changed.connect(self._update_original_image)
            # When plugin is started, start
            plugin._started.connect(self._show)

        self.fig, self.ax = figimage(image)
        self.canvas = self.fig.canvas
        self.canvas.setParent(self)
        self.ax.autoscale(enable=False)

        self._tools = []
        self.useblit = useblit
        if useblit:
            self._blit_manager = BlitManager(self.ax)
        self._event_manager = EventManager(self.ax)

        self._image_plot = self.ax.images[0]
        self._update_original_image(image)
        self.plugins = []

        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.layout.addWidget(self.canvas)

        status_bar = self.statusBar()
        self.status_message = status_bar.showMessage
        sb_size = status_bar.sizeHint()
        cs_size = self.canvas.sizeHint()
        self.resize(cs_size.width(), cs_size.height() + sb_size.height())

        self.connect_event('motion_notify_event', self._update_status_bar)

    def __add__(self, plugin):
        """Add plugin to ImageViewer"""
        plugin.attach(self)
        self.original_image_changed.connect(plugin._update_original_image)

        if plugin.dock:
            location = self.dock_areas[plugin.dock]
            dock_location = Qt.DockWidgetArea(location)
            dock = QtWidgets.QDockWidget()
            dock.setWidget(plugin)
            dock.setWindowTitle(plugin.name)
            self.addDockWidget(dock_location, dock)

            horiz = (self.dock_areas['left'], self.dock_areas['right'])
            dimension = 'width' if location in horiz else 'height'
            self._add_widget_size(plugin, dimension=dimension)

        return self

    def _add_widget_size(self, widget, dimension='width'):
        widget_size = widget.sizeHint()
        viewer_size = self.frameGeometry()

        dx = dy = 0
        if dimension == 'width':
            dx = widget_size.width()
        elif dimension == 'height':
            dy = widget_size.height()

        w = viewer_size.width()
        h = viewer_size.height()
        self.resize(w + dx, h + dy)

    def open_file(self, filename=None):
        """Open image file and display in viewer."""
        if filename is None:
            filename = dialogs.open_file_dialog()
        if filename is None:
            return
        image = io.imread(filename)
        self._update_original_image(image)

    def update_image(self, image):
        """Update displayed image.

        This method can be overridden or extended in subclasses and plugins to
        react to image changes.
        """
        self._update_original_image(image)

    def _update_original_image(self, image):
        self.original_image = image     # update saved image
        self.image = image.copy()       # update displayed image
        self.original_image_changed.emit(image)

    def save_to_file(self, filename=None):
        """Save current image to file.

        The current behavior is not ideal: It saves the image displayed on
        screen, so all images will be converted to RGB, and the image size is
        not preserved (resizing the viewer window will alter the size of the
        saved image).
        """
        if filename is None:
            filename = dialogs.save_file_dialog()
        if filename is None:
            return
        if len(self.ax.images) == 1:
            io.imsave(filename, self.image)
        else:
            underlay = mpl_image_to_rgba(self.ax.images[0])
            overlay = mpl_image_to_rgba(self.ax.images[1])
            alpha = overlay[:, :, 3]

            # alpha can be set by channel of array or by a scalar value.
            # Prefer the alpha channel, but fall back to scalar value.
            if np.all(alpha == 1):
                alpha = np.ones_like(alpha) * self.ax.images[1].get_alpha()

            alpha = alpha[:, :, np.newaxis]
            composite = (overlay[:, :, :3] * alpha +
                         underlay[:, :, :3] * (1 - alpha))
            io.imsave(filename, composite)

    def closeEvent(self, event):
        self.close()

    def _show(self, x=0):
        self.move(x, 0)
        for p in self.plugins:
            p.show()
        super(ImageViewer, self).show()
        self.activateWindow()
        self.raise_()

    def show(self, main_window=True):
        """Show ImageViewer and attached plugins.

        This behaves much like `matplotlib.pyplot.show` and `QWidget.show`.
        """
        self._show()
        if main_window:
            start_qtapp()
        return [p.output() for p in self.plugins]

    def redraw(self):
        if self.useblit:
            self._blit_manager.redraw()
        else:
            self.canvas.draw_idle()

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, image):
        self._img = image
        update_axes_image(self._image_plot, image)

        # update display (otherwise image doesn't fill the canvas)
        h, w = image.shape[:2]
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        # update color range
        clim = dtype_range[image.dtype.type]
        if clim[0] < 0 and image.min() >= 0:
            clim = (0, clim[1])
        self._image_plot.set_clim(clim)

        if self.useblit:
            self._blit_manager.background = None

        self.redraw()

    def reset_image(self):
        self.image = self.original_image.copy()

    def connect_event(self, event, callback):
        """Connect callback function to matplotlib event and return id."""
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        """Disconnect callback by its id (returned by `connect_event`)."""
        self.canvas.mpl_disconnect(callback_id)

    def _update_status_bar(self, event):
        if event.inaxes and event.inaxes.get_navigate():
            self.status_message(self._format_coord(event.xdata, event.ydata))
        else:
            self.status_message('')

    def add_tool(self, tool):
        if self.useblit:
            self._blit_manager.add_artists(tool.artists)
        self._tools.append(tool)
        self._event_manager.attach(tool)

    def remove_tool(self, tool):
        if tool not in self._tools:
            return
        if self.useblit:
            self._blit_manager.remove_artists(tool.artists)
        self._tools.remove(tool)
        self._event_manager.detach(tool)

    def _format_coord(self, x, y):
        # callback function to format coordinate display in status bar
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%4s @ [%4s, %4s]" % (self.image[y, x], x, y)
        except IndexError:
            return ""


class CollectionViewer(ImageViewer):
    """Viewer for displaying image collections.

    Select the displayed frame of the image collection using the slider or
    with the following keyboard shortcuts:

        left/right arrows
            Previous/next image in collection.
        number keys, 0--9
            0% to 90% of collection. For example, "5" goes to the image in the
            middle (i.e. 50%) of the collection.
        home/end keys
            First/last image in collection.

    Parameters
    ----------
    image_collection : list of images
        List of images to be displayed.
    update_on : {'move' | 'release'}
        Control whether image is updated on slide or release of the image
        slider. Using 'on_release' will give smoother behavior when displaying
        large images or when writing a plugin/subclass that requires heavy
        computation.
    """

    def __init__(self, image_collection, update_on='move', **kwargs):
        warn('`CollectionViewer` is deprecated and will be removed in 0.20. '
             'For alternatives, refer to '
             'https://scikit-image.org/docs/stable/user_guide/visualization.html',
             FutureWarning, stacklevel=2)

        self.image_collection = image_collection
        self.index = 0
        self.num_images = len(self.image_collection)

        first_image = image_collection[0]
        super(CollectionViewer, self).__init__(first_image)

        slider_kws = dict(value=0, low=0, high=self.num_images - 1)
        slider_kws['update_on'] = update_on
        slider_kws['callback'] = self.update_index
        slider_kws['value_type'] = 'int'
        self.slider = Slider('frame', **slider_kws)
        self.layout.addWidget(self.slider)

        # TODO: Adjust height to accommodate slider; the following doesn't work
        # s_size = self.slider.sizeHint()
        # cs_size = self.canvas.sizeHint()
        # self.resize(cs_size.width(), cs_size.height() + s_size.height())

    def update_index(self, name, index):
        """Select image on display using index into image collection."""
        index = int(round(index))

        if index == self.index:
            return

        # clip index value to collection limits
        index = max(index, 0)
        index = min(index, self.num_images - 1)

        self.index = index
        self.slider.val = index
        self.update_image(self.image_collection[index])

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            key = event.key()
            # Number keys (code: 0 = key 48, 9 = key 57) move to deciles
            if 48 <= key < 58:
                index = int(0.1 * (key - 48) * self.num_images)
                self.update_index('', index)
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()
