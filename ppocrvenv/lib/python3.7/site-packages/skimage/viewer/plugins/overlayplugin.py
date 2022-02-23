
from ...util.dtype import dtype_range
from .base import Plugin
from ..utils import ClearColormap, update_axes_image

__all__ = ['OverlayPlugin']


class OverlayPlugin(Plugin):
    """Plugin for ImageViewer that displays an overlay on top of main image.

    The base Plugin class displays the filtered image directly on the viewer.
    OverlayPlugin will instead overlay an image with a transparent colormap.

    See base Plugin class for additional details.

    Attributes
    ----------
    overlay : array
        Overlay displayed on top of image. This overlay defaults to a color map
        with alpha values varying linearly from 0 to 1.
    color : int
        Color of overlay.
    """
    colors = {'red': (1, 0, 0),
              'yellow': (1, 1, 0),
              'green': (0, 1, 0),
              'cyan': (0, 1, 1)}

    def __init__(self, **kwargs):
        super(OverlayPlugin, self).__init__(**kwargs)
        self._overlay_plot = None
        self._overlay = None
        self.cmap = None
        self.color_names = sorted(list(self.colors.keys()))

    def attach(self, image_viewer):
        super(OverlayPlugin, self).attach(image_viewer)
        #TODO: `color` doesn't update GUI widget when set manually.
        self.color = 0

    @property
    def overlay(self):
        return self._overlay

    @overlay.setter
    def overlay(self, image):
        self._overlay = image
        ax = self.image_viewer.ax
        if image is None:
            ax.images.remove(self._overlay_plot)
            self._overlay_plot = None
        elif self._overlay_plot is None:
            vmin, vmax = dtype_range[image.dtype.type]
            self._overlay_plot = ax.imshow(image, cmap=self.cmap,
                                           vmin=vmin, vmax=vmax)
        else:
            update_axes_image(self._overlay_plot, image)

        if self.image_viewer.useblit:
            self.image_viewer._blit_manager.background = None

        self.image_viewer.redraw()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, index):
        # Update colormap whenever color is changed.
        if isinstance(index, str) and \
           index not in self.color_names:
            raise ValueError("%s not defined in OverlayPlugin.colors" % index)
        else:
            name = self.color_names[index]
        self._color = name
        rgb = self.colors[name]
        self.cmap = ClearColormap(rgb)

        if self._overlay_plot is not None:
            self._overlay_plot.set_cmap(self.cmap)
        self.image_viewer.redraw()

    @property
    def filtered_image(self):
        """Return filtered image.

        This "filtered image" is used when saving from the plugin.
        """
        return self.overlay

    def display_filtered_image(self, image):
        """Display filtered image as an overlay on top of image in viewer."""
        self.overlay = image

    def closeEvent(self, event):
        # clear overlay from ImageViewer on close
        self.overlay = None
        super(OverlayPlugin, self).closeEvent(event)

    def output(self):
        """Return the overlaid image.

        Returns
        -------
        overlay : array, same shape as image
            The overlay currently displayed.
        data : None
        """
        return (self.overlay, None)

