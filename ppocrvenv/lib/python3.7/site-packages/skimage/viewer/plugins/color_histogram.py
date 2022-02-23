import numpy as np
import matplotlib.pyplot as plt
from ... import color, exposure
from .plotplugin import PlotPlugin
from ..canvastools import RectangleTool


class ColorHistogram(PlotPlugin):
    name = 'Color Histogram'

    def __init__(self, max_pct=0.99, **kwargs):
        super(ColorHistogram, self).__init__(height=400, **kwargs)
        self.max_pct = max_pct

        print(self.help())

    def attach(self, image_viewer):
        super(ColorHistogram, self).attach(image_viewer)

        self.rect_tool = RectangleTool(self,
                                       on_release=self.ab_selected)
        self._on_new_image(image_viewer.image)

    def _on_new_image(self, image):
        self.lab_image = color.rgb2lab(image)

        # Calculate color histogram in the Lab colorspace:
        L, a, b = self.lab_image.T
        left, right = -100, 100
        ab_extents = [left - 0.5, right + 0.5, right + 0.5, left - 0.5]
        self.mask = np.ones(L.shape, bool)
        bins = np.arange(left, right)
        hist, x_edges, y_edges = np.histogram2d(a.flatten(), b.flatten(),
                                                bins, normed=True)
        self.data = {'bins': bins, 'hist': hist, 'edges': (x_edges, y_edges),
                     'extents': (left, right, left, right)}
        # Clip bin heights that dominate a-b histogram
        max_val = pct_total_area(hist, percentile=self.max_pct)
        hist = exposure.rescale_intensity(hist, in_range=(0, max_val))
        self.ax.imshow(hist, extent=ab_extents, cmap=plt.cm.gray)

        self.ax.set_title('Color Histogram')
        self.ax.set_xlabel('b')
        self.ax.set_ylabel('a')

    def help(self):
        helpstr = ("Color Histogram tool:",
                   "Select region of a-b colorspace to highlight on image.")
        return '\n'.join(helpstr)

    def ab_selected(self, extents):
        x0, x1, y0, y1 = extents
        self.data['extents'] = extents

        lab_masked = self.lab_image.copy()
        L, a, b = lab_masked.T

        self.mask = ((a > y0) & (a < y1)) & ((b > x0) & (b < x1))
        lab_masked[..., 1:][~self.mask.T] = 0

        self.image_viewer.image = color.lab2rgb(lab_masked)

    def output(self):
        """Return the image mask and the histogram data.

        Returns
        -------
        mask : array of bool, same shape as image
            The selected pixels.
        data : dict
            The data describing the histogram and the selected region.
            The dictionary contains:

              - 'bins' : array of float
                The bin boundaries for both `a` and `b` channels.
              - 'hist' : 2D array of float
                The normalized histogram.
              - 'edges' : tuple of array of float
                The bin edges along each dimension
              - 'extents' : tuple of float
                The left and right and top and bottom of the selected region.
        """
        return (self.mask, self.data)


def pct_total_area(image, percentile=0.80):
    """Return threshold value based on percentage of total area.

    The specified percent of pixels less than the given intensity threshold.
    """
    idx = int((image.size - 1) * percentile)
    sorted_pixels = np.sort(image.flat)
    return sorted_pixels[idx]
