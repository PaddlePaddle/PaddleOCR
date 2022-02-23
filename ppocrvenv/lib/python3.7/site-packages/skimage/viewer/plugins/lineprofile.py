
import numpy as np
from ...util.dtype import dtype_range
from ... import draw, measure

from .plotplugin import PlotPlugin
from ..canvastools import ThickLineTool


__all__ = ['LineProfile']


class LineProfile(PlotPlugin):
    """Plugin to compute interpolated intensity under a scan line on an image.

    See PlotPlugin and Plugin classes for additional details.

    Parameters
    ----------
    maxdist : float
        Maximum pixel distance allowed when selecting end point of scan line.
    limits : tuple or {None, 'image', 'dtype'}
        (minimum, maximum) intensity limits for plotted profile. The following
        special values are defined:

            None : rescale based on min/max intensity along selected scan line.
            'image' : fixed scale based on min/max intensity in image.
            'dtype' : fixed scale based on min/max intensity of image dtype.
    """
    name = 'Line Profile'

    def __init__(self, maxdist=10, epsilon='deprecated',
                 limits='image', **kwargs):
        super(LineProfile, self).__init__(**kwargs)
        self.maxdist = maxdist
        self._limit_type = limits
        print(self.help())

    def attach(self, image_viewer):
        super(LineProfile, self).attach(image_viewer)

        image = image_viewer.original_image

        if self._limit_type == 'image':
            self.limits = (np.min(image), np.max(image))
        elif self._limit_type == 'dtype':
            self.limits = dtype_range[image.dtype.type]
        elif self._limit_type is None or len(self._limit_type) == 2:
            self.limits = self._limit_type
        else:
            raise ValueError("Unrecognized `limits`: %s" % self._limit_type)

        if not self._limit_type is None:
            self.ax.set_ylim(self.limits)

        h, w = image.shape[0:2]
        x = [w / 3, 2 * w / 3]
        y = [h / 2] * 2

        self.line_tool = ThickLineTool(self.image_viewer,
                                       maxdist=self.maxdist,
                                       on_move=self.line_changed,
                                       on_change=self.line_changed)
        self.line_tool.end_points = np.transpose([x, y])

        scan_data = measure.profile_line(image,
                                         *self.line_tool.end_points[:, ::-1],
                                         mode='reflect')
        self.scan_data = scan_data
        if scan_data.ndim == 1:
            scan_data = scan_data[:, np.newaxis]

        self.reset_axes(scan_data)

        self._autoscale_view()

    def help(self):
        helpstr = ("Line profile tool",
                   "+ and - keys or mouse scroll changes width of scan line.",
                   "Select and drag ends of the scan line to adjust it.")
        return '\n'.join(helpstr)

    def get_profiles(self):
        """Return intensity profile of the selected line.

        Returns
        -------
        end_points: (2, 2) array
            The positions ((x1, y1), (x2, y2)) of the line ends.
        profile: list of 1d arrays
            Profile of intensity values. Length 1 (grayscale) or 3 (rgb).
        """
        self._update_data()
        profiles = [data.get_ydata() for data in self.profile]
        return self.line_tool.end_points, profiles

    def _autoscale_view(self):
        if self.limits is None:
            self.ax.autoscale_view(tight=True)
        else:
            self.ax.autoscale_view(scaley=False, tight=True)

    def line_changed(self, end_points):
        x, y = np.transpose(end_points)
        self.line_tool.end_points = end_points
        self._update_data()
        self.ax.relim()

        self._autoscale_view()
        self.redraw()

    def _update_data(self):
        scan = measure.profile_line(self.image_viewer.image,
                                    *self.line_tool.end_points[:, ::-1],
                                    linewidth=self.line_tool.linewidth,
                                    mode='reflect')
        self.scan_data = scan
        if scan.ndim == 1:
            scan = scan[:, np.newaxis]

        if scan.shape[1] != len(self.profile):
            self.reset_axes(scan)

        for i in range(len(scan[0])):
            self.profile[i].set_xdata(np.arange(scan.shape[0]))
            self.profile[i].set_ydata(scan[:, i])

    def reset_axes(self, scan_data):
        # Clear lines out
        for line in self.ax.lines:
            self.ax.lines = []

        if scan_data.shape[1] == 1:
            self.profile = self.ax.plot(scan_data, 'k-')
        else:
            self.profile = self.ax.plot(scan_data[:, 0], 'r-',
                                        scan_data[:, 1], 'g-',
                                        scan_data[:, 2], 'b-')

    def output(self):
        """Return the drawn line and the resulting scan.

        Returns
        -------
        line_image : (M, N) uint8 array, same shape as image
            An array of 0s with the scanned line set to 255.
            If the linewidth of the line tool is greater than 1,
            sets the values within the profiled polygon to 128.
        scan : (P,) or (P, 3) array of int or float
            The line scan values across the image.
        """
        end_points = self.line_tool.end_points
        line_image = np.zeros(self.image_viewer.image.shape[:2],
                              np.uint8)
        width = self.line_tool.linewidth
        if width > 1:
            rp, cp = measure.profile._line_profile_coordinates(
                *end_points[:, ::-1], linewidth=width)
            # the points are aliased, so create a polygon using the corners
            yp = np.rint(rp[[0, 0, -1, -1],[0, -1, -1, 0]]).astype(int)
            xp = np.rint(cp[[0, 0, -1, -1],[0, -1, -1, 0]]).astype(int)
            rp, cp = draw.polygon(yp, xp, line_image.shape)
            line_image[rp, cp] = 128
        (x1, y1), (x2, y2) = end_points.astype(int)
        rr, cc = draw.line(y1, x1, y2, x2)
        line_image[rr, cc] = 255
        return line_image, self.scan_data
