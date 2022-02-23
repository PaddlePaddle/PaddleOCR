"""
An experimental support for curvilinear grid.
"""
from itertools import chain

import numpy as np

from matplotlib import _api
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, IdentityTransform
from .axislines import AxisArtistHelper, GridHelperBase
from .axis_artist import AxisArtist
from .grid_finder import GridFinder


class FixedAxisArtistHelper(AxisArtistHelper.Fixed):
    """
    Helper class for a fixed axis.
    """

    def __init__(self, grid_helper, side, nth_coord_ticks=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """

        super().__init__(loc=side)

        self.grid_helper = grid_helper
        if nth_coord_ticks is None:
            nth_coord_ticks = self.nth_coord
        self.nth_coord_ticks = nth_coord_ticks

        self.side = side

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)

    @_api.deprecated("3.5")
    def change_tick_coord(self, coord_number=None):
        if coord_number is None:
            self.nth_coord_ticks = 1 - self.nth_coord_ticks
        elif coord_number in [0, 1]:
            self.nth_coord_ticks = coord_number
        else:
            raise Exception("wrong coord number")

    def get_tick_transform(self, axes):
        return axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""
        v1, v2 = axes.get_ylim() if self.nth_coord == 0 else axes.get_xlim()
        if v1 > v2:  # Inverted limits.
            side = {"left": "right", "right": "left",
                    "top": "bottom", "bottom": "top"}[self.side]
        else:
            side = self.side
        g = self.grid_helper
        ti1 = g.get_tick_iterator(self.nth_coord_ticks, side)
        ti2 = g.get_tick_iterator(1-self.nth_coord_ticks, side, minor=True)
        return chain(ti1, ti2), iter([])


class FloatingAxisArtistHelper(AxisArtistHelper.Floating):
    grid_info = _api.deprecate_privatize_attribute("3.5")

    def __init__(self, grid_helper, nth_coord, value, axis_direction=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """
        super().__init__(nth_coord, value)
        self.value = value
        self.grid_helper = grid_helper
        self._extremes = -np.inf, np.inf
        self._line_num_points = 100  # number of points to create a line

    def set_extremes(self, e1, e2):
        if e1 is None:
            e1 = -np.inf
        if e2 is None:
            e2 = np.inf
        self._extremes = e1, e2

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)

        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()
        grid_finder = self.grid_helper.grid_finder
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                              x1, y1, x2, y2)

        lon_min, lon_max, lat_min, lat_max = extremes
        e_min, e_max = self._extremes  # ranges of other coordinates
        if self.nth_coord == 0:
            lat_min = max(e_min, lat_min)
            lat_max = min(e_max, lat_max)
        elif self.nth_coord == 1:
            lon_min = max(e_min, lon_min)
            lon_max = min(e_max, lon_max)

        lon_levs, lon_n, lon_factor = \
            grid_finder.grid_locator1(lon_min, lon_max)
        lat_levs, lat_n, lat_factor = \
            grid_finder.grid_locator2(lat_min, lat_max)

        if self.nth_coord == 0:
            xx0 = np.full(self._line_num_points, self.value)
            yy0 = np.linspace(lat_min, lat_max, self._line_num_points)
            xx, yy = grid_finder.transform_xy(xx0, yy0)
        elif self.nth_coord == 1:
            xx0 = np.linspace(lon_min, lon_max, self._line_num_points)
            yy0 = np.full(self._line_num_points, self.value)
            xx, yy = grid_finder.transform_xy(xx0, yy0)

        self._grid_info = {
            "extremes": (lon_min, lon_max, lat_min, lat_max),
            "lon_info": (lon_levs, lon_n, lon_factor),
            "lat_info": (lat_levs, lat_n, lat_factor),
            "lon_labels": grid_finder.tick_formatter1(
                "bottom", lon_factor, lon_levs),
            "lat_labels": grid_finder.tick_formatter2(
                "bottom", lat_factor, lat_levs),
            "line_xy": (xx, yy),
        }

    def get_axislabel_transform(self, axes):
        return Affine2D()  # axes.transData

    def get_axislabel_pos_angle(self, axes):

        extremes = self._grid_info["extremes"]

        if self.nth_coord == 0:
            xx0 = self.value
            yy0 = (extremes[2] + extremes[3]) / 2
            dxx = 0
            dyy = abs(extremes[2] - extremes[3]) / 1000
        elif self.nth_coord == 1:
            xx0 = (extremes[0] + extremes[1]) / 2
            yy0 = self.value
            dxx = abs(extremes[0] - extremes[1]) / 1000
            dyy = 0

        grid_finder = self.grid_helper.grid_finder
        (xx1,), (yy1,) = grid_finder.transform_xy([xx0], [yy0])

        data_to_axes = axes.transData - axes.transAxes
        p = data_to_axes.transform([xx1, yy1])

        if 0 <= p[0] <= 1 and 0 <= p[1] <= 1:
            xx1c, yy1c = axes.transData.transform([xx1, yy1])
            (xx2,), (yy2,) = grid_finder.transform_xy([xx0 + dxx], [yy0 + dyy])
            xx2c, yy2c = axes.transData.transform([xx2, yy2])
            return (xx1c, yy1c), np.rad2deg(np.arctan2(yy2c-yy1c, xx2c-xx1c))
        else:
            return None, None

    def get_tick_transform(self, axes):
        return IdentityTransform()  # axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label, (optionally) tick_label"""

        grid_finder = self.grid_helper.grid_finder

        lat_levs, lat_n, lat_factor = self._grid_info["lat_info"]
        lat_levs = np.asarray(lat_levs)
        yy0 = lat_levs / lat_factor
        dy = 0.01 / lat_factor

        lon_levs, lon_n, lon_factor = self._grid_info["lon_info"]
        lon_levs = np.asarray(lon_levs)
        xx0 = lon_levs / lon_factor
        dx = 0.01 / lon_factor

        e0, e1 = self._extremes

        if self.nth_coord == 0:
            mask = (e0 <= yy0) & (yy0 <= e1)
            # xx0, yy0 = xx0[mask], yy0[mask]
            yy0 = yy0[mask]
        elif self.nth_coord == 1:
            mask = (e0 <= xx0) & (xx0 <= e1)
            # xx0, yy0 = xx0[mask], yy0[mask]
            xx0 = xx0[mask]

        def transform_xy(x, y):
            trf = grid_finder.get_transform() + axes.transData
            return trf.transform(np.column_stack([x, y])).T

        # find angles
        if self.nth_coord == 0:
            xx0 = np.full_like(yy0, self.value)

            xx1, yy1 = transform_xy(xx0, yy0)

            xx00 = xx0.copy()
            xx00[xx0 + dx > e1] -= dx
            xx1a, yy1a = transform_xy(xx00, yy0)
            xx1b, yy1b = transform_xy(xx00+dx, yy0)

            xx2a, yy2a = transform_xy(xx0, yy0)
            xx2b, yy2b = transform_xy(xx0, yy0+dy)

            labels = self._grid_info["lat_labels"]
            labels = [l for l, m in zip(labels, mask) if m]

        elif self.nth_coord == 1:
            yy0 = np.full_like(xx0, self.value)

            xx1, yy1 = transform_xy(xx0, yy0)

            xx1a, yy1a = transform_xy(xx0, yy0)
            xx1b, yy1b = transform_xy(xx0, yy0+dy)

            xx00 = xx0.copy()
            xx00[xx0 + dx > e1] -= dx
            xx2a, yy2a = transform_xy(xx00, yy0)
            xx2b, yy2b = transform_xy(xx00+dx, yy0)

            labels = self._grid_info["lon_labels"]
            labels = [l for l, m in zip(labels, mask) if m]

        def f1():
            dd = np.arctan2(yy1b-yy1a, xx1b-xx1a)  # angle normal
            dd2 = np.arctan2(yy2b-yy2a, xx2b-xx2a)  # angle tangent
            mm = (yy1b == yy1a) & (xx1b == xx1a)  # mask where dd not defined
            dd[mm] = dd2[mm] + np.pi / 2

            tick_to_axes = self.get_tick_transform(axes) - axes.transAxes
            for x, y, d, d2, lab in zip(xx1, yy1, dd, dd2, labels):
                c2 = tick_to_axes.transform((x, y))
                delta = 0.00001
                if 0-delta <= c2[0] <= 1+delta and 0-delta <= c2[1] <= 1+delta:
                    d1, d2 = np.rad2deg([d, d2])
                    yield [x, y], d1, d2, lab

        return f1(), iter([])

    def get_line_transform(self, axes):
        return axes.transData

    def get_line(self, axes):
        self.update_lim(axes)
        x, y = self._grid_info["line_xy"]
        return Path(np.column_stack([x, y]))


class GridHelperCurveLinear(GridHelperBase):
    grid_info = _api.deprecate_privatize_attribute("3.5")

    def __init__(self, aux_trans,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        """
        aux_trans : a transform from the source (curved) coordinate to
        target (rectilinear) coordinate. An instance of MPL's Transform
        (inverse transform should be defined) or a tuple of two callable
        objects which defines the transform and its inverse. The callables
        need take two arguments of array of source coordinates and
        should return two target coordinates.

        e.g., ``x2, y2 = trans(x1, y1)``
        """
        super().__init__()
        self._grid_info = None
        self._aux_trans = aux_trans
        self.grid_finder = GridFinder(aux_trans,
                                      extreme_finder,
                                      grid_locator1,
                                      grid_locator2,
                                      tick_formatter1,
                                      tick_formatter2)

    def update_grid_finder(self, aux_trans=None, **kw):
        if aux_trans is not None:
            self.grid_finder.update_transform(aux_trans)
        self.grid_finder.update(**kw)
        self._old_limits = None  # Force revalidation.

    def new_fixed_axis(self, loc,
                       nth_coord=None,
                       axis_direction=None,
                       offset=None,
                       axes=None):
        if axes is None:
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        _helper = FixedAxisArtistHelper(self, loc, nth_coord_ticks=nth_coord)
        axisline = AxisArtist(axes, _helper, axis_direction=axis_direction)
        # Why is clip not set on axisline, unlike in new_floating_axis or in
        # the floating_axig.GridHelperCurveLinear subclass?
        return axisline

    def new_floating_axis(self, nth_coord,
                          value,
                          axes=None,
                          axis_direction="bottom"
                          ):

        if axes is None:
            axes = self.axes

        _helper = FloatingAxisArtistHelper(
            self, nth_coord, value, axis_direction)

        axisline = AxisArtist(axes, _helper)

        # _helper = FloatingAxisArtistHelper(self, nth_coord,
        #                                    value,
        #                                    label_direction=label_direction,
        #                                    )

        # axisline = AxisArtistFloating(axes, _helper,
        #                               axis_direction=axis_direction)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        # axisline.major_ticklabels.set_visible(True)
        # axisline.minor_ticklabels.set_visible(False)

        return axisline

    def _update_grid(self, x1, y1, x2, y2):
        self._grid_info = self.grid_finder.get_grid_info(x1, y1, x2, y2)

    def get_gridlines(self, which="major", axis="both"):
        grid_lines = []
        if axis in ["both", "x"]:
            for gl in self._grid_info["lon"]["lines"]:
                grid_lines.extend(gl)
        if axis in ["both", "y"]:
            for gl in self._grid_info["lat"]["lines"]:
                grid_lines.extend(gl)
        return grid_lines

    def get_tick_iterator(self, nth_coord, axis_side, minor=False):

        # axisnr = dict(left=0, bottom=1, right=2, top=3)[axis_side]
        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[axis_side]
        # angle = [0, 90, 180, 270][axisnr]
        lon_or_lat = ["lon", "lat"][nth_coord]
        if not minor:  # major ticks
            for (xy, a), l in zip(
                    self._grid_info[lon_or_lat]["tick_locs"][axis_side],
                    self._grid_info[lon_or_lat]["tick_labels"][axis_side]):
                angle_normal = a
                yield xy, angle_normal, angle_tangent, l
        else:
            for (xy, a), l in zip(
                    self._grid_info[lon_or_lat]["tick_locs"][axis_side],
                    self._grid_info[lon_or_lat]["tick_labels"][axis_side]):
                angle_normal = a
                yield xy, angle_normal, angle_tangent, ""
            # for xy, a, l in self._grid_info[lon_or_lat]["ticks"][axis_side]:
            #     yield xy, a, ""
