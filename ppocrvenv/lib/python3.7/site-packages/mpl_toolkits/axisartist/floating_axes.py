"""
An experimental support for curvilinear grid.
"""

# TODO :
# see if tick_iterator method can be simplified by reusing the parent method.

import numpy as np

from matplotlib import _api, cbook
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.axes as maxes

from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory

from . import axislines, grid_helper_curvelinear
from .axis_artist import AxisArtist
from .grid_finder import ExtremeFinderSimple


class FloatingAxisArtistHelper(
        grid_helper_curvelinear.FloatingAxisArtistHelper):
    pass


class FixedAxisArtistHelper(grid_helper_curvelinear.FloatingAxisArtistHelper):

    def __init__(self, grid_helper, side, nth_coord_ticks=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """
        value, nth_coord = grid_helper.get_data_boundary(side)
        super().__init__(grid_helper, nth_coord, value, axis_direction=side)
        if nth_coord_ticks is None:
            nth_coord_ticks = nth_coord
        self.nth_coord_ticks = nth_coord_ticks

        self.value = value
        self.grid_helper = grid_helper
        self._side = side

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)
        self._grid_info = self.grid_helper._grid_info

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label, (optionally) tick_label"""

        grid_finder = self.grid_helper.grid_finder

        lat_levs, lat_n, lat_factor = self._grid_info["lat_info"]
        lon_levs, lon_n, lon_factor = self._grid_info["lon_info"]

        lon_levs, lat_levs = np.asarray(lon_levs), np.asarray(lat_levs)
        if lat_factor is not None:
            yy0 = lat_levs / lat_factor
            dy = 0.001 / lat_factor
        else:
            yy0 = lat_levs
            dy = 0.001

        if lon_factor is not None:
            xx0 = lon_levs / lon_factor
            dx = 0.001 / lon_factor
        else:
            xx0 = lon_levs
            dx = 0.001

        extremes = self.grid_helper._extremes
        xmin, xmax = sorted(extremes[:2])
        ymin, ymax = sorted(extremes[2:])

        def transform_xy(x, y):
            trf = grid_finder.get_transform() + axes.transData
            return trf.transform(np.column_stack([x, y])).T

        if self.nth_coord == 0:
            mask = (ymin <= yy0) & (yy0 <= ymax)
            yy0 = yy0[mask]
            xx0 = np.full_like(yy0, self.value)
            xx1, yy1 = transform_xy(xx0, yy0)

            xx00 = xx0.astype(float, copy=True)
            xx00[xx0 + dx > xmax] -= dx
            xx1a, yy1a = transform_xy(xx00, yy0)
            xx1b, yy1b = transform_xy(xx00 + dx, yy0)

            yy00 = yy0.astype(float, copy=True)
            yy00[yy0 + dy > ymax] -= dy
            xx2a, yy2a = transform_xy(xx0, yy00)
            xx2b, yy2b = transform_xy(xx0, yy00 + dy)

            labels = self._grid_info["lat_labels"]
            labels = [l for l, m in zip(labels, mask) if m]

        elif self.nth_coord == 1:
            mask = (xmin <= xx0) & (xx0 <= xmax)
            xx0 = xx0[mask]
            yy0 = np.full_like(xx0, self.value)
            xx1, yy1 = transform_xy(xx0, yy0)

            yy00 = yy0.astype(float, copy=True)
            yy00[yy0 + dy > ymax] -= dy
            xx1a, yy1a = transform_xy(xx0, yy00)
            xx1b, yy1b = transform_xy(xx0, yy00 + dy)

            xx00 = xx0.astype(float, copy=True)
            xx00[xx0 + dx > xmax] -= dx
            xx2a, yy2a = transform_xy(xx00, yy0)
            xx2b, yy2b = transform_xy(xx00 + dx, yy0)

            labels = self._grid_info["lon_labels"]
            labels = [l for l, m in zip(labels, mask) if m]

        def f1():
            dd = np.arctan2(yy1b - yy1a, xx1b - xx1a)  # angle normal
            dd2 = np.arctan2(yy2b - yy2a, xx2b - xx2a)  # angle tangent
            mm = (yy1b - yy1a == 0) & (xx1b - xx1a == 0)  # mask not defined dd
            dd[mm] = dd2[mm] + np.pi / 2

            tick_to_axes = self.get_tick_transform(axes) - axes.transAxes
            for x, y, d, d2, lab in zip(xx1, yy1, dd, dd2, labels):
                c2 = tick_to_axes.transform((x, y))
                delta = 0.00001
                if 0-delta <= c2[0] <= 1+delta and 0-delta <= c2[1] <= 1+delta:
                    d1, d2 = np.rad2deg([d, d2])
                    yield [x, y], d1, d2, lab

        return f1(), iter([])

    def get_line(self, axes):
        self.update_lim(axes)
        k, v = dict(left=("lon_lines0", 0),
                    right=("lon_lines0", 1),
                    bottom=("lat_lines0", 0),
                    top=("lat_lines0", 1))[self._side]
        xx, yy = self._grid_info[k][v]
        return Path(np.column_stack([xx, yy]))


class ExtremeFinderFixed(ExtremeFinderSimple):
    # docstring inherited

    def __init__(self, extremes):
        """
        This subclass always returns the same bounding box.

        Parameters
        ----------
        extremes : (float, float, float, float)
            The bounding box that this helper always returns.
        """
        self._extremes = extremes

    def __call__(self, transform_xy, x1, y1, x2, y2):
        # docstring inherited
        return self._extremes


class GridHelperCurveLinear(grid_helper_curvelinear.GridHelperCurveLinear):

    def __init__(self, aux_trans, extremes,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        # docstring inherited
        self._extremes = extremes
        extreme_finder = ExtremeFinderFixed(extremes)
        super().__init__(aux_trans,
                         extreme_finder,
                         grid_locator1=grid_locator1,
                         grid_locator2=grid_locator2,
                         tick_formatter1=tick_formatter1,
                         tick_formatter2=tick_formatter2)

    def get_data_boundary(self, side):
        """
        Return v=0, nth=1.
        """
        lon1, lon2, lat1, lat2 = self._extremes
        return dict(left=(lon1, 0),
                    right=(lon2, 0),
                    bottom=(lat1, 1),
                    top=(lat2, 1))[side]

    def new_fixed_axis(self, loc,
                       nth_coord=None,
                       axis_direction=None,
                       offset=None,
                       axes=None):
        if axes is None:
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        # This is not the same as the FixedAxisArtistHelper class used by
        # grid_helper_curvelinear.GridHelperCurveLinear.new_fixed_axis!
        _helper = FixedAxisArtistHelper(
            self, loc, nth_coord_ticks=nth_coord)
        axisline = AxisArtist(axes, _helper, axis_direction=axis_direction)
        # Perhaps should be moved to the base class?
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        return axisline

    # new_floating_axis will inherit the grid_helper's extremes.

    # def new_floating_axis(self, nth_coord,
    #                       value,
    #                       axes=None,
    #                       axis_direction="bottom"
    #                       ):

    #     axis = super(GridHelperCurveLinear,
    #                  self).new_floating_axis(nth_coord,
    #                                          value, axes=axes,
    #                                          axis_direction=axis_direction)

    #     # set extreme values of the axis helper
    #     if nth_coord == 1:
    #         axis.get_helper().set_extremes(*self._extremes[:2])
    #     elif nth_coord == 0:
    #         axis.get_helper().set_extremes(*self._extremes[2:])

    #     return axis

    def _update_grid(self, x1, y1, x2, y2):
        if self._grid_info is None:
            self._grid_info = dict()

        grid_info = self._grid_info

        grid_finder = self.grid_finder
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                              x1, y1, x2, y2)

        lon_min, lon_max = sorted(extremes[:2])
        lat_min, lat_max = sorted(extremes[2:])
        lon_levs, lon_n, lon_factor = \
            grid_finder.grid_locator1(lon_min, lon_max)
        lat_levs, lat_n, lat_factor = \
            grid_finder.grid_locator2(lat_min, lat_max)
        grid_info["extremes"] = lon_min, lon_max, lat_min, lat_max  # extremes

        grid_info["lon_info"] = lon_levs, lon_n, lon_factor
        grid_info["lat_info"] = lat_levs, lat_n, lat_factor

        grid_info["lon_labels"] = grid_finder.tick_formatter1("bottom",
                                                              lon_factor,
                                                              lon_levs)

        grid_info["lat_labels"] = grid_finder.tick_formatter2("bottom",
                                                              lat_factor,
                                                              lat_levs)

        if lon_factor is None:
            lon_values = np.asarray(lon_levs[:lon_n])
        else:
            lon_values = np.asarray(lon_levs[:lon_n]/lon_factor)
        if lat_factor is None:
            lat_values = np.asarray(lat_levs[:lat_n])
        else:
            lat_values = np.asarray(lat_levs[:lat_n]/lat_factor)

        lon_lines, lat_lines = grid_finder._get_raw_grid_lines(
            lon_values[(lon_min < lon_values) & (lon_values < lon_max)],
            lat_values[(lat_min < lat_values) & (lat_values < lat_max)],
            lon_min, lon_max, lat_min, lat_max)

        grid_info["lon_lines"] = lon_lines
        grid_info["lat_lines"] = lat_lines

        lon_lines, lat_lines = grid_finder._get_raw_grid_lines(
            # lon_min, lon_max, lat_min, lat_max)
            extremes[:2], extremes[2:], *extremes)

        grid_info["lon_lines0"] = lon_lines
        grid_info["lat_lines0"] = lat_lines

    def get_gridlines(self, which="major", axis="both"):
        grid_lines = []
        if axis in ["both", "x"]:
            grid_lines.extend(self._grid_info["lon_lines"])
        if axis in ["both", "y"]:
            grid_lines.extend(self._grid_info["lat_lines"])
        return grid_lines

    @_api.deprecated("3.5")
    def get_boundary(self):
        """
        Return (N, 2) array of (x, y) coordinate of the boundary.
        """
        x0, x1, y0, y1 = self._extremes
        tr = self._aux_trans

        xx = np.linspace(x0, x1, 100)
        yy0 = np.full_like(xx, y0)
        yy1 = np.full_like(xx, y1)
        yy = np.linspace(y0, y1, 100)
        xx0 = np.full_like(yy, x0)
        xx1 = np.full_like(yy, x1)

        xxx = np.concatenate([xx[:-1], xx1[:-1], xx[-1:0:-1], xx0])
        yyy = np.concatenate([yy0[:-1], yy[:-1], yy1[:-1], yy[::-1]])
        t = tr.transform(np.array([xxx, yyy]).transpose())

        return t


class FloatingAxesBase:

    def __init__(self, *args, **kwargs):
        grid_helper = kwargs.get("grid_helper", None)
        if grid_helper is None:
            raise ValueError("FloatingAxes requires grid_helper argument")
        if not hasattr(grid_helper, "get_boundary"):
            raise ValueError("grid_helper must implement get_boundary method")
        super().__init__(*args, **kwargs)
        self.set_aspect(1.)
        self.adjust_axes_lim()

    def _gen_axes_patch(self):
        # docstring inherited
        # Using a public API to access _extremes.
        (x0, _), (x1, _), (y0, _), (y1, _) = map(
            self.get_grid_helper().get_data_boundary,
            ["left", "right", "bottom", "top"])
        patch = mpatches.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        patch.get_path()._interpolation_steps = 100
        return patch

    def cla(self):
        super().cla()
        self.patch.set_transform(
            self.get_grid_helper().grid_finder.get_transform()
            + self.transData)
        # The original patch is not in the draw tree; it is only used for
        # clipping purposes.
        orig_patch = super()._gen_axes_patch()
        orig_patch.set_figure(self.figure)
        orig_patch.set_transform(self.transAxes)
        self.patch.set_clip_path(orig_patch)
        self.gridlines.set_clip_path(orig_patch)

    def adjust_axes_lim(self):
        bbox = self.patch.get_path().get_extents(
            # First transform to pixel coords, then to parent data coords.
            self.patch.get_transform() - self.transData)
        bbox = bbox.expanded(1.02, 1.02)
        self.set_xlim(bbox.xmin, bbox.xmax)
        self.set_ylim(bbox.ymin, bbox.ymax)


floatingaxes_class_factory = cbook._make_class_factory(
    FloatingAxesBase, "Floating{}")
FloatingAxes = floatingaxes_class_factory(
    host_axes_class_factory(axislines.Axes))
FloatingSubplot = maxes.subplot_class_factory(FloatingAxes)
