"""
Axislines includes modified implementation of the Axes class. The
biggest difference is that the artists responsible for drawing the axis spine,
ticks, ticklabels and axis labels are separated out from Matplotlib's Axis
class. Originally, this change was motivated to support curvilinear
grid. Here are a few reasons that I came up with a new axes class:

* "top" and "bottom" x-axis (or "left" and "right" y-axis) can have
  different ticks (tick locations and labels). This is not possible
  with the current Matplotlib, although some twin axes trick can help.

* Curvilinear grid.

* angled ticks.

In the new axes class, xaxis and yaxis is set to not visible by
default, and new set of artist (AxisArtist) are defined to draw axis
line, ticks, ticklabels and axis label. Axes.axis attribute serves as
a dictionary of these artists, i.e., ax.axis["left"] is a AxisArtist
instance responsible to draw left y-axis. The default Axes.axis contains
"bottom", "left", "top" and "right".

AxisArtist can be considered as a container artist and
has following children artists which will draw ticks, labels, etc.

* line
* major_ticks, major_ticklabels
* minor_ticks, minor_ticklabels
* offsetText
* label

Note that these are separate artists from `matplotlib.axis.Axis`, thus most
tick-related functions in Matplotlib won't work. For example, color and
markerwidth of the ``ax.axis["bottom"].major_ticks`` will follow those of
Axes.xaxis unless explicitly specified.

In addition to AxisArtist, the Axes will have *gridlines* attribute,
which obviously draws grid lines. The gridlines needs to be separated
from the axis as some gridlines can never pass any axis.
"""

import numpy as np

from matplotlib import _api, rcParams
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle
from .axis_artist import AxisArtist, GridlinesCollection


class AxisArtistHelper:
    """
    AxisArtistHelper should define
    following method with given APIs. Note that the first axes argument
    will be axes attribute of the caller artist.::


        # LINE (spinal line?)

        def get_line(self, axes):
            # path : Path
            return path

        def get_line_transform(self, axes):
            # ...
            # trans : transform
            return trans

        # LABEL

        def get_label_pos(self, axes):
            # x, y : position
            return (x, y), trans


        def get_label_offset_transform(self,
                axes,
                pad_points, fontprops, renderer,
                bboxes,
                ):
            # va : vertical alignment
            # ha : horizontal alignment
            # a : angle
            return trans, va, ha, a

        # TICK

        def get_tick_transform(self, axes):
            return trans

        def get_tick_iterators(self, axes):
            # iter : iterable object that yields (c, angle, l) where
            # c, angle, l is position, tick angle, and label

            return iter_major, iter_minor
    """

    class _Base:
        """Base class for axis helper."""
        def __init__(self):
            self.delta1, self.delta2 = 0.00001, 0.00001

        def update_lim(self, axes):
            pass

    class Fixed(_Base):
        """Helper class for a fixed (in the axes coordinate) axis."""

        _default_passthru_pt = dict(left=(0, 0),
                                    right=(1, 0),
                                    bottom=(0, 0),
                                    top=(0, 1))

        def __init__(self, loc, nth_coord=None):
            """
            nth_coord = along which coordinate value varies
            in 2D, nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
            """
            _api.check_in_list(["left", "right", "bottom", "top"], loc=loc)
            self._loc = loc

            if nth_coord is None:
                if loc in ["left", "right"]:
                    nth_coord = 1
                elif loc in ["bottom", "top"]:
                    nth_coord = 0

            self.nth_coord = nth_coord

            super().__init__()

            self.passthru_pt = self._default_passthru_pt[loc]

            _verts = np.array([[0., 0.],
                               [1., 1.]])
            fixed_coord = 1 - nth_coord
            _verts[:, fixed_coord] = self.passthru_pt[fixed_coord]

            # axis line in transAxes
            self._path = Path(_verts)

        def get_nth_coord(self):
            return self.nth_coord

        # LINE

        def get_line(self, axes):
            return self._path

        def get_line_transform(self, axes):
            return axes.transAxes

        # LABEL

        def get_axislabel_transform(self, axes):
            return axes.transAxes

        def get_axislabel_pos_angle(self, axes):
            """
            Return the label reference position in transAxes.

            get_label_transform() returns a transform of (transAxes+offset)
            """
            return dict(left=((0., 0.5), 90),  # (position, angle_tangent)
                        right=((1., 0.5), 90),
                        bottom=((0.5, 0.), 0),
                        top=((0.5, 1.), 0))[self._loc]

        # TICK

        def get_tick_transform(self, axes):
            return [axes.get_xaxis_transform(),
                    axes.get_yaxis_transform()][self.nth_coord]

    class Floating(_Base):

        def __init__(self, nth_coord, value):
            self.nth_coord = nth_coord
            self._value = value
            super().__init__()

        def get_nth_coord(self):
            return self.nth_coord

        def get_line(self, axes):
            raise RuntimeError(
                "get_line method should be defined by the derived class")


class AxisArtistHelperRectlinear:

    class Fixed(AxisArtistHelper.Fixed):

        def __init__(self, axes, loc, nth_coord=None):
            """
            nth_coord = along which coordinate value varies
            in 2D, nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
            """
            super().__init__(loc, nth_coord)
            self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]

        # TICK

        def get_tick_iterators(self, axes):
            """tick_loc, tick_angle, tick_label"""

            loc = self._loc

            if loc in ["bottom", "top"]:
                angle_normal, angle_tangent = 90, 0
            else:
                angle_normal, angle_tangent = 0, 90

            major = self.axis.major
            majorLocs = major.locator()
            majorLabels = major.formatter.format_ticks(majorLocs)

            minor = self.axis.minor
            minorLocs = minor.locator()
            minorLabels = minor.formatter.format_ticks(minorLocs)

            tick_to_axes = self.get_tick_transform(axes) - axes.transAxes

            def _f(locs, labels):
                for x, l in zip(locs, labels):

                    c = list(self.passthru_pt)  # copy
                    c[self.nth_coord] = x

                    # check if the tick point is inside axes
                    c2 = tick_to_axes.transform(c)
                    if (0 - self.delta1
                            <= c2[self.nth_coord]
                            <= 1 + self.delta2):
                        yield c, angle_normal, angle_tangent, l

            return _f(majorLocs, majorLabels), _f(minorLocs, minorLabels)

    class Floating(AxisArtistHelper.Floating):
        def __init__(self, axes, nth_coord,
                     passingthrough_point, axis_direction="bottom"):
            super().__init__(nth_coord, passingthrough_point)
            self._axis_direction = axis_direction
            self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]

        def get_line(self, axes):
            _verts = np.array([[0., 0.],
                               [1., 1.]])

            fixed_coord = 1 - self.nth_coord
            data_to_axes = axes.transData - axes.transAxes
            p = data_to_axes.transform([self._value, self._value])
            _verts[:, fixed_coord] = p[fixed_coord]

            return Path(_verts)

        def get_line_transform(self, axes):
            return axes.transAxes

        def get_axislabel_transform(self, axes):
            return axes.transAxes

        def get_axislabel_pos_angle(self, axes):
            """
            Return the label reference position in transAxes.

            get_label_transform() returns a transform of (transAxes+offset)
            """
            angle = [0, 90][self.nth_coord]
            _verts = [0.5, 0.5]
            fixed_coord = 1 - self.nth_coord
            data_to_axes = axes.transData - axes.transAxes
            p = data_to_axes.transform([self._value, self._value])
            _verts[fixed_coord] = p[fixed_coord]
            if 0 <= _verts[fixed_coord] <= 1:
                return _verts, angle
            else:
                return None, None

        def get_tick_transform(self, axes):
            return axes.transData

        def get_tick_iterators(self, axes):
            """tick_loc, tick_angle, tick_label"""
            if self.nth_coord == 0:
                angle_normal, angle_tangent = 90, 0
            else:
                angle_normal, angle_tangent = 0, 90

            major = self.axis.major
            majorLocs = major.locator()
            majorLabels = major.formatter.format_ticks(majorLocs)

            minor = self.axis.minor
            minorLocs = minor.locator()
            minorLabels = minor.formatter.format_ticks(minorLocs)

            data_to_axes = axes.transData - axes.transAxes

            def _f(locs, labels):
                for x, l in zip(locs, labels):
                    c = [self._value, self._value]
                    c[self.nth_coord] = x
                    c1, c2 = data_to_axes.transform(c)
                    if (0 <= c1 <= 1 and 0 <= c2 <= 1
                            and 0 - self.delta1
                                <= [c1, c2][self.nth_coord]
                                <= 1 + self.delta2):
                        yield c, angle_normal, angle_tangent, l

            return _f(majorLocs, majorLabels), _f(minorLocs, minorLabels)


class GridHelperBase:

    def __init__(self):
        self._force_update = True  # Remove together with invalidate()/valid().
        self._old_limits = None
        super().__init__()

    def update_lim(self, axes):
        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()
        if self._force_update or self._old_limits != (x1, x2, y1, y2):
            self._update_grid(x1, y1, x2, y2)
            self._force_update = False
            self._old_limits = (x1, x2, y1, y2)

    def _update_grid(self, x1, y1, x2, y2):
        """Cache relevant computations when the axes limits have changed."""

    @_api.deprecated("3.4")
    def invalidate(self):
        self._force_update = True

    @_api.deprecated("3.4")
    def valid(self):
        return not self._force_update

    def get_gridlines(self, which, axis):
        """
        Return list of grid lines as a list of paths (list of points).

        *which* : "major" or "minor"
        *axis* : "both", "x" or "y"
        """
        return []

    def new_gridlines(self, ax):
        """
        Create and return a new GridlineCollection instance.

        *which* : "major" or "minor"
        *axis* : "both", "x" or "y"

        """
        gridlines = GridlinesCollection(None, transform=ax.transData,
                                        colors=rcParams['grid.color'],
                                        linestyles=rcParams['grid.linestyle'],
                                        linewidths=rcParams['grid.linewidth'])
        ax._set_artist_props(gridlines)
        gridlines.set_grid_helper(self)

        ax.axes._set_artist_props(gridlines)
        # gridlines.set_clip_path(self.axes.patch)
        # set_clip_path need to be deferred after Axes.cla is completed.
        # It is done inside the cla.

        return gridlines


class GridHelperRectlinear(GridHelperBase):

    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def new_fixed_axis(self, loc,
                       nth_coord=None,
                       axis_direction=None,
                       offset=None,
                       axes=None,
                       ):

        if axes is None:
            _api.warn_external(
                "'new_fixed_axis' explicitly requires the axes keyword.")
            axes = self.axes

        _helper = AxisArtistHelperRectlinear.Fixed(axes, loc, nth_coord)

        if axis_direction is None:
            axis_direction = loc
        axisline = AxisArtist(axes, _helper, offset=offset,
                              axis_direction=axis_direction,
                              )

        return axisline

    def new_floating_axis(self, nth_coord, value,
                          axis_direction="bottom",
                          axes=None,
                          ):

        if axes is None:
            _api.warn_external(
                "'new_floating_axis' explicitly requires the axes keyword.")
            axes = self.axes

        _helper = AxisArtistHelperRectlinear.Floating(
            axes, nth_coord, value, axis_direction)

        axisline = AxisArtist(axes, _helper, axis_direction=axis_direction)

        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        return axisline

    def get_gridlines(self, which="major", axis="both"):
        """
        Return list of gridline coordinates in data coordinates.

        *which* : "major" or "minor"
        *axis* : "both", "x" or "y"
        """
        gridlines = []

        if axis in ["both", "x"]:
            locs = []
            y1, y2 = self.axes.get_ylim()
            if which in ["both", "major"]:
                locs.extend(self.axes.xaxis.major.locator())
            if which in ["both", "minor"]:
                locs.extend(self.axes.xaxis.minor.locator())

            for x in locs:
                gridlines.append([[x, x], [y1, y2]])

        if axis in ["both", "y"]:
            x1, x2 = self.axes.get_xlim()
            locs = []
            if self.axes.yaxis._major_tick_kw["gridOn"]:
                locs.extend(self.axes.yaxis.major.locator())
            if self.axes.yaxis._minor_tick_kw["gridOn"]:
                locs.extend(self.axes.yaxis.minor.locator())

            for y in locs:
                gridlines.append([[x1, x2], [y, y]])

        return gridlines


class Axes(maxes.Axes):

    def __call__(self, *args, **kwargs):
        return maxes.Axes.axis(self.axes, *args, **kwargs)

    def __init__(self, *args, grid_helper=None, **kwargs):
        self._axisline_on = True
        self._grid_helper = (grid_helper if grid_helper
                             else GridHelperRectlinear(self))
        super().__init__(*args, **kwargs)
        self.toggle_axisline(True)

    def toggle_axisline(self, b=None):
        if b is None:
            b = not self._axisline_on
        if b:
            self._axisline_on = True
            self.spines[:].set_visible(False)
            self.xaxis.set_visible(False)
            self.yaxis.set_visible(False)
        else:
            self._axisline_on = False
            self.spines[:].set_visible(True)
            self.xaxis.set_visible(True)
            self.yaxis.set_visible(True)

    def _init_axis_artists(self, axes=None):
        if axes is None:
            axes = self

        self._axislines = mpl_axes.Axes.AxisDict(self)
        new_fixed_axis = self.get_grid_helper().new_fixed_axis
        for loc in ["bottom", "top", "left", "right"]:
            self._axislines[loc] = new_fixed_axis(loc=loc, axes=axes,
                                                  axis_direction=loc)

        for axisline in [self._axislines["top"], self._axislines["right"]]:
            axisline.label.set_visible(False)
            axisline.major_ticklabels.set_visible(False)
            axisline.minor_ticklabels.set_visible(False)

    @property
    def axis(self):
        return self._axislines

    def new_gridlines(self, grid_helper=None):
        """
        Create and return a new GridlineCollection instance.

        *which* : "major" or "minor"
        *axis* : "both", "x" or "y"

        """
        if grid_helper is None:
            grid_helper = self.get_grid_helper()

        gridlines = grid_helper.new_gridlines(self)
        return gridlines

    def _init_gridlines(self, grid_helper=None):
        # It is done inside the cla.
        self.gridlines = self.new_gridlines(grid_helper)

    def cla(self):
        # gridlines need to b created before cla() since cla calls grid()
        self._init_gridlines()
        super().cla()

        # the clip_path should be set after Axes.cla() since that's
        # when a patch is created.
        self.gridlines.set_clip_path(self.axes.patch)

        self._init_axis_artists()

    def get_grid_helper(self):
        return self._grid_helper

    @_api.rename_parameter("3.5", "b", "visible")
    def grid(self, visible=None, which='major', axis="both", **kwargs):
        """
        Toggle the gridlines, and optionally set the properties of the lines.
        """
        # There are some discrepancies in the behavior of grid() between
        # axes_grid and Matplotlib, because axes_grid explicitly sets the
        # visibility of the gridlines.
        super().grid(visible, which=which, axis=axis, **kwargs)
        if not self._axisline_on:
            return
        if visible is None:
            visible = (self.axes.xaxis._minor_tick_kw["gridOn"]
                       or self.axes.xaxis._major_tick_kw["gridOn"]
                       or self.axes.yaxis._minor_tick_kw["gridOn"]
                       or self.axes.yaxis._major_tick_kw["gridOn"])
        self.gridlines.set(which=which, axis=axis, visible=visible)
        self.gridlines.set(**kwargs)

    def get_children(self):
        if self._axisline_on:
            children = [*self._axislines.values(), self.gridlines]
        else:
            children = []
        children.extend(super().get_children())
        return children

    @_api.deprecated("3.4")
    def invalidate_grid_helper(self):
        self._grid_helper.invalidate()

    def new_fixed_axis(self, loc, offset=None):
        gh = self.get_grid_helper()
        axis = gh.new_fixed_axis(loc,
                                 nth_coord=None,
                                 axis_direction=None,
                                 offset=offset,
                                 axes=self,
                                 )
        return axis

    def new_floating_axis(self, nth_coord, value, axis_direction="bottom"):
        gh = self.get_grid_helper()
        axis = gh.new_floating_axis(nth_coord, value,
                                    axis_direction=axis_direction,
                                    axes=self)
        return axis


Subplot = maxes.subplot_class_factory(Axes)


class AxesZero(Axes):

    def _init_axis_artists(self):
        super()._init_axis_artists()

        new_floating_axis = self._grid_helper.new_floating_axis
        xaxis_zero = new_floating_axis(nth_coord=0,
                                       value=0.,
                                       axis_direction="bottom",
                                       axes=self)

        xaxis_zero.line.set_clip_path(self.patch)
        xaxis_zero.set_visible(False)
        self._axislines["xzero"] = xaxis_zero

        yaxis_zero = new_floating_axis(nth_coord=1,
                                       value=0.,
                                       axis_direction="left",
                                       axes=self)

        yaxis_zero.line.set_clip_path(self.patch)
        yaxis_zero.set_visible(False)
        self._axislines["yzero"] = yaxis_zero


SubplotZero = maxes.subplot_class_factory(AxesZero)
