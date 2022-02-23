# axis3d.py, original mplot3d version by John Porter
# Created: 23 Sep 2005
# Parts rewritten by Reinier Heeres <reinier@heeres.eu>

import numpy as np

import matplotlib.transforms as mtransforms
from matplotlib import (
    artist, lines as mlines, axis as maxis, patches as mpatches, rcParams)
from . import art3d, proj3d


def move_from_center(coord, centers, deltas, axmask=(True, True, True)):
    """
    For each coordinate where *axmask* is True, move *coord* away from
    *centers* by *deltas*.
    """
    coord = np.asarray(coord)
    return coord + axmask * np.copysign(1, coord - centers) * deltas


def tick_update_position(tick, tickxs, tickys, labelpos):
    """Update tick line and label position and style."""

    tick.label1.set_position(labelpos)
    tick.label2.set_position(labelpos)
    tick.tick1line.set_visible(True)
    tick.tick2line.set_visible(False)
    tick.tick1line.set_linestyle('-')
    tick.tick1line.set_marker('')
    tick.tick1line.set_data(tickxs, tickys)
    tick.gridline.set_data(0, 0)


class Axis(maxis.XAxis):
    """An Axis class for the 3D plots."""
    # These points from the unit cube make up the x, y and z-planes
    _PLANES = (
        (0, 3, 7, 4), (1, 2, 6, 5),     # yz planes
        (0, 1, 5, 4), (3, 2, 6, 7),     # xz planes
        (0, 1, 2, 3), (4, 5, 6, 7),     # xy planes
    )

    # Some properties for the axes
    _AXINFO = {
        'x': {'i': 0, 'tickdir': 1, 'juggled': (1, 0, 2),
              'color': (0.95, 0.95, 0.95, 0.5)},
        'y': {'i': 1, 'tickdir': 0, 'juggled': (0, 1, 2),
              'color': (0.90, 0.90, 0.90, 0.5)},
        'z': {'i': 2, 'tickdir': 0, 'juggled': (0, 2, 1),
              'color': (0.925, 0.925, 0.925, 0.5)},
    }

    def __init__(self, adir, v_intervalx, d_intervalx, axes, *args,
                 rotate_label=None, **kwargs):
        # adir identifies which axes this is
        self.adir = adir

        # This is a temporary member variable.
        # Do not depend on this existing in future releases!
        self._axinfo = self._AXINFO[adir].copy()
        if rcParams['_internal.classic_mode']:
            self._axinfo.update({
                'label': {'va': 'center', 'ha': 'center'},
                'tick': {
                    'inward_factor': 0.2,
                    'outward_factor': 0.1,
                    'linewidth': {
                        True: rcParams['lines.linewidth'],  # major
                        False: rcParams['lines.linewidth'],  # minor
                    }
                },
                'axisline': {'linewidth': 0.75, 'color': (0, 0, 0, 1)},
                'grid': {
                    'color': (0.9, 0.9, 0.9, 1),
                    'linewidth': 1.0,
                    'linestyle': '-',
                },
            })
        else:
            self._axinfo.update({
                'label': {'va': 'center', 'ha': 'center'},
                'tick': {
                    'inward_factor': 0.2,
                    'outward_factor': 0.1,
                    'linewidth': {
                        True: (  # major
                            rcParams['xtick.major.width'] if adir in 'xz' else
                            rcParams['ytick.major.width']),
                        False: (  # minor
                            rcParams['xtick.minor.width'] if adir in 'xz' else
                            rcParams['ytick.minor.width']),
                    }
                },
                'axisline': {
                    'linewidth': rcParams['axes.linewidth'],
                    'color': rcParams['axes.edgecolor'],
                },
                'grid': {
                    'color': rcParams['grid.color'],
                    'linewidth': rcParams['grid.linewidth'],
                    'linestyle': rcParams['grid.linestyle'],
                },
            })

        super().__init__(axes, *args, **kwargs)

        # data and viewing intervals for this direction
        self.d_interval = d_intervalx
        self.v_interval = v_intervalx
        self.set_rotate_label(rotate_label)

    def init3d(self):
        self.line = mlines.Line2D(
            xdata=(0, 0), ydata=(0, 0),
            linewidth=self._axinfo['axisline']['linewidth'],
            color=self._axinfo['axisline']['color'],
            antialiased=True)

        # Store dummy data in Polygon object
        self.pane = mpatches.Polygon(
            np.array([[0, 0], [0, 1], [1, 0], [0, 0]]),
            closed=False, alpha=0.8, facecolor='k', edgecolor='k')
        self.set_pane_color(self._axinfo['color'])

        self.axes._set_artist_props(self.line)
        self.axes._set_artist_props(self.pane)
        self.gridlines = art3d.Line3DCollection([])
        self.axes._set_artist_props(self.gridlines)
        self.axes._set_artist_props(self.label)
        self.axes._set_artist_props(self.offsetText)
        # Need to be able to place the label at the correct location
        self.label._transform = self.axes.transData
        self.offsetText._transform = self.axes.transData

    def get_major_ticks(self, numticks=None):
        ticks = super().get_major_ticks(numticks)
        for t in ticks:
            for obj in [
                    t.tick1line, t.tick2line, t.gridline, t.label1, t.label2]:
                obj.set_transform(self.axes.transData)
        return ticks

    def get_minor_ticks(self, numticks=None):
        ticks = super().get_minor_ticks(numticks)
        for t in ticks:
            for obj in [
                    t.tick1line, t.tick2line, t.gridline, t.label1, t.label2]:
                obj.set_transform(self.axes.transData)
        return ticks

    def set_pane_pos(self, xys):
        xys = np.asarray(xys)
        xys = xys[:, :2]
        self.pane.xy = xys
        self.stale = True

    def set_pane_color(self, color):
        """Set pane color to a RGBA tuple."""
        self._axinfo['color'] = color
        self.pane.set_edgecolor(color)
        self.pane.set_facecolor(color)
        self.pane.set_alpha(color[-1])
        self.stale = True

    def set_rotate_label(self, val):
        """
        Whether to rotate the axis label: True, False or None.
        If set to None the label will be rotated if longer than 4 chars.
        """
        self._rotate_label = val
        self.stale = True

    def get_rotate_label(self, text):
        if self._rotate_label is not None:
            return self._rotate_label
        else:
            return len(text) > 4

    def _get_coord_info(self, renderer):
        mins, maxs = np.array([
            self.axes.get_xbound(),
            self.axes.get_ybound(),
            self.axes.get_zbound(),
        ]).T

        # Get the mean value for each bound:
        centers = 0.5 * (maxs + mins)

        # Add a small offset between min/max point and the edge of the
        # plot:
        deltas = (maxs - mins) / 12
        mins -= 0.25 * deltas
        maxs += 0.25 * deltas

        # Project the bounds along the current position of the cube:
        bounds = mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]
        bounds_proj = self.axes.tunit_cube(bounds, self.axes.M)

        # Determine which one of the parallel planes are higher up:
        highs = np.zeros(3, dtype=bool)
        for i in range(3):
            mean_z0 = np.mean(bounds_proj[self._PLANES[2 * i], 2])
            mean_z1 = np.mean(bounds_proj[self._PLANES[2 * i + 1], 2])
            highs[i] = mean_z0 < mean_z1

        return mins, maxs, centers, deltas, bounds_proj, highs

    def _get_axis_line_edge_points(self, minmax, maxmin):
        """Get the edge points for the black bolded axis line."""
        # When changing vertical axis some of the axes has to be
        # moved to the other plane so it looks the same as if the z-axis
        # was the vertical axis.
        mb = [minmax, maxmin]
        mb_rev = mb[::-1]
        mm = [[mb, mb_rev, mb_rev], [mb_rev, mb_rev, mb], [mb, mb, mb]]
        mm = mm[self.axes._vertical_axis][self._axinfo["i"]]

        juggled = self._axinfo["juggled"]
        edge_point_0 = mm[0].copy()
        edge_point_0[juggled[0]] = mm[1][juggled[0]]

        edge_point_1 = edge_point_0.copy()
        edge_point_1[juggled[1]] = mm[1][juggled[1]]

        return edge_point_0, edge_point_1

    def _get_tickdir(self):
        """
        Get the direction of the tick.

        Returns
        -------
        tickdir : int
            Index which indicates which coordinate the tick line will
            align with.
        """
        # TODO: Move somewhere else where it's triggered less:
        tickdirs_base = [v["tickdir"] for v in self._AXINFO.values()]
        info_i = [v["i"] for v in self._AXINFO.values()]

        i = self._axinfo["i"]
        j = self.axes._vertical_axis - 2
        # tickdir = [[1, 2, 1], [2, 2, 0], [1, 0, 0]][i]
        tickdir = np.roll(info_i, -j)[np.roll(tickdirs_base, j)][i]
        return tickdir

    def draw_pane(self, renderer):
        renderer.open_group('pane3d', gid=self.get_gid())

        mins, maxs, centers, deltas, tc, highs = self._get_coord_info(renderer)

        info = self._axinfo
        index = info['i']
        if not highs[index]:
            plane = self._PLANES[2 * index]
        else:
            plane = self._PLANES[2 * index + 1]
        xys = [tc[p] for p in plane]
        self.set_pane_pos(xys)
        self.pane.draw(renderer)

        renderer.close_group('pane3d')

    @artist.allow_rasterization
    def draw(self, renderer):
        self.label._transform = self.axes.transData
        renderer.open_group("axis3d", gid=self.get_gid())

        ticks = self._update_ticks()

        # Get general axis information:
        info = self._axinfo
        index = info["i"]
        juggled = info["juggled"]

        mins, maxs, centers, deltas, tc, highs = self._get_coord_info(renderer)

        minmax = np.where(highs, maxs, mins)
        maxmin = np.where(~highs, maxs, mins)

        # Create edge points for the black bolded axis line:
        edgep1, edgep2 = self._get_axis_line_edge_points(minmax, maxmin)

        # Project the edge points along the current position and
        # create the line:
        pep = proj3d.proj_trans_points([edgep1, edgep2], self.axes.M)
        pep = np.asarray(pep)
        self.line.set_data(pep[0], pep[1])
        self.line.draw(renderer)

        # Grid points where the planes meet
        xyz0 = np.tile(minmax, (len(ticks), 1))
        xyz0[:, index] = [tick.get_loc() for tick in ticks]

        # Draw labels
        # The transAxes transform is used because the Text object
        # rotates the text relative to the display coordinate system.
        # Therefore, if we want the labels to remain parallel to the
        # axis regardless of the aspect ratio, we need to convert the
        # edge points of the plane to display coordinates and calculate
        # an angle from that.
        # TODO: Maybe Text objects should handle this themselves?
        dx, dy = (self.axes.transAxes.transform([pep[0:2, 1]]) -
                  self.axes.transAxes.transform([pep[0:2, 0]]))[0]

        lxyz = 0.5 * (edgep1 + edgep2)

        # A rough estimate; points are ambiguous since 3D plots rotate
        reltoinches = self.figure.dpi_scale_trans.inverted()
        ax_inches = reltoinches.transform(self.axes.bbox.size)
        ax_points_estimate = sum(72. * ax_inches)
        deltas_per_point = 48 / ax_points_estimate
        default_offset = 21.
        labeldeltas = (
            (self.labelpad + default_offset) * deltas_per_point * deltas)
        axmask = [True, True, True]
        axmask[index] = False
        lxyz = move_from_center(lxyz, centers, labeldeltas, axmask)
        tlx, tly, tlz = proj3d.proj_transform(*lxyz, self.axes.M)
        self.label.set_position((tlx, tly))
        if self.get_rotate_label(self.label.get_text()):
            angle = art3d._norm_text_angle(np.rad2deg(np.arctan2(dy, dx)))
            self.label.set_rotation(angle)
        self.label.set_va(info['label']['va'])
        self.label.set_ha(info['label']['ha'])
        self.label.draw(renderer)

        # Draw Offset text

        # Which of the two edge points do we want to
        # use for locating the offset text?
        if juggled[2] == 2:
            outeredgep = edgep1
            outerindex = 0
        else:
            outeredgep = edgep2
            outerindex = 1

        pos = move_from_center(outeredgep, centers, labeldeltas, axmask)
        olx, oly, olz = proj3d.proj_transform(*pos, self.axes.M)
        self.offsetText.set_text(self.major.formatter.get_offset())
        self.offsetText.set_position((olx, oly))
        angle = art3d._norm_text_angle(np.rad2deg(np.arctan2(dy, dx)))
        self.offsetText.set_rotation(angle)
        # Must set rotation mode to "anchor" so that
        # the alignment point is used as the "fulcrum" for rotation.
        self.offsetText.set_rotation_mode('anchor')

        # ----------------------------------------------------------------------
        # Note: the following statement for determining the proper alignment of
        # the offset text. This was determined entirely by trial-and-error
        # and should not be in any way considered as "the way".  There are
        # still some edge cases where alignment is not quite right, but this
        # seems to be more of a geometry issue (in other words, I might be
        # using the wrong reference points).
        #
        # (TT, FF, TF, FT) are the shorthand for the tuple of
        #   (centpt[info['tickdir']] <= pep[info['tickdir'], outerindex],
        #    centpt[index] <= pep[index, outerindex])
        #
        # Three-letters (e.g., TFT, FTT) are short-hand for the array of bools
        # from the variable 'highs'.
        # ---------------------------------------------------------------------
        centpt = proj3d.proj_transform(*centers, self.axes.M)
        if centpt[info['tickdir']] > pep[info['tickdir'], outerindex]:
            # if FT and if highs has an even number of Trues
            if (centpt[index] <= pep[index, outerindex]
                    and np.count_nonzero(highs) % 2 == 0):
                # Usually, this means align right, except for the FTT case,
                # in which offset for axis 1 and 2 are aligned left.
                if highs.tolist() == [False, True, True] and index in (1, 2):
                    align = 'left'
                else:
                    align = 'right'
            else:
                # The FF case
                align = 'left'
        else:
            # if TF and if highs has an even number of Trues
            if (centpt[index] > pep[index, outerindex]
                    and np.count_nonzero(highs) % 2 == 0):
                # Usually mean align left, except if it is axis 2
                if index == 2:
                    align = 'right'
                else:
                    align = 'left'
            else:
                # The TT case
                align = 'right'

        self.offsetText.set_va('center')
        self.offsetText.set_ha(align)
        self.offsetText.draw(renderer)

        if self.axes._draw_grid and len(ticks):
            # Grid lines go from the end of one plane through the plane
            # intersection (at xyz0) to the end of the other plane.  The first
            # point (0) differs along dimension index-2 and the last (2) along
            # dimension index-1.
            lines = np.stack([xyz0, xyz0, xyz0], axis=1)
            lines[:, 0, index - 2] = maxmin[index - 2]
            lines[:, 2, index - 1] = maxmin[index - 1]
            self.gridlines.set_segments(lines)
            self.gridlines.set_color(info['grid']['color'])
            self.gridlines.set_linewidth(info['grid']['linewidth'])
            self.gridlines.set_linestyle(info['grid']['linestyle'])
            self.gridlines.do_3d_projection()
            self.gridlines.draw(renderer)

        # Draw ticks:
        tickdir = self._get_tickdir()
        tickdelta = deltas[tickdir]
        if highs[tickdir]:
            ticksign = 1
        else:
            ticksign = -1

        for tick in ticks:
            # Get tick line positions
            pos = edgep1.copy()
            pos[index] = tick.get_loc()
            pos[tickdir] = (
                edgep1[tickdir]
                + info['tick']['outward_factor'] * ticksign * tickdelta)
            x1, y1, z1 = proj3d.proj_transform(*pos, self.axes.M)
            pos[tickdir] = (
                edgep1[tickdir]
                - info['tick']['inward_factor'] * ticksign * tickdelta)
            x2, y2, z2 = proj3d.proj_transform(*pos, self.axes.M)

            # Get position of label
            default_offset = 8.  # A rough estimate
            labeldeltas = (
                (tick.get_pad() + default_offset) * deltas_per_point * deltas)

            axmask = [True, True, True]
            axmask[index] = False
            pos[tickdir] = edgep1[tickdir]
            pos = move_from_center(pos, centers, labeldeltas, axmask)
            lx, ly, lz = proj3d.proj_transform(*pos, self.axes.M)

            tick_update_position(tick, (x1, x2), (y1, y2), (lx, ly))
            tick.tick1line.set_linewidth(
                info['tick']['linewidth'][tick._major])
            tick.draw(renderer)

        renderer.close_group('axis3d')
        self.stale = False

    # TODO: Get this to work (more) properly when mplot3d supports the
    #       transforms framework.
    def get_tightbbox(self, renderer, *, for_layout_only=False):
        # docstring inherited
        if not self.get_visible():
            return
        # We have to directly access the internal data structures
        # (and hope they are up to date) because at draw time we
        # shift the ticks and their labels around in (x, y) space
        # based on the projection, the current view port, and their
        # position in 3D space.  If we extend the transforms framework
        # into 3D we would not need to do this different book keeping
        # than we do in the normal axis
        major_locs = self.get_majorticklocs()
        minor_locs = self.get_minorticklocs()

        ticks = [*self.get_minor_ticks(len(minor_locs)),
                 *self.get_major_ticks(len(major_locs))]
        view_low, view_high = self.get_view_interval()
        if view_low > view_high:
            view_low, view_high = view_high, view_low
        interval_t = self.get_transform().transform([view_low, view_high])

        ticks_to_draw = []
        for tick in ticks:
            try:
                loc_t = self.get_transform().transform(tick.get_loc())
            except AssertionError:
                # Transform.transform doesn't allow masked values but
                # some scales might make them, so we need this try/except.
                pass
            else:
                if mtransforms._interval_contains_close(interval_t, loc_t):
                    ticks_to_draw.append(tick)

        ticks = ticks_to_draw

        bb_1, bb_2 = self._get_tick_bboxes(ticks, renderer)
        other = []

        if self.line.get_visible():
            other.append(self.line.get_window_extent(renderer))
        if (self.label.get_visible() and not for_layout_only and
                self.label.get_text()):
            other.append(self.label.get_window_extent(renderer))

        return mtransforms.Bbox.union([*bb_1, *bb_2, *other])

    @property
    def d_interval(self):
        return self.get_data_interval()

    @d_interval.setter
    def d_interval(self, minmax):
        self.set_data_interval(*minmax)

    @property
    def v_interval(self):
        return self.get_view_interval()

    @v_interval.setter
    def v_interval(self, minmax):
        self.set_view_interval(*minmax)


# Use classes to look at different data limits


class XAxis(Axis):
    get_view_interval, set_view_interval = maxis._make_getset_interval(
        "view", "xy_viewLim", "intervalx")
    get_data_interval, set_data_interval = maxis._make_getset_interval(
        "data", "xy_dataLim", "intervalx")


class YAxis(Axis):
    get_view_interval, set_view_interval = maxis._make_getset_interval(
        "view", "xy_viewLim", "intervaly")
    get_data_interval, set_data_interval = maxis._make_getset_interval(
        "data", "xy_dataLim", "intervaly")


class ZAxis(Axis):
    get_view_interval, set_view_interval = maxis._make_getset_interval(
        "view", "zz_viewLim", "intervalx")
    get_data_interval, set_data_interval = maxis._make_getset_interval(
        "data", "zz_dataLim", "intervalx")
