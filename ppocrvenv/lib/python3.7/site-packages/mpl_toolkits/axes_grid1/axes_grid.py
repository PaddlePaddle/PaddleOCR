from numbers import Number
import functools

import numpy as np

from matplotlib import _api, cbook
from matplotlib.gridspec import SubplotSpec

from .axes_divider import Size, SubplotDivider, Divider
from .mpl_axes import Axes


def _tick_only(ax, bottom_on, left_on):
    bottom_off = not bottom_on
    left_off = not left_on
    ax.axis["bottom"].toggle(ticklabels=bottom_off, label=bottom_off)
    ax.axis["left"].toggle(ticklabels=left_off, label=left_off)


class CbarAxesBase:
    def __init__(self, *args, orientation, **kwargs):
        self.orientation = orientation
        self._default_label_on = True
        self._locator = None  # deprecated.
        super().__init__(*args, **kwargs)

    def colorbar(self, mappable, *, ticks=None, **kwargs):
        orientation = (
            "horizontal" if self.orientation in ["top", "bottom"] else
            "vertical")
        cb = self.figure.colorbar(mappable, cax=self, orientation=orientation,
                                  ticks=ticks, **kwargs)
        return cb

    def toggle_label(self, b):
        self._default_label_on = b
        axis = self.axis[self.orientation]
        axis.toggle(ticklabels=b, label=b)

    def cla(self):
        orientation = self.orientation
        super().cla()
        self.orientation = orientation


@_api.deprecated("3.5")
class CbarAxes(CbarAxesBase, Axes):
    pass


_cbaraxes_class_factory = cbook._make_class_factory(CbarAxesBase, "Cbar{}")


class Grid:
    """
    A grid of Axes.

    In Matplotlib, the axes location (and size) is specified in normalized
    figure coordinates. This may not be ideal for images that needs to be
    displayed with a given aspect ratio; for example, it is difficult to
    display multiple images of a same size with some fixed padding between
    them.  AxesGrid can be used in such case.
    """

    _defaultAxesClass = Axes

    def __init__(self, fig,
                 rect,
                 nrows_ncols,
                 ngrids=None,
                 direction="row",
                 axes_pad=0.02,
                 *,
                 share_all=False,
                 share_x=True,
                 share_y=True,
                 label_mode="L",
                 axes_class=None,
                 aspect=False,
                 ):
        """
        Parameters
        ----------
        fig : `.Figure`
            The parent figure.
        rect : (float, float, float, float) or int
            The axes position, as a ``(left, bottom, width, height)`` tuple or
            as a three-digit subplot position code (e.g., "121").
        nrows_ncols : (int, int)
            Number of rows and columns in the grid.
        ngrids : int or None, default: None
            If not None, only the first *ngrids* axes in the grid are created.
        direction : {"row", "column"}, default: "row"
            Whether axes are created in row-major ("row by row") or
            column-major order ("column by column").  This also affects the
            order in which axes are accessed using indexing (``grid[index]``).
        axes_pad : float or (float, float), default: 0.02
            Padding or (horizontal padding, vertical padding) between axes, in
            inches.
        share_all : bool, default: False
            Whether all axes share their x- and y-axis.  Overrides *share_x*
            and *share_y*.
        share_x : bool, default: True
            Whether all axes of a column share their x-axis.
        share_y : bool, default: True
            Whether all axes of a row share their y-axis.
        label_mode : {"L", "1", "all"}, default: "L"
            Determines which axes will get tick labels:

            - "L": All axes on the left column get vertical tick labels;
              all axes on the bottom row get horizontal tick labels.
            - "1": Only the bottom left axes is labelled.
            - "all": all axes are labelled.

        axes_class : subclass of `matplotlib.axes.Axes`, default: None
        aspect : bool, default: False
            Whether the axes aspect ratio follows the aspect ratio of the data
            limits.
        """
        self._nrows, self._ncols = nrows_ncols

        if ngrids is None:
            ngrids = self._nrows * self._ncols
        else:
            if not 0 < ngrids <= self._nrows * self._ncols:
                raise Exception("")

        self.ngrids = ngrids

        self._horiz_pad_size, self._vert_pad_size = map(
            Size.Fixed, np.broadcast_to(axes_pad, 2))

        _api.check_in_list(["column", "row"], direction=direction)
        self._direction = direction

        if axes_class is None:
            axes_class = self._defaultAxesClass
        elif isinstance(axes_class, (list, tuple)):
            cls, kwargs = axes_class
            axes_class = functools.partial(cls, **kwargs)

        kw = dict(horizontal=[], vertical=[], aspect=aspect)
        if isinstance(rect, (str, Number, SubplotSpec)):
            self._divider = SubplotDivider(fig, rect, **kw)
        elif len(rect) == 3:
            self._divider = SubplotDivider(fig, *rect, **kw)
        elif len(rect) == 4:
            self._divider = Divider(fig, rect, **kw)
        else:
            raise Exception("")

        rect = self._divider.get_position()

        axes_array = np.full((self._nrows, self._ncols), None, dtype=object)
        for i in range(self.ngrids):
            col, row = self._get_col_row(i)
            if share_all:
                sharex = sharey = axes_array[0, 0]
            else:
                sharex = axes_array[0, col] if share_x else None
                sharey = axes_array[row, 0] if share_y else None
            axes_array[row, col] = axes_class(
                fig, rect, sharex=sharex, sharey=sharey)
        self.axes_all = axes_array.ravel(
            order="C" if self._direction == "row" else "F").tolist()
        self.axes_column = axes_array.T.tolist()
        self.axes_row = axes_array.tolist()
        self.axes_llc = self.axes_column[0][-1]

        self._init_locators()

        for ax in self.axes_all:
            fig.add_axes(ax)

        self.set_label_mode(label_mode)

    def _init_locators(self):

        h = []
        h_ax_pos = []
        for _ in range(self._ncols):
            if h:
                h.append(self._horiz_pad_size)
            h_ax_pos.append(len(h))
            sz = Size.Scaled(1)
            h.append(sz)

        v = []
        v_ax_pos = []
        for _ in range(self._nrows):
            if v:
                v.append(self._vert_pad_size)
            v_ax_pos.append(len(v))
            sz = Size.Scaled(1)
            v.append(sz)

        for i in range(self.ngrids):
            col, row = self._get_col_row(i)
            locator = self._divider.new_locator(
                nx=h_ax_pos[col], ny=v_ax_pos[self._nrows - 1 - row])
            self.axes_all[i].set_axes_locator(locator)

        self._divider.set_horizontal(h)
        self._divider.set_vertical(v)

    def _get_col_row(self, n):
        if self._direction == "column":
            col, row = divmod(n, self._nrows)
        else:
            row, col = divmod(n, self._ncols)

        return col, row

    # Good to propagate __len__ if we have __getitem__
    def __len__(self):
        return len(self.axes_all)

    def __getitem__(self, i):
        return self.axes_all[i]

    def get_geometry(self):
        """
        Return the number of rows and columns of the grid as (nrows, ncols).
        """
        return self._nrows, self._ncols

    def set_axes_pad(self, axes_pad):
        """
        Set the padding between the axes.

        Parameters
        ----------
        axes_pad : (float, float)
            The padding (horizontal pad, vertical pad) in inches.
        """
        self._horiz_pad_size.fixed_size = axes_pad[0]
        self._vert_pad_size.fixed_size = axes_pad[1]

    def get_axes_pad(self):
        """
        Return the axes padding.

        Returns
        -------
        hpad, vpad
            Padding (horizontal pad, vertical pad) in inches.
        """
        return (self._horiz_pad_size.fixed_size,
                self._vert_pad_size.fixed_size)

    def set_aspect(self, aspect):
        """Set the aspect of the SubplotDivider."""
        self._divider.set_aspect(aspect)

    def get_aspect(self):
        """Return the aspect of the SubplotDivider."""
        return self._divider.get_aspect()

    def set_label_mode(self, mode):
        """
        Define which axes have tick labels.

        Parameters
        ----------
        mode : {"L", "1", "all"}
            The label mode:

            - "L": All axes on the left column get vertical tick labels;
              all axes on the bottom row get horizontal tick labels.
            - "1": Only the bottom left axes is labelled.
            - "all": all axes are labelled.
        """
        if mode == "all":
            for ax in self.axes_all:
                _tick_only(ax, False, False)
        elif mode == "L":
            # left-most axes
            for ax in self.axes_column[0][:-1]:
                _tick_only(ax, bottom_on=True, left_on=False)
            # lower-left axes
            ax = self.axes_column[0][-1]
            _tick_only(ax, bottom_on=False, left_on=False)

            for col in self.axes_column[1:]:
                # axes with no labels
                for ax in col[:-1]:
                    _tick_only(ax, bottom_on=True, left_on=True)

                # bottom
                ax = col[-1]
                _tick_only(ax, bottom_on=False, left_on=True)

        elif mode == "1":
            for ax in self.axes_all:
                _tick_only(ax, bottom_on=True, left_on=True)

            ax = self.axes_llc
            _tick_only(ax, bottom_on=False, left_on=False)

    def get_divider(self):
        return self._divider

    def set_axes_locator(self, locator):
        self._divider.set_locator(locator)

    def get_axes_locator(self):
        return self._divider.get_locator()

    @_api.deprecated("3.5")
    def get_vsize_hsize(self):
        return self._divider.get_vsize_hsize()


class ImageGrid(Grid):
    # docstring inherited

    def __init__(self, fig,
                 rect,
                 nrows_ncols,
                 ngrids=None,
                 direction="row",
                 axes_pad=0.02,
                 *,
                 share_all=False,
                 aspect=True,
                 label_mode="L",
                 cbar_mode=None,
                 cbar_location="right",
                 cbar_pad=None,
                 cbar_size="5%",
                 cbar_set_cax=True,
                 axes_class=None,
                 ):
        """
        Parameters
        ----------
        fig : `.Figure`
            The parent figure.
        rect : (float, float, float, float) or int
            The axes position, as a ``(left, bottom, width, height)`` tuple or
            as a three-digit subplot position code (e.g., "121").
        nrows_ncols : (int, int)
            Number of rows and columns in the grid.
        ngrids : int or None, default: None
            If not None, only the first *ngrids* axes in the grid are created.
        direction : {"row", "column"}, default: "row"
            Whether axes are created in row-major ("row by row") or
            column-major order ("column by column").  This also affects the
            order in which axes are accessed using indexing (``grid[index]``).
        axes_pad : float or (float, float), default: 0.02in
            Padding or (horizontal padding, vertical padding) between axes, in
            inches.
        share_all : bool, default: False
            Whether all axes share their x- and y-axis.
        aspect : bool, default: True
            Whether the axes aspect ratio follows the aspect ratio of the data
            limits.
        label_mode : {"L", "1", "all"}, default: "L"
            Determines which axes will get tick labels:

            - "L": All axes on the left column get vertical tick labels;
              all axes on the bottom row get horizontal tick labels.
            - "1": Only the bottom left axes is labelled.
            - "all": all axes are labelled.

        cbar_mode : {"each", "single", "edge", None}, default: None
            Whether to create a colorbar for "each" axes, a "single" colorbar
            for the entire grid, colorbars only for axes on the "edge"
            determined by *cbar_location*, or no colorbars.  The colorbars are
            stored in the :attr:`cbar_axes` attribute.
        cbar_location : {"left", "right", "bottom", "top"}, default: "right"
        cbar_pad : float, default: None
            Padding between the image axes and the colorbar axes.
        cbar_size : size specification (see `.Size.from_any`), default: "5%"
            Colorbar size.
        cbar_set_cax : bool, default: True
            If True, each axes in the grid has a *cax* attribute that is bound
            to associated *cbar_axes*.
        axes_class : subclass of `matplotlib.axes.Axes`, default: None
        """
        self._colorbar_mode = cbar_mode
        self._colorbar_location = cbar_location
        self._colorbar_pad = cbar_pad
        self._colorbar_size = cbar_size
        # The colorbar axes are created in _init_locators().

        super().__init__(
            fig, rect, nrows_ncols, ngrids,
            direction=direction, axes_pad=axes_pad,
            share_all=share_all, share_x=True, share_y=True, aspect=aspect,
            label_mode=label_mode, axes_class=axes_class)

        for ax in self.cbar_axes:
            fig.add_axes(ax)

        if cbar_set_cax:
            if self._colorbar_mode == "single":
                for ax in self.axes_all:
                    ax.cax = self.cbar_axes[0]
            elif self._colorbar_mode == "edge":
                for index, ax in enumerate(self.axes_all):
                    col, row = self._get_col_row(index)
                    if self._colorbar_location in ("left", "right"):
                        ax.cax = self.cbar_axes[row]
                    else:
                        ax.cax = self.cbar_axes[col]
            else:
                for ax, cax in zip(self.axes_all, self.cbar_axes):
                    ax.cax = cax

    def _init_locators(self):
        # Slightly abusing this method to inject colorbar creation into init.

        if self._colorbar_pad is None:
            # horizontal or vertical arrangement?
            if self._colorbar_location in ("left", "right"):
                self._colorbar_pad = self._horiz_pad_size.fixed_size
            else:
                self._colorbar_pad = self._vert_pad_size.fixed_size
        self.cbar_axes = [
            _cbaraxes_class_factory(self._defaultAxesClass)(
                self.axes_all[0].figure, self._divider.get_position(),
                orientation=self._colorbar_location)
            for _ in range(self.ngrids)]

        cb_mode = self._colorbar_mode
        cb_location = self._colorbar_location

        h = []
        v = []

        h_ax_pos = []
        h_cb_pos = []
        if cb_mode == "single" and cb_location in ("left", "bottom"):
            if cb_location == "left":
                sz = self._nrows * Size.AxesX(self.axes_llc)
                h.append(Size.from_any(self._colorbar_size, sz))
                h.append(Size.from_any(self._colorbar_pad, sz))
                locator = self._divider.new_locator(nx=0, ny=0, ny1=-1)
            elif cb_location == "bottom":
                sz = self._ncols * Size.AxesY(self.axes_llc)
                v.append(Size.from_any(self._colorbar_size, sz))
                v.append(Size.from_any(self._colorbar_pad, sz))
                locator = self._divider.new_locator(nx=0, nx1=-1, ny=0)
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(False)
            self.cbar_axes[0].set_axes_locator(locator)
            self.cbar_axes[0].set_visible(True)

        for col, ax in enumerate(self.axes_row[0]):
            if h:
                h.append(self._horiz_pad_size)

            if ax:
                sz = Size.AxesX(ax, aspect="axes", ref_ax=self.axes_all[0])
            else:
                sz = Size.AxesX(self.axes_all[0],
                                aspect="axes", ref_ax=self.axes_all[0])

            if (cb_location == "left"
                    and (cb_mode == "each"
                         or (cb_mode == "edge" and col == 0))):
                h_cb_pos.append(len(h))
                h.append(Size.from_any(self._colorbar_size, sz))
                h.append(Size.from_any(self._colorbar_pad, sz))

            h_ax_pos.append(len(h))
            h.append(sz)

            if (cb_location == "right"
                    and (cb_mode == "each"
                         or (cb_mode == "edge" and col == self._ncols - 1))):
                h.append(Size.from_any(self._colorbar_pad, sz))
                h_cb_pos.append(len(h))
                h.append(Size.from_any(self._colorbar_size, sz))

        v_ax_pos = []
        v_cb_pos = []
        for row, ax in enumerate(self.axes_column[0][::-1]):
            if v:
                v.append(self._vert_pad_size)

            if ax:
                sz = Size.AxesY(ax, aspect="axes", ref_ax=self.axes_all[0])
            else:
                sz = Size.AxesY(self.axes_all[0],
                                aspect="axes", ref_ax=self.axes_all[0])

            if (cb_location == "bottom"
                    and (cb_mode == "each"
                         or (cb_mode == "edge" and row == 0))):
                v_cb_pos.append(len(v))
                v.append(Size.from_any(self._colorbar_size, sz))
                v.append(Size.from_any(self._colorbar_pad, sz))

            v_ax_pos.append(len(v))
            v.append(sz)

            if (cb_location == "top"
                    and (cb_mode == "each"
                         or (cb_mode == "edge" and row == self._nrows - 1))):
                v.append(Size.from_any(self._colorbar_pad, sz))
                v_cb_pos.append(len(v))
                v.append(Size.from_any(self._colorbar_size, sz))

        for i in range(self.ngrids):
            col, row = self._get_col_row(i)
            locator = self._divider.new_locator(nx=h_ax_pos[col],
                                                ny=v_ax_pos[self._nrows-1-row])
            self.axes_all[i].set_axes_locator(locator)

            if cb_mode == "each":
                if cb_location in ("right", "left"):
                    locator = self._divider.new_locator(
                        nx=h_cb_pos[col], ny=v_ax_pos[self._nrows - 1 - row])

                elif cb_location in ("top", "bottom"):
                    locator = self._divider.new_locator(
                        nx=h_ax_pos[col], ny=v_cb_pos[self._nrows - 1 - row])

                self.cbar_axes[i].set_axes_locator(locator)
            elif cb_mode == "edge":
                if (cb_location == "left" and col == 0
                        or cb_location == "right" and col == self._ncols - 1):
                    locator = self._divider.new_locator(
                        nx=h_cb_pos[0], ny=v_ax_pos[self._nrows - 1 - row])
                    self.cbar_axes[row].set_axes_locator(locator)
                elif (cb_location == "bottom" and row == self._nrows - 1
                      or cb_location == "top" and row == 0):
                    locator = self._divider.new_locator(nx=h_ax_pos[col],
                                                        ny=v_cb_pos[0])
                    self.cbar_axes[col].set_axes_locator(locator)

        if cb_mode == "single":
            if cb_location == "right":
                sz = self._nrows * Size.AxesX(self.axes_llc)
                h.append(Size.from_any(self._colorbar_pad, sz))
                h.append(Size.from_any(self._colorbar_size, sz))
                locator = self._divider.new_locator(nx=-2, ny=0, ny1=-1)
            elif cb_location == "top":
                sz = self._ncols * Size.AxesY(self.axes_llc)
                v.append(Size.from_any(self._colorbar_pad, sz))
                v.append(Size.from_any(self._colorbar_size, sz))
                locator = self._divider.new_locator(nx=0, nx1=-1, ny=-2)
            if cb_location in ("right", "top"):
                for i in range(self.ngrids):
                    self.cbar_axes[i].set_visible(False)
                self.cbar_axes[0].set_axes_locator(locator)
                self.cbar_axes[0].set_visible(True)
        elif cb_mode == "each":
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(True)
        elif cb_mode == "edge":
            if cb_location in ("right", "left"):
                count = self._nrows
            else:
                count = self._ncols
            for i in range(count):
                self.cbar_axes[i].set_visible(True)
            for j in range(i + 1, self.ngrids):
                self.cbar_axes[j].set_visible(False)
        else:
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(False)
                self.cbar_axes[i].set_position([1., 1., 0.001, 0.001],
                                               which="active")

        self._divider.set_horizontal(h)
        self._divider.set_vertical(v)


AxesGrid = ImageGrid
