"""
Helper classes to adjust the positions of multiple axes at drawing time.
"""

import numpy as np

import matplotlib as mpl
from matplotlib import _api
from matplotlib.axes import SubplotBase
from matplotlib.gridspec import SubplotSpec, GridSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size


class Divider:
    """
    An Axes positioning class.

    The divider is initialized with lists of horizontal and vertical sizes
    (:mod:`mpl_toolkits.axes_grid1.axes_size`) based on which a given
    rectangular area will be divided.

    The `new_locator` method then creates a callable object
    that can be used as the *axes_locator* of the axes.
    """

    def __init__(self, fig, pos, horizontal, vertical,
                 aspect=None, anchor="C"):
        """
        Parameters
        ----------
        fig : Figure
        pos : tuple of 4 floats
            Position of the rectangle that will be divided.
        horizontal : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
            Sizes for horizontal division.
        vertical : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
            Sizes for vertical division.
        aspect : bool
            Whether overall rectangular area is reduced so that the relative
            part of the horizontal and vertical scales have the same scale.
        anchor : {'C', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW', 'W'}
            Placement of the reduced rectangle, when *aspect* is True.
        """

        self._fig = fig
        self._pos = pos
        self._horizontal = horizontal
        self._vertical = vertical
        self._anchor = anchor
        self._aspect = aspect
        self._xrefindex = 0
        self._yrefindex = 0
        self._locator = None

    def get_horizontal_sizes(self, renderer):
        return [s.get_size(renderer) for s in self.get_horizontal()]

    def get_vertical_sizes(self, renderer):
        return [s.get_size(renderer) for s in self.get_vertical()]

    @_api.deprecated("3.5")
    def get_vsize_hsize(self):
        vsize = Size.AddList(self.get_vertical())
        hsize = Size.AddList(self.get_horizontal())
        return vsize, hsize

    @staticmethod
    def _calc_k(l, total_size):

        rs_sum, as_sum = 0., 0.

        for _rs, _as in l:
            rs_sum += _rs
            as_sum += _as

        if rs_sum != 0.:
            k = (total_size - as_sum) / rs_sum
            return k
        else:
            return 0.

    @staticmethod
    def _calc_offsets(l, k):
        offsets = [0.]
        for _rs, _as in l:
            offsets.append(offsets[-1] + _rs*k + _as)
        return offsets

    def set_position(self, pos):
        """
        Set the position of the rectangle.

        Parameters
        ----------
        pos : tuple of 4 floats
            position of the rectangle that will be divided
        """
        self._pos = pos

    def get_position(self):
        """Return the position of the rectangle."""
        return self._pos

    def set_anchor(self, anchor):
        """
        Parameters
        ----------
        anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
            Either an (*x*, *y*) pair of relative coordinates (0 is left or
            bottom, 1 is right or top), 'C' (center), or a cardinal direction
            ('SW', southwest, is bottom left, etc.).

        See Also
        --------
        .Axes.set_anchor
        """
        if len(anchor) != 2:
            _api.check_in_list(mtransforms.Bbox.coefs, anchor=anchor)
        self._anchor = anchor

    def get_anchor(self):
        """Return the anchor."""
        return self._anchor

    def set_horizontal(self, h):
        """
        Parameters
        ----------
        h : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
            sizes for horizontal division
        """
        self._horizontal = h

    def get_horizontal(self):
        """Return horizontal sizes."""
        return self._horizontal

    def set_vertical(self, v):
        """
        Parameters
        ----------
        v : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
            sizes for vertical division
        """
        self._vertical = v

    def get_vertical(self):
        """Return vertical sizes."""
        return self._vertical

    def set_aspect(self, aspect=False):
        """
        Parameters
        ----------
        aspect : bool
        """
        self._aspect = aspect

    def get_aspect(self):
        """Return aspect."""
        return self._aspect

    def set_locator(self, _locator):
        self._locator = _locator

    def get_locator(self):
        return self._locator

    def get_position_runtime(self, ax, renderer):
        if self._locator is None:
            return self.get_position()
        else:
            return self._locator(ax, renderer).bounds

    def locate(self, nx, ny, nx1=None, ny1=None, axes=None, renderer=None):
        """
        Parameters
        ----------
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        ny, ny1 : int
            Same as *nx* and *nx1*, but for row positions.
        axes
        renderer
        """

        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)

        hsizes = self.get_horizontal_sizes(renderer)
        vsizes = self.get_vertical_sizes(renderer)
        k_h = self._calc_k(hsizes, fig_w * w)
        k_v = self._calc_k(vsizes, fig_h * h)

        if self.get_aspect():
            k = min(k_h, k_v)
            ox = self._calc_offsets(hsizes, k)
            oy = self._calc_offsets(vsizes, k)

            ww = (ox[-1] - ox[0]) / fig_w
            hh = (oy[-1] - oy[0]) / fig_h
            pb = mtransforms.Bbox.from_bounds(x, y, w, h)
            pb1 = mtransforms.Bbox.from_bounds(x, y, ww, hh)
            pb1_anchored = pb1.anchored(self.get_anchor(), pb)
            x0, y0 = pb1_anchored.x0, pb1_anchored.y0

        else:
            ox = self._calc_offsets(hsizes, k_h)
            oy = self._calc_offsets(vsizes, k_v)
            x0, y0 = x, y

        if nx1 is None:
            _api.warn_deprecated(
                "3.5", message="Support for passing nx1=None to mean nx+1 is "
                "deprecated since %(since)s; in a future version, nx1=None "
                "will mean 'up to the last cell'.")
            nx1 = nx + 1
        if ny1 is None:
            _api.warn_deprecated(
                "3.5", message="Support for passing ny1=None to mean ny+1 is "
                "deprecated since %(since)s; in a future version, ny1=None "
                "will mean 'up to the last cell'.")
            ny1 = ny + 1

        x1, w1 = x0 + ox[nx] / fig_w, (ox[nx1] - ox[nx]) / fig_w
        y1, h1 = y0 + oy[ny] / fig_h, (oy[ny1] - oy[ny]) / fig_h

        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)

    def new_locator(self, nx, ny, nx1=None, ny1=None):
        """
        Return a new `AxesLocator` for the specified cell.

        Parameters
        ----------
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        ny, ny1 : int
            Same as *nx* and *nx1*, but for row positions.
        """
        return AxesLocator(
            self, nx, ny,
            nx1 if nx1 is not None else nx + 1,
            ny1 if ny1 is not None else ny + 1)

    def append_size(self, position, size):
        if position == "left":
            self._horizontal.insert(0, size)
            self._xrefindex += 1
        elif position == "right":
            self._horizontal.append(size)
        elif position == "bottom":
            self._vertical.insert(0, size)
            self._yrefindex += 1
        elif position == "top":
            self._vertical.append(size)
        else:
            _api.check_in_list(["left", "right", "bottom", "top"],
                               position=position)

    def add_auto_adjustable_area(self, use_axes, pad=0.1, adjust_dirs=None):
        if adjust_dirs is None:
            adjust_dirs = ["left", "right", "bottom", "top"]
        from .axes_size import Padded, SizeFromFunc, GetExtentHelper
        for d in adjust_dirs:
            helper = GetExtentHelper(use_axes, d)
            size = SizeFromFunc(helper)
            padded_size = Padded(size, pad)  # pad in inch
            self.append_size(d, padded_size)


class AxesLocator:
    """
    A callable object which returns the position and size of a given
    AxesDivider cell.
    """

    def __init__(self, axes_divider, nx, ny, nx1=None, ny1=None):
        """
        Parameters
        ----------
        axes_divider : AxesDivider
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        ny, ny1 : int
            Same as *nx* and *nx1*, but for row positions.
        """
        self._axes_divider = axes_divider

        _xrefindex = axes_divider._xrefindex
        _yrefindex = axes_divider._yrefindex

        self._nx, self._ny = nx - _xrefindex, ny - _yrefindex

        if nx1 is None:
            _api.warn_deprecated(
                "3.5", message="Support for passing nx1=None to mean nx+1 is "
                "deprecated since %(since)s; in a future version, nx1=None "
                "will mean 'up to the last cell'.")
            nx1 = nx + 1
        if ny1 is None:
            _api.warn_deprecated(
                "3.5", message="Support for passing ny1=None to mean ny+1 is "
                "deprecated since %(since)s; in a future version, ny1=None "
                "will mean 'up to the last cell'.")
            ny1 = ny + 1

        self._nx1 = nx1 - _xrefindex
        self._ny1 = ny1 - _yrefindex

    def __call__(self, axes, renderer):

        _xrefindex = self._axes_divider._xrefindex
        _yrefindex = self._axes_divider._yrefindex

        return self._axes_divider.locate(self._nx + _xrefindex,
                                         self._ny + _yrefindex,
                                         self._nx1 + _xrefindex,
                                         self._ny1 + _yrefindex,
                                         axes,
                                         renderer)

    def get_subplotspec(self):
        if hasattr(self._axes_divider, "get_subplotspec"):
            return self._axes_divider.get_subplotspec()
        else:
            return None


class SubplotDivider(Divider):
    """
    The Divider class whose rectangle area is specified as a subplot geometry.
    """

    def __init__(self, fig, *args, horizontal=None, vertical=None,
                 aspect=None, anchor='C'):
        """
        Parameters
        ----------
        fig : `matplotlib.figure.Figure`

        *args : tuple (*nrows*, *ncols*, *index*) or int
            The array of subplots in the figure has dimensions ``(nrows,
            ncols)``, and *index* is the index of the subplot being created.
            *index* starts at 1 in the upper left corner and increases to the
            right.

            If *nrows*, *ncols*, and *index* are all single digit numbers, then
            *args* can be passed as a single 3-digit number (e.g. 234 for
            (2, 3, 4)).
        """
        self.figure = fig
        super().__init__(fig, [0, 0, 1, 1],
                         horizontal=horizontal or [], vertical=vertical or [],
                         aspect=aspect, anchor=anchor)
        self.set_subplotspec(SubplotSpec._from_subplot_args(fig, args))

    def get_position(self):
        """Return the bounds of the subplot box."""
        return self.get_subplotspec().get_position(self.figure).bounds

    @_api.deprecated("3.4")
    @property
    def figbox(self):
        return self.get_subplotspec().get_position(self.figure)

    @_api.deprecated("3.4")
    def update_params(self):
        pass

    @_api.deprecated(
        "3.4", alternative="get_subplotspec",
        addendum="(get_subplotspec returns a SubplotSpec instance.)")
    def get_geometry(self):
        """Get the subplot geometry, e.g., (2, 2, 3)."""
        rows, cols, num1, num2 = self.get_subplotspec().get_geometry()
        return rows, cols, num1 + 1  # for compatibility

    @_api.deprecated("3.4", alternative="set_subplotspec")
    def change_geometry(self, numrows, numcols, num):
        """Change subplot geometry, e.g., from (1, 1, 1) to (2, 2, 3)."""
        self._subplotspec = GridSpec(numrows, numcols)[num-1]
        self.update_params()
        self.set_position(self.figbox)

    def get_subplotspec(self):
        """Get the SubplotSpec instance."""
        return self._subplotspec

    def set_subplotspec(self, subplotspec):
        """Set the SubplotSpec instance."""
        self._subplotspec = subplotspec
        self.set_position(subplotspec.get_position(self.figure))


class AxesDivider(Divider):
    """
    Divider based on the pre-existing axes.
    """

    def __init__(self, axes, xref=None, yref=None):
        """
        Parameters
        ----------
        axes : :class:`~matplotlib.axes.Axes`
        xref
        yref
        """
        self._axes = axes
        if xref is None:
            self._xref = Size.AxesX(axes)
        else:
            self._xref = xref
        if yref is None:
            self._yref = Size.AxesY(axes)
        else:
            self._yref = yref

        super().__init__(fig=axes.get_figure(), pos=None,
                         horizontal=[self._xref], vertical=[self._yref],
                         aspect=None, anchor="C")

    def _get_new_axes(self, *, axes_class=None, **kwargs):
        axes = self._axes
        if axes_class is None:
            if isinstance(axes, SubplotBase):
                axes_class = axes._axes_class
            else:
                axes_class = type(axes)
        return axes_class(axes.get_figure(), axes.get_position(original=True),
                          **kwargs)

    def new_horizontal(self, size, pad=None, pack_start=False, **kwargs):
        """
        Add a new axes on the right (or left) side of the main axes.

        Parameters
        ----------
        size : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            The axes width.  float or str arguments are interpreted as
            ``axes_size.from_any(size, AxesX(<main_axes>))``.
        pad : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            Padding between the axes.  float or str arguments are interpreted
            as ``axes_size.from_any(size, AxesX(<main_axes>))``.  Defaults to
            :rc:`figure.subplot.wspace` times the main axes width.
        pack_start : bool
            If False, the new axes is appended at the end
            of the list, i.e., it became the right-most axes. If True, it is
            inserted at the start of the list, and becomes the left-most axes.
        **kwargs
            All extra keywords arguments are passed to the created axes.
            If *axes_class* is given, the new axes will be created as an
            instance of the given class. Otherwise, the same class of the
            main axes will be used.
        """
        if pad is None:
            pad = mpl.rcParams["figure.subplot.wspace"] * self._xref
        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad, fraction_ref=self._xref)
            if pack_start:
                self._horizontal.insert(0, pad)
                self._xrefindex += 1
            else:
                self._horizontal.append(pad)
        if not isinstance(size, Size._Base):
            size = Size.from_any(size, fraction_ref=self._xref)
        if pack_start:
            self._horizontal.insert(0, size)
            self._xrefindex += 1
            locator = self.new_locator(nx=0, ny=self._yrefindex)
        else:
            self._horizontal.append(size)
            locator = self.new_locator(
                nx=len(self._horizontal) - 1, ny=self._yrefindex)
        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)
        return ax

    def new_vertical(self, size, pad=None, pack_start=False, **kwargs):
        """
        Add a new axes on the top (or bottom) side of the main axes.

        Parameters
        ----------
        size : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            The axes height.  float or str arguments are interpreted as
            ``axes_size.from_any(size, AxesY(<main_axes>))``.
        pad : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            Padding between the axes.  float or str arguments are interpreted
            as ``axes_size.from_any(size, AxesY(<main_axes>))``.  Defaults to
            :rc:`figure.subplot.hspace` times the main axes height.
        pack_start : bool
            If False, the new axes is appended at the end
            of the list, i.e., it became the right-most axes. If True, it is
            inserted at the start of the list, and becomes the left-most axes.
        **kwargs
            All extra keywords arguments are passed to the created axes.
            If *axes_class* is given, the new axes will be created as an
            instance of the given class. Otherwise, the same class of the
            main axes will be used.
        """
        if pad is None:
            pad = mpl.rcParams["figure.subplot.hspace"] * self._yref
        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad, fraction_ref=self._yref)
            if pack_start:
                self._vertical.insert(0, pad)
                self._yrefindex += 1
            else:
                self._vertical.append(pad)
        if not isinstance(size, Size._Base):
            size = Size.from_any(size, fraction_ref=self._yref)
        if pack_start:
            self._vertical.insert(0, size)
            self._yrefindex += 1
            locator = self.new_locator(nx=self._xrefindex, ny=0)
        else:
            self._vertical.append(size)
            locator = self.new_locator(
                nx=self._xrefindex, ny=len(self._vertical)-1)
        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)
        return ax

    @_api.delete_parameter("3.5", "add_to_figure", alternative="ax.remove()")
    def append_axes(self, position, size, pad=None, add_to_figure=True,
                    **kwargs):
        """
        Create an axes at the given *position* with the same height
        (or width) of the main axes.

         *position*
           ["left"|"right"|"bottom"|"top"]

         *size* and *pad* should be axes_grid.axes_size compatible.
        """
        if position == "left":
            ax = self.new_horizontal(size, pad, pack_start=True, **kwargs)
        elif position == "right":
            ax = self.new_horizontal(size, pad, pack_start=False, **kwargs)
        elif position == "bottom":
            ax = self.new_vertical(size, pad, pack_start=True, **kwargs)
        elif position == "top":
            ax = self.new_vertical(size, pad, pack_start=False, **kwargs)
        else:
            _api.check_in_list(["left", "right", "bottom", "top"],
                               position=position)
        if add_to_figure:
            self._fig.add_axes(ax)
        return ax

    def get_aspect(self):
        if self._aspect is None:
            aspect = self._axes.get_aspect()
            if aspect == "auto":
                return False
            else:
                return True
        else:
            return self._aspect

    def get_position(self):
        if self._pos is None:
            bbox = self._axes.get_position(original=True)
            return bbox.bounds
        else:
            return self._pos

    def get_anchor(self):
        if self._anchor is None:
            return self._axes.get_anchor()
        else:
            return self._anchor

    def get_subplotspec(self):
        if hasattr(self._axes, "get_subplotspec"):
            return self._axes.get_subplotspec()
        else:
            return None


# Helper for HBoxDivider/VBoxDivider.
# The variable names are written for a horizontal layout, but the calculations
# work identically for vertical layouts (and likewise for the helpers below).
def _determine_karray(summed_widths, equal_heights, total_width, max_height):
    n = len(equal_heights)
    eq_rs, eq_as = np.asarray(equal_heights).T
    sm_rs, sm_as = np.asarray(summed_widths).T
    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)
    np.fill_diagonal(A[:n, :n], eq_rs)
    A[:n, -1] = -1
    A[-1, :-1] = sm_rs
    B[:n] = -eq_as
    B[-1] = total_width - sum(sm_as)
    # A @ K = B: This solves for {k_0, ..., k_{N-1}, H} so that
    #   eq_r_i * k_i + eq_a_i = H for all i: all axes have the same height
    #   sum(sm_r_i * k_i + sm_a_i) = total_summed_width: fixed total width
    # (foo_r_i * k_i + foo_a_i will end up being the size of foo.)
    karray_and_height = np.linalg.solve(A, B)
    karray = karray_and_height[:-1]
    height = karray_and_height[-1]
    if height > max_height:  # Additionally, upper-bound the height.
        karray = (max_height - eq_as) / eq_rs
    return karray


# Helper for HBoxDivider/VBoxDivider (see above re: variable naming).
def _calc_offsets(summed_sizes, karray):
    offsets = [0.]
    for (r, a), k in zip(summed_sizes, karray):
        offsets.append(offsets[-1] + r*k + a)
    return offsets


# Helper for HBoxDivider/VBoxDivider (see above re: variable naming).
def _locate(x, y, w, h, summed_widths, equal_heights, fig_w, fig_h, anchor):
    karray = _determine_karray(
        summed_widths, equal_heights,
        total_width=fig_w * w, max_height=fig_h * h)
    ox = _calc_offsets(summed_widths, karray)

    ww = (ox[-1] - ox[0]) / fig_w
    h0_r, h0_a = equal_heights[0]
    hh = (karray[0]*h0_r + h0_a) / fig_h
    pb = mtransforms.Bbox.from_bounds(x, y, w, h)
    pb1 = mtransforms.Bbox.from_bounds(x, y, ww, hh)
    pb1_anchored = pb1.anchored(anchor, pb)
    x0, y0 = pb1_anchored.x0, pb1_anchored.y0

    return x0, y0, ox, hh


class HBoxDivider(SubplotDivider):
    """
    A `SubplotDivider` for laying out axes horizontally, while ensuring that
    they have equal heights.

    Examples
    --------
    .. plot:: gallery/axes_grid1/demo_axes_hbox_divider.py
    """

    def new_locator(self, nx, nx1=None):
        """
        Create a new `AxesLocator` for the specified cell.

        Parameters
        ----------
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        """
        return AxesLocator(self, nx, 0, nx1 if nx1 is not None else nx + 1, 1)

    def locate(self, nx, ny, nx1=None, ny1=None, axes=None, renderer=None):
        # docstring inherited
        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)
        summed_ws = self.get_horizontal_sizes(renderer)
        equal_hs = self.get_vertical_sizes(renderer)
        x0, y0, ox, hh = _locate(
            x, y, w, h, summed_ws, equal_hs, fig_w, fig_h, self.get_anchor())
        if nx1 is None:
            _api.warn_deprecated(
                "3.5", message="Support for passing nx1=None to mean nx+1 is "
                "deprecated since %(since)s; in a future version, nx1=None "
                "will mean 'up to the last cell'.")
            nx1 = nx + 1
        x1, w1 = x0 + ox[nx] / fig_w, (ox[nx1] - ox[nx]) / fig_w
        y1, h1 = y0, hh
        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)


class VBoxDivider(SubplotDivider):
    """
    A `SubplotDivider` for laying out axes vertically, while ensuring that they
    have equal widths.
    """

    def new_locator(self, ny, ny1=None):
        """
        Create a new `AxesLocator` for the specified cell.

        Parameters
        ----------
        ny, ny1 : int
            Integers specifying the row-position of the
            cell. When *ny1* is None, a single *ny*-th row is
            specified. Otherwise location of rows spanning between *ny*
            to *ny1* (but excluding *ny1*-th row) is specified.
        """
        return AxesLocator(self, 0, ny, 1, ny1 if ny1 is not None else ny + 1)

    def locate(self, nx, ny, nx1=None, ny1=None, axes=None, renderer=None):
        # docstring inherited
        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)
        summed_hs = self.get_vertical_sizes(renderer)
        equal_ws = self.get_horizontal_sizes(renderer)
        y0, x0, oy, ww = _locate(
            y, x, h, w, summed_hs, equal_ws, fig_h, fig_w, self.get_anchor())
        if ny1 is None:
            _api.warn_deprecated(
                "3.5", message="Support for passing ny1=None to mean ny+1 is "
                "deprecated since %(since)s; in a future version, ny1=None "
                "will mean 'up to the last cell'.")
            ny1 = ny + 1
        x1, w1 = x0, ww
        y1, h1 = y0 + oy[ny] / fig_h, (oy[ny1] - oy[ny]) / fig_h
        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)


def make_axes_locatable(axes):
    divider = AxesDivider(axes)
    locator = divider.new_locator(nx=0, ny=0)
    axes.set_axes_locator(locator)

    return divider


def make_axes_area_auto_adjustable(ax,
                                   use_axes=None, pad=0.1,
                                   adjust_dirs=None):
    if adjust_dirs is None:
        adjust_dirs = ["left", "right", "bottom", "top"]
    divider = make_axes_locatable(ax)

    if use_axes is None:
        use_axes = ax

    divider.add_auto_adjustable_area(use_axes=use_axes, pad=pad,
                                     adjust_dirs=adjust_dirs)
