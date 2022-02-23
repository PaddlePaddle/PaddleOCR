import numpy as np

from matplotlib import docstring
from matplotlib.contour import ContourSet
from matplotlib.tri.triangulation import Triangulation


@docstring.dedent_interpd
class TriContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions for
    a triangular grid.

    This class is typically not instantiated directly by the user but by
    `~.Axes.tricontour` and `~.Axes.tricontourf`.

    %(contour_set_attributes)s
    """
    def __init__(self, ax, *args, **kwargs):
        """
        Draw triangular grid contour lines or filled regions,
        depending on whether keyword arg 'filled' is False
        (default) or True.

        The first argument of the initializer must be an axes
        object.  The remaining arguments and keyword arguments
        are described in the docstring of `~.Axes.tricontour`.
        """
        super().__init__(ax, *args, **kwargs)

    def _process_args(self, *args, **kwargs):
        """
        Process args and kwargs.
        """
        if isinstance(args[0], TriContourSet):
            C = args[0]._contour_generator
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
        else:
            from matplotlib import _tri
            tri, z = self._contour_args(args, kwargs)
            C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
            self._mins = [tri.x.min(), tri.y.min()]
            self._maxs = [tri.x.max(), tri.y.max()]

        self._contour_generator = C
        return kwargs

    def _contour_args(self, args, kwargs):
        if self.filled:
            fn = 'contourf'
        else:
            fn = 'contour'
        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args,
                                                                   **kwargs)
        z = np.ma.asarray(args[0])
        if z.shape != tri.x.shape:
            raise ValueError('z array must have same length as triangulation x'
                             ' and y arrays')

        # z values must be finite, only need to check points that are included
        # in the triangulation.
        z_check = z[np.unique(tri.get_masked_triangles())]
        if np.ma.is_masked(z_check):
            raise ValueError('z must not contain masked points within the '
                             'triangulation')
        if not np.isfinite(z_check).all():
            raise ValueError('z array must not contain non-finite values '
                             'within the triangulation')

        z = np.ma.masked_invalid(z, copy=False)
        self.zmax = float(z_check.max())
        self.zmin = float(z_check.min())
        if self.logscale and self.zmin <= 0:
            raise ValueError('Cannot %s log of negative values.' % fn)
        self._process_contour_level_args(args[1:])
        return (tri, z)


docstring.interpd.update(_tricontour_doc="""
Draw contour %(type)s on an unstructured triangular grid.

The triangulation can be specified in one of two ways; either ::

    %(func)s(triangulation, ...)

where *triangulation* is a `.Triangulation` object, or ::

    %(func)s(x, y, ...)
    %(func)s(x, y, triangles, ...)
    %(func)s(x, y, triangles=triangles, ...)
    %(func)s(x, y, mask=mask, ...)
    %(func)s(x, y, triangles, mask=mask, ...)

in which case a `.Triangulation` object will be created.  See that class'
docstring for an explanation of these cases.

The remaining arguments may be::

    %(func)s(..., Z)

where *Z* is the array of values to contour, one per point in the
triangulation.  The level values are chosen automatically.

::

    %(func)s(..., Z, levels)

contour up to *levels+1* automatically chosen contour levels (*levels*
intervals).

::

    %(func)s(..., Z, levels)

draw contour %(type)s at the values specified in sequence *levels*, which must
be in increasing order.

::

    %(func)s(Z, **kwargs)

Use keyword arguments to control colors, linewidth, origin, cmap ... see below
for more details.

Parameters
----------
triangulation : `.Triangulation`, optional
    The unstructured triangular grid.

    If specified, then *x*, *y*, *triangles*, and *mask* are not accepted.

x, y : array-like, optional
    The coordinates of the values in *Z*.

triangles : (ntri, 3) array-like of int, optional
    For each triangle, the indices of the three points that make up the
    triangle, ordered in an anticlockwise manner.  If not specified, the
    Delaunay triangulation is calculated.

mask : (ntri,) array-like of bool, optional
    Which triangles are masked out.

Z : 2D array-like
    The height values over which the contour is drawn.

levels : int or array-like, optional
    Determines the number and positions of the contour lines / regions.

    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries to
    automatically choose no more than *n+1* "nice" contour levels between
    *vmin* and *vmax*.

    If array-like, draw contour lines at the specified levels.  The values must
    be in increasing order.

Returns
-------
`~matplotlib.tri.TriContourSet`

Other Parameters
----------------
colors : color string or sequence of colors, optional
    The colors of the levels, i.e., the contour %(type)s.

    The sequence is cycled for the levels in ascending order. If the sequence
    is shorter than the number of levels, it's repeated.

    As a shortcut, single color strings may be used in place of one-element
    lists, i.e. ``'red'`` instead of ``['red']`` to color all levels with the
    same color. This shortcut does only work for color strings, not for other
    ways of specifying colors.

    By default (value *None*), the colormap specified by *cmap* will be used.

alpha : float, default: 1
    The alpha blending value, between 0 (transparent) and 1 (opaque).

cmap : str or `.Colormap`, default: :rc:`image.cmap`
    A `.Colormap` instance or registered colormap name. The colormap maps the
    level values to colors.

    If both *colors* and *cmap* are given, an error is raised.

norm : `~matplotlib.colors.Normalize`, optional
    If a colormap is used, the `.Normalize` instance scales the level values to
    the canonical colormap range [0, 1] for mapping to colors. If not given,
    the default linear scaling is used.

vmin, vmax : float, optional
    If not *None*, either or both of these values will be supplied to
    the `.Normalize` instance, overriding the default color scaling
    based on *levels*.

origin : {*None*, 'upper', 'lower', 'image'}, default: None
    Determines the orientation and exact position of *Z* by specifying the
    position of ``Z[0, 0]``.  This is only relevant, if *X*, *Y* are not given.

    - *None*: ``Z[0, 0]`` is at X=0, Y=0 in the lower left corner.
    - 'lower': ``Z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
    - 'upper': ``Z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.
    - 'image': Use the value from :rc:`image.origin`.

extent : (x0, x1, y0, y1), optional
    If *origin* is not *None*, then *extent* is interpreted as in `.imshow`: it
    gives the outer pixel boundaries. In this case, the position of Z[0, 0] is
    the center of the pixel, not a corner. If *origin* is *None*, then
    (*x0*, *y0*) is the position of Z[0, 0], and (*x1*, *y1*) is the position
    of Z[-1, -1].

    This argument is ignored if *X* and *Y* are specified in the call to
    contour.

locator : ticker.Locator subclass, optional
    The locator is used to determine the contour levels if they are not given
    explicitly via *levels*.
    Defaults to `~.ticker.MaxNLocator`.

extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
    Determines the ``%(func)s``-coloring of values that are outside the
    *levels* range.

    If 'neither', values outside the *levels* range are not colored.  If 'min',
    'max' or 'both', color the values below, above or below and above the
    *levels* range.

    Values below ``min(levels)`` and above ``max(levels)`` are mapped to the
    under/over values of the `.Colormap`. Note that most colormaps do not have
    dedicated colors for these by default, so that the over and under values
    are the edge values of the colormap.  You may want to set these values
    explicitly using `.Colormap.set_under` and `.Colormap.set_over`.

    .. note::

        An existing `.TriContourSet` does not get notified if properties of its
        colormap are changed. Therefore, an explicit call to
        `.ContourSet.changed()` is needed after modifying the colormap. The
        explicit call can be left out, if a colorbar is assigned to the
        `.TriContourSet` because it internally calls `.ContourSet.changed()`.

xunits, yunits : registered units, optional
    Override axis units by specifying an instance of a
    :class:`matplotlib.units.ConversionInterface`.

antialiased : bool, optional
    Enable antialiasing, overriding the defaults.  For
    filled contours, the default is *True*.  For line contours,
    it is taken from :rc:`lines.antialiased`.""")


@docstring.Substitution(func='tricontour', type='lines')
@docstring.dedent_interpd
def tricontour(ax, *args, **kwargs):
    """
    %(_tricontour_doc)s

    linewidths : float or array-like, default: :rc:`contour.linewidth`
        The line width of the contour lines.

        If a number, all levels will be plotted with this linewidth.

        If a sequence, the levels in ascending order will be plotted with
        the linewidths in the order specified.

        If None, this falls back to :rc:`lines.linewidth`.

    linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
        If *linestyles* is *None*, the default is 'solid' unless the lines are
        monochrome.  In that case, negative contours will take their linestyle
        from :rc:`contour.negative_linestyle` setting.

        *linestyles* can also be an iterable of the above strings specifying a
        set of linestyles to be used. If this iterable is shorter than the
        number of contour levels it will be repeated as necessary.
    """
    kwargs['filled'] = False
    return TriContourSet(ax, *args, **kwargs)


@docstring.Substitution(func='tricontourf', type='regions')
@docstring.dedent_interpd
def tricontourf(ax, *args, **kwargs):
    """
    %(_tricontour_doc)s

    hatches : list[str], optional
        A list of cross hatch patterns to use on the filled areas.
        If None, no hatching will be added to the contour.
        Hatching is supported in the PostScript, PDF, SVG and Agg
        backends only.

    Notes
    -----
    `.tricontourf` fills intervals that are closed at the top; that is, for
    boundaries *z1* and *z2*, the filled region is::

        z1 < Z <= z2

    except for the lowest interval, which is closed on both sides (i.e. it
    includes the lowest value).
    """
    kwargs['filled'] = True
    return TriContourSet(ax, *args, **kwargs)
