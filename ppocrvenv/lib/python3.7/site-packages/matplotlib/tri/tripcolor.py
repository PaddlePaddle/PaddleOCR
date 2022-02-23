import numpy as np

from matplotlib import _api
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.colors import Normalize
from matplotlib.tri.triangulation import Triangulation


def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None,
              vmax=None, shading='flat', facecolors=None, **kwargs):
    """
    Create a pseudocolor plot of an unstructured triangular grid.

    The triangulation can be specified in one of two ways; either::

      tripcolor(triangulation, ...)

    where triangulation is a `.Triangulation` object, or

    ::

      tripcolor(x, y, ...)
      tripcolor(x, y, triangles, ...)
      tripcolor(x, y, triangles=triangles, ...)
      tripcolor(x, y, mask=mask, ...)
      tripcolor(x, y, triangles, mask=mask, ...)

    in which case a Triangulation object will be created.  See `.Triangulation`
    for a explanation of these possibilities.

    The next argument must be *C*, the array of color values, either
    one per point in the triangulation if color values are defined at
    points, or one per triangle in the triangulation if color values
    are defined at triangles. If there are the same number of points
    and triangles in the triangulation it is assumed that color
    values are defined at points; to force the use of color values at
    triangles use the kwarg ``facecolors=C`` instead of just ``C``.

    *shading* may be 'flat' (the default) or 'gouraud'. If *shading*
    is 'flat' and C values are defined at points, the color values
    used for each triangle are from the mean C of the triangle's
    three points. If *shading* is 'gouraud' then color values must be
    defined at points.

    The remaining kwargs are the same as for `~.Axes.pcolor`.
    """
    _api.check_in_list(['flat', 'gouraud'], shading=shading)

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)

    # C is the colors array defined at either points or faces (i.e. triangles).
    # If facecolors is None, C are defined at points.
    # If facecolors is not None, C are defined at faces.
    if facecolors is not None:
        C = facecolors
    else:
        C = np.asarray(args[0])

    # If there are a different number of points and triangles in the
    # triangulation, can omit facecolors kwarg as it is obvious from
    # length of C whether it refers to points or faces.
    # Do not do this for gouraud shading.
    if (facecolors is None and len(C) == len(tri.triangles) and
            len(C) != len(tri.x) and shading != 'gouraud'):
        facecolors = C

    # Check length of C is OK.
    if ((facecolors is None and len(C) != len(tri.x)) or
            (facecolors is not None and len(C) != len(tri.triangles))):
        raise ValueError('Length of color values array must be the same '
                         'as either the number of triangulation points '
                         'or triangles')

    # Handling of linewidths, shading, edgecolors and antialiased as
    # in Axes.pcolor
    linewidths = (0.25,)
    if 'linewidth' in kwargs:
        kwargs['linewidths'] = kwargs.pop('linewidth')
    kwargs.setdefault('linewidths', linewidths)

    edgecolors = 'none'
    if 'edgecolor' in kwargs:
        kwargs['edgecolors'] = kwargs.pop('edgecolor')
    ec = kwargs.setdefault('edgecolors', edgecolors)

    if 'antialiased' in kwargs:
        kwargs['antialiaseds'] = kwargs.pop('antialiased')
    if 'antialiaseds' not in kwargs and ec.lower() == "none":
        kwargs['antialiaseds'] = False

    if shading == 'gouraud':
        if facecolors is not None:
            raise ValueError('Gouraud shading does not support the use '
                             'of facecolors kwarg')
        if len(C) != len(tri.x):
            raise ValueError('For gouraud shading, the length of color '
                             'values array must be the same as the '
                             'number of triangulation points')
        collection = TriMesh(tri, **kwargs)
    else:
        # Vertices of triangles.
        maskedTris = tri.get_masked_triangles()
        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)

        # Color values.
        if facecolors is None:
            # One color per triangle, the mean of the 3 vertex color values.
            C = C[maskedTris].mean(axis=1)
        elif tri.mask is not None:
            # Remove color values of masked triangles.
            C = C[~tri.mask]

        collection = PolyCollection(verts, **kwargs)

    collection.set_alpha(alpha)
    collection.set_array(C)
    _api.check_isinstance((Normalize, None), norm=norm)
    collection.set_cmap(cmap)
    collection.set_norm(norm)
    collection._scale_norm(norm, vmin, vmax)
    ax.grid(False)

    minx = tri.x.min()
    maxx = tri.x.max()
    miny = tri.y.min()
    maxy = tri.y.max()
    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(collection)
    return collection
