"""Affine transforms, both in general and specific, named transforms."""

from math import sin, cos, tan, pi
from shapely.errors import GeometryTypeError

__all__ = ['affine_transform', 'rotate', 'scale', 'skew', 'translate']


def affine_transform(geom, matrix):
    r"""Returns a transformed geometry using an affine transformation matrix.

    The coefficient matrix is provided as a list or tuple with 6 or 12 items
    for 2D or 3D transformations, respectively.

    For 2D affine transformations, the 6 parameter matrix is::

        [a, b, d, e, xoff, yoff]

    which represents the augmented matrix::

        [x']   / a  b xoff \ [x]
        [y'] = | d  e yoff | [y]
        [1 ]   \ 0  0   1  / [1]

    or the equations for the transformed coordinates::

        x' = a * x + b * y + xoff
        y' = d * x + e * y + yoff

    For 3D affine transformations, the 12 parameter matrix is::

        [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]

    which represents the augmented matrix::

        [x']   / a  b  c xoff \ [x]
        [y'] = | d  e  f yoff | [y]
        [z']   | g  h  i zoff | [z]
        [1 ]   \ 0  0  0   1  / [1]

    or the equations for the transformed coordinates::

        x' = a * x + b * y + c * z + xoff
        y' = d * x + e * y + f * z + yoff
        z' = g * x + h * y + i * z + zoff
    """
    if geom.is_empty:
        return geom
    if len(matrix) == 6:
        ndim = 2
        a, b, d, e, xoff, yoff = matrix
        if geom.has_z:
            ndim = 3
            i = 1.0
            c = f = g = h = zoff = 0.0
            matrix = a, b, c, d, e, f, g, h, i, xoff, yoff, zoff
    elif len(matrix) == 12:
        ndim = 3
        a, b, c, d, e, f, g, h, i, xoff, yoff, zoff = matrix
        if not geom.has_z:
            ndim = 2
            matrix = a, b, d, e, xoff, yoff
    else:
        raise ValueError("'matrix' expects either 6 or 12 coefficients")

    def affine_pts(pts):
        """Internal function to yield affine transform of coordinate tuples"""
        if ndim == 2:
            for x, y in pts:
                xp = a * x + b * y + xoff
                yp = d * x + e * y + yoff
                yield (xp, yp)
        elif ndim == 3:
            for x, y, z in pts:
                xp = a * x + b * y + c * z + xoff
                yp = d * x + e * y + f * z + yoff
                zp = g * x + h * y + i * z + zoff
                yield (xp, yp, zp)

    # Process coordinates from each supported geometry type
    if geom.type in ('Point', 'LineString', 'LinearRing'):
        return type(geom)(list(affine_pts(geom.coords)))
    elif geom.type == 'Polygon':
        ring = geom.exterior
        shell = type(ring)(list(affine_pts(ring.coords)))
        holes = list(geom.interiors)
        for pos, ring in enumerate(holes):
            holes[pos] = type(ring)(list(affine_pts(ring.coords)))
        return type(geom)(shell, holes)
    elif geom.type.startswith('Multi') or geom.type == 'GeometryCollection':
        # Recursive call
        # TODO: fix GeometryCollection constructor
        return type(geom)([affine_transform(part, matrix)
                           for part in geom.geoms])
    else:
        raise GeometryTypeError('Type %r not recognized' % geom.type)


def interpret_origin(geom, origin, ndim):
    """Returns interpreted coordinate tuple for origin parameter.

    This is a helper function for other transform functions.

    The point of origin can be a keyword 'center' for the 2D bounding box
    center, 'centroid' for the geometry's 2D centroid, a Point object or a
    coordinate tuple (x0, y0, z0).
    """
    # get coordinate tuple from 'origin' from keyword or Point type
    if origin == 'center':
        # bounding box center
        minx, miny, maxx, maxy = geom.bounds
        origin = ((maxx + minx)/2.0, (maxy + miny)/2.0)
    elif origin == 'centroid':
        origin = geom.centroid.coords[0]
    elif isinstance(origin, str):
        raise ValueError("'origin' keyword %r is not recognized" % origin)
    elif hasattr(origin, 'type') and origin.type == 'Point':
        origin = origin.coords[0]

    # origin should now be tuple-like
    if len(origin) not in (2, 3):
        raise ValueError("Expected number of items in 'origin' to be "
                         "either 2 or 3")
    if ndim == 2:
        return origin[0:2]
    else:  # 3D coordinate
        if len(origin) == 2:
            return origin + (0.0,)
        else:
            return origin


def rotate(geom, angle, origin='center', use_radians=False):
    r"""Returns a rotated geometry on a 2D plane.

    The angle of rotation can be specified in either degrees (default) or
    radians by setting ``use_radians=True``. Positive angles are
    counter-clockwise and negative are clockwise rotations.

    The point of origin can be a keyword 'center' for the bounding box
    center (default), 'centroid' for the geometry's centroid, a Point object
    or a coordinate tuple (x0, y0).

    The affine transformation matrix for 2D rotation is:

      / cos(r) -sin(r) xoff \ 
      | sin(r)  cos(r) yoff |
      \   0       0      1  /

    where the offsets are calculated from the origin Point(x0, y0):

        xoff = x0 - x0 * cos(r) + y0 * sin(r)
        yoff = y0 - x0 * sin(r) - y0 * cos(r)
    """
    if geom.is_empty:
        return geom
    if not use_radians:  # convert from degrees
        angle = angle * pi/180.0
    cosp = cos(angle)
    sinp = sin(angle)
    if abs(cosp) < 2.5e-16:
        cosp = 0.0
    if abs(sinp) < 2.5e-16:
        sinp = 0.0
    x0, y0 = interpret_origin(geom, origin, 2)

    matrix = (cosp, -sinp, 0.0,
              sinp,  cosp, 0.0,
              0.0,    0.0, 1.0,
              x0 - x0 * cosp + y0 * sinp, y0 - x0 * sinp - y0 * cosp, 0.0)
    return affine_transform(geom, matrix)


def scale(geom, xfact=1.0, yfact=1.0, zfact=1.0, origin='center'):
    r"""Returns a scaled geometry, scaled by factors along each dimension.

    The point of origin can be a keyword 'center' for the 2D bounding box
    center (default), 'centroid' for the geometry's 2D centroid, a Point
    object or a coordinate tuple (x0, y0, z0).

    Negative scale factors will mirror or reflect coordinates.

    The general 3D affine transformation matrix for scaling is:

        / xfact  0    0   xoff \ 
        |   0  yfact  0   yoff |
        |   0    0  zfact zoff |
        \   0    0    0     1  /

    where the offsets are calculated from the origin Point(x0, y0, z0):

        xoff = x0 - x0 * xfact
        yoff = y0 - y0 * yfact
        zoff = z0 - z0 * zfact
    """
    if geom.is_empty:
        return geom
    x0, y0, z0 = interpret_origin(geom, origin, 3)

    matrix = (xfact, 0.0, 0.0,
              0.0, yfact, 0.0,
              0.0, 0.0, zfact,
              x0 - x0 * xfact, y0 - y0 * yfact, z0 - z0 * zfact)
    return affine_transform(geom, matrix)


def skew(geom, xs=0.0, ys=0.0, origin='center', use_radians=False):
    r"""Returns a skewed geometry, sheared by angles along x and y dimensions.

    The shear angle can be specified in either degrees (default) or radians
    by setting ``use_radians=True``.

    The point of origin can be a keyword 'center' for the bounding box
    center (default), 'centroid' for the geometry's centroid, a Point object
    or a coordinate tuple (x0, y0).

    The general 2D affine transformation matrix for skewing is:

        /   1    tan(xs) xoff \ 
        | tan(ys)  1     yoff |
        \   0      0       1  /

    where the offsets are calculated from the origin Point(x0, y0):

        xoff = -y0 * tan(xs)
        yoff = -x0 * tan(ys)
    """
    if geom.is_empty:
        return geom
    if not use_radians:  # convert from degrees
        xs = xs * pi/180.0
        ys = ys * pi/180.0
    tanx = tan(xs)
    tany = tan(ys)
    if abs(tanx) < 2.5e-16:
        tanx = 0.0
    if abs(tany) < 2.5e-16:
        tany = 0.0
    x0, y0 = interpret_origin(geom, origin, 2)

    matrix = (1.0, tanx, 0.0,
              tany, 1.0, 0.0,
              0.0,  0.0, 1.0,
              -y0 * tanx, -x0 * tany, 0.0)
    return affine_transform(geom, matrix)


def translate(geom, xoff=0.0, yoff=0.0, zoff=0.0):
    r"""Returns a translated geometry shifted by offsets along each dimension.

    The general 3D affine transformation matrix for translation is:

        / 1  0  0 xoff \ 
        | 0  1  0 yoff |
        | 0  0  1 zoff |
        \ 0  0  0   1  /
    """
    if geom.is_empty:
        return geom
    matrix = (1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0,
              xoff, yoff, zoff)
    return affine_transform(geom, matrix)
