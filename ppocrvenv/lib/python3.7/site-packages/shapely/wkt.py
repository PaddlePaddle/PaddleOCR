"""Load/dump geometries using the well-known text (WKT) format

Also provides pickle-like convenience functions.
"""

from shapely import geos


def loads(data):
    """
    Load a geometry from a WKT string.

    Parameters
    ----------
    data : str
        A WKT string

    Returns
    -------
    Shapely geometry object
    """
    return geos.WKTReader(geos.lgeos).read(data)


def load(fp):
    """
    Load a geometry from an open file.

    Parameters
    ----------
    fp :
        A file-like object which implements a `read` method.

    Returns
    -------
    Shapely geometry object
    """
    data = fp.read()
    return loads(data)


def dumps(ob, trim=False, **kw):
    """
    Dump a WKT representation of a geometry to a string.

    Parameters
    ----------
    ob :
        A geometry object of any type to be dumped to WKT.
    trim : bool, default False
        Remove excess decimals from the WKT.
    rounding_precision : int
        Round output to the specified number of digits.
        Default behavior returns full precision.
    output_dimension : int, default 3
        Force removal of dimensions above the one specified.

    Returns
    -------
    input geometry as WKT string
    """
    return geos.WKTWriter(geos.lgeos, trim=trim, **kw).write(ob)


def dump(ob, fp, **settings):
    """
    Dump a geometry to an open file.

    Parameters
    ----------
    ob :
        A geometry object of any type to be dumped to WKT.
    fp :
        A file-like object which implements a `write` method.
    trim : bool, default False
        Remove excess decimals from the WKT.
    rounding_precision : int
        Round output to the specified number of digits.
        Default behavior returns full precision.
    output_dimension : int, default 3
        Force removal of dimensions above the one specified.

    Returns
    -------
    None
    """
    fp.write(dumps(ob, **settings))
