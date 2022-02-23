"""Load/dump geometries using the well-known binary (WKB) format.

Also provides pickle-like convenience functions.
"""

from shapely.geos import WKBReader, WKBWriter, lgeos
from shapely.geometry.base import geom_factory


def loads(data, hex=False):
    """Load a geometry from a WKB byte string, or hex-encoded string if
    ``hex=True``.

    Raises
    ------
    WKBReadingError, UnicodeDecodeError
        If ``data`` contains an invalid geometry.
    """
    reader = WKBReader(lgeos)
    if hex:
        return reader.read_hex(data)
    else:
        return reader.read(data)


def load(fp, hex=False):
    """Load a geometry from an open file.

    Raises
    ------
    WKBReadingError, UnicodeDecodeError
        If the given file contains an invalid geometry.
    """
    data = fp.read()
    return loads(data, hex=hex)


def dumps(ob, hex=False, srid=None, **kw):
    """Dump a WKB representation of a geometry to a byte string, or a
    hex-encoded string if ``hex=True``.
    
    Parameters
    ----------
    ob : geometry
        The geometry to export to well-known binary (WKB) representation
    hex : bool
        If true, export the WKB as a hexadecimal string. The default is to
        return a binary string/bytes object.
    srid : int
        Spatial reference system ID to include in the output. The default value
        means no SRID is included.
    **kw : kwargs
        See available keyword output settings in ``shapely.geos.WKBWriter``.
    """
    if srid is not None:
        # clone the object and set the SRID before dumping
        geom = lgeos.GEOSGeom_clone(ob._geom)
        lgeos.GEOSSetSRID(geom, srid)
        ob = geom_factory(geom)
        kw["include_srid"] = True
    writer = WKBWriter(lgeos, **kw)
    if hex:
        return writer.write_hex(ob)
    else:
        return writer.write(ob)


def dump(ob, fp, hex=False, **kw):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob, hex=hex, **kw))
