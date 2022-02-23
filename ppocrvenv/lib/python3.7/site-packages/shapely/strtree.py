"""
strtree
=======

Index geometry objects for efficient lookup of nearby or
nearest neighbors. Home of the `STRtree` class which is
an interface to the query-only GEOS R-tree packed using
the Sort-Tile-Recursive algorithm [1]_.

.. autoclass:: STRtree
    :members:

References
----------
  .. [1]  Leutenegger, Scott & Lopez, Mario & Edgington, Jeffrey. (1997).
     "STR: A Simple and Efficient Algorithm for R-Tree Packing." Proc.
     VLDB Conf. 497-506. 10.1109/ICDE.1997.582015.
     https://www.cs.odu.edu/~mln/ltrs-pdfs/icase-1997-14.pdf
"""

import ctypes
import logging
from typing import Any, ItemsView, Iterable, Iterator, Optional, Sequence, Tuple, Union
import sys

from shapely.geometry.base import BaseGeometry
from shapely.geos import lgeos

log = logging.getLogger(__name__)


class STRtree:
    """An STR-packed R-tree spatial index.

    An index is initialized from a sequence of geometry objects and
    optionally an sequence of items. The items, if provided, are stored
    in nodes of the tree. If items are not provided, the indices of the
    geometry sequence will be used instead.

    Stored items and corresponding geometry objects can be spatially
    queried using another geometric object.

    The tree is immutable and query-only, meaning that once created
    nodes cannot be added or removed.

    Parameters
    ----------
    geoms : sequence
        A sequence of geometry objects.
    items : sequence, optional
        A sequence of objects which typically serve as identifiers in an
        application. This sequence must have the same length as geoms.

    Attributes
    ----------
    node_capacity : int
        The maximum number of items per node. Default: 10.

    Examples
    --------
    Creating an index of polygons:

    >>> from shapely.strtree import STRtree
    >>> from shapely.geometry import Polygon
    >>>
    >>> polys = [Polygon(((0, 0), (1, 0), (1, 1))),
    ...          Polygon(((0, 1), (0, 0), (1, 0))),
    ...          Polygon(((100, 100), (101, 100), (101, 101)))]
    >>> tree = STRtree(polys)
    >>> query_geom = Polygon(((-1, -1), (2, 0), (2, 2), (-1, 2)))
    >>> result = tree.query(query_geom)
    >>> polys[0] in result
    True
    >>> polys[1] in result
    True
    >>> polys[2] in result
    False

    Notes
    -----
    The class maintains a reverse mapping of items to geometries. These
    items must therefore be hashable. The tree is filled using the
    Sort-Tile-Recursive [1]_ algorithm.

    References
    ----------
    .. [1] Leutenegger, Scott T.; Edgington, Jeffrey M.; Lopez, Mario A.
       (February 1997). "STR: A Simple and Efficient Algorithm for
       R-Tree Packing".
       https://ia600900.us.archive.org/27/items/nasa_techdoc_19970016975/19970016975.pdf

    """

    def __init__(
        self,
        geoms: Iterable[BaseGeometry],
        items: Iterable[Any] = None,
        node_capacity: int = 10,
    ):
        self.node_capacity = node_capacity

        # Keep references to geoms
        self._geoms = list(geoms)
        # Default enumeration index to store in the tree
        self._idxs = list(range(len(self._geoms)))

        # handle items
        self._has_custom_items = items is not None
        if not self._has_custom_items:
            items = self._idxs
        self._items = items

        # initialize GEOS STRtree
        self._tree = lgeos.GEOSSTRtree_create(self.node_capacity)
        i = 0
        for idx, geom in zip(self._idxs, self._geoms):
            # filter empty geometries out of the input
            if geom is not None and not geom.is_empty:
                lgeos.GEOSSTRtree_insert(self._tree, geom._geom, ctypes.py_object(idx))
                i += 1
        self._n_geoms = i

    def __reduce__(self):
        if self._has_custom_items:
            return STRtree, (self._geoms, self._items)
        else:
            return STRtree, (self._geoms, )

    def __del__(self):
        if self._tree is not None:
            try:
                lgeos.GEOSSTRtree_destroy(self._tree)
            except AttributeError:
                pass  # lgeos might be empty on shutdown.

            self._tree = None

    def _query(self, geom):
        if self._n_geoms == 0:
            return []

        result = []

        def callback(item, userdata):
            idx = ctypes.cast(item, ctypes.py_object).value
            result.append(idx)

        lgeos.GEOSSTRtree_query(self._tree, geom._geom, lgeos.GEOSQueryCallback(callback), None)
        return result

    def query_items(self, geom: BaseGeometry) -> Sequence[Any]:
        """Query for nodes which intersect the geom's envelope to get
        stored items.

        Items are integers serving as identifiers for an application.

        Parameters
        ----------
        geom : geometry object
            The query geometry.

        Returns
        -------
        An array or list of items stored in the tree.

        Note
        ----
        A geometry object's "envelope" is its minimum xy bounding
        rectangle.

        Examples
        --------
        A buffer around a point can be used to control the extent
        of the query.

        >>> from shapely.strtree import STRtree
        >>> from shapely.geometry import Point
        >>> points = [Point(i, i) for i in range(10)]
        >>> tree = STRtree(points)
        >>> query_geom = Point(2,2).buffer(0.99)
        >>> [o.wkt for o in tree.query(query_geom)]
        ['POINT (2 2)']
        >>> query_geom = Point(2, 2).buffer(1.0)
        >>> [o.wkt for o in tree.query(query_geom)]
        ['POINT (1 1)', 'POINT (2 2)', 'POINT (3 3)']

        A subsequent search through the returned subset using the
        desired binary predicate (eg. intersects, crosses, contains,
        overlaps) may be necessary to further filter the results
        according to their specific spatial relationships.

        >>> [o.wkt for o in tree.query(query_geom) if o.intersects(query_geom)]
        ['POINT (2 2)']

        """
        result = self._query(geom)
        if self._has_custom_items:
            return [self._items[i] for i in result]
        else:
            return result

    def query_geoms(self, geom: BaseGeometry) -> Sequence[BaseGeometry]:
        """Query for nodes which intersect the geom's envelope to get
        geometries corresponding to the items stored in the nodes.

        Parameters
        ----------
        geom : geometry object
            The query geometry.

        Returns
        -------
        An array or list of geometry objects.

        """
        result = self._query(geom)
        return [self._geoms[i] for i in result]

    def query(self, geom: BaseGeometry) -> Sequence[BaseGeometry]:
        """Query for nodes which intersect the geom's envelope to get
        geometries corresponding to the items stored in the nodes.

        This method is an alias for query_geoms. It may be removed in
        version 2.0.

        Parameters
        ----------
        geom : geometry object
            The query geometry.

        Returns
        -------
        An array or list of geometry objects.

        """
        return self.query_geoms(geom)

    def _nearest(self, geom, exclusive):
        envelope = geom.envelope

        def callback(item1, item2, distance, userdata):
            try:
                callback_userdata = ctypes.cast(userdata, ctypes.py_object).value
                idx = ctypes.cast(item1, ctypes.py_object).value
                geom2 = ctypes.cast(item2, ctypes.py_object).value
                dist = ctypes.cast(distance, ctypes.POINTER(ctypes.c_double))
                if callback_userdata["exclusive"] and self._geoms[idx].equals(geom2):
                    dist[0] = sys.float_info.max
                else:
                    lgeos.GEOSDistance(self._geoms[idx]._geom, geom2._geom, dist)
                
                return 1
            except Exception:
                log.exception("Caught exception")
                return 0

        item = lgeos.GEOSSTRtree_nearest_generic(
            self._tree,
            ctypes.py_object(geom),
            envelope._geom,
            lgeos.GEOSDistanceCallback(callback),
            ctypes.py_object({"exclusive": exclusive}),
        )
        return ctypes.cast(item, ctypes.py_object).value

    def nearest_item(
        self, geom: BaseGeometry, exclusive: bool = False
    ) -> Union[Any, None]:
        """Query the tree for the node nearest to geom and get the item
        stored in the node.

        Items are integers serving as identifiers for an application.

        Parameters
        ----------
        geom : geometry object
            The query geometry.
        exclusive : bool, optional
            Whether to exclude the item corresponding to the given geom
            from results or not.  Default: False.

        Returns
        -------
        Stored item or None.

        None is returned if this index is empty. This may change in
        version 2.0.

        Examples
        --------
        >>> from shapely.strtree import STRtree
        >>> from shapely.geometry import Point
        >>> tree = STRtree([Point(i, i) for i in range(10)])
        >>> tree.nearest(Point(2.2, 2.2)).wkt
        'POINT (2 2)'

        Will only return one object:

        >>> tree = STRtree ([Point(0, 0), Point(0, 0)])
        >>> tree.nearest(Point(0, 0)).wkt
        'POINT (0 0)'

        """
        if self._n_geoms == 0:
            return None

        result = self._nearest(geom, exclusive)
        if self._has_custom_items:
            return self._items[result]
        else:
            return result

    def nearest_geom(
        self, geom: BaseGeometry, exclusive: bool = False
    ) -> Union[BaseGeometry, None]:
        """Query the tree for the node nearest to geom and get the
        geometry corresponding to the item stored in the node.

        Parameters
        ----------
        geom : geometry object
            The query geometry.
        exclusive : bool, optional
            Whether to exclude the given geom from results or not.
            Default: False.

        Returns
        -------
        BaseGeometry or None.

        None is returned if this index is empty. This may change in
        version 2.0.

        """
        result = self._nearest(geom, exclusive)
        return self._geoms[result]

    def nearest(
        self, geom: BaseGeometry, exclusive: bool = False
    ) -> Union[BaseGeometry, None]:
        """Query the tree for the node nearest to geom and get the
        geometry corresponding to the item stored in the node.

        This method is an alias for nearest_geom. It may be removed in
        version 2.0.

        Parameters
        ----------
        geom : geometry object
            The query geometry.
        exclusive : bool, optional
            Whether to exclude the given geom from results or not.
            Default: False.

        Returns
        -------
        BaseGeometry or None.

        None is returned if this index is empty. This may change in
        version 2.0.

        """
        return self.nearest_geom(geom, exclusive=exclusive)
