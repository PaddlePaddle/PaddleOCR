from ..geometry import Point, LineString
from ..geos import TopologicalError
from heapq import heappush, heappop


class Cell:
    """A `Cell`'s centroid property is a potential solution to finding the pole
    of inaccessibility for a given polygon. Rich comparison operators are used
    for sorting `Cell` objects in a priority queue based on the potential
    maximum distance of any theoretical point within a cell to a given
    polygon's exterior boundary.
    """
    def __init__(self, x, y, h, polygon):
        self.x = x
        self.y = y
        self.h = h  # half of cell size
        self.centroid = Point(x, y)  # cell centroid, potential solution

        # distance from cell centroid to polygon exterior
        self.distance = self._dist(polygon)

        # max distance to polygon exterior within a cell
        self.max_distance = self.distance + h * 1.4142135623730951  # sqrt(2)

    # rich comparison operators for sorting in minimum priority queue
    def __lt__(self, other):
        return self.max_distance > other.max_distance

    def __le__(self, other):
        return self.max_distance >= other.max_distance

    def __eq__(self, other):
        return self.max_distance == other.max_distance

    def __ne__(self, other):
        return self.max_distance != other.max_distance

    def __gt__(self, other):
        return self.max_distance < other.max_distance

    def __ge__(self, other):
        return self.max_distance <= other.max_distance

    def _dist(self, polygon):
        """Signed distance from Cell centroid to polygon outline. The returned
        value is negative if the point is outside of the polygon exterior
        boundary.
        """
        inside = polygon.contains(self.centroid)
        distance = self.centroid.distance(polygon.exterior)
        for interior in polygon.interiors:
            distance = min(distance, self.centroid.distance(interior))
        if inside:
            return distance
        return -distance


def polylabel(polygon, tolerance=1.0):
    """Finds pole of inaccessibility for a given polygon. Based on
    Vladimir Agafonkin's https://github.com/mapbox/polylabel

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
    tolerance : int or float, optional
                `tolerance` represents the highest resolution in units of the
                input geometry that will be considered for a solution. (default
                value is 1.0).

    Returns
    -------
    shapely.geometry.Point
        A point representing the pole of inaccessibility for the given input
        polygon.

    Raises
    ------
    shapely.geos.TopologicalError
        If the input polygon is not a valid geometry.

    Example
    -------
    >>> polygon = LineString([(0, 0), (50, 200), (100, 100), (20, 50),
    ... (-100, -20), (-150, -200)]).buffer(100)
    >>> label = polylabel(polygon, tolerance=10)
    >>> label.wkt
    'POINT (59.35615556364569 121.8391962974644)'
    """
    if not polygon.is_valid:
        raise TopologicalError('Invalid polygon')
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    cell_size = min(width, height)
    h = cell_size / 2.0
    cell_queue = []

    # First best cell approximation is one constructed from the centroid
    # of the polygon
    x, y = polygon.centroid.coords[0]
    best_cell = Cell(x, y, 0, polygon)

    # Special case for rectangular polygons avoiding floating point error
    bbox_cell = Cell(minx + width / 2.0, miny + height / 2, 0, polygon)
    if bbox_cell.distance > best_cell.distance:
        best_cell = bbox_cell

    # build a regular square grid covering the polygon
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            heappush(cell_queue, Cell(x + h, y + h, h, polygon))
            y += cell_size
        x += cell_size

    # minimum priority queue
    while cell_queue:
        cell = heappop(cell_queue)

        # update the best cell if we find a better one
        if cell.distance > best_cell.distance:
            best_cell = cell

        # continue to the next iteration if we cant find a better solution
        # based on tolerance
        if cell.max_distance - best_cell.distance <= tolerance:
            continue

        # split the cell into quadrants
        h = cell.h / 2.0
        heappush(cell_queue, Cell(cell.x - h, cell.y - h, h, polygon))
        heappush(cell_queue, Cell(cell.x + h, cell.y - h, h, polygon))
        heappush(cell_queue, Cell(cell.x - h, cell.y + h, h, polygon))
        heappush(cell_queue, Cell(cell.x + h, cell.y + h, h, polygon))

    return best_cell.centroid
