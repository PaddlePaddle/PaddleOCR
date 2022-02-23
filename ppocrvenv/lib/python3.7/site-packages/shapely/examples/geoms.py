from numpy import asarray
import pylab
from shapely.geometry import Point, LineString, Polygon

polygon = Polygon(((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)))

point_r = Point(-1.5, 1.2)
point_g = Point(-1.0, 1.0)
point_b = Point(-0.5, 0.5)

line_r = LineString(((-0.5, 0.5), (0.5, 0.5)))
line_g = LineString(((1.0, -1.0), (1.8, 0.5)))
line_b = LineString(((-1.8, -1.2), (1.8, 0.5)))

def plot_point(g, o, l):
    pylab.plot([g.x], [g.y], o, label=l)

def plot_line(g, o):
    a = asarray(g)
    pylab.plot(a[:,0], a[:,1], o)

def fill_polygon(g, o):
    a = asarray(g.exterior)
    pylab.fill(a[:,0], a[:,1], o, alpha=0.5)

def fill_multipolygon(g, o):
    for g in g.geoms:
        fill_polygon(g, o)

if __name__ == "__main__":
    from numpy import asarray
    import pylab
    
    fig = pylab.figure(1, figsize=(4, 3), dpi=150)
    #pylab.axis([-2.0, 2.0, -1.5, 1.5])
    pylab.axis('tight')

    a = asarray(polygon.exterior)
    pylab.fill(a[:,0], a[:,1], 'c')

    plot_point(point_r, 'ro', 'b')
    plot_point(point_g, 'go', 'c')
    plot_point(point_b, 'bo', 'd')

    plot_line(line_r, 'r')
    plot_line(line_g, 'g')
    plot_line(line_b, 'b')

    pylab.show()


