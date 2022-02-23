# intersect.py
#
# Demonstrate how Shapely can be used to analyze and plot the intersection of
# a trajectory and regions in space.

from functools import partial
import random

import pylab

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

# Build patches as in dissolved.py
r = partial(random.uniform, -20.0, 20.0)
points = [Point(r(), r()) for i in range(100)]
spots = [p.buffer(2.5) for p in points]
patches = unary_union(spots)

# Represent the following geolocation parameters
#
# initial position: -25, -25
# heading: 45.0
# speed: 50*sqrt(2)
#
# as a line
vector = LineString(((-25.0, -25.0), (25.0, 25.0)))

# Find intercepted and missed patches. List the former so we can count them
# later
intercepts = [patch for patch in patches.geoms if vector.intersects(patch)]
misses = (patch for patch in patches.geoms if not vector.intersects(patch))

# Plot the intersection
intersection = vector.intersection(patches)
assert intersection.geom_type in ['MultiLineString']

if __name__ == "__main__":
    # Illustrate the results using matplotlib's pylab interface
    pylab.figure(num=None, figsize=(4, 4), dpi=180)
    
    # Plot the misses
    for spot in misses:
        x, y = spot.exterior.xy
        pylab.fill(x, y, color='#cccccc', aa=True) 
        pylab.plot(x, y, color='#999999', aa=True, lw=1.0)
    
        # Do the same for the holes of the patch
        for hole in spot.interiors:
            x, y = hole.xy
            pylab.fill(x, y, color='#ffffff', aa=True) 
            pylab.plot(x, y, color='#999999', aa=True, lw=1.0)
    
    # Plot the intercepts
    for spot in intercepts:
        x, y = spot.exterior.xy
        pylab.fill(x, y, color='red', alpha=0.25, aa=True) 
        pylab.plot(x, y, color='red', alpha=0.5, aa=True, lw=1.0)
    
        # Do the same for the holes of the patch
        for hole in spot.interiors:
            x, y = hole.xy
            pylab.fill(x, y, color='#ffffff', aa=True) 
            pylab.plot(x, y, color='red', alpha=0.5, aa=True, lw=1.0)
    
    # Draw the projected trajectory
    pylab.arrow(-25, -25, 50, 50, color='#999999', aa=True,
        head_width=1.0, head_length=1.0)
    
    for segment in intersection.geoms:
        x, y = segment.xy
        pylab.plot(x, y, color='red', aa=True, lw=1.5)
    
    # Write the number of patches and the total patch area to the figure
    pylab.text(-28, 25, 
        "Patches: %d/%d (%d), total length: %.1f" \
         % (len(intercepts), len(patches.geoms), 
            len(intersection.geoms), intersection.length))
    
    pylab.savefig('intersect.png')
    
