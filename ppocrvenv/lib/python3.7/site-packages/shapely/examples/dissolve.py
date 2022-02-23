# dissolve.py
#
# Demonstrate how Shapely can be used to build up a collection of patches by 
# dissolving circular regions and how Shapely supports plotting of the results.

from functools import partial
import random

import pylab

from shapely.geometry import Point
from shapely.ops import unary_union

# Use a partial function to make 100 points uniformly distributed in a 40x40 
# box centered on 0,0.
r = partial(random.uniform, -20.0, 20.0)
points = [Point(r(), r()) for i in range(100)]

# Buffer the points, producing 100 polygon spots
spots = [p.buffer(2.5) for p in points]

# Perform a unary union of the polygon spots, dissolving them into a
# collection of polygon patches
patches = unary_union(spots)

if __name__ == "__main__":
    # Illustrate the results using matplotlib's pylab interface
    pylab.figure(num=None, figsize=(4, 4), dpi=180)
    
    for patch in patches.geoms:
        assert patch.geom_type in ['Polygon']
        assert patch.is_valid
    
        # Fill and outline each patch
        x, y = patch.exterior.xy
        pylab.fill(x, y, color='#cccccc', aa=True) 
        pylab.plot(x, y, color='#666666', aa=True, lw=1.0)
    
        # Do the same for the holes of the patch
        for hole in patch.interiors:
            x, y = hole.xy
            pylab.fill(x, y, color='#ffffff', aa=True) 
            pylab.plot(x, y, color='#999999', aa=True, lw=1.0)
    
    # Plot the original points
    pylab.plot([p.x for p in points], [p.y for p in points], 'b,', alpha=0.75)
    
    # Write the number of patches and the total patch area to the figure
    pylab.text(-25, 25, 
        "Patches: %d, total area: %.2f" % (len(patches.geoms), patches.area))
    
    pylab.savefig('dissolve.png')
    
