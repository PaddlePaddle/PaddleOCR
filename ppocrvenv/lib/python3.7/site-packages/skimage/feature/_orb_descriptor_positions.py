import os
import numpy as np

# Putting this in cython was giving strange bugs for different versions
# of cython which seemed to indicate troubles with the __file__ variable
# not being defined. Keeping it in pure python makes it more reliable
this_dir = os.path.dirname(__file__)
POS = np.loadtxt(os.path.join(this_dir, "orb_descriptor_positions.txt"),
                 dtype=np.int8)
POS0 = np.ascontiguousarray(POS[:, :2])
POS1 = np.ascontiguousarray(POS[:, 2:])
