#!/usr/bin/env python

"""Helper script for creating image .dat files by numpy.save

Usage:

    python create_dat.py <name of image file> <name of dat file>

Example (to create aero.dat):

    python create_dat.py aero.png aero.dat

Requires Scipy and PIL.
"""

from __future__ import print_function

import sys

import numpy as np


def main():
    from scipy.misc import imread

    if len(sys.argv) != 3:
        print(__doc__)
        exit()

    image_fname = sys.argv[1]
    dat_fname = sys.argv[2]

    data = imread(image_fname)

    np.savez_compressed(dat_fname, data=data)


if __name__ == "__main__":
    main()
