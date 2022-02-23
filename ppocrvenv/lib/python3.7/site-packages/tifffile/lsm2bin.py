#!/usr/bin/env python3
# tifffile/lsm2bin.py

"""Convert TZCYX LSM file to series of BIN files.

Usage: lsm2bin lsm_filename [bin_filename]

"""

import sys

try:
    from .tifffile import lsm2bin
except ImportError:
    try:
        from tifffile.tifffile import lsm2bin
    except ImportError:
        from tifffile import lsm2bin


def main(argv=None):
    """Lsm2bin command line usage main function."""
    if argv is None:
        argv = sys.argv
    if len(argv) > 1:
        lsm2bin(argv[1], argv[2] if len(argv) > 2 else None)
    else:
        print()
        print(__doc__.strip())


if __name__ == '__main__':
    sys.exit(main())
