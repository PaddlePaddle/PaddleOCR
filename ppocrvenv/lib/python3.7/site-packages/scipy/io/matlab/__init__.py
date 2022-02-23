"""
Utilities for dealing with MATLAB(R) files

Notes
-----
MATLAB(R) is a registered trademark of The MathWorks, Inc., 3 Apple Hill
Drive, Natick, MA 01760-2098, USA.

"""
# Matlab file read and write utilities
from .mio import loadmat, savemat, whosmat
from . import byteordercodes

__all__ = ['loadmat', 'savemat', 'whosmat', 'byteordercodes']

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
