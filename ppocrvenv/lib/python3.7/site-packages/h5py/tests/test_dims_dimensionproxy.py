# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py.Dataset.dims.DimensionProxy class.
"""

import numpy as np
import h5py

from .common import ut, TestCase

class TestItems(TestCase):

    def test_empty(self):
        """ no dimension scales -> empty list """
        dset = self.f.create_dataset('x', (10,))
        self.assertEqual(dset.dims[0].items(), [])
