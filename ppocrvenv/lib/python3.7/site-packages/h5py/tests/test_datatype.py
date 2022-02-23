# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    File-resident datatype tests.

    Tests "committed" file-resident datatype objects.
"""

import numpy as np

from .common import ut, TestCase

from h5py import Datatype

class TestCreation(TestCase):

    """
        Feature: repr() works sensibly on datatype objects
    """

    def test_repr(self):
        """ repr() on datatype objects """
        self.f['foo'] = np.dtype('S10')
        dt = self.f['foo']
        self.assertIsInstance(repr(dt), str)
        self.f.close()
        self.assertIsInstance(repr(dt), str)


    def test_appropriate_low_level_id(self):
        " Binding a group to a non-TypeID identifier fails with ValueError "
        with self.assertRaises(ValueError):
            Datatype(self.f['/'].id)
