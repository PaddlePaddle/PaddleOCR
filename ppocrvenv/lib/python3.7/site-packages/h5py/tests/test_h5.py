# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

from h5py import h5

from .common import TestCase

def fixnames():
    cfg = h5.get_config()
    cfg.complex_names = ('r','i')

class TestH5(TestCase):

    def test_config(self):
        cfg = h5.get_config()
        self.assertIsInstance(cfg, h5.H5PYConfig)
        cfg2 = h5.get_config()
        self.assertIs(cfg, cfg2)

    def test_cnames_get(self):
        cfg = h5.get_config()
        self.assertEqual(cfg.complex_names, ('r','i'))

    def test_cnames_set(self):
        self.addCleanup(fixnames)
        cfg = h5.get_config()
        cfg.complex_names = ('q','x')
        self.assertEqual(cfg.complex_names, ('q','x'))

    def test_cnames_set_exc(self):
        self.addCleanup(fixnames)
        cfg = h5.get_config()
        with self.assertRaises(TypeError):
            cfg.complex_names = ('q','i','v')
        self.assertEqual(cfg.complex_names, ('r','i'))

    def test_repr(self):
        cfg = h5.get_config()
        repr(cfg)
