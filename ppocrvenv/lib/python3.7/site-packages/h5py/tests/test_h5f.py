# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

import tempfile
import shutil
import os
import numpy as np
from h5py import File, special_dtype

from .common import TestCase


class TestFileID(TestCase):
    def test_descriptor_core(self):
        with File('TestFileID.test_descriptor_core', driver='core',
                  backing_store=False, mode='x') as f:
            assert isinstance(f.id.get_vfd_handle(), int)

    def test_descriptor_sec2(self):
        dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.test_descriptor_sec2')
        fn_h5 = os.path.join(dn_tmp, 'test.h5')
        try:
            with File(fn_h5, driver='sec2', mode='x') as f:
                descriptor = f.id.get_vfd_handle()
                self.assertNotEqual(descriptor, 0)
                os.fsync(descriptor)
        finally:
            shutil.rmtree(dn_tmp)


class TestCacheConfig(TestCase):
    def test_simple_gets(self):
        dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.TestCacheConfig.test_simple_gets')
        fn_h5 = os.path.join(dn_tmp, 'test.h5')
        try:
            with File(fn_h5, mode='x') as f:
                hit_rate = f._id.get_mdc_hit_rate()
                mdc_size = f._id.get_mdc_size()

        finally:
            shutil.rmtree(dn_tmp)

    def test_hitrate_reset(self):
        dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.TestCacheConfig.test_hitrate_reset')
        fn_h5 = os.path.join(dn_tmp, 'test.h5')
        try:
            with File(fn_h5, mode='x') as f:
                hit_rate = f._id.get_mdc_hit_rate()
                f._id.reset_mdc_hit_rate_stats()
                hit_rate = f._id.get_mdc_hit_rate()
                assert hit_rate == 0

        finally:
            shutil.rmtree(dn_tmp)

    def test_mdc_config_get(self):
        dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.TestCacheConfig.test_mdc_config_get')
        fn_h5 = os.path.join(dn_tmp, 'test.h5')
        try:
            with File(fn_h5, mode='x') as f:
                conf = f._id.get_mdc_config()
                f._id.set_mdc_config(conf)
        finally:
            shutil.rmtree(dn_tmp)


class TestVlenData(TestCase):
    def test_vlen_strings(self):
        # Create file with dataset containing vlen arrays of vlen strings
        dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestVlenStrings.test_vlen_strings')
        fn_h5 = os.path.join(dn_tmp, 'test.h5')
        try:
            with File(fn_h5, mode='w') as h:
                vlen_str = special_dtype(vlen=str)
                vlen_vlen_str = special_dtype(vlen=vlen_str)

                ds = h.create_dataset('/com', (2,), dtype=vlen_vlen_str)
                ds[0] = (np.array(["a", "b", "c"], dtype=vlen_vlen_str))
                ds[1] = (np.array(["d", "e", "f","g"], dtype=vlen_vlen_str))

            with File(fn_h5, "r") as h:
                ds = h["com"]
                assert ds[0].tolist() == [b'a', b'b', b'c']
                assert ds[1].tolist() == [b'd', b'e', b'f', b'g']

        finally:
            shutil.rmtree(dn_tmp)
