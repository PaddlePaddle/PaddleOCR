# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py._hl.filters module.

"""
import os
import numpy as np
import h5py
import pytest

from .common import ut, TestCase, insubprocess


class TestFilters(TestCase):

    def setUp(self):
        """ like TestCase.setUp but also store the file path """
        self.path = self.mktemp()
        self.f = h5py.File(self.path, 'w')

    @ut.skipUnless(h5py.h5z.filter_avail(h5py.h5z.FILTER_SZIP), 'szip filter required')
    def test_wr_szip_fletcher32_64bit(self):
        """ test combination of szip, fletcher32, and 64bit arrays

        The fletcher32 checksum must be computed after the szip
        compression is applied.

        References:
        - GitHub issue #953
        - https://lists.hdfgroup.org/pipermail/
          hdf-forum_lists.hdfgroup.org/2018-January/010753.html
        """
        self.f.create_dataset("test_data",
                              data=np.zeros(10000, dtype=np.float64),
                              fletcher32=True,
                              compression="szip",
                              )
        self.f.close()

        with h5py.File(self.path, "r") as h5:
            # Access the data which will compute the fletcher32
            # checksum and raise an OSError if something is wrong.
            h5["test_data"][0]

    def test_wr_scaleoffset_fletcher32(self):
        """ make sure that scaleoffset + fletcher32 is prevented
        """
        data = np.linspace(0, 1, 100)
        with self.assertRaises(ValueError):
            self.f.create_dataset("test_data",
                                  data=data,
                                  fletcher32=True,
                                  # retain 3 digits after the decimal point
                                  scaleoffset=3,
                                  )


@ut.skipIf('gzip' not in h5py.filters.encode, "DEFLATE is not installed")
def test_filter_ref_obj(writable_file):
    gzip8 = h5py.filters.Gzip(level=8)
    # **kwargs unpacking (compatible with earlier h5py versions)
    assert dict(**gzip8) == {
        'compression': h5py.h5z.FILTER_DEFLATE,
        'compression_opts': (8,)
    }

    # Pass object as compression argument (new in h5py 3.0)
    ds = writable_file.create_dataset(
        'x', shape=(100,), dtype=np.uint32, compression=gzip8
    )
    assert ds.compression == 'gzip'
    assert ds.compression_opts == 8


def test_filter_ref_obj_eq():
    gzip8 = h5py.filters.Gzip(level=8)

    assert gzip8 == h5py.filters.Gzip(level=8)
    assert gzip8 != h5py.filters.Gzip(level=7)


@pytest.mark.mpi_skip
@insubprocess
def test_unregister_filter(request):
    if h5py.h5z.filter_avail(h5py.h5z.FILTER_LZF):
        res = h5py.h5z.unregister_filter(h5py.h5z.FILTER_LZF)
        assert res


@ut.skipIf(not os.getenv('H5PY_TEST_CHECK_FILTERS'),  "H5PY_TEST_CHECK_FILTERS not set")
def test_filters_available():
    assert 'gzip' in h5py.filters.decode
    assert 'gzip' in h5py.filters.encode
    assert 'lzf' in h5py.filters.decode
    assert 'lzf' in h5py.filters.encode
