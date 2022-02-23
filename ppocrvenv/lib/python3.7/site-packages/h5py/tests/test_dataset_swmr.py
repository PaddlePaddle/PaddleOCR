import numpy as np
import h5py

from .common import ut, TestCase


@ut.skipUnless(h5py.version.hdf5_version_tuple < (1, 9, 178), 'SWMR is available. Skipping backwards compatible tests')
class TestSwmrNotAvailable(TestCase):
    """ Test backwards compatibility behaviour when using SWMR functions with
    an older version of HDF5 which does not have this feature available.
    Skip this test if SWMR features *are* available in the HDF5 library.
    """

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.arange(13).astype('f')
        self.dset = self.f.create_dataset('data', chunks=(13,), maxshape=(None,), data=self.data)

    def test_open_swmr_raises(self):
        fname = self.f.filename
        self.f.close()

        with self.assertRaises(ValueError):
            self.f = h5py.File(fname, 'r', swmr=True)

    def test_refresh_raises(self):
        """ If the SWMR feature is not available then Dataset.refresh() should throw an AttributeError
        """
        with self.assertRaises(AttributeError):
            self.dset.refresh()

    def test_flush_raises(self):
        """ If the SWMR feature is not available the Dataset.flush() should
        throw an AttributeError
        """
        with self.assertRaises(AttributeError):
            self.dset.flush()

    def test_swmr_mode_false(self):
        """ The SWMR getter should just be False
        """
        assert not self.f.swmr_mode

    def test_set_swmr_mode_raises(self):
        """ If the SWMR feature is not available, setting swmr_mode = True
        should raise a RuntimeError
        """
        with self.assertRaises(RuntimeError):
            self.f.swmr_mode = True
        assert not self.f.swmr_mode

@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 9, 178), 'SWMR requires HDF5 >= 1.9.178')
class TestDatasetSwmrRead(TestCase):
    """ Testing SWMR functions when reading a dataset.
    Skip this test if the HDF5 library does not have the SWMR features.
    """

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.arange(13).astype('f')
        self.dset = self.f.create_dataset('data', chunks=(13,), maxshape=(None,), data=self.data)
        fname = self.f.filename
        self.f.close()

        self.f = h5py.File(fname, 'r', swmr=True)
        self.dset = self.f['data']

    def test_initial_swmr_mode_on(self):
        """ Verify that the file is initially in SWMR mode"""
        self.assertTrue(self.f.swmr_mode)

    def test_read_data(self):
        self.assertArrayEqual(self.dset, self.data)

    def test_refresh(self):
        self.dset.refresh()

    def test_force_swmr_mode_on_raises(self):
        """ Verify when reading a file cannot be forcibly switched to swmr mode.
        When reading with SWMR the file must be opened with swmr=True."""
        with self.assertRaises(Exception):
            self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)

    def test_force_swmr_mode_off_raises(self):
        """ Switching SWMR write mode off is only possible by closing the file.
        Attempts to forcibly switch off the SWMR mode should raise a ValueError.
        """
        with self.assertRaises(ValueError):
            self.f.swmr_mode = False
        self.assertTrue(self.f.swmr_mode)

@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 9, 178), 'SWMR requires HDF5 >= 1.9.178')
class TestDatasetSwmrWrite(TestCase):
    """ Testing SWMR functions when reading a dataset.
    Skip this test if the HDF5 library does not have the SWMR features.
    """

    def setUp(self):
        """ First setup a file with a small chunked and empty dataset.
        No data written yet.
        """

        # Note that when creating the file, the swmr=True is not required for
        # write, but libver='latest' is required.
        self.f = h5py.File(self.mktemp(), 'w', libver='latest')

        self.data = np.arange(4).astype('f')
        self.dset = self.f.create_dataset('data', shape=(0,), dtype=self.data.dtype, chunks=(2,), maxshape=(None,))


    def test_initial_swmr_mode_off(self):
        """ Verify that the file is not initially in SWMR mode"""
        self.assertFalse(self.f.swmr_mode)

    def test_switch_swmr_mode_on(self):
        """ Switch to SWMR mode and verify """
        self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)

    def test_switch_swmr_mode_off_raises(self):
        """ Switching SWMR write mode off is only possible by closing the file.
        Attempts to forcibly switch off the SWMR mode should raise a ValueError.
        """
        self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)
        with self.assertRaises(ValueError):
            self.f.swmr_mode = False
        self.assertTrue(self.f.swmr_mode)

    def test_extend_dset(self):
        """ Extend and flush a SWMR dataset
        """
        self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)

        self.dset.resize( self.data.shape )
        self.dset[:] = self.data
        self.dset.flush()

        # Refresh and read back data for assertion
        self.dset.refresh()
        self.assertArrayEqual(self.dset, self.data)

    def test_extend_dset_multiple(self):

        self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)

        self.dset.resize( (4,) )
        self.dset[0:] = self.data
        self.dset.flush()

        # Refresh and read back 1st data block for assertion
        self.dset.refresh()
        self.assertArrayEqual(self.dset, self.data)

        self.dset.resize( (8,) )
        self.dset[4:] = self.data
        self.dset.flush()

        # Refresh and read back 1st data block for assertion
        self.dset.refresh()
        self.assertArrayEqual(self.dset[0:4], self.data)
        self.assertArrayEqual(self.dset[4:8], self.data)
