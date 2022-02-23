'''
Unit test for the high level vds interface for eiger
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''
import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile

import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support


@ut.skipUnless(vds_support,
               'VDS requires HDF5 >= 1.9.233')
class TestEigerHighLevel(ut.TestCase):
    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['raw_file_1.h5', 'raw_file_2.h5', 'raw_file_3.h5']
        for k, outfile in enumerate(self.fname):
            filename = osp.join(self.working_dir, outfile)
            f = h5.File(filename, 'w')
            f['data'] = np.ones((20, 200, 200)) * k
            f.close()

        f = h5.File(osp.join(self.working_dir, 'raw_file_4.h5'), 'w')
        f['data'] = np.ones((18, 200, 200)) * 3
        self.fname.append('raw_file_4.h5')
        self.fname = [osp.join(self.working_dir, ix) for ix in self.fname]
        f.close()

    def test_eiger_high_level(self):
        outfile = osp.join(self.working_dir, 'eiger.h5')
        layout = h5.VirtualLayout(shape=(78, 200, 200), dtype=float)

        M_minus_1 = 0
        # Create the virtual dataset file
        with h5.File(outfile, 'w', libver='latest') as f:
            for foo in self.fname:
                in_data = h5.File(foo, 'r')['data']
                src_shape = in_data.shape
                in_data.file.close()
                M = M_minus_1 + src_shape[0]
                vsource = h5.VirtualSource(foo, 'data', shape=src_shape)
                layout[M_minus_1:M, :, :] = vsource
                M_minus_1 = M
            f.create_virtual_dataset('data', layout, fillvalue=45)

        f = h5.File(outfile, 'r')['data']
        self.assertEqual(f[10, 100, 10], 0.0)
        self.assertEqual(f[30, 100, 100], 1.0)
        self.assertEqual(f[50, 100, 100], 2.0)
        self.assertEqual(f[70, 100, 100], 3.0)
        f.file.close()

    def tearDown(self):
        shutil.rmtree(self.working_dir)

'''
Unit test for the high level vds interface for excalibur
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''

class ExcaliburData(object):
    FEM_PIXELS_PER_CHIP_X = 256
    FEM_PIXELS_PER_CHIP_Y = 256
    FEM_CHIPS_PER_STRIPE_X = 8
    FEM_CHIPS_PER_STRIPE_Y = 1
    FEM_STRIPES_PER_MODULE = 2

    @property
    def sensor_module_dimensions(self):
        x_pixels = self.FEM_PIXELS_PER_CHIP_X * self.FEM_CHIPS_PER_STRIPE_X
        y_pixels = self.FEM_PIXELS_PER_CHIP_Y * self.FEM_CHIPS_PER_STRIPE_Y * self.FEM_STRIPES_PER_MODULE
        return y_pixels, x_pixels,

    @property
    def fem_stripe_dimensions(self):
        x_pixels = self.FEM_PIXELS_PER_CHIP_X * self.FEM_CHIPS_PER_STRIPE_X
        y_pixels = self.FEM_PIXELS_PER_CHIP_Y * self.FEM_CHIPS_PER_STRIPE_Y
        return y_pixels, x_pixels,

    def generate_sensor_module_image(self, value, dtype='uint16'):
        dset = np.empty(shape=self.sensor_module_dimensions, dtype=dtype)
        dset.fill(value)
        return dset

    def generate_fem_stripe_image(self, value, dtype='uint16'):
        dset = np.empty(shape=self.fem_stripe_dimensions, dtype=dtype)
        dset.fill(value)
        return dset


@ut.skipUnless(vds_support,
               'VDS requires HDF5 >= 1.9.233')
class TestExcaliburHighLevel(ut.TestCase):
    def create_excalibur_fem_stripe_datafile(self, fname, nframes, excalibur_data,scale):
        shape = (nframes,) + excalibur_data.fem_stripe_dimensions
        max_shape = shape#(None,) + excalibur_data.fem_stripe_dimensions
        chunk = (1,) + excalibur_data.fem_stripe_dimensions
        with h5.File(fname, 'w', libver='latest') as f:
            dset = f.create_dataset('data', shape=shape, maxshape=max_shape, chunks=chunk, dtype='uint16')
            for data_value_index in np.arange(nframes):
                dset[data_value_index] = excalibur_data.generate_fem_stripe_image(data_value_index*scale)

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ["stripe_%d.h5" % stripe for stripe in range(1,7)]
        self.fname = [osp.join(self.working_dir, f) for f in self.fname]
        nframes = 5
        self.edata = ExcaliburData()
        for k, raw_file in enumerate(self.fname):
            self.create_excalibur_fem_stripe_datafile(raw_file, nframes, self.edata,k)

    def test_excalibur_high_level(self):
        outfile = osp.join(self.working_dir, 'excalibur.h5')
        f = h5.File(outfile,'w',libver='latest') # create an output file.
        in_key = 'data' # where is the data at the input?
        in_sh = h5.File(self.fname[0],'r')[in_key].shape # get the input shape
        dtype = h5.File(self.fname[0],'r')[in_key].dtype # get the datatype

        # now generate the output shape
        vertical_gap = 10 # pixels spacing in the vertical
        nfiles = len(self.fname)
        nframes = in_sh[0]
        width = in_sh[2]
        height = (in_sh[1]*nfiles) + (vertical_gap*(nfiles-1))
        out_sh = (nframes, height, width)

        # Virtual layout is a representation of the output dataset
        layout = h5.VirtualLayout(shape=out_sh, dtype=dtype)
        offset = 0 # initial offset
        for i, filename in enumerate(self.fname):
            # A representation of the input dataset
            vsource = h5.VirtualSource(filename, in_key, shape=in_sh)
            layout[:, offset:(offset + in_sh[1]), :] = vsource # map them with indexing
            offset += in_sh[1] + vertical_gap # increment the offset

        # pass the fill value and list of maps
        f.create_virtual_dataset('data', layout, fillvalue=0x1)
        f.close()

        f = h5.File(outfile,'r')['data']
        self.assertEqual(f[3,100,0], 0.0)
        self.assertEqual(f[3,260,0], 1.0)
        self.assertEqual(f[3,350,0], 3.0)
        self.assertEqual(f[3,650,0], 6.0)
        self.assertEqual(f[3,900,0], 9.0)
        self.assertEqual(f[3,1150,0], 12.0)
        self.assertEqual(f[3,1450,0], 15.0)
        f.file.close()

    def tearDown(self):
        shutil.rmtree(self.working_dir)


'''
Unit test for the high level vds interface for percival
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''


@ut.skipUnless(vds_support,
               'VDS requires HDF5 >= 1.9.233')
class TestPercivalHighLevel(ut.TestCase):

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['raw_file_1.h5','raw_file_2.h5','raw_file_3.h5']
        k = 0
        for outfile in self.fname:
            filename = osp.join(self.working_dir, outfile)
            f = h5.File(filename,'w')
            f['data'] = np.ones((20,200,200))*k
            k +=1
            f.close()

        f = h5.File(osp.join(self.working_dir, 'raw_file_4.h5'), 'w')
        f['data'] = np.ones((19,200,200))*3
        self.fname.append('raw_file_4.h5')
        self.fname = [osp.join(self.working_dir, ix) for ix in self.fname]
        f.close()

    def test_percival_high_level(self):
        outfile = osp.join(self.working_dir,  'percival.h5')

        # Virtual layout is a representation of the output dataset
        layout = h5.VirtualLayout(shape=(79, 200, 200), dtype=np.float64)
        for k, filename in enumerate(self.fname):
            dim1 = 19 if k == 3 else 20
            vsource = h5.VirtualSource(filename, 'data',shape=(dim1, 200, 200))
            layout[k:79:4, :, :] = vsource[:, :, :]

        # Create the virtual dataset file
        with h5.File(outfile, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-5)

        foo = np.array(2 * list(range(4)))
        with h5.File(outfile,'r') as f:
            ds = f['data']
            line = ds[:8,100,100]
            self.assertEqual(ds.shape, (79,200,200),)
            assert_array_equal(line, foo)

    def test_percival_source_from_dataset(self):
        outfile = osp.join(self.working_dir,  'percival.h5')

        # Virtual layout is a representation of the output dataset
        layout = h5.VirtualLayout(shape=(79, 200, 200), dtype=np.float64)
        for k, filename in enumerate(self.fname):
            with h5.File(filename, 'r') as f:
                vsource = h5.VirtualSource(f['data'])
                layout[k:79:4, :, :] = vsource

        # Create the virtual dataset file
        with h5.File(outfile, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-5)

        foo = np.array(2 * list(range(4)))
        with h5.File(outfile,'r') as f:
            ds = f['data']
            line = ds[:8,100,100]
            self.assertEqual(ds.shape, (79,200,200),)
            assert_array_equal(line, foo)

    def tearDown(self):
        shutil.rmtree(self.working_dir)

@ut.skipUnless(vds_support,
               'VDS requires HDF5 >= 1.9.233')
class SlicingTestCase(ut.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create source files (1.h5 to 4.h5)
        for n in range(1, 5):
            with h5.File(osp.join(self.tmpdir, '{}.h5'.format(n)), 'w') as f:
                d = f.create_dataset('data', (100,), 'i4')
                d[:] = np.arange(100) + n

    def make_virtual_ds(self):
        # Assemble virtual dataset
        layout = h5.VirtualLayout((4, 100), 'i4', maxshape=(4, None))

        for n in range(1, 5):
            filename = osp.join(self.tmpdir, "{}.h5".format(n))
            vsource = h5.VirtualSource(filename, 'data', shape=(100,))
            # Fill the first half with positions 0, 2, 4... from the source
            layout[n - 1, :50] = vsource[0:100:2]
            # Fill the second half with places 1, 3, 5... from the source
            layout[n - 1, 50:] = vsource[1:100:2]

        outfile = osp.join(self.tmpdir, 'VDS.h5')

        # Add virtual dataset to output file
        with h5.File(outfile, 'w', libver='latest') as f:
            f.create_virtual_dataset('/group/data', layout, fillvalue=-5)

        return outfile

    def test_slice_source(self):
        outfile = self.make_virtual_ds()

        with h5.File(outfile, 'r') as f:
            assert_array_equal(f['/group/data'][0][:3], [1, 3, 5])
            assert_array_equal(f['/group/data'][0][50:53], [2, 4, 6])
            assert_array_equal(f['/group/data'][3][:3], [4, 6, 8])
            assert_array_equal(f['/group/data'][3][50:53], [5, 7, 9])

    def test_inspection(self):
        with h5.File(osp.join(self.tmpdir, '1.h5'), 'r') as f:
            assert not f['data'].is_virtual

        outfile = self.make_virtual_ds()

        with h5.File(outfile, 'r') as f:
            ds = f['/group/data']
            assert ds.is_virtual

            src_files = {osp.join(self.tmpdir, '{}.h5'.format(n))
                         for n in range(1, 5)}
            assert {s.file_name for s in ds.virtual_sources()} == src_files

    def test_mismatched_selections(self):
        layout = h5.VirtualLayout((4, 100), 'i4', maxshape=(4, None))

        filename = osp.join(self.tmpdir, "1.h5")
        vsource = h5.VirtualSource(filename, 'data', shape=(100,))
        with self.assertRaisesRegex(ValueError, r'different number'):
            layout[0, :49] = vsource[0:100:2]

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

@ut.skipUnless(vds_support,
               'VDS requires HDF5 >= 1.9.233')
class IndexingTestCase(ut.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create source file (1.h5)
        with h5.File(osp.join(self.tmpdir, '1.h5'), 'w') as f:
            d = f.create_dataset('data', (10,), 'i4')
            d[:] = np.arange(10)*10

    def test_index_layout(self):
        # Assemble virtual dataset (indexing target)
        layout = h5.VirtualLayout((100,), 'i4')

        inds = [3,6,20,25,33,47,70,75,96,98]

        filename = osp.join(self.tmpdir, "1.h5")
        vsource = h5.VirtualSource(filename, 'data', shape=(10,))
        layout[inds] = vsource

        outfile = osp.join(self.tmpdir, 'VDS.h5')

        # Assembly virtual dataset (indexing source)
        layout2 = h5.VirtualLayout((6,), 'i4')

        inds2 = [0,1,4,5,8]
        layout2[1:] = vsource[inds2]

        # Add virtual datasets to output file and close
        with h5.File(outfile, 'w', libver='latest') as f:
            f.create_virtual_dataset('/data', layout, fillvalue=-5)
            f.create_virtual_dataset(b'/data2', layout2, fillvalue=-3)

        # Read data from virtual datasets
        with h5.File(outfile, 'r') as f:
            data = f['/data'][()]
            data2 = f['/data2'][()]

        # Verify
        assert_array_equal(data[inds], np.arange(10)*10)
        assert_array_equal(data2[1:], [0,10,40,50,80])

        mask = np.zeros(100)
        mask[inds] = 1
        self.assertEqual(data[mask == 0].min(), -5)
        self.assertEqual(data[mask == 0].max(), -5)
        self.assertEqual(data2[0], -3)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

@ut.skipUnless(vds_support,
               'VDS requires HDF5 >= 1.9.233')
class RelativeLinkTestCase(ut.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.f1 = osp.join(self.tmpdir, 'testfile1.h5')
        self.f2 = osp.join(self.tmpdir, 'testfile2.h5')

        self.data1 = np.arange(10)
        self.data2 = np.arange(10) * -1

        with h5.File(self.f1, 'w') as f:
            # dataset
            ds = f.create_dataset('data', (10,), 'f4')
            ds[:] = self.data1

        with h5.File(self.f2, 'w') as f:
            # dataset
            ds = f.create_dataset('data', (10,), 'f4')
            ds[:] = self.data2
            self.make_vds(f)

    def make_vds(self, f):
        # virtual dataset
        layout = h5.VirtualLayout((2, 10), 'f4')
        vsource1 = h5.VirtualSource(self.f1, 'data', shape=(10,))
        vsource2 = h5.VirtualSource(self.f2, 'data', shape=(10,))
        layout[0] = vsource1
        layout[1] = vsource2
        f.create_virtual_dataset('virtual', layout)

    def test_relative_vds(self):
        with h5.File(self.f2) as f:
            data = f['virtual'][:]
            np.testing.assert_array_equal(data[0], self.data1)
            np.testing.assert_array_equal(data[1], self.data2)

        # move f2 -> f3
        f3 = osp.join(self.tmpdir, 'testfile3.h5')
        os.rename(self.f2, f3)

        with h5.File(f3) as f:
            data = f['virtual'][:]
            assert data.dtype == 'f4'
            np.testing.assert_array_equal(data[0], self.data1)
            np.testing.assert_array_equal(data[1], self.data2)

        # moving other file
        f4 = osp.join(self.tmpdir, 'testfile4.h5')
        os.rename(self.f1, f4)

        with h5.File(f3) as f:
            data = f['virtual'][:]
            assert data.dtype == 'f4'
            # unavailable data is silently converted to default value
            np.testing.assert_array_equal(data[0], 0)
            np.testing.assert_array_equal(data[1], self.data2)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

class RelativeLinkBuildVDSTestCase(RelativeLinkTestCase):
    # Test a link to the same file with the virtual dataset created by
    # File.build_virtual_dataset()
    def make_vds(self, f):
        with f.build_virtual_dataset('virtual', (2, 10), dtype='f4') as layout:
            layout[0] = h5.VirtualSource(self.f1, 'data', shape=(10,))
            layout[1] = h5.VirtualSource(self.f2, 'data', shape=(10,))

@ut.skipUnless(vds_support,
               'VDS requires HDF5 >= 1.9.233')
class VDSUnlimitedTestCase(ut.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = osp.join(self.tmpdir, "resize.h5")
        with h5.File(self.path, "w") as f:
            source_dset = f.create_dataset(
                "source",
                data=np.arange(20),
                shape=(10, 2),
                maxshape=(None, 2),
                chunks=(10, 1),
                fillvalue=-1
            )
            self.layout = h5.VirtualLayout((10, 1), int, maxshape=(None, 1))
            layout_source = h5.VirtualSource(source_dset)
            self.layout[:h5.UNLIMITED, 0] = layout_source[:h5.UNLIMITED, 1]

            f.create_virtual_dataset("virtual", self.layout)

    def test_unlimited_axis(self):
        comp1 = np.arange(1, 20, 2).reshape(10, 1)
        comp2 = np.vstack((
            comp1,
            np.full(shape=(10, 1), fill_value=-1)
        ))
        comp3 = np.vstack((
            comp1,
            np.full(shape=(10, 1), fill_value=0)
        ))
        with h5.File(self.path, "a") as f:
            source_dset = f['source']
            virtual_dset = f['virtual']
            np.testing.assert_array_equal(comp1, virtual_dset)
            source_dset.resize(20, axis=0)
            np.testing.assert_array_equal(comp2, virtual_dset)
            source_dset[10:, 1] = np.zeros((10,), dtype=int)
            np.testing.assert_array_equal(comp3, virtual_dset)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

if __name__ == "__main__":
    ut.main()
