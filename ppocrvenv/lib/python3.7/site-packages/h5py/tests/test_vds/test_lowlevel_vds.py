'''
Unit test for the low level vds interface for eiger
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''


from ..common import ut
import numpy as np
import h5py as h5
import tempfile


@ut.skipUnless(h5.version.hdf5_version_tuple >= (1, 9, 233),
               'VDS requires HDF5 >= 1.9.233')
class TestEigerLowLevel(ut.TestCase):
    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['raw_file_1.h5', 'raw_file_2.h5', 'raw_file_3.h5']
        k = 0
        for outfile in self.fname:
            filename = self.working_dir + outfile
            f = h5.File(filename, 'w')
            f['data'] = np.ones((20, 200, 200))*k
            k += 1
            f.close()

        f = h5.File(self.working_dir+'raw_file_4.h5', 'w')
        f['data'] = np.ones((18, 200, 200))*3
        self.fname.append('raw_file_4.h5')
        self.fname = [self.working_dir+ix for ix in self.fname]
        f.close()

    def test_eiger_low_level(self):
        self.outfile = self.working_dir + 'eiger.h5'
        with h5.File(self.outfile, 'w', libver='latest') as f:
            vdset_shape = (78, 200, 200)
            vdset_max_shape = vdset_shape
            virt_dspace = h5.h5s.create_simple(vdset_shape, vdset_max_shape)
            dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
            dcpl.set_fill_value(np.array([-1]))
            # Create the source dataset dataspace
            k = 0
            for foo in self.fname:
                in_data = h5.File(foo, 'r')['data']
                src_shape = in_data.shape
                max_src_shape = src_shape
                in_data.file.close()
                src_dspace = h5.h5s.create_simple(src_shape, max_src_shape)
                # Select the source dataset hyperslab
                src_dspace.select_hyperslab(start=(0, 0, 0),
                                            stride=(1, 1, 1),
                                            count=(1, 1, 1),
                                            block=src_shape)

                virt_dspace.select_hyperslab(start=(k, 0, 0),
                                             stride=(1, 1, 1),
                                             count=(1, 1, 1),
                                             block=src_shape)

                dcpl.set_virtual(virt_dspace, foo.encode('utf-8'),
                                 b'data', src_dspace)
                k += src_shape[0]

            # Create the virtual dataset
            h5.h5d.create(f.id, name=b"data", tid=h5.h5t.NATIVE_INT16,
                          space=virt_dspace, dcpl=dcpl)

        f = h5.File(self.outfile, 'r')['data']
        self.assertEqual(f[10, 100, 10], 0.0)
        self.assertEqual(f[30, 100, 100], 1.0)
        self.assertEqual(f[50, 100, 100], 2.0)
        self.assertEqual(f[70, 100, 100], 3.0)
        f.file.close()

    def tearDown(self):
        import os
        for f in self.fname:
            os.remove(f)
        os.remove(self.outfile)


if __name__ == "__main__":
    ut.main()
'''
Unit test for the low level vds interface for excalibur
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


@ut.skipUnless(h5.version.hdf5_version_tuple >= (1, 9, 233),
               'VDS requires HDF5 >= 1.9.233')
class TestExcaliburLowLevel(ut.TestCase):
    def create_excalibur_fem_stripe_datafile(self, fname, nframes, excalibur_data,scale):
        shape = (nframes,) + excalibur_data.fem_stripe_dimensions
        max_shape = (nframes,) + excalibur_data.fem_stripe_dimensions
        chunk = (1,) + excalibur_data.fem_stripe_dimensions
        with h5.File(fname, 'w', libver='latest') as f:
            dset = f.create_dataset('data', shape=shape, maxshape=max_shape, chunks=chunk, dtype='uint16')
            for data_value_index in np.arange(nframes):
                dset[data_value_index] = excalibur_data.generate_fem_stripe_image(data_value_index*scale)

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ["stripe_%d.h5" % stripe for stripe in range(1,7)]
        self.fname = [self.working_dir+ix for ix in self.fname]
        nframes = 5
        self.edata = ExcaliburData()
        k=0
        for raw_file in self.fname:
            self.create_excalibur_fem_stripe_datafile(raw_file, nframes, self.edata,k)
            k+=1

    def test_excalibur_low_level(self):

        excalibur_data = self.edata
        self.outfile = self.working_dir+'excalibur.h5'
        vdset_stripe_shape = (1,) + excalibur_data.fem_stripe_dimensions
        vdset_stripe_max_shape = (5, ) + excalibur_data.fem_stripe_dimensions
        vdset_shape = (5,
                       excalibur_data.fem_stripe_dimensions[0] * len(self.fname) + (10 * (len(self.fname)-1)),
                       excalibur_data.fem_stripe_dimensions[1])
        vdset_max_shape = (5,
                           excalibur_data.fem_stripe_dimensions[0] * len(self.fname) + (10 * (len(self.fname)-1)),
                           excalibur_data.fem_stripe_dimensions[1])
        vdset_y_offset = 0

        # Create the virtual dataset file
        with h5.File(self.outfile, 'w', libver='latest') as f:

            # Create the source dataset dataspace
            src_dspace = h5.h5s.create_simple(vdset_stripe_shape, vdset_stripe_max_shape)
            # Create the virtual dataset dataspace
            virt_dspace = h5.h5s.create_simple(vdset_shape, vdset_max_shape)

            # Create the virtual dataset property list
            dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
            dcpl.set_fill_value(np.array([0x01]))

            # Select the source dataset hyperslab
            src_dspace.select_hyperslab(start=(0, 0, 0), count=(1, 1, 1), block=vdset_stripe_max_shape)

            for raw_file in self.fname:
                # Select the virtual dataset hyperslab (for the source dataset)
                virt_dspace.select_hyperslab(start=(0, vdset_y_offset, 0),
                                             count=(1, 1, 1),
                                             block=vdset_stripe_max_shape)
                # Set the virtual dataset hyperslab to point to the real first dataset
                dcpl.set_virtual(virt_dspace, raw_file.encode('utf-8'),
                                 b"/data", src_dspace)
                vdset_y_offset += vdset_stripe_shape[1] + 10

            # Create the virtual dataset
            dset = h5.h5d.create(f.id, name=b"data",
                                 tid=h5.h5t.NATIVE_INT16, space=virt_dspace, dcpl=dcpl)
            assert(f['data'].fillvalue == 0x01)

        f = h5.File(self.outfile,'r')['data']
        self.assertEqual(f[3,100,0], 0.0)
        self.assertEqual(f[3,260,0], 1.0)
        self.assertEqual(f[3,350,0], 3.0)
        self.assertEqual(f[3,650,0], 6.0)
        self.assertEqual(f[3,900,0], 9.0)
        self.assertEqual(f[3,1150,0], 12.0)
        self.assertEqual(f[3,1450,0], 15.0)
        f.file.close()

    def tearDown(self):
        import os
        for f in self.fname:
            os.remove(f)
        os.remove(self.outfile)

'''
Unit test for the low level vds interface for percival
https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf
'''


@ut.skipUnless(h5.version.hdf5_version_tuple >= (1, 9, 233),
               'VDS requires HDF5 >= 1.9.233')
class TestPercivalLowLevel(ut.TestCase):

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['raw_file_1.h5','raw_file_2.h5','raw_file_3.h5']
        k = 0
        for outfile in self.fname:
            filename = self.working_dir + outfile
            f = h5.File(filename,'w')
            f['data'] = np.ones((20,200,200))*k
            k +=1
            f.close()

        f = h5.File(self.working_dir+'raw_file_4.h5','w')
        f['data'] = np.ones((19,200,200))*3
        self.fname.append('raw_file_4.h5')
        self.fname = [self.working_dir+ix for ix in self.fname]
        f.close()

    def test_percival_low_level(self):
        self.outfile = self.working_dir + 'percival.h5'
        with h5.File(self.outfile, 'w', libver='latest') as f:
            vdset_shape = (1,200,200)
            num = h5.h5s.UNLIMITED
            vdset_max_shape = (num,)+vdset_shape[1:]
            virt_dspace = h5.h5s.create_simple(vdset_shape, vdset_max_shape)
            dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
            dcpl.set_fill_value(np.array([-1]))
            # Create the source dataset dataspace
            k = 0
            for foo in self.fname:
                in_data = h5.File(foo, 'r')['data']
                src_shape = in_data.shape
                max_src_shape = (num,)+src_shape[1:]
                in_data.file.close()
                src_dspace = h5.h5s.create_simple(src_shape, max_src_shape)
                # Select the source dataset hyperslab
                src_dspace.select_hyperslab(start=(0, 0, 0),
                                            stride=(1,1,1),
                                            count=(num, 1, 1),
                                            block=(1,)+src_shape[1:])

                virt_dspace.select_hyperslab(start=(k, 0, 0),
                                             stride=(4,1,1),
                                             count=(num, 1, 1),
                                             block=(1,)+src_shape[1:])

                dcpl.set_virtual(virt_dspace, foo.encode('utf-8'), b'data', src_dspace)
                k+=1

            # Create the virtual dataset
            dset = h5.h5d.create(f.id, name=b"data", tid=h5.h5t.NATIVE_INT16, space=virt_dspace, dcpl=dcpl)

            f = h5.File(self.outfile,'r')
            sh = f['data'].shape
            line = f['data'][:8,100,100]
            foo = np.array(2*list(range(4)))
            f.close()
            self.assertEqual(sh,(79,200,200),)
            np.testing.assert_array_equal(line,foo)

    def tearDown(self):
        import os
        for f in self.fname:
            os.remove(f)
        os.remove(self.outfile)


@ut.skipUnless(h5.version.hdf5_version_tuple >= (1, 10, 2),
               'get_ / set_virtual_prefix requires HDF5 >= 1.10.2')
def test_virtual_prefix(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    src_file = h5.File(tmp_path / 'a' / 'src.h5', 'w')
    src_file['data'] = np.arange(10)

    vds_file = h5.File(tmp_path / 'b' / 'vds.h5', 'w')
    layout = h5.VirtualLayout(shape=(10,), dtype=np.int64)
    layout[:] = h5.VirtualSource('src.h5', 'data', shape=(10,))
    vds_file.create_virtual_dataset('data', layout, fillvalue=-1)

    # Path doesn't resolve
    np.testing.assert_array_equal(vds_file['data'], np.full(10, fill_value=-1))

    path_a = bytes(tmp_path / 'a')
    dapl = h5.h5p.create(h5.h5p.DATASET_ACCESS)
    dapl.set_virtual_prefix(path_a)
    vds_id = h5.h5d.open(vds_file.id, b'data', dapl=dapl)
    vds = h5.Dataset(vds_id)

    # Now it should find the source file and read the data correctly
    np.testing.assert_array_equal(vds[:], np.arange(10))
    # Check that get_virtual_prefix gives back what we put in
    assert vds.id.get_access_plist().get_virtual_prefix() == path_a
