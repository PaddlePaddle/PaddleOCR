# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

import numpy as np

import h5py
from h5py import h5t

from .common import TestCase, ut


class TestCompound(ut.TestCase):

    """
        Feature: Compound types can be created from Python dtypes
    """

    def test_ref(self):
        """ Reference types are correctly stored in compound types (issue 144)
        """
        dt = np.dtype([('a', h5py.ref_dtype), ('b', '<f4')])
        tid = h5t.py_create(dt, logical=True)
        t1, t2 = tid.get_member_type(0), tid.get_member_type(1)
        self.assertEqual(t1, h5t.STD_REF_OBJ)
        self.assertEqual(t2, h5t.IEEE_F32LE)
        self.assertEqual(tid.get_member_offset(0), 0)
        self.assertEqual(tid.get_member_offset(1), h5t.STD_REF_OBJ.get_size())

    def test_out_of_order_offsets(self):
        size = 20
        type_dict = {
            'names': ['f1', 'f2', 'f3'],
            'formats': ['<f4', '<i4', '<f8'],
            'offsets': [0, 16, 8]
        }

        expected_dtype = np.dtype(type_dict)

        tid = h5t.create(h5t.COMPOUND, size)
        for name, offset, dt in zip(
                type_dict["names"], type_dict["offsets"], type_dict["formats"]
        ):
            tid.insert(
                name.encode("utf8") if isinstance(name, str) else name,
                offset,
                h5t.py_create(dt)
            )

        self.assertEqual(tid.dtype, expected_dtype)
        self.assertEqual(tid.dtype.itemsize, size)


class TestTypeFloatID(TestCase):
    """Test TypeFloatID."""

    def test_custom_float_promotion(self):
        """Custom floats are correctly promoted to standard floats on read."""

        # This test uses the low-level API, so we need names as byte strings
        test_filename = self.mktemp().encode()
        dataset = b'DS1'
        dataset2 = b'DS2'
        dataset3 = b'DS3'
        dataset4 = b'DS4'
        dataset5 = b'DS5'

        dims = (4, 7)

        wdata = np.array([[-1.50066626e-09,   1.40062184e-09,   1.81216819e-10,
                           4.01087163e-10,   4.27917257e-10,  -7.04858394e-11,
                           5.74800652e-10],
                          [-1.50066626e-09,   4.86579665e-10,   3.42879503e-10,
                           5.12045517e-10,   5.10226528e-10,   2.24190444e-10,
                           3.93356459e-10],
                          [-1.50066626e-09,   5.24778443e-10,   8.19454726e-10,
                           1.28966349e-09,   1.68483894e-10,   5.71276360e-11,
                           -1.08684617e-10],
                          [-1.50066626e-09,  -1.08343556e-10,  -1.58934199e-10,
                           8.52196536e-10,   6.18456397e-10,   6.16637408e-10,
                           1.31694833e-09]], dtype=np.float32)

        wdata2 = np.array([[-1.50066626e-09,   5.63886715e-10,  -8.74251782e-11,
                            1.32558853e-10,   1.59161573e-10,   2.29420039e-10,
                            -7.24185156e-11],
                           [-1.50066626e-09,   1.87810656e-10,   7.74889486e-10,
                            3.95630195e-10,   9.42236511e-10,   8.38554115e-10,
                            -8.71978045e-11],
                           [-1.50066626e-09,   6.20275387e-10,   7.34871719e-10,
                            6.64840627e-10,   2.64662958e-10,   1.05319486e-09,
                            1.68256520e-10],
                           [-1.50066626e-09,   1.67347025e-10,   5.12045517e-10,
                            3.36513040e-10,   1.02545528e-10,   1.28784450e-09,
                            4.06089384e-10]], dtype=np.float32)

        # Create a new file using the default properties.
        fid = h5py.h5f.create(test_filename)
        # Create the dataspace.  No maximum size parameter needed.
        space = h5py.h5s.create_simple(dims)

        # create a custom type with larger bias
        mytype = h5t.IEEE_F16LE.copy()
        mytype.set_fields(14, 9, 5, 0, 9)
        mytype.set_size(2)
        mytype.set_ebias(53)
        mytype.lock()

        dset = h5py.h5d.create(fid, dataset, mytype, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata)

        del dset

        # create a custom type with larger exponent
        mytype2 = h5t.IEEE_F16LE.copy()
        mytype2.set_fields(15, 9, 6, 0, 9)
        mytype2.set_size(2)
        mytype2.set_ebias(53)
        mytype2.lock()

        dset = h5py.h5d.create(fid, dataset2, mytype2, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)

        del dset

        # create a custom type which reimplements 16-bit floats
        mytype3 = h5t.IEEE_F16LE.copy()
        mytype3.set_fields(15, 10, 5, 0, 10)
        mytype3.set_size(2)
        mytype3.set_ebias(15)
        mytype3.lock()

        dset = h5py.h5d.create(fid, dataset3, mytype3, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)

        del dset

        # create a custom type with larger bias
        mytype4 = h5t.IEEE_F16LE.copy()
        mytype4.set_fields(15, 10, 5, 0, 10)
        mytype4.set_size(2)
        mytype4.set_ebias(258)
        mytype4.lock()

        dset = h5py.h5d.create(fid, dataset4, mytype4, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)

        del dset

        # create a dataset with long doubles
        dset = h5py.h5d.create(fid, dataset5, h5t.NATIVE_LDOUBLE, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)

        # Explicitly close and release resources.
        del space
        del dset
        del fid

        f = h5py.File(test_filename, 'r')

        # ebias promotion to float32
        values = f[dataset][:]
        np.testing.assert_array_equal(values, wdata)
        self.assertEqual(values.dtype, np.dtype('<f4'))

        # esize promotion to float32
        values = f[dataset2][:]
        np.testing.assert_array_equal(values, wdata2)
        self.assertEqual(values.dtype, np.dtype('<f4'))

        # regular half floats
        dset = f[dataset3]
        try:
            self.assertEqual(dset.dtype, np.dtype('<f2'))
        except AttributeError:
            self.assertEqual(dset.dtype, np.dtype('<f4'))

        # ebias promotion to float64
        dset = f[dataset4]
        self.assertEqual(dset.dtype, np.dtype('<f8'))

        # long double floats
        dset = f[dataset5]
        self.assertEqual(dset.dtype, np.longdouble)
