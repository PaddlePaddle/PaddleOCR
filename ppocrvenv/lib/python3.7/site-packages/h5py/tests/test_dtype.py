"""
    Tests for converting between numpy dtypes and h5py data types
"""

from itertools import count
import platform
import numpy as np
import h5py
try:
    import tables
except ImportError:
    tables = None

from .common import ut, TestCase

UNSUPPORTED_LONG_DOUBLE = ('i386', 'i486', 'i586', 'i686', 'ppc64le')
UNSUPPORTED_LONG_DOUBLE_TYPES = ('float96', 'float128', 'complex192',
                                 'complex256')


class TestVlen(TestCase):

    """
        Check that storage of vlen strings is carried out correctly.
    """
    def assertVlenArrayEqual(self, dset, arr, message=None, precision=None):
        assert dset.shape == arr.shape, \
            "Shape mismatch (%s vs %s)%s" % (dset.shape, arr.shape, message)
        for (i, d, a) in zip(count(), dset, arr):
            self.assertArrayEqual(d, a, message, precision)

    def test_compound(self):

        fields = []
        fields.append(('field_1', h5py.string_dtype()))
        fields.append(('field_2', np.int32))
        dt = np.dtype(fields)
        self.f['mytype'] = np.dtype(dt)
        dt_out = self.f['mytype'].dtype.fields['field_1'][0]
        string_inf = h5py.check_string_dtype(dt_out)
        self.assertEqual(string_inf.encoding, 'utf-8')

    def test_compound_vlen_bool(self):
        vidt = h5py.vlen_dtype(np.uint8)
        def a(items):
            return np.array(items, dtype=np.uint8)

        f = self.f

        dt_vb = np.dtype([
            ('foo', vidt),
            ('logical', bool)])
        vb = f.create_dataset('dt_vb', shape=(4,), dtype=dt_vb)
        data = np.array([(a([1, 2, 3]), True),
                         (a([1    ]), False),
                         (a([1, 5  ]), True),
                         (a([],), False), ],
                     dtype=dt_vb)
        vb[:] = data
        actual = f['dt_vb'][:]
        self.assertVlenArrayEqual(data['foo'], actual['foo'])
        self.assertArrayEqual(data['logical'], actual['logical'])

        dt_vv = np.dtype([
            ('foo', vidt),
            ('bar', vidt)])
        f.create_dataset('dt_vv', shape=(4,), dtype=dt_vv)

        dt_vvb = np.dtype([
            ('foo', vidt),
            ('bar', vidt),
            ('logical', bool)])
        vvb = f.create_dataset('dt_vvb', shape=(2,), dtype=dt_vvb)

        dt_bvv = np.dtype([
            ('logical', bool),
            ('foo', vidt),
            ('bar', vidt)])
        bvv = f.create_dataset('dt_bvv', shape=(2,), dtype=dt_bvv)
        data = np.array([(True, a([1, 2, 3]), a([1, 2])),
                         (False, a([]), a([2, 4, 6])), ],
                         dtype=bvv)
        bvv[:] = data
        actual = bvv[:]
        self.assertVlenArrayEqual(data['foo'], actual['foo'])
        self.assertVlenArrayEqual(data['bar'], actual['bar'])
        self.assertArrayEqual(data['logical'], actual['logical'])

    def test_compound_vlen_enum(self):
        eidt = h5py.enum_dtype({'OFF': 0, 'ON': 1}, basetype=np.uint8)
        vidt = h5py.vlen_dtype(np.uint8)
        def a(items):
            return np.array(items, dtype=np.uint8)

        f = self.f

        dt_vve = np.dtype([
            ('foo', vidt),
            ('bar', vidt),
            ('switch', eidt)])
        vve = f.create_dataset('dt_vve', shape=(2,), dtype=dt_vve)
        data = np.array([(a([1, 2, 3]), a([1, 2]), 1),
                         (a([]), a([2, 4, 6]), 0), ],
                         dtype=dt_vve)
        vve[:] = data
        actual = vve[:]
        self.assertVlenArrayEqual(data['foo'], actual['foo'])
        self.assertVlenArrayEqual(data['bar'], actual['bar'])
        self.assertArrayEqual(data['switch'], actual['switch'])

    def test_vlen_enum(self):
        fname = self.mktemp()
        arr1 = [[1], [1, 2]]
        dt1 = h5py.vlen_dtype(h5py.enum_dtype(dict(foo=1, bar=2), 'i'))

        with h5py.File(fname, 'w') as f:
            df1 = f.create_dataset('test', (len(arr1),), dtype=dt1)
            df1[:] = np.array(arr1, dtype=object)

        with h5py.File(fname, 'r') as f:
            df2 = f['test']
            dt2 = df2.dtype
            arr2 = [e.tolist() for e in df2[:]]

        self.assertEqual(arr1, arr2)
        self.assertEqual(h5py.check_enum_dtype(h5py.check_vlen_dtype(dt1)),
                         h5py.check_enum_dtype(h5py.check_vlen_dtype(dt2)))


class TestEmptyVlen(TestCase):
    def test_write_empty_vlen(self):
        fname = self.mktemp()
        with h5py.File(fname, 'w') as f:
            d = np.core.records.fromarrays([[], []], names='a,b', formats='|V16,O')
            dset = f.create_dataset('test', data=d, dtype=[('a', '|V16'), ('b', h5py.special_dtype(vlen=np.float_))])
            self.assertEqual(dset.size, 0)


class TestExplicitCast(TestCase):
    def test_f2_casting(self):
        fname = self.mktemp()

        np.random.seed(1)
        A = np.random.rand(1500, 20)

        # Save to HDF5 file
        with h5py.File(fname, "w") as Fid:
            Fid.create_dataset("Data", data=A, dtype='f2')

        with h5py.File(fname, "r") as Fid:
            B = Fid["Data"][:]

        # Compare
        self.assertTrue(np.all(A.astype('f2') == B))


class TestOffsets(TestCase):
    """
        Check that compound members with aligned or manual offsets are handled
        correctly.
    """

    def test_compound_vlen(self):
        vidt = h5py.vlen_dtype(np.uint8)
        eidt = h5py.enum_dtype({'OFF': 0, 'ON': 1}, basetype=np.uint8)

        for np_align in (False, True):
            dt = np.dtype([
                ('a', eidt),
                ('foo', vidt),
                ('bar', vidt),
                ('switch', eidt)], align=np_align)
            np_offsets = [dt.fields[i][1] for i in dt.names]

            for logical in (False, True):
                if logical and np_align:
                    # Vlen types have different size in the numpy struct
                    self.assertRaises(TypeError, h5py.h5t.py_create, dt,
                            logical=logical)
                else:
                    ht = h5py.h5t.py_create(dt, logical=logical)
                    offsets = [ht.get_member_offset(i)
                               for i in range(ht.get_nmembers())]
                    if np_align:
                        self.assertEqual(np_offsets, offsets)

    def test_aligned_offsets(self):
        dt = np.dtype('i4,i8,i2', align=True)
        ht = h5py.h5t.py_create(dt)
        self.assertEqual(dt.itemsize, ht.get_size())
        self.assertEqual(
            [dt.fields[i][1] for i in dt.names],
            [ht.get_member_offset(i) for i in range(ht.get_nmembers())]
        )

    def test_aligned_data(self):
        dt = np.dtype('i4,f8,i2', align=True)
        data = np.zeros(10, dtype=dt)

        data['f0'] = np.array(np.random.randint(-100, 100, size=data.size),
                              dtype='i4')
        data['f1'] = np.random.rand(data.size)
        data['f2'] = np.array(np.random.randint(-100, 100, size=data.size),
                              dtype='i2')

        fname = self.mktemp()

        with h5py.File(fname, 'w') as f:
            f['data'] = data

        with h5py.File(fname, 'r') as f:
            self.assertArrayEqual(f['data'], data)

    def test_compound_robustness(self):
        # make an out of order compound type with gaps in it, and larger itemsize than minimum
        # Idea is to be robust to type descriptions we *could* get out of HDF5 files, from custom descriptions
        # of types in addition to numpy's flakey history on unaligned fields with non-standard or padded layouts.
        fields = [
            ('f0', np.float64, 25),
            ('f1', np.uint64, 9),
            ('f2', np.uint32, 0),
            ('f3', np.uint16, 5)
        ]
        lastfield = fields[np.argmax([ x[2] for x in fields ])]
        itemsize = lastfield[2] + np.dtype(lastfield[1]).itemsize + 6
        extract_index = lambda index, sequence: [ x[index] for x in sequence ]

        dt = np.dtype({
            'names' : extract_index(0, fields),
            'formats' : extract_index(1, fields),
            'offsets' : extract_index(2, fields),
            # 'aligned': False, - already defaults to False
            'itemsize': itemsize
        })

        self.assertTrue(dt.itemsize == itemsize)
        data = np.zeros(10, dtype=dt)

        # don't trust numpy struct handling, keep fields out of band in case content insertion is erroneous
        # yes... this has also been known to happen.
        f1 = np.array([1 + i * 4 for i in range(data.shape[0])], dtype=dt.fields['f1'][0])
        f2 = np.array([2 + i * 4 for i in range(data.shape[0])], dtype=dt.fields['f2'][0])
        f3 = np.array([3 + i * 4 for i in range(data.shape[0])], dtype=dt.fields['f3'][0])
        f0c = 3.14
        data['f0'] = f0c
        data['f3'] = f3
        data['f1'] = f1
        data['f2'] = f2

        # numpy consistency checks
        self.assertTrue(np.all(data['f0'] == f0c))
        self.assertArrayEqual(data['f3'], f3)
        self.assertArrayEqual(data['f1'], f1)
        self.assertArrayEqual(data['f2'], f2)

        fname = self.mktemp()

        with h5py.File(fname, 'w') as fd:
            fd.create_dataset('data', data=data)

        with h5py.File(fname, 'r') as fd:
            readback = fd['data']
            self.assertTrue(readback.dtype == dt)
            self.assertArrayEqual(readback, data)
            self.assertTrue(np.all(readback['f0'] == f0c))
            self.assertArrayEqual(readback['f1'], f1)
            self.assertArrayEqual(readback['f2'], f2)
            self.assertArrayEqual(readback['f3'], f3)

    def test_out_of_order_offsets(self):
        dt = np.dtype({
            'names' : ['f1', 'f2', 'f3'],
            'formats' : ['<f4', '<i4', '<f8'],
            'offsets' : [0, 16, 8]
        })
        data = np.zeros(10, dtype=dt)
        data['f1'] = np.random.rand(data.size)
        data['f2'] = np.random.randint(-10, 11, data.size)
        data['f3'] = np.random.rand(data.size) * -1

        fname = self.mktemp()

        with h5py.File(fname, 'w') as fd:
            fd.create_dataset('data', data=data)

        with h5py.File(fname, 'r') as fd:
            self.assertArrayEqual(fd['data'], data)

    def test_float_round_tripping(self):
        dtypes = set(f for f in np.sctypeDict.values()
                     if (np.issubdtype(f, np.floating) or
                         np.issubdtype(f, np.complexfloating)))

        unsupported_types = []
        if platform.machine() in UNSUPPORTED_LONG_DOUBLE:
            for x in UNSUPPORTED_LONG_DOUBLE_TYPES:
                if hasattr(np, x):
                    unsupported_types.append(getattr(np, x))
        dtype_dset_map = {str(j): d
                          for j, d in enumerate(dtypes)
                          if d not in unsupported_types}

        fname = self.mktemp()

        with h5py.File(fname, 'w') as f:
            for n, d in dtype_dset_map.items():
                data = np.zeros(10, dtype=d)
                data[...] = np.arange(10)

                f.create_dataset(n, data=data)

        with h5py.File(fname, 'r') as f:
            for n, d in dtype_dset_map.items():
                ldata = f[n][:]
                self.assertEqual(ldata.dtype, d)


class TestStrings(TestCase):
    def test_vlen_utf8(self):
        dt = h5py.string_dtype()

        string_info = h5py.check_string_dtype(dt)
        assert string_info.encoding == 'utf-8'
        assert string_info.length is None
        assert h5py.check_vlen_dtype(dt) is str

    def test_vlen_ascii(self):
        dt = h5py.string_dtype(encoding='ascii')

        string_info = h5py.check_string_dtype(dt)
        assert string_info.encoding == 'ascii'
        assert string_info.length is None
        assert h5py.check_vlen_dtype(dt) is bytes

    def test_fixed_utf8(self):
        dt = h5py.string_dtype(length=10)

        string_info = h5py.check_string_dtype(dt)
        assert string_info.encoding == 'utf-8'
        assert string_info.length == 10
        assert h5py.check_vlen_dtype(dt) is None

    def test_fixed_ascii(self):
        dt = h5py.string_dtype(encoding='ascii', length=10)

        string_info = h5py.check_string_dtype(dt)
        assert string_info.encoding == 'ascii'
        assert string_info.length == 10
        assert h5py.check_vlen_dtype(dt) is None

class TestDateTime(TestCase):
    datetime_units = [
        # Dates
        'Y', 'M', 'D',
        # Times
        'h', 'm', 's', 'ms', 'us',
        'ns', 'ps', 'fs', 'as',
    ]

    def test_datetime(self):
        fname = self.mktemp()

        for dt_unit in self.datetime_units:
            for dt_order in ['<', '>']:
                dt_descr = f'{dt_order}M8[{dt_unit}]'
                dt = h5py.opaque_dtype(np.dtype(dt_descr))
                arr = np.array([0], dtype=np.int64).view(dtype=dt)

                with h5py.File(fname, 'w') as f:
                    dset = f.create_dataset("default", data=arr, dtype=dt)
                    self.assertArrayEqual(arr, dset)
                    self.assertEqual(arr.dtype, dset.dtype)

    def test_timedelta(self):
        fname = self.mktemp()

        for dt_unit in self.datetime_units:
            for dt_order in ['<', '>']:
                dt_descr = f'{dt_order}m8[{dt_unit}]'
                dt = h5py.opaque_dtype(np.dtype(dt_descr))
                arr = np.array([np.timedelta64(500, dt_unit)], dtype=dt)

                with h5py.File(fname, 'w') as f:
                    dset = f.create_dataset("default", data=arr, dtype=dt)
                    self.assertArrayEqual(arr, dset)
                    self.assertEqual(arr.dtype, dset.dtype)

@ut.skipUnless(tables is not None, 'tables is required')
class TestBitfield(TestCase):

    """
    Test H5T_NATIVE_B8 reading
    """

    def test_b8_bool(self):
        arr1 = np.array([False, True], dtype=bool)
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.uint8
        )
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.uint8,
            cast_dtype=np.uint8
        )

    def test_b8_bool_compound(self):
        arr1 = np.array([(False,), (True,)], dtype=np.dtype([('x', '?')]))
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.dtype([('x', 'u1')])
        )
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.dtype([('x', 'u1')]),
            cast_dtype=np.dtype([('x', 'u1')])
        )

    def test_b8_bool_compound_nested(self):
        arr1 = np.array(
            [(True, (True, False)), (True, (False, True))],
            dtype=np.dtype([('x', '?'), ('y', [('a', '?'), ('b', '?')])]),
        )
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.dtype(
                [('x', 'u1'), ('y', [('a', 'u1'), ('b', 'u1')])]
            )
        )
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.dtype(
                [('x', 'u1'), ('y', [('a', 'u1'), ('b', 'u1')])]
            ),
            cast_dtype=np.dtype([('x', 'u1'), ('y', [('a', 'u1'), ('b', 'u1')])]),
        )

    def test_b8_bool_compound_mixed_types(self):
        arr1 = np.array(
            [(True, 0.5), (False, 0.2)], dtype=np.dtype([('x','?'), ('y', '<f8')])
        )
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.dtype([('x', 'u1'), ('y', '<f8')])
        )
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.dtype([('x', 'u1'), ('y', '<f8')]),
            cast_dtype=np.dtype([('x', 'u1'), ('y', '<f8')])
        )

    def test_b8_bool_array(self):
        arr1 = np.array(
            [((True, True, False),), ((True, False, True),)],
            dtype=np.dtype([('x', ('?', (3,)))]),
        )
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.dtype([('x', ('u1', (3,)))])
        )
        self._test_b8(
            arr1,
            expected_default_cast_dtype=np.dtype([('x', ('u1', (3,)))]),
            cast_dtype=np.dtype([('x', ('?', (3,)))]),
        )

    def _test_b8(self, arr1, expected_default_cast_dtype, cast_dtype=None):
        path = self.mktemp()

        with tables.open_file(path, 'w') as f:
            if arr1.dtype.names:
                f.create_table('/', 'test', obj=arr1)
            else:
                f.create_array('/', 'test', obj=arr1)

        with h5py.File(path, 'r') as f:
            dset = f['test']

            # This should do an implicit uint8 cast
            # Expect that the "No NumPy equivalent for TypeBitfieldID exists"
            # error is not thrown.
            arr2 = dset[:]

            self.assertArrayEqual(
                arr2,
                arr1.astype(expected_default_cast_dtype, copy=False)
            )

            # read cast dset and make sure it's equal
            if cast_dtype is None:
                cast_dtype = arr1.dtype
            arr3 = dset.astype(cast_dtype)[:]
            self.assertArrayEqual(arr3, arr1.astype(cast_dtype, copy=False))

    def test_b16_uint16(self):
        arr1 = np.arange(10, dtype=np.uint16)
        path = self.mktemp()
        with h5py.File(path, 'w') as f:
            space = h5py.h5s.create_simple(arr1.shape)
            dset_id = h5py.h5d.create(f.id, b'test', h5py.h5t.STD_B16LE, space)
            dset = h5py.Dataset(dset_id)
            dset[:] = arr1

        with h5py.File(path, 'r') as f:
            dset = f['test']
            self.assertArrayEqual(dset[:], arr1)

def test_opaque(writable_file):
    # opaque without an h5py tag corresponds to numpy void dtypes
    arr = np.zeros(3, dtype='V2')
    ds = writable_file.create_dataset('v', data=arr)
    assert isinstance(ds.id.get_type(), h5py.h5t.TypeOpaqueID)
    assert ds.id.get_type().get_size() == 2
    np.testing.assert_array_equal(ds[:], arr)
