from tempfile import mkdtemp, mktemp
import os
import io
import shutil
import textwrap

import numpy as np
from numpy import array, transpose, pi
from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_equal, assert_array_almost_equal)
import pytest
from pytest import raises as assert_raises

import scipy.sparse
from scipy.io.mmio import mminfo, mmread, mmwrite

parametrize_args = [('integer', 'int'),
                    ('unsigned-integer', 'uint')]


class TestMMIOArray:
    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def check(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(a, b)

    def check_exact(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_equal(a, b)

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_integer(self, typeval, dtype):
        self.check_exact(array([[1, 2], [3, 4]], dtype=dtype),
                         (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_32bit_integer(self, typeval, dtype):
        a = array([[2**31-1, 2**31-2], [2**31-3, 2**31-4]], dtype=dtype)
        self.check_exact(a, (2, 2, 4, 'array', typeval, 'general'))

    def test_64bit_integer(self):
        a = array([[2**31, 2**32], [2**63-2, 2**63-1]], dtype=np.int64)
        if (np.intp(0).itemsize < 8):
            assert_raises(OverflowError, mmwrite, self.fn, a)
        else:
            self.check_exact(a, (2, 2, 4, 'array', 'integer', 'general'))

    def test_64bit_unsigned_integer(self):
        a = array([[2**31, 2**32], [2**64-2, 2**64-1]], dtype=np.uint64)
        self.check_exact(a, (2, 2, 4, 'array', 'unsigned-integer', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_upper_triangle_integer(self, typeval, dtype):
        self.check_exact(array([[0, 1], [0, 0]], dtype=dtype),
                         (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_lower_triangle_integer(self, typeval, dtype):
        self.check_exact(array([[0, 0], [1, 0]], dtype=dtype),
                         (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_rectangular_integer(self, typeval, dtype):
        self.check_exact(array([[1, 2, 3], [4, 5, 6]], dtype=dtype),
                         (2, 3, 6, 'array', typeval, 'general'))

    def test_simple_rectangular_float(self):
        self.check([[1, 2], [3.5, 4], [5, 6]],
                   (3, 2, 6, 'array', 'real', 'general'))

    def test_simple_float(self):
        self.check([[1, 2], [3, 4.0]],
                   (2, 2, 4, 'array', 'real', 'general'))

    def test_simple_complex(self):
        self.check([[1, 2], [3, 4j]],
                   (2, 2, 4, 'array', 'complex', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_symmetric_integer(self, typeval, dtype):
        self.check_exact(array([[1, 2], [2, 4]], dtype=dtype),
                         (2, 2, 4, 'array', typeval, 'symmetric'))

    def test_simple_skew_symmetric_integer(self):
        self.check_exact([[0, 2], [-2, 0]],
                         (2, 2, 4, 'array', 'integer', 'skew-symmetric'))

    def test_simple_skew_symmetric_float(self):
        self.check(array([[0, 2], [-2.0, 0.0]], 'f'),
                   (2, 2, 4, 'array', 'real', 'skew-symmetric'))

    def test_simple_hermitian_complex(self):
        self.check([[1, 2+3j], [2-3j, 4]],
                   (2, 2, 4, 'array', 'complex', 'hermitian'))

    def test_random_symmetric_float(self):
        sz = (20, 20)
        a = np.random.random(sz)
        a = a + transpose(a)
        self.check(a, (20, 20, 400, 'array', 'real', 'symmetric'))

    def test_random_rectangular_float(self):
        sz = (20, 15)
        a = np.random.random(sz)
        self.check(a, (20, 15, 300, 'array', 'real', 'general'))

    def test_bad_number_of_array_header_fields(self):
        s = """\
            %%MatrixMarket matrix array real general
              3  3 999
            1.0
            2.0
            3.0
            4.0
            5.0
            6.0
            7.0
            8.0
            9.0
            """
        text = textwrap.dedent(s).encode('ascii')
        with pytest.raises(ValueError, match='not of length 2'):
            scipy.io.mmread(io.BytesIO(text))

    def test_gh13634_non_skew_symmetric_int(self):
        self.check_exact(array([[1, 2], [-2, 99]], dtype=np.int32),
                         (2, 2, 4, 'array', 'integer', 'general'))

    def test_gh13634_non_skew_symmetric_float(self):
        self.check(array([[1, 2], [-2, 99.]], dtype=np.float32),
                   (2, 2, 4, 'array', 'real', 'general'))


class TestMMIOSparseCSR(TestMMIOArray):
    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def check(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(a.todense(), b.todense())

    def check_exact(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_equal(a.todense(), b.todense())

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[1, 2], [3, 4]], dtype=dtype),
                         (2, 2, 4, 'coordinate', typeval, 'general'))

    def test_32bit_integer(self):
        a = scipy.sparse.csr_matrix(array([[2**31-1, -2**31+2],
                                           [2**31-3, 2**31-4]],
                                          dtype=np.int32))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    def test_64bit_integer(self):
        a = scipy.sparse.csr_matrix(array([[2**32+1, 2**32+1],
                                           [-2**63+2, 2**63-2]],
                                          dtype=np.int64))
        if (np.intp(0).itemsize < 8):
            assert_raises(OverflowError, mmwrite, self.fn, a)
        else:
            self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    def test_32bit_unsigned_integer(self):
        a = scipy.sparse.csr_matrix(array([[2**31-1, 2**31-2],
                                           [2**31-3, 2**31-4]],
                                          dtype=np.uint32))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'unsigned-integer', 'general'))

    def test_64bit_unsigned_integer(self):
        a = scipy.sparse.csr_matrix(array([[2**32+1, 2**32+1],
                                           [2**64-2, 2**64-1]],
                                          dtype=np.uint64))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'unsigned-integer', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_upper_triangle_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[0, 1], [0, 0]], dtype=dtype),
                         (2, 2, 1, 'coordinate', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_lower_triangle_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[0, 0], [1, 0]], dtype=dtype),
                         (2, 2, 1, 'coordinate', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_rectangular_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]], dtype=dtype),
                         (2, 3, 6, 'coordinate', typeval, 'general'))

    def test_simple_rectangular_float(self):
        self.check(scipy.sparse.csr_matrix([[1, 2], [3.5, 4], [5, 6]]),
                   (3, 2, 6, 'coordinate', 'real', 'general'))

    def test_simple_float(self):
        self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4.0]]),
                   (2, 2, 4, 'coordinate', 'real', 'general'))

    def test_simple_complex(self):
        self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4j]]),
                   (2, 2, 4, 'coordinate', 'complex', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_symmetric_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[1, 2], [2, 4]], dtype=dtype),
                         (2, 2, 3, 'coordinate', typeval, 'symmetric'))

    def test_simple_skew_symmetric_integer(self):
        self.check_exact(scipy.sparse.csr_matrix([[0, 2], [-2, 0]]),
                         (2, 2, 1, 'coordinate', 'integer', 'skew-symmetric'))

    def test_simple_skew_symmetric_float(self):
        self.check(scipy.sparse.csr_matrix(array([[0, 2], [-2.0, 0]], 'f')),
                   (2, 2, 1, 'coordinate', 'real', 'skew-symmetric'))

    def test_simple_hermitian_complex(self):
        self.check(scipy.sparse.csr_matrix([[1, 2+3j], [2-3j, 4]]),
                   (2, 2, 3, 'coordinate', 'complex', 'hermitian'))

    def test_random_symmetric_float(self):
        sz = (20, 20)
        a = np.random.random(sz)
        a = a + transpose(a)
        a = scipy.sparse.csr_matrix(a)
        self.check(a, (20, 20, 210, 'coordinate', 'real', 'symmetric'))

    def test_random_rectangular_float(self):
        sz = (20, 15)
        a = np.random.random(sz)
        a = scipy.sparse.csr_matrix(a)
        self.check(a, (20, 15, 300, 'coordinate', 'real', 'general'))

    def test_simple_pattern(self):
        a = scipy.sparse.csr_matrix([[0, 1.5], [3.0, 2.5]])
        p = np.zeros_like(a.todense())
        p[a.todense() > 0] = 1
        info = (2, 2, 3, 'coordinate', 'pattern', 'general')
        mmwrite(self.fn, a, field='pattern')
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(p, b.todense())

    def test_gh13634_non_skew_symmetric_int(self):
        a = scipy.sparse.csr_matrix([[1, 2], [-2, 99]], dtype=np.int32)
        self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    def test_gh13634_non_skew_symmetric_float(self):
        a = scipy.sparse.csr_matrix([[1, 2], [-2, 99.]], dtype=np.float32)
        self.check(a, (2, 2, 4, 'coordinate', 'real', 'general'))


_32bit_integer_dense_example = '''\
%%MatrixMarket matrix array integer general
2  2
2147483647
2147483646
2147483647
2147483646
'''

_32bit_integer_sparse_example = '''\
%%MatrixMarket matrix coordinate integer symmetric
2  2  2
1  1  2147483647
2  2  2147483646
'''

_64bit_integer_dense_example = '''\
%%MatrixMarket matrix array integer general
2  2
          2147483648
-9223372036854775806
         -2147483648
 9223372036854775807
'''

_64bit_integer_sparse_general_example = '''\
%%MatrixMarket matrix coordinate integer general
2  2  3
1  1           2147483648
1  2  9223372036854775807
2  2  9223372036854775807
'''

_64bit_integer_sparse_symmetric_example = '''\
%%MatrixMarket matrix coordinate integer symmetric
2  2  3
1  1            2147483648
1  2  -9223372036854775807
2  2   9223372036854775807
'''

_64bit_integer_sparse_skew_example = '''\
%%MatrixMarket matrix coordinate integer skew-symmetric
2  2  3
1  1            2147483648
1  2  -9223372036854775807
2  2   9223372036854775807
'''

_over64bit_integer_dense_example = '''\
%%MatrixMarket matrix array integer general
2  2
         2147483648
9223372036854775807
         2147483648
9223372036854775808
'''

_over64bit_integer_sparse_example = '''\
%%MatrixMarket matrix coordinate integer symmetric
2  2  2
1  1            2147483648
2  2  19223372036854775808
'''


class TestMMIOReadLargeIntegers:
    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def check_read(self, example, a, info, dense, over32, over64):
        with open(self.fn, 'w') as f:
            f.write(example)
        assert_equal(mminfo(self.fn), info)
        if (over32 and (np.intp(0).itemsize < 8)) or over64:
            assert_raises(OverflowError, mmread, self.fn)
        else:
            b = mmread(self.fn)
            if not dense:
                b = b.todense()
            assert_equal(a, b)

    def test_read_32bit_integer_dense(self):
        a = array([[2**31-1, 2**31-1],
                   [2**31-2, 2**31-2]], dtype=np.int64)
        self.check_read(_32bit_integer_dense_example,
                        a,
                        (2, 2, 4, 'array', 'integer', 'general'),
                        dense=True,
                        over32=False,
                        over64=False)

    def test_read_32bit_integer_sparse(self):
        a = array([[2**31-1, 0],
                   [0, 2**31-2]], dtype=np.int64)
        self.check_read(_32bit_integer_sparse_example,
                        a,
                        (2, 2, 2, 'coordinate', 'integer', 'symmetric'),
                        dense=False,
                        over32=False,
                        over64=False)

    def test_read_64bit_integer_dense(self):
        a = array([[2**31, -2**31],
                   [-2**63+2, 2**63-1]], dtype=np.int64)
        self.check_read(_64bit_integer_dense_example,
                        a,
                        (2, 2, 4, 'array', 'integer', 'general'),
                        dense=True,
                        over32=True,
                        over64=False)

    def test_read_64bit_integer_sparse_general(self):
        a = array([[2**31, 2**63-1],
                   [0, 2**63-1]], dtype=np.int64)
        self.check_read(_64bit_integer_sparse_general_example,
                        a,
                        (2, 2, 3, 'coordinate', 'integer', 'general'),
                        dense=False,
                        over32=True,
                        over64=False)

    def test_read_64bit_integer_sparse_symmetric(self):
        a = array([[2**31, -2**63+1],
                   [-2**63+1, 2**63-1]], dtype=np.int64)
        self.check_read(_64bit_integer_sparse_symmetric_example,
                        a,
                        (2, 2, 3, 'coordinate', 'integer', 'symmetric'),
                        dense=False,
                        over32=True,
                        over64=False)

    def test_read_64bit_integer_sparse_skew(self):
        a = array([[2**31, -2**63+1],
                   [2**63-1, 2**63-1]], dtype=np.int64)
        self.check_read(_64bit_integer_sparse_skew_example,
                        a,
                        (2, 2, 3, 'coordinate', 'integer', 'skew-symmetric'),
                        dense=False,
                        over32=True,
                        over64=False)

    def test_read_over64bit_integer_dense(self):
        self.check_read(_over64bit_integer_dense_example,
                        None,
                        (2, 2, 4, 'array', 'integer', 'general'),
                        dense=True,
                        over32=True,
                        over64=True)

    def test_read_over64bit_integer_sparse(self):
        self.check_read(_over64bit_integer_sparse_example,
                        None,
                        (2, 2, 2, 'coordinate', 'integer', 'symmetric'),
                        dense=False,
                        over32=True,
                        over64=True)


_general_example = '''\
%%MatrixMarket matrix coordinate real general
%=================================================================================
%
% This ASCII file represents a sparse MxN matrix with L
% nonzeros in the following Matrix Market format:
%
% +----------------------------------------------+
% |%%MatrixMarket matrix coordinate real general | <--- header line
% |%                                             | <--+
% |% comments                                    |    |-- 0 or more comment lines
% |%                                             | <--+
% |    M  N  L                                   | <--- rows, columns, entries
% |    I1  J1  A(I1, J1)                         | <--+
% |    I2  J2  A(I2, J2)                         |    |
% |    I3  J3  A(I3, J3)                         |    |-- L lines
% |        . . .                                 |    |
% |    IL JL  A(IL, JL)                          | <--+
% +----------------------------------------------+
%
% Indices are 1-based, i.e. A(1,1) is the first element.
%
%=================================================================================
  5  5  8
    1     1   1.000e+00
    2     2   1.050e+01
    3     3   1.500e-02
    1     4   6.000e+00
    4     2   2.505e+02
    4     4  -2.800e+02
    4     5   3.332e+01
    5     5   1.200e+01
'''

_hermitian_example = '''\
%%MatrixMarket matrix coordinate complex hermitian
  5  5  7
    1     1     1.0      0
    2     2    10.5      0
    4     2   250.5     22.22
    3     3     1.5e-2   0
    4     4    -2.8e2    0
    5     5    12.       0
    5     4     0       33.32
'''

_skew_example = '''\
%%MatrixMarket matrix coordinate real skew-symmetric
  5  5  7
    1     1     1.0
    2     2    10.5
    4     2   250.5
    3     3     1.5e-2
    4     4    -2.8e2
    5     5    12.
    5     4     0
'''

_symmetric_example = '''\
%%MatrixMarket matrix coordinate real symmetric
  5  5  7
    1     1     1.0
    2     2    10.5
    4     2   250.5
    3     3     1.5e-2
    4     4    -2.8e2
    5     5    12.
    5     4     8
'''

_symmetric_pattern_example = '''\
%%MatrixMarket matrix coordinate pattern symmetric
  5  5  7
    1     1
    2     2
    4     2
    3     3
    4     4
    5     5
    5     4
'''

# example (without comment lines) from Figure 1 in
# https://math.nist.gov/MatrixMarket/reports/MMformat.ps
_empty_lines_example = '''\
%%MatrixMarket  MATRIX    Coordinate    Real General

   5  5         8

1 1  1.0
2 2       10.5
3 3             1.5e-2
4 4                     -2.8E2
5 5                              12.
     1      4      6
     4      2      250.5
     4      5      33.32

'''


class TestMMIOCoordinate:
    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def check_read(self, example, a, info):
        f = open(self.fn, 'w')
        f.write(example)
        f.close()
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn).todense()
        assert_array_almost_equal(a, b)

    def test_read_general(self):
        a = [[1, 0, 0, 6, 0],
             [0, 10.5, 0, 0, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5, 0, -280, 33.32],
             [0, 0, 0, 0, 12]]
        self.check_read(_general_example, a,
                        (5, 5, 8, 'coordinate', 'real', 'general'))

    def test_read_hermitian(self):
        a = [[1, 0, 0, 0, 0],
             [0, 10.5, 0, 250.5 - 22.22j, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5 + 22.22j, 0, -280, -33.32j],
             [0, 0, 0, 33.32j, 12]]
        self.check_read(_hermitian_example, a,
                        (5, 5, 7, 'coordinate', 'complex', 'hermitian'))

    def test_read_skew(self):
        a = [[1, 0, 0, 0, 0],
             [0, 10.5, 0, -250.5, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5, 0, -280, 0],
             [0, 0, 0, 0, 12]]
        self.check_read(_skew_example, a,
                        (5, 5, 7, 'coordinate', 'real', 'skew-symmetric'))

    def test_read_symmetric(self):
        a = [[1, 0, 0, 0, 0],
             [0, 10.5, 0, 250.5, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5, 0, -280, 8],
             [0, 0, 0, 8, 12]]
        self.check_read(_symmetric_example, a,
                        (5, 5, 7, 'coordinate', 'real', 'symmetric'))

    def test_read_symmetric_pattern(self):
        a = [[1, 0, 0, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 1, 1],
             [0, 0, 0, 1, 1]]
        self.check_read(_symmetric_pattern_example, a,
                        (5, 5, 7, 'coordinate', 'pattern', 'symmetric'))

    def test_read_empty_lines(self):
        a = [[1, 0, 0, 6, 0],
             [0, 10.5, 0, 0, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5, 0, -280, 33.32],
             [0, 0, 0, 0, 12]]
        self.check_read(_empty_lines_example, a,
                        (5, 5, 8, 'coordinate', 'real', 'general'))

    def test_empty_write_read(self):
        # https://github.com/scipy/scipy/issues/1410 (Trac #883)

        b = scipy.sparse.coo_matrix((10, 10))
        mmwrite(self.fn, b)

        assert_equal(mminfo(self.fn),
                     (10, 10, 0, 'coordinate', 'real', 'symmetric'))
        a = b.todense()
        b = mmread(self.fn).todense()
        assert_array_almost_equal(a, b)

    def test_bzip2_py3(self):
        # test if fix for #2152 works
        try:
            # bz2 module isn't always built when building Python.
            import bz2
        except ImportError:
            return
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])

        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))

        mmwrite(self.fn, b)

        fn_bzip2 = "%s.bz2" % self.fn
        with open(self.fn, 'rb') as f_in:
            f_out = bz2.BZ2File(fn_bzip2, 'wb')
            f_out.write(f_in.read())
            f_out.close()

        a = mmread(fn_bzip2).todense()
        assert_array_almost_equal(a, b.todense())

    def test_gzip_py3(self):
        # test if fix for #2152 works
        try:
            # gzip module can be missing from Python installation
            import gzip
        except ImportError:
            return
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])

        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))

        mmwrite(self.fn, b)

        fn_gzip = "%s.gz" % self.fn
        with open(self.fn, 'rb') as f_in:
            f_out = gzip.open(fn_gzip, 'wb')
            f_out.write(f_in.read())
            f_out.close()

        a = mmread(fn_gzip).todense()
        assert_array_almost_equal(a, b.todense())

    def test_real_write_read(self):
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])

        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))

        mmwrite(self.fn, b)

        assert_equal(mminfo(self.fn),
                     (5, 5, 8, 'coordinate', 'real', 'general'))
        a = b.todense()
        b = mmread(self.fn).todense()
        assert_array_almost_equal(a, b)

    def test_complex_write_read(self):
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        V = array([1.0 + 3j, 6.0 + 2j, 10.50 + 0.9j, 0.015 + -4.4j,
                   250.5 + 0j, -280.0 + 5j, 33.32 + 6.4j, 12.00 + 0.8j])

        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))

        mmwrite(self.fn, b)

        assert_equal(mminfo(self.fn),
                     (5, 5, 8, 'coordinate', 'complex', 'general'))
        a = b.todense()
        b = mmread(self.fn).todense()
        assert_array_almost_equal(a, b)

    def test_sparse_formats(self):
        mats = []

        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])

        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
        mats.append(scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5)))

        V = array([1.0 + 3j, 6.0 + 2j, 10.50 + 0.9j, 0.015 + -4.4j,
                   250.5 + 0j, -280.0 + 5j, 33.32 + 6.4j, 12.00 + 0.8j])
        mats.append(scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5)))

        for mat in mats:
            expected = mat.todense()
            for fmt in ['csr', 'csc', 'coo']:
                fn = mktemp(dir=self.tmpdir)  # safe, we own tmpdir
                mmwrite(fn, mat.asformat(fmt))

                result = mmread(fn).todense()
                assert_array_almost_equal(result, expected)

    def test_precision(self):
        test_values = [pi] + [10**(i) for i in range(0, -10, -1)]
        test_precisions = range(1, 10)
        for value in test_values:
            for precision in test_precisions:
                # construct sparse matrix with test value at last main diagonal
                n = 10**precision + 1
                A = scipy.sparse.dok_matrix((n, n))
                A[n-1, n-1] = value
                # write matrix with test precision and read again
                mmwrite(self.fn, A, precision=precision)
                A = scipy.io.mmread(self.fn)
                # check for right entries in matrix
                assert_array_equal(A.row, [n-1])
                assert_array_equal(A.col, [n-1])
                assert_allclose(A.data, [float('%%.%dg' % precision % value)])

    def test_bad_number_of_coordinate_header_fields(self):
        s = """\
            %%MatrixMarket matrix coordinate real general
              5  5  8 999
                1     1   1.000e+00
                2     2   1.050e+01
                3     3   1.500e-02
                1     4   6.000e+00
                4     2   2.505e+02
                4     4  -2.800e+02
                4     5   3.332e+01
                5     5   1.200e+01
            """
        text = textwrap.dedent(s).encode('ascii')
        with pytest.raises(ValueError, match='not of length 3'):
            scipy.io.mmread(io.BytesIO(text))


def test_gh11389():
    mmread(io.StringIO("%%MatrixMarket matrix coordinate complex symmetric\n"
                       " 1 1 1\n"
                       "1 1 -2.1846000000000e+02  0.0000000000000e+00"))
