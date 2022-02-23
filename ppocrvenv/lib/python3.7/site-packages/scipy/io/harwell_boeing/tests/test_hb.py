from io import StringIO
import tempfile

import numpy as np

from numpy.testing import assert_equal, \
    assert_array_almost_equal_nulp

from scipy.sparse import coo_matrix, csc_matrix, rand

from scipy.io import hb_read, hb_write


SIMPLE = """\
No Title                                                                |No Key
             9             4             1             4
RUA                      100           100            10             0
(26I3)          (26I3)          (3E23.15)
1  2  2  2  2  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3
3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3
3  3  3  3  3  3  3  4  4  4  6  6  6  6  6  6  6  6  6  6  6  8  9  9  9  9
9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9 11
37 71 89 18 30 45 70 19 25 52
2.971243799687726e-01  3.662366682877375e-01  4.786962174699534e-01
6.490068647991184e-01  6.617490424831662e-02  8.870370343191623e-01
4.196478590163001e-01  5.649603072111251e-01  9.934423887087086e-01
6.912334991524289e-01
"""

SIMPLE_MATRIX = coo_matrix(
    ((0.297124379969, 0.366236668288, 0.47869621747, 0.649006864799,
      0.0661749042483, 0.887037034319, 0.419647859016,
      0.564960307211, 0.993442388709, 0.691233499152,),
     (np.array([[36, 70, 88, 17, 29, 44, 69, 18, 24, 51],
                [0, 4, 58, 61, 61, 72, 72, 73, 99, 99]]))))


def assert_csc_almost_equal(r, l):
    r = csc_matrix(r)
    l = csc_matrix(l)
    assert_equal(r.indptr, l.indptr)
    assert_equal(r.indices, l.indices)
    assert_array_almost_equal_nulp(r.data, l.data, 10000)


class TestHBReader:
    def test_simple(self):
        m = hb_read(StringIO(SIMPLE))
        assert_csc_almost_equal(m, SIMPLE_MATRIX)


class TestHBReadWrite:

    def check_save_load(self, value):
        with tempfile.NamedTemporaryFile(mode='w+t') as file:
            hb_write(file, value)
            file.file.seek(0)
            value_loaded = hb_read(file)
        assert_csc_almost_equal(value, value_loaded)

    def test_simple(self):
        random_matrix = rand(10, 100, 0.1)
        for matrix_format in ('coo', 'csc', 'csr', 'bsr', 'dia', 'dok', 'lil'):
            matrix = random_matrix.asformat(matrix_format, copy=False)
            self.check_save_load(matrix)
