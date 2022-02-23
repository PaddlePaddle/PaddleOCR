''' Tests for fortran sequential files '''

import tempfile
import shutil
from os import path
from glob import iglob
import re

from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest

from scipy.io import (FortranFile,
                      _test_fortran,
                      FortranEOFError,
                      FortranFormattingError)


DATA_PATH = path.join(path.dirname(__file__), 'data')


def test_fortranfiles_read():
    for filename in iglob(path.join(DATA_PATH, "fortran-*-*x*x*.dat")):
        m = re.search(r'fortran-([^-]+)-(\d+)x(\d+)x(\d+).dat', filename, re.I)
        if not m:
            raise RuntimeError("Couldn't match %s filename to regex" % filename)

        dims = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

        dtype = m.group(1).replace('s', '<')

        f = FortranFile(filename, 'r', '<u4')
        data = f.read_record(dtype=dtype).reshape(dims, order='F')
        f.close()

        expected = np.arange(np.prod(dims)).reshape(dims).astype(dtype)
        assert_equal(data, expected)


def test_fortranfiles_mixed_record():
    filename = path.join(DATA_PATH, "fortran-mixed.dat")
    with FortranFile(filename, 'r', '<u4') as f:
        record = f.read_record('<i4,<f4,<i8,(2)<f8')

    assert_equal(record['f0'][0], 1)
    assert_allclose(record['f1'][0], 2.3)
    assert_equal(record['f2'][0], 4)
    assert_allclose(record['f3'][0], [5.6, 7.8])


def test_fortranfiles_write():
    for filename in iglob(path.join(DATA_PATH, "fortran-*-*x*x*.dat")):
        m = re.search(r'fortran-([^-]+)-(\d+)x(\d+)x(\d+).dat', filename, re.I)
        if not m:
            raise RuntimeError("Couldn't match %s filename to regex" % filename)
        dims = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

        dtype = m.group(1).replace('s', '<')
        data = np.arange(np.prod(dims)).reshape(dims).astype(dtype)

        tmpdir = tempfile.mkdtemp()
        try:
            testFile = path.join(tmpdir,path.basename(filename))
            f = FortranFile(testFile, 'w','<u4')
            f.write_record(data.T)
            f.close()
            originalfile = open(filename, 'rb')
            newfile = open(testFile, 'rb')
            assert_equal(originalfile.read(), newfile.read(),
                         err_msg=filename)
            originalfile.close()
            newfile.close()
        finally:
            shutil.rmtree(tmpdir)


def test_fortranfile_read_mixed_record():
    # The data file fortran-3x3d-2i.dat contains the program that
    # produced it at the end.
    #
    # double precision :: a(3,3)
    # integer :: b(2)
    # ...
    # open(1, file='fortran-3x3d-2i.dat', form='unformatted')
    # write(1) a, b
    # close(1)
    #

    filename = path.join(DATA_PATH, "fortran-3x3d-2i.dat")
    with FortranFile(filename, 'r', '<u4') as f:
        record = f.read_record('(3,3)<f8', '2<i4')

    ax = np.arange(3*3).reshape(3, 3).astype(np.double)
    bx = np.array([-1, -2], dtype=np.int32)

    assert_equal(record[0], ax.T)
    assert_equal(record[1], bx.T)


def test_fortranfile_write_mixed_record(tmpdir):
    tf = path.join(str(tmpdir), 'test.dat')

    records = [
        (('f4', 'f4', 'i4'), (np.float32(2), np.float32(3), np.int32(100))),
        (('4f4', '(3,3)f4', '8i4'), (np.random.randint(255, size=[4]).astype(np.float32),
                                     np.random.randint(255, size=[3, 3]).astype(np.float32),
                                     np.random.randint(255, size=[8]).astype(np.int32)))
    ]

    for dtype, a in records:
        with FortranFile(tf, 'w') as f:
            f.write_record(*a)

        with FortranFile(tf, 'r') as f:
            b = f.read_record(*dtype)

        assert_equal(len(a), len(b))

        for aa, bb in zip(a, b):
            assert_equal(bb, aa)


def test_fortran_roundtrip(tmpdir):
    filename = path.join(str(tmpdir), 'test.dat')

    np.random.seed(1)

    # double precision
    m, n, k = 5, 3, 2
    a = np.random.randn(m, n, k)
    with FortranFile(filename, 'w') as f:
        f.write_record(a.T)
    a2 = _test_fortran.read_unformatted_double(m, n, k, filename)
    with FortranFile(filename, 'r') as f:
        a3 = f.read_record('(2,3,5)f8').T
    assert_equal(a2, a)
    assert_equal(a3, a)

    # integer
    m, n, k = 5, 3, 2
    a = np.random.randn(m, n, k).astype(np.int32)
    with FortranFile(filename, 'w') as f:
        f.write_record(a.T)
    a2 = _test_fortran.read_unformatted_int(m, n, k, filename)
    with FortranFile(filename, 'r') as f:
        a3 = f.read_record('(2,3,5)i4').T
    assert_equal(a2, a)
    assert_equal(a3, a)

    # mixed
    m, n, k = 5, 3, 2
    a = np.random.randn(m, n)
    b = np.random.randn(k).astype(np.intc)
    with FortranFile(filename, 'w') as f:
        f.write_record(a.T, b.T)
    a2, b2 = _test_fortran.read_unformatted_mixed(m, n, k, filename)
    with FortranFile(filename, 'r') as f:
        a3, b3 = f.read_record('(3,5)f8', '2i4')
        a3 = a3.T
    assert_equal(a2, a)
    assert_equal(a3, a)
    assert_equal(b2, b)
    assert_equal(b3, b)


def test_fortran_eof_ok(tmpdir):
    filename = path.join(str(tmpdir), "scratch")
    np.random.seed(1)
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    with FortranFile(filename, 'r') as f:
        assert len(f.read_reals()) == 5
        assert len(f.read_reals()) == 3
        with pytest.raises(FortranEOFError):
            f.read_reals()


def test_fortran_eof_broken_size(tmpdir):
    filename = path.join(str(tmpdir), "scratch")
    np.random.seed(1)
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    with open(filename, "ab") as f:
        f.write(b"\xff")
    with FortranFile(filename, 'r') as f:
        assert len(f.read_reals()) == 5
        assert len(f.read_reals()) == 3
        with pytest.raises(FortranFormattingError):
            f.read_reals()


def test_fortran_bogus_size(tmpdir):
    filename = path.join(str(tmpdir), "scratch")
    np.random.seed(1)
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    with open(filename, "w+b") as f:
        f.write(b"\xff\xff")
    with FortranFile(filename, 'r') as f:
        with pytest.raises(FortranFormattingError):
            f.read_reals()


def test_fortran_eof_broken_record(tmpdir):
    filename = path.join(str(tmpdir), "scratch")
    np.random.seed(1)
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    with open(filename, "ab") as f:
        f.truncate(path.getsize(filename)-20)
    with FortranFile(filename, 'r') as f:
        assert len(f.read_reals()) == 5
        with pytest.raises(FortranFormattingError):
            f.read_reals()


def test_fortran_eof_multidimensional(tmpdir):
    filename = path.join(str(tmpdir), "scratch")
    n, m, q = 3, 5, 7
    dt = np.dtype([("field", np.float64, (n, m))])
    a = np.zeros(q, dtype=dt)
    with FortranFile(filename, 'w') as f:
        f.write_record(a[0])
        f.write_record(a)
        f.write_record(a)
    with open(filename, "ab") as f:
        f.truncate(path.getsize(filename)-20)
    with FortranFile(filename, 'r') as f:
        assert len(f.read_record(dtype=dt)) == 1
        assert len(f.read_record(dtype=dt)) == q
        with pytest.raises(FortranFormattingError):
            f.read_record(dtype=dt)
