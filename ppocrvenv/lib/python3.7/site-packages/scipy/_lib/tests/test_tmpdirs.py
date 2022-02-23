""" Test tmpdirs module """
from os import getcwd
from os.path import realpath, abspath, dirname, isfile, join as pjoin, exists

from scipy._lib._tmpdirs import tempdir, in_tempdir, in_dir

from numpy.testing import assert_, assert_equal

MY_PATH = abspath(__file__)
MY_DIR = dirname(MY_PATH)


def test_tempdir():
    with tempdir() as tmpdir:
        fname = pjoin(tmpdir, 'example_file.txt')
        with open(fname, 'wt') as fobj:
            fobj.write('a string\\n')
    assert_(not exists(tmpdir))


def test_in_tempdir():
    my_cwd = getcwd()
    with in_tempdir() as tmpdir:
        with open('test.txt', 'wt') as f:
            f.write('some text')
        assert_(isfile('test.txt'))
        assert_(isfile(pjoin(tmpdir, 'test.txt')))
    assert_(not exists(tmpdir))
    assert_equal(getcwd(), my_cwd)


def test_given_directory():
    # Test InGivenDirectory
    cwd = getcwd()
    with in_dir() as tmpdir:
        assert_equal(tmpdir, abspath(cwd))
        assert_equal(tmpdir, abspath(getcwd()))
    with in_dir(MY_DIR) as tmpdir:
        assert_equal(tmpdir, MY_DIR)
        assert_equal(realpath(MY_DIR), realpath(abspath(getcwd())))
    # We were deleting the given directory! Check not so now.
    assert_(isfile(MY_PATH))
