""" Test reading of files not conforming to matlab specification

We try and read any file that matlab reads, these files included
"""
from os.path import dirname, join as pjoin

from numpy.testing import assert_
from pytest import raises as assert_raises

from scipy.io.matlab.mio import loadmat

TEST_DATA_PATH = pjoin(dirname(__file__), 'data')


def test_multiple_fieldnames():
    # Example provided by Dharhas Pothina
    # Extracted using mio5.varmats_from_mat
    multi_fname = pjoin(TEST_DATA_PATH, 'nasty_duplicate_fieldnames.mat')
    vars = loadmat(multi_fname)
    funny_names = vars['Summary'].dtype.names
    assert_(set(['_1_Station_Q', '_2_Station_Q',
                     '_3_Station_Q']).issubset(funny_names))


def test_malformed1():
    # Example from gh-6072
    # Contains malformed header data, which previously resulted into a
    # buffer overflow.
    #
    # Should raise an exception, not segfault
    fname = pjoin(TEST_DATA_PATH, 'malformed1.mat')
    with open(fname, 'rb') as f:
        assert_raises(ValueError, loadmat, f)
