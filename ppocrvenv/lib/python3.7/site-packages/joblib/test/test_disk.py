"""
Unit tests for the disk utilities.
"""

# Authors: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
#          Lars Buitinck
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.

from __future__ import with_statement
import array
import os

from joblib.disk import disk_used, memstr_to_bytes, mkdirp, rm_subdirs
from joblib.testing import parametrize, raises

###############################################################################


def test_disk_used(tmpdir):
    cachedir = tmpdir.strpath
    # Not write a file that is 1M big in this directory, and check the
    # size. The reason we use such a big file is that it makes us robust
    # to errors due to block allocation.
    a = array.array('i')
    sizeof_i = a.itemsize
    target_size = 1024
    n = int(target_size * 1024 / sizeof_i)
    a = array.array('i', n * (1,))
    with open(os.path.join(cachedir, 'test'), 'wb') as output:
        a.tofile(output)
    assert disk_used(cachedir) >= target_size
    assert disk_used(cachedir) < target_size + 12


@parametrize('text,value',
             [('80G', 80 * 1024 ** 3),
              ('1.4M', int(1.4 * 1024 ** 2)),
              ('120M', 120 * 1024 ** 2),
              ('53K', 53 * 1024)])
def test_memstr_to_bytes(text, value):
    assert memstr_to_bytes(text) == value


@parametrize('text,exception,regex',
             [('fooG', ValueError, r'Invalid literal for size.*fooG.*'),
              ('1.4N', ValueError, r'Invalid literal for size.*1.4N.*')])
def test_memstr_to_bytes_exception(text, exception, regex):
    with raises(exception) as excinfo:
        memstr_to_bytes(text)
    assert excinfo.match(regex)


def test_mkdirp(tmpdir):
    mkdirp(os.path.join(tmpdir.strpath, 'ham'))
    mkdirp(os.path.join(tmpdir.strpath, 'ham'))
    mkdirp(os.path.join(tmpdir.strpath, 'spam', 'spam'))

    # Not all OSErrors are ignored
    with raises(OSError):
        mkdirp('')


def test_rm_subdirs(tmpdir):
    sub_path = os.path.join(tmpdir.strpath, "am", "stram")
    full_path = os.path.join(sub_path, "gram")
    mkdirp(os.path.join(full_path))

    rm_subdirs(sub_path)
    assert os.path.exists(sub_path)
    assert not os.path.exists(full_path)
