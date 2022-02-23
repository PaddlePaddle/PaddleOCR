# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2019 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

import pytest

import h5py
from h5py import h5pl
from h5py.tests.common import insubprocess, subproc_env


# pytestmark is a special name - the skipif marker applies to the whole file
pytestmark = pytest.mark.skipif(
    h5py.version.hdf5_version_tuple < (1, 10, 1), reason='HDF5 1.10.1+ required'
)


@pytest.mark.mpi_skip
@insubprocess
@subproc_env({'HDF5_PLUGIN_PATH': 'h5py_plugin_test'})
def test_default(request):
    assert h5pl.size() == 1
    assert h5pl.get(0) == b'h5py_plugin_test'


@pytest.mark.mpi_skip
@insubprocess
@subproc_env({'HDF5_PLUGIN_PATH': 'h5py_plugin_test'})
def test_append(request):
    h5pl.append(b'/opt/hdf5/vendor-plugin')
    assert h5pl.size() == 2
    assert h5pl.get(0) == b'h5py_plugin_test'
    assert h5pl.get(1) == b'/opt/hdf5/vendor-plugin'


@pytest.mark.mpi_skip
@insubprocess
@subproc_env({'HDF5_PLUGIN_PATH': 'h5py_plugin_test'})
def test_prepend(request):
    h5pl.prepend(b'/opt/hdf5/vendor-plugin')
    assert h5pl.size() == 2
    assert h5pl.get(0) == b'/opt/hdf5/vendor-plugin'
    assert h5pl.get(1) == b'h5py_plugin_test'


@pytest.mark.mpi_skip
@insubprocess
@subproc_env({'HDF5_PLUGIN_PATH': 'h5py_plugin_test'})
def test_insert(request):
    h5pl.insert(b'/opt/hdf5/vendor-plugin', 0)
    assert h5pl.size() == 2
    assert h5pl.get(0) == b'/opt/hdf5/vendor-plugin'
    assert h5pl.get(1) == b'h5py_plugin_test'


@pytest.mark.mpi_skip
@insubprocess
@subproc_env({'HDF5_PLUGIN_PATH': 'h5py_plugin_test'})
def test_replace(request):
    h5pl.replace(b'/opt/hdf5/vendor-plugin', 0)
    assert  h5pl.size() == 1
    assert  h5pl.get(0) == b'/opt/hdf5/vendor-plugin'


@pytest.mark.mpi_skip
@insubprocess
def test_remove(request):
    h5pl.remove(0)
    assert h5pl.size() == 0
