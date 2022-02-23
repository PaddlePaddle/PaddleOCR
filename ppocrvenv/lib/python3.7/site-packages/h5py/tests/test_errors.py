# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py.File object.
"""

import threading
import h5py


def _access_not_existing_object(filename):
    """Create a file and access not existing key"""
    with h5py.File(filename, 'w') as newfile:
        try:
            doesnt_exist = newfile['doesnt_exist'].value
        except KeyError:
            pass


def test_unsilence_errors(tmp_path, capfd):
    """Check that HDF5 errors can be muted/unmuted from h5py"""
    filename = tmp_path / 'test.h5'

    # Unmute HDF5 errors
    try:
        h5py._errors.unsilence_errors()

        _access_not_existing_object(filename)
        captured = capfd.readouterr()
        assert captured.err != ''
        assert captured.out == ''

    # Mute HDF5 errors
    finally:
        h5py._errors.silence_errors()

    _access_not_existing_object(filename)
    captured = capfd.readouterr()
    assert captured.err == ''
    assert captured.out == ''


def test_thread_hdf5_silence_error_membership(tmp_path, capfd):
    """Verify the error printing is squashed in all threads.

    No console messages should be shown from membership tests
    """
    th = threading.Thread(target=_access_not_existing_object,
                          args=(tmp_path / 'test.h5',))
    th.start()
    th.join()

    captured = capfd.readouterr()
    assert captured.err == ''
    assert captured.out == ''


def test_thread_hdf5_silence_error_attr(tmp_path, capfd):
    """Verify the error printing is squashed in all threads.

    No console messages should be shown for non-existing attributes
    """
    def test():
        with h5py.File(tmp_path/'test.h5', 'w') as newfile:
            newfile['newdata'] = [1, 2, 3]
            try:
                nonexistent_attr = newfile['newdata'].attrs['nonexistent_attr']
            except KeyError:
                pass

    th = threading.Thread(target=test)
    th.start()
    th.join()

    captured = capfd.readouterr()
    assert captured.err == ''
    assert captured.out == ''
