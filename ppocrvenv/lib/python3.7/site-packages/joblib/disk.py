"""
Disk management utilities.
"""

# Authors: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
#          Lars Buitinck
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.


import os
import sys
import time
import errno
import shutil

from multiprocessing import util


try:
    WindowsError
except NameError:
    WindowsError = OSError


def disk_used(path):
    """ Return the disk usage in a directory."""
    size = 0
    for file in os.listdir(path) + ['.']:
        stat = os.stat(os.path.join(path, file))
        if hasattr(stat, 'st_blocks'):
            size += stat.st_blocks * 512
        else:
            # on some platform st_blocks is not available (e.g., Windows)
            # approximate by rounding to next multiple of 512
            size += (stat.st_size // 512 + 1) * 512
    # We need to convert to int to avoid having longs on some systems (we
    # don't want longs to avoid problems we SQLite)
    return int(size / 1024.)


def memstr_to_bytes(text):
    """ Convert a memory text to its value in bytes.
    """
    kilo = 1024
    units = dict(K=kilo, M=kilo ** 2, G=kilo ** 3)
    try:
        size = int(units[text[-1]] * float(text[:-1]))
    except (KeyError, ValueError) as e:
        raise ValueError(
            "Invalid literal for size give: %s (type %s) should be "
            "alike '10G', '500M', '50K'." % (text, type(text))) from e
    return size


def mkdirp(d):
    """Ensure directory d exists (like mkdir -p on Unix)
    No guarantee that the directory is writable.
    """
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# if a rmtree operation fails in rm_subdirs, wait for this much time (in secs),
# then retry up to RM_SUBDIRS_N_RETRY times. If it still fails, raise the
# exception. this mecanism ensures that the sub-process gc have the time to
# collect and close the memmaps before we fail.
RM_SUBDIRS_RETRY_TIME = 0.1
RM_SUBDIRS_N_RETRY = 5


def rm_subdirs(path, onerror=None):
    """Remove all subdirectories in this path.

    The directory indicated by `path` is left in place, and its subdirectories
    are erased.

    If onerror is set, it is called to handle the error with arguments (func,
    path, exc_info) where func is os.listdir, os.remove, or os.rmdir;
    path is the argument to that function that caused it to fail; and
    exc_info is a tuple returned by sys.exc_info().  If onerror is None,
    an exception is raised.
    """

    # NOTE this code is adapted from the one in shutil.rmtree, and is
    # just as fast

    names = []
    try:
        names = os.listdir(path)
    except os.error:
        if onerror is not None:
            onerror(os.listdir, path, sys.exc_info())
        else:
            raise

    for name in names:
        fullname = os.path.join(path, name)
        delete_folder(fullname, onerror=onerror)


def delete_folder(folder_path, onerror=None, allow_non_empty=True):
    """Utility function to cleanup a temporary folder if it still exists."""
    if os.path.isdir(folder_path):
        if onerror is not None:
            shutil.rmtree(folder_path, False, onerror)
        else:
            # allow the rmtree to fail once, wait and re-try.
            # if the error is raised again, fail
            err_count = 0
            while True:
                files = os.listdir(folder_path)
                try:
                    if len(files) == 0 or allow_non_empty:
                        shutil.rmtree(
                            folder_path, ignore_errors=False, onerror=None
                        )
                        util.debug(
                            "Sucessfully deleted {}".format(folder_path))
                        break
                    else:
                        raise OSError(
                            "Expected empty folder {} but got {} "
                            "files.".format(folder_path, len(files))
                        )
                except (OSError, WindowsError):
                    err_count += 1
                    if err_count > RM_SUBDIRS_N_RETRY:
                        # the folder cannot be deleted right now. It maybe
                        # because some temporary files have not been deleted
                        # yet.
                        raise
                time.sleep(RM_SUBDIRS_RETRY_TIME)
