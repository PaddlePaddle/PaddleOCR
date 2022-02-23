from contextlib import contextmanager
import os
import tempfile

import pytest

from pandas.io.pytables import HDFStore

tables = pytest.importorskip("tables")
# set these parameters so we don't have file sharing
tables.parameters.MAX_NUMEXPR_THREADS = 1
tables.parameters.MAX_BLOSC_THREADS = 1
tables.parameters.MAX_THREADS = 1


def safe_remove(path):
    if path is not None:
        try:
            os.remove(path)  # noqa: PDF008
        except OSError:
            pass


def safe_close(store):
    try:
        if store is not None:
            store.close()
    except OSError:
        pass


def create_tempfile(path):
    """create an unopened named temporary file"""
    return os.path.join(tempfile.gettempdir(), path)


# contextmanager to ensure the file cleanup
@contextmanager
def ensure_clean_store(path, mode="a", complevel=None, complib=None, fletcher32=False):

    try:

        # put in the temporary path if we don't have one already
        if not len(os.path.dirname(path)):
            path = create_tempfile(path)

        store = HDFStore(
            path, mode=mode, complevel=complevel, complib=complib, fletcher32=False
        )
        yield store
    finally:
        safe_close(store)
        if mode == "w" or mode == "a":
            safe_remove(path)


@contextmanager
def ensure_clean_path(path):
    """
    return essentially a named temporary file that is not opened
    and deleted on exiting; if path is a list, then create and
    return list of filenames
    """
    try:
        if isinstance(path, list):
            filenames = [create_tempfile(p) for p in path]
            yield filenames
        else:
            filenames = [create_tempfile(path)]
            yield filenames[0]
    finally:
        for f in filenames:
            safe_remove(f)


def _maybe_remove(store, key):
    """
    For tests using tables, try removing the table to be sure there is
    no content from previous tests using the same table name.
    """
    try:
        store.remove(key)
    except (ValueError, KeyError):
        pass
