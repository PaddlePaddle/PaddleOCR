"""
Backports of fixes for joblib dependencies
"""
import os
import time

from distutils.version import LooseVersion
from os.path import basename
from multiprocessing import util


try:
    import numpy as np

    def make_memmap(filename, dtype='uint8', mode='r+', offset=0,
                    shape=None, order='C', unlink_on_gc_collect=False):
        """Custom memmap constructor compatible with numpy.memmap.

        This function:
        - is a backport the numpy memmap offset fix (See
          https://github.com/numpy/numpy/pull/8443 for more details.
          The numpy fix is available starting numpy 1.13)
        - adds ``unlink_on_gc_collect``, which specifies  explicitly whether
          the process re-constructing the memmap owns a reference to the
          underlying file. If set to True, it adds a finalizer to the
          newly-created memmap that sends a maybe_unlink request for the
          memmaped file to resource_tracker.
        """
        util.debug(
            "[MEMMAP READ] creating a memmap (shape {}, filename {}, "
            "pid {})".format(shape, basename(filename), os.getpid())
        )

        mm = np.memmap(filename, dtype=dtype, mode=mode, offset=offset,
                       shape=shape, order=order)
        if LooseVersion(np.__version__) < '1.13':
            mm.offset = offset
        if unlink_on_gc_collect:
            from ._memmapping_reducer import add_maybe_unlink_finalizer
            add_maybe_unlink_finalizer(mm)
        return mm
except ImportError:
    def make_memmap(filename, dtype='uint8', mode='r+', offset=0,
                    shape=None, order='C', unlink_on_gc_collect=False):
        raise NotImplementedError(
            "'joblib.backports.make_memmap' should not be used "
            'if numpy is not installed.')


if os.name == 'nt':
    # https://github.com/joblib/joblib/issues/540
    access_denied_errors = (5, 13)
    from os import replace

    def concurrency_safe_rename(src, dst):
        """Renames ``src`` into ``dst`` overwriting ``dst`` if it exists.

        On Windows os.replace can yield permission errors if executed by two
        different processes.
        """
        max_sleep_time = 1
        total_sleep_time = 0
        sleep_time = 0.001
        while total_sleep_time < max_sleep_time:
            try:
                replace(src, dst)
                break
            except Exception as exc:
                if getattr(exc, 'winerror', None) in access_denied_errors:
                    time.sleep(sleep_time)
                    total_sleep_time += sleep_time
                    sleep_time *= 2
                else:
                    raise
        else:
            raise
else:
    from os import replace as concurrency_safe_rename  # noqa
