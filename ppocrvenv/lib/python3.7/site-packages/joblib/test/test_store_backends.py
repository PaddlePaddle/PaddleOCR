try:
    # Python 2.7: use the C pickle to speed up
    # test_concurrency_safe_write which pickles big python objects
    import cPickle as cpickle
except ImportError:
    import pickle as cpickle
import functools
import time

from joblib.testing import parametrize, timeout
from joblib.test.common import with_multiprocessing
from joblib.backports import concurrency_safe_rename
from joblib import Parallel, delayed
from joblib._store_backends import concurrency_safe_write


def write_func(output, filename):
    with open(filename, 'wb') as f:
        cpickle.dump(output, f)


def load_func(expected, filename):
    for i in range(10):
        try:
            with open(filename, 'rb') as f:
                reloaded = cpickle.load(f)
            break
        except (OSError, IOError):
            # On Windows you can have WindowsError ([Error 5] Access
            # is denied or [Error 13] Permission denied) when reading the file,
            # probably because a writer process has a lock on the file
            time.sleep(0.1)
    else:
        raise
    assert expected == reloaded


def concurrency_safe_write_rename(to_write, filename, write_func):
    temporary_filename = concurrency_safe_write(to_write,
                                                filename, write_func)
    concurrency_safe_rename(temporary_filename, filename)


@timeout(0)  # No timeout as this test can be long
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky', 'threading'])
def test_concurrency_safe_write(tmpdir, backend):
    # Add one item to cache
    filename = tmpdir.join('test.pkl').strpath

    obj = {str(i): i for i in range(int(1e5))}
    funcs = [functools.partial(concurrency_safe_write_rename,
                               write_func=write_func)
             if i % 3 != 2 else load_func for i in range(12)]
    Parallel(n_jobs=2, backend=backend)(
        delayed(func)(obj, filename) for func in funcs)
