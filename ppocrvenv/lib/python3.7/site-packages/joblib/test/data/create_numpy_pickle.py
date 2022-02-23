"""
This script is used to generate test data for joblib/test/test_numpy_pickle.py
"""

import sys
import re

# pytest needs to be able to import this module even when numpy is
# not installed
try:
    import numpy as np
except ImportError:
    np = None

import joblib


def get_joblib_version(joblib_version=joblib.__version__):
    """Normalize joblib version by removing suffix.

    >>> get_joblib_version('0.8.4')
    '0.8.4'
    >>> get_joblib_version('0.8.4b1')
    '0.8.4'
    >>> get_joblib_version('0.9.dev0')
    '0.9'
    """
    matches = [re.match(r'(\d+).*', each)
               for each in joblib_version.split('.')]
    return '.'.join([m.group(1) for m in matches if m is not None])


def write_test_pickle(to_pickle, args):
    kwargs = {}
    compress = args.compress
    method = args.method
    joblib_version = get_joblib_version()
    py_version = '{0[0]}{0[1]}'.format(sys.version_info)
    numpy_version = ''.join(np.__version__.split('.')[:2])

    # The game here is to generate the right filename according to the options.
    body = '_compressed' if (compress and method == 'zlib') else ''
    if compress:
        if method == 'zlib':
            kwargs['compress'] = True
            extension = '.gz'
        else:
            kwargs['compress'] = (method, 3)
            extension = '.pkl.{}'.format(method)
        if args.cache_size:
            kwargs['cache_size'] = 0
            body += '_cache_size'
    else:
        extension = '.pkl'

    pickle_filename = 'joblib_{}{}_pickle_py{}_np{}{}'.format(
        joblib_version, body, py_version, numpy_version, extension)

    try:
        joblib.dump(to_pickle, pickle_filename, **kwargs)
    except Exception as e:
        # With old python version (=< 3.3.), we can arrive there when
        # dumping compressed pickle with LzmaFile.
        print("Error: cannot generate file '{}' with arguments '{}'. "
              "Error was: {}".format(pickle_filename, kwargs, e))
    else:
        print("File '{}' generated successfuly.".format(pickle_filename))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Joblib pickle data "
                                                 "generator.")
    parser.add_argument('--cache_size', action="store_true",
                        help="Force creation of companion numpy "
                             "files for pickled arrays.")
    parser.add_argument('--compress', action="store_true",
                        help="Generate compress pickles.")
    parser.add_argument('--method', type=str, default='zlib',
                        choices=['zlib', 'gzip', 'bz2', 'xz', 'lzma', 'lz4'],
                        help="Set compression method.")
    # We need to be specific about dtypes in particular endianness
    # because the pickles can be generated on one architecture and
    # the tests run on another one. See
    # https://github.com/joblib/joblib/issues/279.
    to_pickle = [np.arange(5, dtype=np.dtype('<i8')),
                 np.arange(5, dtype=np.dtype('<f8')),
                 np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'),
                 # all possible bytes as a byte string
                 np.arange(256, dtype=np.uint8).tobytes(),
                 np.matrix([0, 1, 2], dtype=np.dtype('<i8')),
                 # unicode string with non-ascii chars
                 u"C'est l'\xe9t\xe9 !"]

    write_test_pickle(to_pickle, parser.parse_args())
