"""
Tests of the ._exceptions module. Primarily for exercising the __str__ methods.
"""

import pickle

import numpy as np

_ArrayMemoryError = np.core._exceptions._ArrayMemoryError
_UFuncNoLoopError = np.core._exceptions._UFuncNoLoopError

class TestArrayMemoryError:
    def test_pickling(self):
        """ Test that _ArrayMemoryError can be pickled """
        error = _ArrayMemoryError((1023,), np.dtype(np.uint8))
        res = pickle.loads(pickle.dumps(error))
        assert res._total_size == error._total_size

    def test_str(self):
        e = _ArrayMemoryError((1023,), np.dtype(np.uint8))
        str(e)  # not crashing is enough

    # testing these properties is easier than testing the full string repr
    def test__size_to_string(self):
        """ Test e._size_to_string """
        f = _ArrayMemoryError._size_to_string
        Ki = 1024
        assert f(0) == '0 bytes'
        assert f(1) == '1 bytes'
        assert f(1023) == '1023 bytes'
        assert f(Ki) == '1.00 KiB'
        assert f(Ki+1) == '1.00 KiB'
        assert f(10*Ki) == '10.0 KiB'
        assert f(int(999.4*Ki)) == '999. KiB'
        assert f(int(1023.4*Ki)) == '1023. KiB'
        assert f(int(1023.5*Ki)) == '1.00 MiB'
        assert f(Ki*Ki) == '1.00 MiB'

        # 1023.9999 Mib should round to 1 GiB
        assert f(int(Ki*Ki*Ki*0.9999)) == '1.00 GiB'
        assert f(Ki*Ki*Ki*Ki*Ki*Ki) == '1.00 EiB'
        # larger than sys.maxsize, adding larger prefices isn't going to help
        # anyway.
        assert f(Ki*Ki*Ki*Ki*Ki*Ki*123456) == '123456. EiB'

    def test__total_size(self):
        """ Test e._total_size """
        e = _ArrayMemoryError((1,), np.dtype(np.uint8))
        assert e._total_size == 1

        e = _ArrayMemoryError((2, 4), np.dtype((np.uint64, 16)))
        assert e._total_size == 1024


class TestUFuncNoLoopError:
    def test_pickling(self):
        """ Test that _UFuncNoLoopError can be pickled """
        assert isinstance(pickle.dumps(_UFuncNoLoopError), bytes)
