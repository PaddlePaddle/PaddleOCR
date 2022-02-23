""" Test fast_dict.
"""
import numpy as np

from sklearn.utils._fast_dict import IntFloatDict, argmin


def test_int_float_dict():
    rng = np.random.RandomState(0)
    keys = np.unique(rng.randint(100, size=10).astype(np.intp))
    values = rng.rand(len(keys))

    d = IntFloatDict(keys, values)
    for key, value in zip(keys, values):
        assert d[key] == value
    assert len(d) == len(keys)

    d.append(120, 3.0)
    assert d[120] == 3.0
    assert len(d) == len(keys) + 1
    for i in range(2000):
        d.append(i + 1000, 4.0)
    assert d[1100] == 4.0


def test_int_float_dict_argmin():
    # Test the argmin implementation on the IntFloatDict
    keys = np.arange(100, dtype=np.intp)
    values = np.arange(100, dtype=np.float64)
    d = IntFloatDict(keys, values)
    assert argmin(d) == (0, 0)
