# Author: Olivier Grisel <olivier.grisel@ensta.org>
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils.murmurhash import murmurhash3_32
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


def test_mmhash3_int():
    assert murmurhash3_32(3) == 847579505
    assert murmurhash3_32(3, seed=0) == 847579505
    assert murmurhash3_32(3, seed=42) == -1823081949

    assert murmurhash3_32(3, positive=False) == 847579505
    assert murmurhash3_32(3, seed=0, positive=False) == 847579505
    assert murmurhash3_32(3, seed=42, positive=False) == -1823081949

    assert murmurhash3_32(3, positive=True) == 847579505
    assert murmurhash3_32(3, seed=0, positive=True) == 847579505
    assert murmurhash3_32(3, seed=42, positive=True) == 2471885347


def test_mmhash3_int_array():
    rng = np.random.RandomState(42)
    keys = rng.randint(-5342534, 345345, size=3 * 2 * 1).astype(np.int32)
    keys = keys.reshape((3, 2, 1))

    for seed in [0, 42]:
        expected = np.array([murmurhash3_32(int(k), seed) for k in keys.flat])
        expected = expected.reshape(keys.shape)
        assert_array_equal(murmurhash3_32(keys, seed), expected)

    for seed in [0, 42]:
        expected = np.array([murmurhash3_32(k, seed, positive=True) for k in keys.flat])
        expected = expected.reshape(keys.shape)
        assert_array_equal(murmurhash3_32(keys, seed, positive=True), expected)


def test_mmhash3_bytes():
    assert murmurhash3_32(b"foo", 0) == -156908512
    assert murmurhash3_32(b"foo", 42) == -1322301282

    assert murmurhash3_32(b"foo", 0, positive=True) == 4138058784
    assert murmurhash3_32(b"foo", 42, positive=True) == 2972666014


def test_mmhash3_unicode():
    assert murmurhash3_32("foo", 0) == -156908512
    assert murmurhash3_32("foo", 42) == -1322301282

    assert murmurhash3_32("foo", 0, positive=True) == 4138058784
    assert murmurhash3_32("foo", 42, positive=True) == 2972666014


def test_no_collision_on_byte_range():
    previous_hashes = set()
    for i in range(100):
        h = murmurhash3_32(" " * i, 0)
        assert h not in previous_hashes, "Found collision on growing empty string"


def test_uniform_distribution():
    n_bins, n_samples = 10, 100000
    bins = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_samples):
        bins[murmurhash3_32(i, positive=True) % n_bins] += 1

    means = bins / n_samples
    expected = np.full(n_bins, 1.0 / n_bins)

    assert_array_almost_equal(means / expected, np.ones(n_bins), 2)
