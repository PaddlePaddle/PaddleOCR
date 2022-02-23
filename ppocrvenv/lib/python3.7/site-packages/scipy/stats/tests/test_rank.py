import numpy as np
from numpy.testing import assert_equal, assert_array_equal

from scipy.stats import rankdata, tiecorrect
import pytest


class TestTieCorrect:

    def test_empty(self):
        """An empty array requires no correction, should return 1.0."""
        ranks = np.array([], dtype=np.float64)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)

    def test_one(self):
        """A single element requires no correction, should return 1.0."""
        ranks = np.array([1.0], dtype=np.float64)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)

    def test_no_correction(self):
        """Arrays with no ties require no correction."""
        ranks = np.arange(2.0)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)
        ranks = np.arange(3.0)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)

    def test_basic(self):
        """Check a few basic examples of the tie correction factor."""
        # One tie of two elements
        ranks = np.array([1.0, 2.5, 2.5])
        c = tiecorrect(ranks)
        T = 2.0
        N = ranks.size
        expected = 1.0 - (T**3 - T) / (N**3 - N)
        assert_equal(c, expected)

        # One tie of two elements (same as above, but tie is not at the end)
        ranks = np.array([1.5, 1.5, 3.0])
        c = tiecorrect(ranks)
        T = 2.0
        N = ranks.size
        expected = 1.0 - (T**3 - T) / (N**3 - N)
        assert_equal(c, expected)

        # One tie of three elements
        ranks = np.array([1.0, 3.0, 3.0, 3.0])
        c = tiecorrect(ranks)
        T = 3.0
        N = ranks.size
        expected = 1.0 - (T**3 - T) / (N**3 - N)
        assert_equal(c, expected)

        # Two ties, lengths 2 and 3.
        ranks = np.array([1.5, 1.5, 4.0, 4.0, 4.0])
        c = tiecorrect(ranks)
        T1 = 2.0
        T2 = 3.0
        N = ranks.size
        expected = 1.0 - ((T1**3 - T1) + (T2**3 - T2)) / (N**3 - N)
        assert_equal(c, expected)

    def test_overflow(self):
        ntie, k = 2000, 5
        a = np.repeat(np.arange(k), ntie)
        n = a.size  # ntie * k
        out = tiecorrect(rankdata(a))
        assert_equal(out, 1.0 - k * (ntie**3 - ntie) / float(n**3 - n))


class TestRankData:

    def test_empty(self):
        """stats.rankdata([]) should return an empty array."""
        a = np.array([], dtype=int)
        r = rankdata(a)
        assert_array_equal(r, np.array([], dtype=np.float64))
        r = rankdata([])
        assert_array_equal(r, np.array([], dtype=np.float64))

    def test_one(self):
        """Check stats.rankdata with an array of length 1."""
        data = [100]
        a = np.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, np.array([1.0], dtype=np.float64))
        r = rankdata(data)
        assert_array_equal(r, np.array([1.0], dtype=np.float64))

    def test_basic(self):
        """Basic tests of stats.rankdata."""
        data = [100, 10, 50]
        expected = np.array([3.0, 1.0, 2.0], dtype=np.float64)
        a = np.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, expected)
        r = rankdata(data)
        assert_array_equal(r, expected)

        data = [40, 10, 30, 10, 50]
        expected = np.array([4.0, 1.5, 3.0, 1.5, 5.0], dtype=np.float64)
        a = np.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, expected)
        r = rankdata(data)
        assert_array_equal(r, expected)

        data = [20, 20, 20, 10, 10, 10]
        expected = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float64)
        a = np.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, expected)
        r = rankdata(data)
        assert_array_equal(r, expected)
        # The docstring states explicitly that the argument is flattened.
        a2d = a.reshape(2, 3)
        r = rankdata(a2d)
        assert_array_equal(r, expected)

    def test_rankdata_object_string(self):
        min_rank = lambda a: [1 + sum(i < j for i in a) for j in a]
        max_rank = lambda a: [sum(i <= j for i in a) for j in a]
        ordinal_rank = lambda a: min_rank([(x, i) for i, x in enumerate(a)])

        def average_rank(a):
            return [(i + j) / 2.0 for i, j in zip(min_rank(a), max_rank(a))]

        def dense_rank(a):
            b = np.unique(a)
            return [1 + sum(i < j for i in b) for j in a]

        rankf = dict(min=min_rank, max=max_rank, ordinal=ordinal_rank,
                     average=average_rank, dense=dense_rank)

        def check_ranks(a):
            for method in 'min', 'max', 'dense', 'ordinal', 'average':
                out = rankdata(a, method=method)
                assert_array_equal(out, rankf[method](a))

        val = ['foo', 'bar', 'qux', 'xyz', 'abc', 'efg', 'ace', 'qwe', 'qaz']
        check_ranks(np.random.choice(val, 200))
        check_ranks(np.random.choice(val, 200).astype('object'))

        val = np.array([0, 1, 2, 2.718, 3, 3.141], dtype='object')
        check_ranks(np.random.choice(val, 200).astype('object'))

    def test_large_int(self):
        data = np.array([2**60, 2**60+1], dtype=np.uint64)
        r = rankdata(data)
        assert_array_equal(r, [1.0, 2.0])

        data = np.array([2**60, 2**60+1], dtype=np.int64)
        r = rankdata(data)
        assert_array_equal(r, [1.0, 2.0])

        data = np.array([2**60, -2**60+1], dtype=np.int64)
        r = rankdata(data)
        assert_array_equal(r, [2.0, 1.0])

    def test_big_tie(self):
        for n in [10000, 100000, 1000000]:
            data = np.ones(n, dtype=int)
            r = rankdata(data)
            expected_rank = 0.5 * (n + 1)
            assert_array_equal(r, expected_rank * data,
                               "test failed with n=%d" % n)

    def test_axis(self):
        data = [[0, 2, 1],
                [4, 2, 2]]
        expected0 = [[1., 1.5, 1.],
                     [2., 1.5, 2.]]
        r0 = rankdata(data, axis=0)
        assert_array_equal(r0, expected0)
        expected1 = [[1., 3., 2.],
                     [3., 1.5, 1.5]]
        r1 = rankdata(data, axis=1)
        assert_array_equal(r1, expected1)

    methods = ["average", "min", "max", "dense", "ordinal"]
    dtypes = [np.float64] + [np.int_]*4

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("method, dtype", zip(methods, dtypes))
    def test_size_0_axis(self, axis, method, dtype):
        shape = (3, 0)
        data = np.zeros(shape)
        r = rankdata(data, method=method, axis=axis)
        assert_equal(r.shape, shape)
        assert_equal(r.dtype, dtype)


_cases = (
    # values, method, expected
    ([], 'average', []),
    ([], 'min', []),
    ([], 'max', []),
    ([], 'dense', []),
    ([], 'ordinal', []),
    #
    ([100], 'average', [1.0]),
    ([100], 'min', [1.0]),
    ([100], 'max', [1.0]),
    ([100], 'dense', [1.0]),
    ([100], 'ordinal', [1.0]),
    #
    ([100, 100, 100], 'average', [2.0, 2.0, 2.0]),
    ([100, 100, 100], 'min', [1.0, 1.0, 1.0]),
    ([100, 100, 100], 'max', [3.0, 3.0, 3.0]),
    ([100, 100, 100], 'dense', [1.0, 1.0, 1.0]),
    ([100, 100, 100], 'ordinal', [1.0, 2.0, 3.0]),
    #
    ([100, 300, 200], 'average', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'min', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'max', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'dense', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'ordinal', [1.0, 3.0, 2.0]),
    #
    ([100, 200, 300, 200], 'average', [1.0, 2.5, 4.0, 2.5]),
    ([100, 200, 300, 200], 'min', [1.0, 2.0, 4.0, 2.0]),
    ([100, 200, 300, 200], 'max', [1.0, 3.0, 4.0, 3.0]),
    ([100, 200, 300, 200], 'dense', [1.0, 2.0, 3.0, 2.0]),
    ([100, 200, 300, 200], 'ordinal', [1.0, 2.0, 4.0, 3.0]),
    #
    ([100, 200, 300, 200, 100], 'average', [1.5, 3.5, 5.0, 3.5, 1.5]),
    ([100, 200, 300, 200, 100], 'min', [1.0, 3.0, 5.0, 3.0, 1.0]),
    ([100, 200, 300, 200, 100], 'max', [2.0, 4.0, 5.0, 4.0, 2.0]),
    ([100, 200, 300, 200, 100], 'dense', [1.0, 2.0, 3.0, 2.0, 1.0]),
    ([100, 200, 300, 200, 100], 'ordinal', [1.0, 3.0, 5.0, 4.0, 2.0]),
    #
    ([10] * 30, 'ordinal', np.arange(1.0, 31.0)),
)


def test_cases():
    for values, method, expected in _cases:
        r = rankdata(values, method=method)
        assert_array_equal(r, expected)
