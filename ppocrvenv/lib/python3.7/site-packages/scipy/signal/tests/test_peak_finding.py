import copy

import numpy as np
from numpy.testing import (
    assert_,
    assert_equal,
    assert_allclose,
    assert_array_equal
)
import pytest
from pytest import raises, warns

from scipy.signal._peak_finding import (
    argrelmax,
    argrelmin,
    peak_prominences,
    peak_widths,
    _unpack_condition_args,
    find_peaks,
    find_peaks_cwt,
    _identify_ridge_lines
)
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning


def _gen_gaussians(center_locs, sigmas, total_length):
    xdata = np.arange(0, total_length).astype(float)
    out_data = np.zeros(total_length, dtype=float)
    for ind, sigma in enumerate(sigmas):
        tmp = (xdata - center_locs[ind]) / sigma
        out_data += np.exp(-(tmp**2))
    return out_data


def _gen_gaussians_even(sigmas, total_length):
    num_peaks = len(sigmas)
    delta = total_length / (num_peaks + 1)
    center_locs = np.linspace(delta, total_length - delta, num=num_peaks).astype(int)
    out_data = _gen_gaussians(center_locs, sigmas, total_length)
    return out_data, center_locs


def _gen_ridge_line(start_locs, max_locs, length, distances, gaps):
    """
    Generate coordinates for a ridge line.

    Will be a series of coordinates, starting a start_loc (length 2).
    The maximum distance between any adjacent columns will be
    `max_distance`, the max distance between adjacent rows
    will be `map_gap'.

    `max_locs` should be the size of the intended matrix. The
    ending coordinates are guaranteed to be less than `max_locs`,
    although they may not approach `max_locs` at all.
    """

    def keep_bounds(num, max_val):
        out = max(num, 0)
        out = min(out, max_val)
        return out

    gaps = copy.deepcopy(gaps)
    distances = copy.deepcopy(distances)

    locs = np.zeros([length, 2], dtype=int)
    locs[0, :] = start_locs
    total_length = max_locs[0] - start_locs[0] - sum(gaps)
    if total_length < length:
        raise ValueError('Cannot generate ridge line according to constraints')
    dist_int = length / len(distances) - 1
    gap_int = length / len(gaps) - 1
    for ind in range(1, length):
        nextcol = locs[ind - 1, 1]
        nextrow = locs[ind - 1, 0] + 1
        if (ind % dist_int == 0) and (len(distances) > 0):
            nextcol += ((-1)**ind)*distances.pop()
        if (ind % gap_int == 0) and (len(gaps) > 0):
            nextrow += gaps.pop()
        nextrow = keep_bounds(nextrow, max_locs[0])
        nextcol = keep_bounds(nextcol, max_locs[1])
        locs[ind, :] = [nextrow, nextcol]

    return [locs[:, 0], locs[:, 1]]


class TestLocalMaxima1d:

    def test_empty(self):
        """Test with empty signal."""
        x = np.array([], dtype=np.float64)
        for array in _local_maxima_1d(x):
            assert_equal(array, np.array([]))
            assert_(array.base is None)

    def test_linear(self):
        """Test with linear signal."""
        x = np.linspace(0, 100)
        for array in _local_maxima_1d(x):
            assert_equal(array, np.array([]))
            assert_(array.base is None)

    def test_simple(self):
        """Test with simple signal."""
        x = np.linspace(-10, 10, 50)
        x[2::3] += 1
        expected = np.arange(2, 50, 3)
        for array in _local_maxima_1d(x):
            # For plateaus of size 1, the edges are identical with the
            # midpoints
            assert_equal(array, expected)
            assert_(array.base is None)

    def test_flat_maxima(self):
        """Test if flat maxima are detected correctly."""
        x = np.array([-1.3, 0, 1, 0, 2, 2, 0, 3, 3, 3, 2.99, 4, 4, 4, 4, -10,
                      -5, -5, -5, -5, -5, -10])
        midpoints, left_edges, right_edges = _local_maxima_1d(x)
        assert_equal(midpoints, np.array([2, 4, 8, 12, 18]))
        assert_equal(left_edges, np.array([2, 4, 7, 11, 16]))
        assert_equal(right_edges, np.array([2, 5, 9, 14, 20]))

    @pytest.mark.parametrize('x', [
        np.array([1., 0, 2]),
        np.array([3., 3, 0, 4, 4]),
        np.array([5., 5, 5, 0, 6, 6, 6]),
    ])
    def test_signal_edges(self, x):
        """Test if behavior on signal edges is correct."""
        for array in _local_maxima_1d(x):
            assert_equal(array, np.array([]))
            assert_(array.base is None)

    def test_exceptions(self):
        """Test input validation and raised exceptions."""
        with raises(ValueError, match="wrong number of dimensions"):
            _local_maxima_1d(np.ones((1, 1)))
        with raises(ValueError, match="expected 'float64_t'"):
            _local_maxima_1d(np.ones(1, dtype=int))
        with raises(TypeError, match="list"):
            _local_maxima_1d([1., 2.])
        with raises(TypeError, match="'x' must not be None"):
            _local_maxima_1d(None)


class TestRidgeLines:

    def test_empty(self):
        test_matr = np.zeros([20, 100])
        lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
        assert_(len(lines) == 0)

    def test_minimal(self):
        test_matr = np.zeros([20, 100])
        test_matr[0, 10] = 1
        lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
        assert_(len(lines) == 1)

        test_matr = np.zeros([20, 100])
        test_matr[0:2, 10] = 1
        lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
        assert_(len(lines) == 1)

    def test_single_pass(self):
        distances = [0, 1, 2, 5]
        gaps = [0, 1, 2, 0, 1]
        test_matr = np.zeros([20, 50]) + 1e-12
        length = 12
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_distances = np.full(20, max(distances))
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
        assert_array_equal(identified_lines, [line])

    def test_single_bigdist(self):
        distances = [0, 1, 2, 5]
        gaps = [0, 1, 2, 4]
        test_matr = np.zeros([20, 50])
        length = 12
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_dist = 3
        max_distances = np.full(20, max_dist)
        #This should get 2 lines, since the distance is too large
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
        assert_(len(identified_lines) == 2)

        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)

            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)

    def test_single_biggap(self):
        distances = [0, 1, 2, 5]
        max_gap = 3
        gaps = [0, 4, 2, 1]
        test_matr = np.zeros([20, 50])
        length = 12
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_dist = 6
        max_distances = np.full(20, max_dist)
        #This should get 2 lines, since the gap is too large
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
        assert_(len(identified_lines) == 2)

        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)

            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)

    def test_single_biggaps(self):
        distances = [0]
        max_gap = 1
        gaps = [3, 6]
        test_matr = np.zeros([50, 50])
        length = 30
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_dist = 1
        max_distances = np.full(50, max_dist)
        #This should get 3 lines, since the gaps are too large
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
        assert_(len(identified_lines) == 3)

        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)

            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)


class TestArgrel:

    def test_empty(self):
        # Regression test for gh-2832.
        # When there are no relative extrema, make sure that
        # the number of empty arrays returned matches the
        # dimension of the input.

        empty_array = np.array([], dtype=int)

        z1 = np.zeros(5)

        i = argrelmin(z1)
        assert_equal(len(i), 1)
        assert_array_equal(i[0], empty_array)

        z2 = np.zeros((3,5))

        row, col = argrelmin(z2, axis=0)
        assert_array_equal(row, empty_array)
        assert_array_equal(col, empty_array)

        row, col = argrelmin(z2, axis=1)
        assert_array_equal(row, empty_array)
        assert_array_equal(col, empty_array)

    def test_basic(self):
        # Note: the docstrings for the argrel{min,max,extrema} functions
        # do not give a guarantee of the order of the indices, so we'll
        # sort them before testing.

        x = np.array([[1, 2, 2, 3, 2],
                      [2, 1, 2, 2, 3],
                      [3, 2, 1, 2, 2],
                      [2, 3, 2, 1, 2],
                      [1, 2, 3, 2, 1]])

        row, col = argrelmax(x, axis=0)
        order = np.argsort(row)
        assert_equal(row[order], [1, 2, 3])
        assert_equal(col[order], [4, 0, 1])

        row, col = argrelmax(x, axis=1)
        order = np.argsort(row)
        assert_equal(row[order], [0, 3, 4])
        assert_equal(col[order], [3, 1, 2])

        row, col = argrelmin(x, axis=0)
        order = np.argsort(row)
        assert_equal(row[order], [1, 2, 3])
        assert_equal(col[order], [1, 2, 3])

        row, col = argrelmin(x, axis=1)
        order = np.argsort(row)
        assert_equal(row[order], [1, 2, 3])
        assert_equal(col[order], [1, 2, 3])

    def test_highorder(self):
        order = 2
        sigmas = [1.0, 2.0, 10.0, 5.0, 15.0]
        test_data, act_locs = _gen_gaussians_even(sigmas, 500)
        test_data[act_locs + order] = test_data[act_locs]*0.99999
        test_data[act_locs - order] = test_data[act_locs]*0.99999
        rel_max_locs = argrelmax(test_data, order=order, mode='clip')[0]

        assert_(len(rel_max_locs) == len(act_locs))
        assert_((rel_max_locs == act_locs).all())

    def test_2d_gaussians(self):
        sigmas = [1.0, 2.0, 10.0]
        test_data, act_locs = _gen_gaussians_even(sigmas, 100)
        rot_factor = 20
        rot_range = np.arange(0, len(test_data)) - rot_factor
        test_data_2 = np.vstack([test_data, test_data[rot_range]])
        rel_max_rows, rel_max_cols = argrelmax(test_data_2, axis=1, order=1)

        for rw in range(0, test_data_2.shape[0]):
            inds = (rel_max_rows == rw)

            assert_(len(rel_max_cols[inds]) == len(act_locs))
            assert_((act_locs == (rel_max_cols[inds] - rot_factor*rw)).all())


class TestPeakProminences:

    def test_empty(self):
        """
        Test if an empty array is returned if no peaks are provided.
        """
        out = peak_prominences([1, 2, 3], [])
        for arr, dtype in zip(out, [np.float64, np.intp, np.intp]):
            assert_(arr.size == 0)
            assert_(arr.dtype == dtype)

        out = peak_prominences([], [])
        for arr, dtype in zip(out, [np.float64, np.intp, np.intp]):
            assert_(arr.size == 0)
            assert_(arr.dtype == dtype)

    def test_basic(self):
        """
        Test if height of prominences is correctly calculated in signal with
        rising baseline (peak widths are 1 sample).
        """
        # Prepare basic signal
        x = np.array([-1, 1.2, 1.2, 1, 3.2, 1.3, 2.88, 2.1])
        peaks = np.array([1, 2, 4, 6])
        lbases = np.array([0, 0, 0, 5])
        rbases = np.array([3, 3, 5, 7])
        proms = x[peaks] - np.max([x[lbases], x[rbases]], axis=0)
        # Test if calculation matches handcrafted result
        out = peak_prominences(x, peaks)
        assert_equal(out[0], proms)
        assert_equal(out[1], lbases)
        assert_equal(out[2], rbases)

    def test_edge_cases(self):
        """
        Test edge cases.
        """
        # Peaks have same height, prominence and bases
        x = [0, 2, 1, 2, 1, 2, 0]
        peaks = [1, 3, 5]
        proms, lbases, rbases = peak_prominences(x, peaks)
        assert_equal(proms, [2, 2, 2])
        assert_equal(lbases, [0, 0, 0])
        assert_equal(rbases, [6, 6, 6])

        # Peaks have same height & prominence but different bases
        x = [0, 1, 0, 1, 0, 1, 0]
        peaks = np.array([1, 3, 5])
        proms, lbases, rbases = peak_prominences(x, peaks)
        assert_equal(proms, [1, 1, 1])
        assert_equal(lbases, peaks - 1)
        assert_equal(rbases, peaks + 1)

    def test_non_contiguous(self):
        """
        Test with non-C-contiguous input arrays.
        """
        x = np.repeat([-9, 9, 9, 0, 3, 1], 2)
        peaks = np.repeat([1, 2, 4], 2)
        proms, lbases, rbases = peak_prominences(x[::2], peaks[::2])
        assert_equal(proms, [9, 9, 2])
        assert_equal(lbases, [0, 0, 3])
        assert_equal(rbases, [3, 3, 5])

    def test_wlen(self):
        """
        Test if wlen actually shrinks the evaluation range correctly.
        """
        x = [0, 1, 2, 3, 1, 0, -1]
        peak = [3]
        # Test rounding behavior of wlen
        assert_equal(peak_prominences(x, peak), [3., 0, 6])
        for wlen, i in [(8, 0), (7, 0), (6, 0), (5, 1), (3.2, 1), (3, 2), (1.1, 2)]:
            assert_equal(peak_prominences(x, peak, wlen), [3. - i, 0 + i, 6 - i])

    def test_exceptions(self):
        """
        Verify that exceptions and warnings are raised.
        """
        # x with dimension > 1
        with raises(ValueError, match='1-D array'):
            peak_prominences([[0, 1, 1, 0]], [1, 2])
        # peaks with dimension > 1
        with raises(ValueError, match='1-D array'):
            peak_prominences([0, 1, 1, 0], [[1, 2]])
        # x with dimension < 1
        with raises(ValueError, match='1-D array'):
            peak_prominences(3, [0,])

        # empty x with supplied
        with raises(ValueError, match='not a valid index'):
            peak_prominences([], [0])
        # invalid indices with non-empty x
        for p in [-100, -1, 3, 1000]:
            with raises(ValueError, match='not a valid index'):
                peak_prominences([1, 0, 2], [p])

        # peaks is not cast-able to np.intp
        with raises(TypeError, match='cannot safely cast'):
            peak_prominences([0, 1, 1, 0], [1.1, 2.3])

        # wlen < 3
        with raises(ValueError, match='wlen'):
            peak_prominences(np.arange(10), [3, 5], wlen=1)

    def test_warnings(self):
        """
        Verify that appropriate warnings are raised.
        """
        msg = "some peaks have a prominence of 0"
        for p in [0, 1, 2]:
            with warns(PeakPropertyWarning, match=msg):
                peak_prominences([1, 0, 2], [p,])
        with warns(PeakPropertyWarning, match=msg):
            peak_prominences([0, 1, 1, 1, 0], [2], wlen=2)


class TestPeakWidths:

    def test_empty(self):
        """
        Test if an empty array is returned if no peaks are provided.
        """
        widths = peak_widths([], [])[0]
        assert_(isinstance(widths, np.ndarray))
        assert_equal(widths.size, 0)
        widths = peak_widths([1, 2, 3], [])[0]
        assert_(isinstance(widths, np.ndarray))
        assert_equal(widths.size, 0)
        out = peak_widths([], [])
        for arr in out:
            assert_(isinstance(arr, np.ndarray))
            assert_equal(arr.size, 0)

    @pytest.mark.filterwarnings("ignore:some peaks have a width of 0")
    def test_basic(self):
        """
        Test a simple use case with easy to verify results at different relative
        heights.
        """
        x = np.array([1, 0, 1, 2, 1, 0, -1])
        prominence = 2
        for rel_height, width_true, lip_true, rip_true in [
            (0., 0., 3., 3.),  # raises warning
            (0.25, 1., 2.5, 3.5),
            (0.5, 2., 2., 4.),
            (0.75, 3., 1.5, 4.5),
            (1., 4., 1., 5.),
            (2., 5., 1., 6.),
            (3., 5., 1., 6.)
        ]:
            width_calc, height, lip_calc, rip_calc = peak_widths(
                x, [3], rel_height)
            assert_allclose(width_calc, width_true)
            assert_allclose(height, 2 - rel_height * prominence)
            assert_allclose(lip_calc, lip_true)
            assert_allclose(rip_calc, rip_true)

    def test_non_contiguous(self):
        """
        Test with non-C-contiguous input arrays.
        """
        x = np.repeat([0, 100, 50], 4)
        peaks = np.repeat([1], 3)
        result = peak_widths(x[::4], peaks[::3])
        assert_equal(result, [0.75, 75, 0.75, 1.5])

    def test_exceptions(self):
        """
        Verify that argument validation works as intended.
        """
        with raises(ValueError, match='1-D array'):
            # x with dimension > 1
            peak_widths(np.zeros((3, 4)), np.ones(3))
        with raises(ValueError, match='1-D array'):
            # x with dimension < 1
            peak_widths(3, [0])
        with raises(ValueError, match='1-D array'):
            # peaks with dimension > 1
            peak_widths(np.arange(10), np.ones((3, 2), dtype=np.intp))
        with raises(ValueError, match='1-D array'):
            # peaks with dimension < 1
            peak_widths(np.arange(10), 3)
        with raises(ValueError, match='not a valid index'):
            # peak pos exceeds x.size
            peak_widths(np.arange(10), [8, 11])
        with raises(ValueError, match='not a valid index'):
            # empty x with peaks supplied
            peak_widths([], [1, 2])
        with raises(TypeError, match='cannot safely cast'):
            # peak cannot be safely casted to intp
            peak_widths(np.arange(10), [1.1, 2.3])
        with raises(ValueError, match='rel_height'):
            # rel_height is < 0
            peak_widths([0, 1, 0, 1, 0], [1, 3], rel_height=-1)
        with raises(TypeError, match='None'):
            # prominence data contains None
            peak_widths([1, 2, 1], [1], prominence_data=(None, None, None))

    def test_warnings(self):
        """
        Verify that appropriate warnings are raised.
        """
        msg = "some peaks have a width of 0"
        with warns(PeakPropertyWarning, match=msg):
            # Case: rel_height is 0
            peak_widths([0, 1, 0], [1], rel_height=0)
        with warns(PeakPropertyWarning, match=msg):
            # Case: prominence is 0 and bases are identical
            peak_widths(
                [0, 1, 1, 1, 0], [2],
                prominence_data=(np.array([0.], np.float64),
                                 np.array([2], np.intp),
                                 np.array([2], np.intp))
            )

    def test_mismatching_prominence_data(self):
        """Test with mismatching peak and / or prominence data."""
        x = [0, 1, 0]
        peak = [1]
        for i, (prominences, left_bases, right_bases) in enumerate([
            ((1.,), (-1,), (2,)),  # left base not in x
            ((1.,), (0,), (3,)),  # right base not in x
            ((1.,), (2,), (0,)),  # swapped bases same as peak
            ((1., 1.), (0, 0), (2, 2)),  # array shapes don't match peaks
            ((1., 1.), (0,), (2,)),  # arrays with different shapes
            ((1.,), (0, 0), (2,)),  # arrays with different shapes
            ((1.,), (0,), (2, 2))  # arrays with different shapes
        ]):
            # Make sure input is matches output of signal.peak_prominences
            prominence_data = (np.array(prominences, dtype=np.float64),
                               np.array(left_bases, dtype=np.intp),
                               np.array(right_bases, dtype=np.intp))
            # Test for correct exception
            if i < 3:
                match = "prominence data is invalid for peak"
            else:
                match = "arrays in `prominence_data` must have the same shape"
            with raises(ValueError, match=match):
                peak_widths(x, peak, prominence_data=prominence_data)

    @pytest.mark.filterwarnings("ignore:some peaks have a width of 0")
    def test_intersection_rules(self):
        """Test if x == eval_height counts as an intersection."""
        # Flatt peak with two possible intersection points if evaluated at 1
        x = [0, 1, 2, 1, 3, 3, 3, 1, 2, 1, 0]
        # relative height is 0 -> width is 0 as well, raises warning
        assert_allclose(peak_widths(x, peaks=[5], rel_height=0),
                        [(0.,), (3.,), (5.,), (5.,)])
        # width_height == x counts as intersection -> nearest 1 is chosen
        assert_allclose(peak_widths(x, peaks=[5], rel_height=2/3),
                        [(4.,), (1.,), (3.,), (7.,)])


def test_unpack_condition_args():
    """
    Verify parsing of condition arguments for `scipy.signal.find_peaks` function.
    """
    x = np.arange(10)
    amin_true = x
    amax_true = amin_true + 10
    peaks = amin_true[1::2]

    # Test unpacking with None or interval
    assert_((None, None) == _unpack_condition_args((None, None), x, peaks))
    assert_((1, None) == _unpack_condition_args(1, x, peaks))
    assert_((1, None) == _unpack_condition_args((1, None), x, peaks))
    assert_((None, 2) == _unpack_condition_args((None, 2), x, peaks))
    assert_((3., 4.5) == _unpack_condition_args((3., 4.5), x, peaks))

    # Test if borders are correctly reduced with `peaks`
    amin_calc, amax_calc = _unpack_condition_args((amin_true, amax_true), x, peaks)
    assert_equal(amin_calc, amin_true[peaks])
    assert_equal(amax_calc, amax_true[peaks])

    # Test raises if array borders don't match x
    with raises(ValueError, match="array size of lower"):
        _unpack_condition_args(amin_true, np.arange(11), peaks)
    with raises(ValueError, match="array size of upper"):
        _unpack_condition_args((None, amin_true), np.arange(11), peaks)


class TestFindPeaks:

    # Keys of optionally returned properties
    property_keys = {'peak_heights', 'left_thresholds', 'right_thresholds',
                     'prominences', 'left_bases', 'right_bases', 'widths',
                     'width_heights', 'left_ips', 'right_ips'}

    def test_constant(self):
        """
        Test behavior for signal without local maxima.
        """
        open_interval = (None, None)
        peaks, props = find_peaks(np.ones(10),
                                  height=open_interval, threshold=open_interval,
                                  prominence=open_interval, width=open_interval)
        assert_(peaks.size == 0)
        for key in self.property_keys:
            assert_(props[key].size == 0)

    def test_plateau_size(self):
        """
        Test plateau size condition for peaks.
        """
        # Prepare signal with peaks with peak_height == plateau_size
        plateau_sizes = np.array([1, 2, 3, 4, 8, 20, 111])
        x = np.zeros(plateau_sizes.size * 2 + 1)
        x[1::2] = plateau_sizes
        repeats = np.ones(x.size, dtype=int)
        repeats[1::2] = x[1::2]
        x = np.repeat(x, repeats)

        # Test full output
        peaks, props = find_peaks(x, plateau_size=(None, None))
        assert_equal(peaks, [1, 3, 7, 11, 18, 33, 100])
        assert_equal(props["plateau_sizes"], plateau_sizes)
        assert_equal(props["left_edges"], peaks - (plateau_sizes - 1) // 2)
        assert_equal(props["right_edges"], peaks + plateau_sizes // 2)

        # Test conditions
        assert_equal(find_peaks(x, plateau_size=4)[0], [11, 18, 33, 100])
        assert_equal(find_peaks(x, plateau_size=(None, 3.5))[0], [1, 3, 7])
        assert_equal(find_peaks(x, plateau_size=(5, 50))[0], [18, 33])

    def test_height_condition(self):
        """
        Test height condition for peaks.
        """
        x = (0., 1/3, 0., 2.5, 0, 4., 0)
        peaks, props = find_peaks(x, height=(None, None))
        assert_equal(peaks, np.array([1, 3, 5]))
        assert_equal(props['peak_heights'], np.array([1/3, 2.5, 4.]))
        assert_equal(find_peaks(x, height=0.5)[0], np.array([3, 5]))
        assert_equal(find_peaks(x, height=(None, 3))[0], np.array([1, 3]))
        assert_equal(find_peaks(x, height=(2, 3))[0], np.array([3]))

    def test_threshold_condition(self):
        """
        Test threshold condition for peaks.
        """
        x = (0, 2, 1, 4, -1)
        peaks, props = find_peaks(x, threshold=(None, None))
        assert_equal(peaks, np.array([1, 3]))
        assert_equal(props['left_thresholds'], np.array([2, 3]))
        assert_equal(props['right_thresholds'], np.array([1, 5]))
        assert_equal(find_peaks(x, threshold=2)[0], np.array([3]))
        assert_equal(find_peaks(x, threshold=3.5)[0], np.array([]))
        assert_equal(find_peaks(x, threshold=(None, 5))[0], np.array([1, 3]))
        assert_equal(find_peaks(x, threshold=(None, 4))[0], np.array([1]))
        assert_equal(find_peaks(x, threshold=(2, 4))[0], np.array([]))

    def test_distance_condition(self):
        """
        Test distance condition for peaks.
        """
        # Peaks of different height with constant distance 3
        peaks_all = np.arange(1, 21, 3)
        x = np.zeros(21)
        x[peaks_all] += np.linspace(1, 2, peaks_all.size)

        # Test if peaks with "minimal" distance are still selected (distance = 3)
        assert_equal(find_peaks(x, distance=3)[0], peaks_all)

        # Select every second peak (distance > 3)
        peaks_subset = find_peaks(x, distance=3.0001)[0]
        # Test if peaks_subset is subset of peaks_all
        assert_(
            np.setdiff1d(peaks_subset, peaks_all, assume_unique=True).size == 0
        )
        # Test if every second peak was removed
        assert_equal(np.diff(peaks_subset), 6)

        # Test priority of peak removal
        x = [-2, 1, -1, 0, -3]
        peaks_subset = find_peaks(x, distance=10)[0]  # use distance > x size
        assert_(peaks_subset.size == 1 and peaks_subset[0] == 1)

    def test_prominence_condition(self):
        """
        Test prominence condition for peaks.
        """
        x = np.linspace(0, 10, 100)
        peaks_true = np.arange(1, 99, 2)
        offset = np.linspace(1, 10, peaks_true.size)
        x[peaks_true] += offset
        prominences = x[peaks_true] - x[peaks_true + 1]
        interval = (3, 9)
        keep = np.nonzero(
            (interval[0] <= prominences) & (prominences <= interval[1]))

        peaks_calc, properties = find_peaks(x, prominence=interval)
        assert_equal(peaks_calc, peaks_true[keep])
        assert_equal(properties['prominences'], prominences[keep])
        assert_equal(properties['left_bases'], 0)
        assert_equal(properties['right_bases'], peaks_true[keep] + 1)

    def test_width_condition(self):
        """
        Test width condition for peaks.
        """
        x = np.array([1, 0, 1, 2, 1, 0, -1, 4, 0])
        peaks, props = find_peaks(x, width=(None, 2), rel_height=0.75)
        assert_equal(peaks.size, 1)
        assert_equal(peaks, 7)
        assert_allclose(props['widths'], 1.35)
        assert_allclose(props['width_heights'], 1.)
        assert_allclose(props['left_ips'], 6.4)
        assert_allclose(props['right_ips'], 7.75)

    def test_properties(self):
        """
        Test returned properties.
        """
        open_interval = (None, None)
        x = [0, 1, 0, 2, 1.5, 0, 3, 0, 5, 9]
        peaks, props = find_peaks(x,
                                  height=open_interval, threshold=open_interval,
                                  prominence=open_interval, width=open_interval)
        assert_(len(props) == len(self.property_keys))
        for key in self.property_keys:
            assert_(peaks.size == props[key].size)

    def test_raises(self):
        """
        Test exceptions raised by function.
        """
        with raises(ValueError, match="1-D array"):
            find_peaks(np.array(1))
        with raises(ValueError, match="1-D array"):
            find_peaks(np.ones((2, 2)))
        with raises(ValueError, match="distance"):
            find_peaks(np.arange(10), distance=-1)

    @pytest.mark.filterwarnings("ignore:some peaks have a prominence of 0",
                                "ignore:some peaks have a width of 0")
    def test_wlen_smaller_plateau(self):
        """
        Test behavior of prominence and width calculation if the given window
        length is smaller than a peak's plateau size.

        Regression test for gh-9110.
        """
        peaks, props = find_peaks([0, 1, 1, 1, 0], prominence=(None, None),
                                  width=(None, None), wlen=2)
        assert_equal(peaks, 2)
        assert_equal(props["prominences"], 0)
        assert_equal(props["widths"], 0)
        assert_equal(props["width_heights"], 1)
        for key in ("left_bases", "right_bases", "left_ips", "right_ips"):
            assert_equal(props[key], peaks)


class TestFindPeaksCwt:

    def test_find_peaks_exact(self):
        """
        Generate a series of gaussians and attempt to find the peak locations.
        """
        sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
        num_points = 500
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        widths = np.arange(0.1, max(sigmas))
        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=0,
                                         min_length=None)
        np.testing.assert_array_equal(found_locs, act_locs,
                        "Found maximum locations did not equal those expected")

    def test_find_peaks_withnoise(self):
        """
        Verify that peak locations are (approximately) found
        for a series of gaussians with added noise.
        """
        sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
        num_points = 500
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        widths = np.arange(0.1, max(sigmas))
        noise_amp = 0.07
        np.random.seed(18181911)
        test_data += (np.random.rand(num_points) - 0.5)*(2*noise_amp)
        found_locs = find_peaks_cwt(test_data, widths, min_length=15,
                                         gap_thresh=1, min_snr=noise_amp / 5)

        np.testing.assert_equal(len(found_locs), len(act_locs), 'Different number' +
                                'of peaks found than expected')
        diffs = np.abs(found_locs - act_locs)
        max_diffs = np.array(sigmas) / 5
        np.testing.assert_array_less(diffs, max_diffs, 'Maximum location differed' +
                                     'by more than %s' % (max_diffs))

    def test_find_peaks_nopeak(self):
        """
        Verify that no peak is found in
        data that's just noise.
        """
        noise_amp = 1.0
        num_points = 100
        np.random.seed(181819141)
        test_data = (np.random.rand(num_points) - 0.5)*(2*noise_amp)
        widths = np.arange(10, 50)
        found_locs = find_peaks_cwt(test_data, widths, min_snr=5, noise_perc=30)
        np.testing.assert_equal(len(found_locs), 0)

    def test_find_peaks_window_size(self):
        """
        Verify that window_size is passed correctly to private function and
        affects the result.
        """
        sigmas = [2.0, 2.0]
        num_points = 1000
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        widths = np.arange(0.1, max(sigmas), 0.2)
        noise_amp = 0.05
        np.random.seed(18181911)
        test_data += (np.random.rand(num_points) - 0.5)*(2*noise_amp)

        # Possibly contrived negative region to throw off peak finding
        # when window_size is too large
        test_data[250:320] -= 1

        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=3,
                                    min_length=None, window_size=None)
        with pytest.raises(AssertionError):
            assert found_locs.size == act_locs.size

        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=3,
                                    min_length=None, window_size=20)
        assert found_locs.size == act_locs.size

    def test_find_peaks_with_one_width(self):
        """
        Verify that the `width` argument
        in `find_peaks_cwt` can be a float
        """
        xs = np.arange(0, np.pi, 0.05)
        test_data = np.sin(xs)
        widths = 1
        found_locs = find_peaks_cwt(test_data, widths)

        np.testing.assert_equal(found_locs, 32)
