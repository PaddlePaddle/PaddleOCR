"""
Functions for identifying peaks in signals.
"""
import math
import numpy as np

from scipy.signal.wavelets import cwt, ricker
from scipy.stats import scoreatpercentile

from ._peak_finding_utils import (
    _local_maxima_1d,
    _select_by_peak_distance,
    _peak_prominences,
    _peak_widths
)


__all__ = ['argrelmin', 'argrelmax', 'argrelextrema', 'peak_prominences',
           'peak_widths', 'find_peaks', 'find_peaks_cwt']


def _boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'. See numpy.take.

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.

    See also
    --------
    argrelmax, argrelmin

    Examples
    --------
    >>> testdata = np.array([1,2,3,2,1])
    >>> _boolrelextrema(testdata, np.greater, axis=0)
    array([False, False,  True, False, False], dtype=bool)

    """
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if(~results.any()):
            return results
    return results


def argrelmin(data, axis=0, order=1, mode='clip'):
    """
    Calculate the relative minima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative minima.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.
        Available options are 'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'. See numpy.take.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the minima in arrays of integers. ``extrema[k]`` is
        the array of indices of axis `k` of `data`. Note that the
        return value is a tuple even when `data` is 1-D.

    See Also
    --------
    argrelextrema, argrelmax, find_peaks

    Notes
    -----
    This function uses `argrelextrema` with np.less as comparator. Therefore, it
    requires a strict inequality on both sides of a value to consider it a
    minimum. This means flat minima (more than one sample wide) are not detected.
    In case of 1-D `data` `find_peaks` can be used to detect all
    local minima, including flat ones, by calling it with negated `data`.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy.signal import argrelmin
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmin(x)
    (array([1, 5]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmin(y, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    return argrelextrema(data, np.less, axis, order, mode)


def argrelmax(data, axis=0, order=1, mode='clip'):
    """
    Calculate the relative maxima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative maxima.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.
        Available options are 'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'. See `numpy.take`.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers. ``extrema[k]`` is
        the array of indices of axis `k` of `data`. Note that the
        return value is a tuple even when `data` is 1-D.

    See Also
    --------
    argrelextrema, argrelmin, find_peaks

    Notes
    -----
    This function uses `argrelextrema` with np.greater as comparator. Therefore,
    it  requires a strict inequality on both sides of a value to consider it a
    maximum. This means flat maxima (more than one sample wide) are not detected.
    In case of 1-D `data` `find_peaks` can be used to detect all
    local maxima, including flat ones.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy.signal import argrelmax
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmax(x)
    (array([3, 6]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmax(y, axis=1)
    (array([0]), array([1]))
    """
    return argrelextrema(data, np.greater, axis, order, mode)


def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default is 'clip'. See `numpy.take`.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers. ``extrema[k]`` is
        the array of indices of axis `k` of `data`. Note that the
        return value is a tuple even when `data` is 1-D.

    See Also
    --------
    argrelmin, argrelmax

    Notes
    -----

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy.signal import argrelextrema
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, np.greater)
    (array([3, 6]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, np.less, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    results = _boolrelextrema(data, comparator,
                              axis, order, mode)
    return np.nonzero(results)


def _arg_x_as_expected(value):
    """Ensure argument `x` is a 1-D C-contiguous array of dtype('float64').

    Used in `find_peaks`, `peak_prominences` and `peak_widths` to make `x`
    compatible with the signature of the wrapped Cython functions.

    Returns
    -------
    value : ndarray
        A 1-D C-contiguous array with dtype('float64').
    """
    value = np.asarray(value, order='C', dtype=np.float64)
    if value.ndim != 1:
        raise ValueError('`x` must be a 1-D array')
    return value


def _arg_peaks_as_expected(value):
    """Ensure argument `peaks` is a 1-D C-contiguous array of dtype('intp').

    Used in `peak_prominences` and `peak_widths` to make `peaks` compatible
    with the signature of the wrapped Cython functions.

    Returns
    -------
    value : ndarray
        A 1-D C-contiguous array with dtype('intp').
    """
    value = np.asarray(value)
    if value.size == 0:
        # Empty arrays default to np.float64 but are valid input
        value = np.array([], dtype=np.intp)
    try:
        # Safely convert to C-contiguous array of type np.intp
        value = value.astype(np.intp, order='C', casting='safe',
                             subok=False, copy=False)
    except TypeError as e:
        raise TypeError("cannot safely cast `peaks` to dtype('intp')") from e
    if value.ndim != 1:
        raise ValueError('`peaks` must be a 1-D array')
    return value


def _arg_wlen_as_expected(value):
    """Ensure argument `wlen` is of type `np.intp` and larger than 1.

    Used in `peak_prominences` and `peak_widths`.

    Returns
    -------
    value : np.intp
        The original `value` rounded up to an integer or -1 if `value` was
        None.
    """
    if value is None:
        # _peak_prominences expects an intp; -1 signals that no value was
        # supplied by the user
        value = -1
    elif 1 < value:
        # Round up to a positive integer
        if not np.can_cast(value, np.intp, "safe"):
            value = math.ceil(value)
        value = np.intp(value)
    else:
        raise ValueError('`wlen` must be larger than 1, was {}'
                         .format(value))
    return value


def peak_prominences(x, peaks, wlen=None):
    """
    Calculate the prominence of each peak in a signal.

    The prominence of a peak measures how much a peak stands out from the
    surrounding baseline of the signal and is defined as the vertical distance
    between the peak and its lowest contour line.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    peaks : sequence
        Indices of peaks in `x`.
    wlen : int, optional
        A window length in samples that optionally limits the evaluated area for
        each peak to a subset of `x`. The peak is always placed in the middle of
        the window therefore the given length is rounded up to the next odd
        integer. This parameter can speed up the calculation (see Notes).

    Returns
    -------
    prominences : ndarray
        The calculated prominences for each peak in `peaks`.
    left_bases, right_bases : ndarray
        The peaks' bases as indices in `x` to the left and right of each peak.
        The higher base of each pair is a peak's lowest contour line.

    Raises
    ------
    ValueError
        If a value in `peaks` is an invalid index for `x`.

    Warns
    -----
    PeakPropertyWarning
        For indices in `peaks` that don't point to valid local maxima in `x`,
        the returned prominence will be 0 and this warning is raised. This
        also happens if `wlen` is smaller than the plateau size of a peak.

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks
        Find peaks inside a signal based on peak properties.
    peak_widths
        Calculate the width of peaks.

    Notes
    -----
    Strategy to compute a peak's prominence:

    1. Extend a horizontal line from the current peak to the left and right
       until the line either reaches the window border (see `wlen`) or
       intersects the signal again at the slope of a higher peak. An
       intersection with a peak of the same height is ignored.
    2. On each side find the minimal signal value within the interval defined
       above. These points are the peak's bases.
    3. The higher one of the two bases marks the peak's lowest contour line. The
       prominence can then be calculated as the vertical difference between the
       peaks height itself and its lowest contour line.

    Searching for the peak's bases can be slow for large `x` with periodic
    behavior because large chunks or even the full signal need to be evaluated
    for the first algorithmic step. This evaluation area can be limited with the
    parameter `wlen` which restricts the algorithm to a window around the
    current peak and can shorten the calculation time if the window length is
    short in relation to `x`.
    However, this may stop the algorithm from finding the true global contour
    line if the peak's true bases are outside this window. Instead, a higher
    contour line is found within the restricted window leading to a smaller
    calculated prominence. In practice, this is only relevant for the highest set
    of peaks in `x`. This behavior may even be used intentionally to calculate
    "local" prominences.

    .. versionadded:: 1.1.0

    References
    ----------
    .. [1] Wikipedia Article for Topographic Prominence:
       https://en.wikipedia.org/wiki/Topographic_prominence

    Examples
    --------
    >>> from scipy.signal import find_peaks, peak_prominences
    >>> import matplotlib.pyplot as plt

    Create a test signal with two overlayed harmonics

    >>> x = np.linspace(0, 6 * np.pi, 1000)
    >>> x = np.sin(x) + 0.6 * np.sin(2.6 * x)

    Find all peaks and calculate prominences

    >>> peaks, _ = find_peaks(x)
    >>> prominences = peak_prominences(x, peaks)[0]
    >>> prominences
    array([1.24159486, 0.47840168, 0.28470524, 3.10716793, 0.284603  ,
           0.47822491, 2.48340261, 0.47822491])

    Calculate the height of each peak's contour line and plot the results

    >>> contour_heights = x[peaks] - prominences
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.vlines(x=peaks, ymin=contour_heights, ymax=x[peaks])
    >>> plt.show()

    Let's evaluate a second example that demonstrates several edge cases for
    one peak at index 5.

    >>> x = np.array([0, 1, 0, 3, 1, 3, 0, 4, 0])
    >>> peaks = np.array([5])
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()
    >>> peak_prominences(x, peaks)  # -> (prominences, left_bases, right_bases)
    (array([3.]), array([2]), array([6]))

    Note how the peak at index 3 of the same height is not considered as a
    border while searching for the left base. Instead, two minima at 0 and 2
    are found in which case the one closer to the evaluated peak is always
    chosen. On the right side, however, the base must be placed at 6 because the
    higher peak represents the right border to the evaluated area.

    >>> peak_prominences(x, peaks, wlen=3.1)
    (array([2.]), array([4]), array([6]))

    Here, we restricted the algorithm to a window from 3 to 7 (the length is 5
    samples because `wlen` was rounded up to the next odd integer). Thus, the
    only two candidates in the evaluated area are the two neighboring samples
    and a smaller prominence is calculated.
    """
    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    wlen = _arg_wlen_as_expected(wlen)
    return _peak_prominences(x, peaks, wlen)


def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None):
    """
    Calculate the width of each peak in a signal.

    This function calculates the width of a peak in samples at a relative
    distance to the peak's height and prominence.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    peaks : sequence
        Indices of peaks in `x`.
    rel_height : float, optional
        Chooses the relative height at which the peak width is measured as a
        percentage of its prominence. 1.0 calculates the width of the peak at
        its lowest contour line while 0.5 evaluates at half the prominence
        height. Must be at least 0. See notes for further explanation.
    prominence_data : tuple, optional
        A tuple of three arrays matching the output of `peak_prominences` when
        called with the same arguments `x` and `peaks`. This data are calculated
        internally if not provided.
    wlen : int, optional
        A window length in samples passed to `peak_prominences` as an optional
        argument for internal calculation of `prominence_data`. This argument
        is ignored if `prominence_data` is given.

    Returns
    -------
    widths : ndarray
        The widths for each peak in samples.
    width_heights : ndarray
        The height of the contour lines at which the `widths` where evaluated.
    left_ips, right_ips : ndarray
        Interpolated positions of left and right intersection points of a
        horizontal line at the respective evaluation height.

    Raises
    ------
    ValueError
        If `prominence_data` is supplied but doesn't satisfy the condition
        ``0 <= left_base <= peak <= right_base < x.shape[0]`` for each peak,
        has the wrong dtype, is not C-contiguous or does not have the same
        shape.

    Warns
    -----
    PeakPropertyWarning
        Raised if any calculated width is 0. This may stem from the supplied
        `prominence_data` or if `rel_height` is set to 0.

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks
        Find peaks inside a signal based on peak properties.
    peak_prominences
        Calculate the prominence of peaks.

    Notes
    -----
    The basic algorithm to calculate a peak's width is as follows:

    * Calculate the evaluation height :math:`h_{eval}` with the formula
      :math:`h_{eval} = h_{Peak} - P \\cdot R`, where :math:`h_{Peak}` is the
      height of the peak itself, :math:`P` is the peak's prominence and
      :math:`R` a positive ratio specified with the argument `rel_height`.
    * Draw a horizontal line at the evaluation height to both sides, starting at
      the peak's current vertical position until the lines either intersect a
      slope, the signal border or cross the vertical position of the peak's
      base (see `peak_prominences` for an definition). For the first case,
      intersection with the signal, the true intersection point is estimated
      with linear interpolation.
    * Calculate the width as the horizontal distance between the chosen
      endpoints on both sides. As a consequence of this the maximal possible
      width for each peak is the horizontal distance between its bases.

    As shown above to calculate a peak's width its prominence and bases must be
    known. You can supply these yourself with the argument `prominence_data`.
    Otherwise, they are internally calculated (see `peak_prominences`).

    .. versionadded:: 1.1.0

    Examples
    --------
    >>> from scipy.signal import chirp, find_peaks, peak_widths
    >>> import matplotlib.pyplot as plt

    Create a test signal with two overlayed harmonics

    >>> x = np.linspace(0, 6 * np.pi, 1000)
    >>> x = np.sin(x) + 0.6 * np.sin(2.6 * x)

    Find all peaks and calculate their widths at the relative height of 0.5
    (contour line at half the prominence height) and 1 (at the lowest contour
    line at full prominence height).

    >>> peaks, _ = find_peaks(x)
    >>> results_half = peak_widths(x, peaks, rel_height=0.5)
    >>> results_half[0]  # widths
    array([ 64.25172825,  41.29465463,  35.46943289, 104.71586081,
            35.46729324,  41.30429622, 181.93835853,  45.37078546])
    >>> results_full = peak_widths(x, peaks, rel_height=1)
    >>> results_full[0]  # widths
    array([181.9396084 ,  72.99284945,  61.28657872, 373.84622694,
        61.78404617,  72.48822812, 253.09161876,  79.36860878])

    Plot signal, peaks and contour lines at which the widths where calculated

    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.hlines(*results_half[1:], color="C2")
    >>> plt.hlines(*results_full[1:], color="C3")
    >>> plt.show()
    """
    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    if prominence_data is None:
        # Calculate prominence if not supplied and use wlen if supplied.
        wlen = _arg_wlen_as_expected(wlen)
        prominence_data = _peak_prominences(x, peaks, wlen)
    return _peak_widths(x, peaks, rel_height, *prominence_data)


def _unpack_condition_args(interval, x, peaks):
    """
    Parse condition arguments for `find_peaks`.

    Parameters
    ----------
    interval : number or ndarray or sequence
        Either a number or ndarray or a 2-element sequence of the former. The
        first value is always interpreted as `imin` and the second, if supplied,
        as `imax`.
    x : ndarray
        The signal with `peaks`.
    peaks : ndarray
        An array with indices used to reduce `imin` and / or `imax` if those are
        arrays.

    Returns
    -------
    imin, imax : number or ndarray or None
        Minimal and maximal value in `argument`.

    Raises
    ------
    ValueError :
        If interval border is given as array and its size does not match the size
        of `x`.

    Notes
    -----

    .. versionadded:: 1.1.0
    """
    try:
        imin, imax = interval
    except (TypeError, ValueError):
        imin, imax = (interval, None)

    # Reduce arrays if arrays
    if isinstance(imin, np.ndarray):
        if imin.size != x.size:
            raise ValueError('array size of lower interval border must match x')
        imin = imin[peaks]
    if isinstance(imax, np.ndarray):
        if imax.size != x.size:
            raise ValueError('array size of upper interval border must match x')
        imax = imax[peaks]

    return imin, imax


def _select_by_property(peak_properties, pmin, pmax):
    """
    Evaluate where the generic property of peaks confirms to an interval.

    Parameters
    ----------
    peak_properties : ndarray
        An array with properties for each peak.
    pmin : None or number or ndarray
        Lower interval boundary for `peak_properties`. ``None`` is interpreted as
        an open border.
    pmax : None or number or ndarray
        Upper interval boundary for `peak_properties`. ``None`` is interpreted as
        an open border.

    Returns
    -------
    keep : bool
        A boolean mask evaluating to true where `peak_properties` confirms to the
        interval.

    See Also
    --------
    find_peaks

    Notes
    -----

    .. versionadded:: 1.1.0
    """
    keep = np.ones(peak_properties.size, dtype=bool)
    if pmin is not None:
        keep &= (pmin <= peak_properties)
    if pmax is not None:
        keep &= (peak_properties <= pmax)
    return keep


def _select_by_peak_threshold(x, peaks, tmin, tmax):
    """
    Evaluate which peaks fulfill the threshold condition.

    Parameters
    ----------
    x : ndarray
        A 1-D array which is indexable by `peaks`.
    peaks : ndarray
        Indices of peaks in `x`.
    tmin, tmax : scalar or ndarray or None
         Minimal and / or maximal required thresholds. If supplied as ndarrays
         their size must match `peaks`. ``None`` is interpreted as an open
         border.

    Returns
    -------
    keep : bool
        A boolean mask evaluating to true where `peaks` fulfill the threshold
        condition.
    left_thresholds, right_thresholds : ndarray
        Array matching `peak` containing the thresholds of each peak on
        both sides.

    Notes
    -----

    .. versionadded:: 1.1.0
    """
    # Stack thresholds on both sides to make min / max operations easier:
    # tmin is compared with the smaller, and tmax with the greater thresold to
    # each peak's side
    stacked_thresholds = np.vstack([x[peaks] - x[peaks - 1],
                                    x[peaks] - x[peaks + 1]])
    keep = np.ones(peaks.size, dtype=bool)
    if tmin is not None:
        min_thresholds = np.min(stacked_thresholds, axis=0)
        keep &= (tmin <= min_thresholds)
    if tmax is not None:
        max_thresholds = np.max(stacked_thresholds, axis=0)
        keep &= (max_thresholds <= tmax)

    return keep, stacked_thresholds[0], stacked_thresholds[1]


def find_peaks(x, height=None, threshold=None, distance=None,
               prominence=None, width=None, wlen=None, rel_height=0.5,
               plateau_size=None):
    """
    Find peaks inside a signal based on peak properties.

    This function takes a 1-D array and finds all local maxima by
    simple comparison of neighboring values. Optionally, a subset of these
    peaks can be selected by specifying conditions for a peak's properties.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, ``None``, an array matching
        `x` or a 2-element sequence of the former. The first element is
        always interpreted as the  minimal and the second, if supplied, as the
        maximal required height.
    threshold : number or ndarray or sequence, optional
        Required threshold of peaks, the vertical distance to its neighboring
        samples. Either a number, ``None``, an array matching `x` or a
        2-element sequence of the former. The first element is always
        interpreted as the  minimal and the second, if supplied, as the maximal
        required threshold.
    distance : number, optional
        Required minimal horizontal distance (>= 1) in samples between
        neighbouring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks.
    prominence : number or ndarray or sequence, optional
        Required prominence of peaks. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required prominence.
    width : number or ndarray or sequence, optional
        Required width of peaks in samples. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required width.
    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used if
        one of the arguments `prominence` or `width` is given. See argument
        `wlen` in `peak_prominences` for a full description of its effects.
    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if `width`
        is given. See argument  `rel_height` in `peak_widths` for a full
        description of its effects.
    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number,
        ``None``, an array matching `x` or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second,
        if supplied as the maximal required plateau size.

        .. versionadded:: 1.2.0

    Returns
    -------
    peaks : ndarray
        Indices of peaks in `x` that satisfy all given conditions.
    properties : dict
        A dictionary containing properties of the returned peaks which were
        calculated as intermediate results during evaluation of the specified
        conditions:

        * 'peak_heights'
              If `height` is given, the height of each peak in `x`.
        * 'left_thresholds', 'right_thresholds'
              If `threshold` is given, these keys contain a peaks vertical
              distance to its neighbouring samples.
        * 'prominences', 'right_bases', 'left_bases'
              If `prominence` is given, these keys are accessible. See
              `peak_prominences` for a description of their content.
        * 'width_heights', 'left_ips', 'right_ips'
              If `width` is given, these keys are accessible. See `peak_widths`
              for a description of their content.
        * 'plateau_sizes', left_edges', 'right_edges'
              If `plateau_size` is given, these keys are accessible and contain
              the indices of a peak's edges (edges are still part of the
              plateau) and the calculated plateau sizes.

              .. versionadded:: 1.2.0

        To calculate and return properties without excluding peaks, provide the
        open interval ``(None, None)`` as a value to the appropriate argument
        (excluding `distance`).

    Warns
    -----
    PeakPropertyWarning
        Raised if a peak's properties have unexpected values (see
        `peak_prominences` and `peak_widths`).

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks_cwt
        Find peaks using the wavelet transformation.
    peak_prominences
        Directly calculate the prominence of peaks.
    peak_widths
        Directly calculate the width of peaks.

    Notes
    -----
    In the context of this function, a peak or local maximum is defined as any
    sample whose two direct neighbours have a smaller amplitude. For flat peaks
    (more than one sample of equal amplitude wide) the index of the middle
    sample is returned (rounded down in case the number of samples is even).
    For noisy signals the peak locations can be off because the noise might
    change the position of local maxima. In those cases consider smoothing the
    signal before searching for peaks or use other peak finding and fitting
    methods (like `find_peaks_cwt`).

    Some additional comments on specifying conditions:

    * Almost all conditions (excluding `distance`) can be given as half-open or
      closed intervals, e.g., ``1`` or ``(1, None)`` defines the half-open
      interval :math:`[1, \\infty]` while ``(None, 1)`` defines the interval
      :math:`[-\\infty, 1]`. The open interval ``(None, None)`` can be specified
      as well, which returns the matching properties without exclusion of peaks.
    * The border is always included in the interval used to select valid peaks.
    * For several conditions the interval borders can be specified with
      arrays matching `x` in shape which enables dynamic constrains based on
      the sample position.
    * The conditions are evaluated in the following order: `plateau_size`,
      `height`, `threshold`, `distance`, `prominence`, `width`. In most cases
      this order is the fastest one because faster operations are applied first
      to reduce the number of peaks that need to be evaluated later.
    * While indices in `peaks` are guaranteed to be at least `distance` samples
      apart, edges of flat peaks may be closer than the allowed `distance`.
    * Use `wlen` to reduce the time it takes to evaluate the conditions for
      `prominence` or `width` if `x` is large or has many local maxima
      (see `peak_prominences`).

    .. versionadded:: 1.1.0

    Examples
    --------
    To demonstrate this function's usage we use a signal `x` supplied with
    SciPy (see `scipy.misc.electrocardiogram`). Let's find all peaks (local
    maxima) in `x` whose amplitude lies above 0.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.misc import electrocardiogram
    >>> from scipy.signal import find_peaks
    >>> x = electrocardiogram()[2000:4000]
    >>> peaks, _ = find_peaks(x, height=0)
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.plot(np.zeros_like(x), "--", color="gray")
    >>> plt.show()

    We can select peaks below 0 with ``height=(None, 0)`` or use arrays matching
    `x` in size to reflect a changing condition for different parts of the
    signal.

    >>> border = np.sin(np.linspace(0, 3 * np.pi, x.size))
    >>> peaks, _ = find_peaks(x, height=(-border, border))
    >>> plt.plot(x)
    >>> plt.plot(-border, "--", color="gray")
    >>> plt.plot(border, ":", color="gray")
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()

    Another useful condition for periodic signals can be given with the
    `distance` argument. In this case, we can easily select the positions of
    QRS complexes within the electrocardiogram (ECG) by demanding a distance of
    at least 150 samples.

    >>> peaks, _ = find_peaks(x, distance=150)
    >>> np.diff(peaks)
    array([186, 180, 177, 171, 177, 169, 167, 164, 158, 162, 172])
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()

    Especially for noisy signals peaks can be easily grouped by their
    prominence (see `peak_prominences`). E.g., we can select all peaks except
    for the mentioned QRS complexes by limiting the allowed prominence to 0.6.

    >>> peaks, properties = find_peaks(x, prominence=(None, 0.6))
    >>> properties["prominences"].max()
    0.5049999999999999
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()

    And, finally, let's examine a different section of the ECG which contains
    beat forms of different shape. To select only the atypical heart beats, we
    combine two conditions: a minimal prominence of 1 and width of at least 20
    samples.

    >>> x = electrocardiogram()[17000:18000]
    >>> peaks, properties = find_peaks(x, prominence=1, width=20)
    >>> properties["prominences"], properties["widths"]
    (array([1.495, 2.3  ]), array([36.93773946, 39.32723577]))
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
    ...            ymax = x[peaks], color = "C1")
    >>> plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
    ...            xmax=properties["right_ips"], color = "C1")
    >>> plt.show()
    """
    # _argmaxima1d expects array of dtype 'float64'
    x = _arg_x_as_expected(x)
    if distance is not None and distance < 1:
        raise ValueError('`distance` must be greater or equal to 1')

    peaks, left_edges, right_edges = _local_maxima_1d(x)
    properties = {}

    if plateau_size is not None:
        # Evaluate plateau size
        plateau_sizes = right_edges - left_edges + 1
        pmin, pmax = _unpack_condition_args(plateau_size, x, peaks)
        keep = _select_by_property(plateau_sizes, pmin, pmax)
        peaks = peaks[keep]
        properties["plateau_sizes"] = plateau_sizes
        properties["left_edges"] = left_edges
        properties["right_edges"] = right_edges
        properties = {key: array[keep] for key, array in properties.items()}

    if height is not None:
        # Evaluate height condition
        peak_heights = x[peaks]
        hmin, hmax = _unpack_condition_args(height, x, peaks)
        keep = _select_by_property(peak_heights, hmin, hmax)
        peaks = peaks[keep]
        properties["peak_heights"] = peak_heights
        properties = {key: array[keep] for key, array in properties.items()}

    if threshold is not None:
        # Evaluate threshold condition
        tmin, tmax = _unpack_condition_args(threshold, x, peaks)
        keep, left_thresholds, right_thresholds = _select_by_peak_threshold(
            x, peaks, tmin, tmax)
        peaks = peaks[keep]
        properties["left_thresholds"] = left_thresholds
        properties["right_thresholds"] = right_thresholds
        properties = {key: array[keep] for key, array in properties.items()}

    if distance is not None:
        # Evaluate distance condition
        keep = _select_by_peak_distance(peaks, x[peaks], distance)
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if prominence is not None or width is not None:
        # Calculate prominence (required for both conditions)
        wlen = _arg_wlen_as_expected(wlen)
        properties.update(zip(
            ['prominences', 'left_bases', 'right_bases'],
            _peak_prominences(x, peaks, wlen=wlen)
        ))

    if prominence is not None:
        # Evaluate prominence condition
        pmin, pmax = _unpack_condition_args(prominence, x, peaks)
        keep = _select_by_property(properties['prominences'], pmin, pmax)
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if width is not None:
        # Calculate widths
        properties.update(zip(
            ['widths', 'width_heights', 'left_ips', 'right_ips'],
            _peak_widths(x, peaks, rel_height, properties['prominences'],
                         properties['left_bases'], properties['right_bases'])
        ))
        # Evaluate width condition
        wmin, wmax = _unpack_condition_args(width, x, peaks)
        keep = _select_by_property(properties['widths'], wmin, wmax)
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    return peaks, properties


def _identify_ridge_lines(matr, max_distances, gap_thresh):
    """
    Identify ridges in the 2-D matrix.

    Expect that the width of the wavelet feature increases with increasing row
    number.

    Parameters
    ----------
    matr : 2-D ndarray
        Matrix in which to identify ridge lines.
    max_distances : 1-D sequence
        At each row, a ridge line is only connected
        if the relative max at row[n] is within
        `max_distances`[n] from the relative max at row[n+1].
    gap_thresh : int
        If a relative maximum is not found within `max_distances`,
        there will be a gap. A ridge line is discontinued if
        there are more than `gap_thresh` points without connecting
        a new relative maximum.

    Returns
    -------
    ridge_lines : tuple
        Tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the
        ii-th ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none
        found.  Each ridge-line will be sorted by row (increasing), but the
        order of the ridge lines is not specified.

    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
       :doi:`10.1093/bioinformatics/btl355`

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> data = rng.random((5,5))
    >>> ridge_lines = _identify_ridge_lines(data, 1, 1)

    Notes
    -----
    This function is intended to be used in conjunction with `cwt`
    as part of `find_peaks_cwt`.

    """
    if(len(max_distances) < matr.shape[0]):
        raise ValueError('Max_distances must have at least as many rows '
                         'as matr')

    all_max_cols = _boolrelextrema(matr, np.greater, axis=1, order=1)
    # Highest row for which there are any relative maxima
    has_relmax = np.nonzero(all_max_cols.any(axis=1))[0]
    if(len(has_relmax) == 0):
        return []
    start_row = has_relmax[-1]
    # Each ridge line is a 3-tuple:
    # rows, cols,Gap number
    ridge_lines = [[[start_row],
                   [col],
                   0] for col in np.nonzero(all_max_cols[start_row])[0]]
    final_lines = []
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    for row in rows:
        this_max_cols = cols[all_max_cols[row]]

        # Increment gap number of each line,
        # set it to zero later if appropriate
        for line in ridge_lines:
            line[2] += 1

        # XXX These should always be all_max_cols[row]
        # But the order might be different. Might be an efficiency gain
        # to make sure the order is the same and avoid this iteration
        prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
        # Look through every relative maximum found at current row
        # Attempt to connect them with existing ridge lines.
        for ind, col in enumerate(this_max_cols):
            # If there is a previous ridge line within
            # the max_distance to connect to, do so.
            # Otherwise start a new one.
            line = None
            if(len(prev_ridge_cols) > 0):
                diffs = np.abs(col - prev_ridge_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = ridge_lines[closest]
            if(line is not None):
                # Found a point close enough, extend current ridge line
                line[1].append(col)
                line[0].append(row)
                line[2] = 0
            else:
                new_line = [[row],
                            [col],
                            0]
                ridge_lines.append(new_line)

        # Remove the ridge lines with gap_number too high
        # XXX Modifying a list while iterating over it.
        # Should be safe, since we iterate backwards, but
        # still tacky.
        for ind in range(len(ridge_lines) - 1, -1, -1):
            line = ridge_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del ridge_lines[ind]

    out_lines = []
    for line in (final_lines + ridge_lines):
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])

    return out_lines


def _filter_ridge_lines(cwt, ridge_lines, window_size=None, min_length=None,
                        min_snr=1, noise_perc=10):
    """
    Filter ridge lines according to prescribed criteria. Intended
    to be used for finding relative maxima.

    Parameters
    ----------
    cwt : 2-D ndarray
        Continuous wavelet transform from which the `ridge_lines` were defined.
    ridge_lines : 1-D sequence
        Each element should contain 2 sequences, the rows and columns
        of the ridge line (respectively).
    window_size : int, optional
        Size of window to use to calculate noise floor.
        Default is ``cwt.shape[1] / 20``.
    min_length : int, optional
        Minimum length a ridge line needs to be acceptable.
        Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.
    min_snr : float, optional
        Minimum SNR ratio. Default 1. The signal is the value of
        the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
        noise is the `noise_perc`th percentile of datapoints contained within a
        window of `window_size` around ``cwt[0, loc]``.
    noise_perc : float, optional
        When calculating the noise floor, percentile of data points
        examined below which to consider noise. Calculated using
        scipy.stats.scoreatpercentile.

    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
       :doi:`10.1093/bioinformatics/btl355`

    """
    num_points = cwt.shape[1]
    if min_length is None:
        min_length = np.ceil(cwt.shape[0] / 4)
    if window_size is None:
        window_size = np.ceil(num_points / 20)

    window_size = int(window_size)
    hf_window, odd = divmod(window_size, 2)

    # Filter based on SNR
    row_one = cwt[0, :]
    noises = np.empty_like(row_one)
    for ind, val in enumerate(row_one):
        window_start = max(ind - hf_window, 0)
        window_end = min(ind + hf_window + odd, num_points)
        noises[ind] = scoreatpercentile(row_one[window_start:window_end],
                                        per=noise_perc)

    def filt_func(line):
        if len(line[0]) < min_length:
            return False
        snr = abs(cwt[line[0][0], line[1][0]] / noises[line[1][0]])
        if snr < min_snr:
            return False
        return True

    return list(filter(filt_func, ridge_lines))


def find_peaks_cwt(vector, widths, wavelet=None, max_distances=None,
                   gap_thresh=None, min_length=None,
                   min_snr=1, noise_perc=10, window_size=None):
    """
    Find peaks in a 1-D array with wavelet transformation.

    The general approach is to smooth `vector` by convolving it with
    `wavelet(width)` for each width in `widths`. Relative maxima which
    appear at enough length scales, and with sufficiently high SNR, are
    accepted.

    Parameters
    ----------
    vector : ndarray
        1-D array in which to find the peaks.
    widths : float or sequence
        Single width or 1-D array-like of widths to use for calculating
        the CWT matrix. In general,
        this range should cover the expected width of peaks of interest.
    wavelet : callable, optional
        Should take two parameters and return a 1-D array to convolve
        with `vector`. The first parameter determines the number of points
        of the returned wavelet array, the second parameter is the scale
        (`width`) of the wavelet. Should be normalized and symmetric.
        Default is the ricker wavelet.
    max_distances : ndarray, optional
        At each row, a ridge line is only connected if the relative max at
        row[n] is within ``max_distances[n]`` from the relative max at
        ``row[n+1]``.  Default value is ``widths/4``.
    gap_thresh : float, optional
        If a relative maximum is not found within `max_distances`,
        there will be a gap. A ridge line is discontinued if there are more
        than `gap_thresh` points without connecting a new relative maximum.
        Default is the first value of the widths array i.e. widths[0].
    min_length : int, optional
        Minimum length a ridge line needs to be acceptable.
        Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.
    min_snr : float, optional
        Minimum SNR ratio. Default 1. The signal is the value of
        the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
        noise is the `noise_perc`th percentile of datapoints contained within a
        window of `window_size` around ``cwt[0, loc]``.
    noise_perc : float, optional
        When calculating the noise floor, percentile of data points
        examined below which to consider noise. Calculated using
        `stats.scoreatpercentile`.  Default is 10.
    window_size : int, optional
        Size of window to use to calculate noise floor.
        Default is ``cwt.shape[1] / 20``.

    Returns
    -------
    peaks_indices : ndarray
        Indices of the locations in the `vector` where peaks were found.
        The list is sorted.

    See Also
    --------
    cwt
        Continuous wavelet transform.
    find_peaks
        Find peaks inside a signal based on peak properties.

    Notes
    -----
    This approach was designed for finding sharp peaks among noisy data,
    however with proper parameter selection it should function well for
    different peak shapes.

    The algorithm is as follows:
     1. Perform a continuous wavelet transform on `vector`, for the supplied
        `widths`. This is a convolution of `vector` with `wavelet(width)` for
        each width in `widths`. See `cwt`.
     2. Identify "ridge lines" in the cwt matrix. These are relative maxima
        at each row, connected across adjacent rows. See identify_ridge_lines
     3. Filter the ridge_lines using filter_ridge_lines.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
       :doi:`10.1093/bioinformatics/btl355`

    Examples
    --------
    >>> from scipy import signal
    >>> xs = np.arange(0, np.pi, 0.05)
    >>> data = np.sin(xs)
    >>> peakind = signal.find_peaks_cwt(data, np.arange(1,10))
    >>> peakind, xs[peakind], data[peakind]
    ([32], array([ 1.6]), array([ 0.9995736]))

    """
    widths = np.array(widths, copy=False, ndmin=1)

    if gap_thresh is None:
        gap_thresh = np.ceil(widths[0])
    if max_distances is None:
        max_distances = widths / 4.0
    if wavelet is None:
        wavelet = ricker

    cwt_dat = cwt(vector, wavelet, widths, window_size=window_size)
    ridge_lines = _identify_ridge_lines(cwt_dat, max_distances, gap_thresh)
    filtered = _filter_ridge_lines(cwt_dat, ridge_lines, min_length=min_length,
                                   window_size=window_size, min_snr=min_snr,
                                   noise_perc=noise_perc)
    max_locs = np.asarray([x[1][0] for x in filtered])
    max_locs.sort()

    return max_locs
