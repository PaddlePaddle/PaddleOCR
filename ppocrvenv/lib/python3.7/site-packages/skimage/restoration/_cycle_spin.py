from itertools import product
import numpy as np
from .._shared import utils
from .._shared.utils import warn

try:
    import dask
    dask_available = True
except ImportError:
    dask_available = False


def _generate_shifts(ndim, multichannel, max_shifts, shift_steps=1):
    """Returns all combinations of shifts in n dimensions over the specified
    max_shifts and step sizes.

    Examples
    --------
    >>> s = list(_generate_shifts(2, False, max_shifts=(1, 2), shift_steps=1))
    >>> print(s)
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    """
    mc = int(multichannel)
    if np.isscalar(max_shifts):
        max_shifts = (max_shifts, ) * (ndim - mc) + (0, ) * mc
    elif multichannel and len(max_shifts) == ndim - 1:
        max_shifts = tuple(max_shifts) + (0, )
    elif len(max_shifts) != ndim:
        raise ValueError("max_shifts should have length ndim")

    if np.isscalar(shift_steps):
        shift_steps = (shift_steps, ) * (ndim - mc) + (1, ) * mc
    elif multichannel and len(shift_steps) == ndim - 1:
        shift_steps = tuple(shift_steps) + (1, )
    elif len(shift_steps) != ndim:
        raise ValueError("max_shifts should have length ndim")

    if any(s < 1 for s in shift_steps):
        raise ValueError("shift_steps must all be >= 1")

    if multichannel and max_shifts[-1] != 0:
        raise ValueError(
            "Multichannel cycle spinning should not have shifts along the "
            "last axis.")

    return product(*[range(0, s + 1, t) for
                     s, t in zip(max_shifts, shift_steps)])


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg(multichannel_position=5)
def cycle_spin(x, func, max_shifts, shift_steps=1, num_workers=None,
               multichannel=False, func_kw={}, *, channel_axis=-1):
    """Cycle spinning (repeatedly apply func to shifted versions of x).

    Parameters
    ----------
    x : array-like
        Data for input to ``func``.
    func : function
        A function to apply to circularly shifted versions of ``x``.  Should
        take ``x`` as its first argument. Any additional arguments can be
        supplied via ``func_kw``.
    max_shifts : int or tuple
        If an integer, shifts in ``range(0, max_shifts+1)`` will be used along
        each axis of ``x``. If a tuple, ``range(0, max_shifts[i]+1)`` will be
        along axis i.
    shift_steps : int or tuple, optional
        The step size for the shifts applied along axis, i, are::
        ``range((0, max_shifts[i]+1, shift_steps[i]))``. If an integer is
        provided, the same step size is used for all axes.
    num_workers : int or None, optional
        The number of parallel threads to use during cycle spinning. If set to
        ``None``, the full set of available cores are used.
    multichannel : bool, optional
        Whether to treat the final axis as channels (no cycle shifts are
        performed over the channels axis). This argument is deprecated: specify
        `channel_axis` instead.
    func_kw : dict, optional
        Additional keyword arguments to supply to ``func``.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    avg_y : np.ndarray
        The output of ``func(x, **func_kw)`` averaged over all combinations of
        the specified axis shifts.

    Notes
    -----
    Cycle spinning was proposed as a way to approach shift-invariance via
    performing several circular shifts of a shift-variant transform [1]_.

    For a n-level discrete wavelet transforms, one may wish to perform all
    shifts up to ``max_shifts = 2**n - 1``. In practice, much of the benefit
    can often be realized with only a small number of shifts per axis.

    For transforms such as the blockwise discrete cosine transform, one may
    wish to evaluate shifts up to the block size used by the transform.

    References
    ----------
    .. [1] R.R. Coifman and D.L. Donoho.  "Translation-Invariant De-Noising".
           Wavelets and Statistics, Lecture Notes in Statistics, vol.103.
           Springer, New York, 1995, pp.125-150.
           :DOI:`10.1007/978-1-4612-2544-7_9`

    Examples
    --------
    >>> import skimage.data
    >>> from skimage import img_as_float
    >>> from skimage.restoration import denoise_wavelet, cycle_spin
    >>> img = img_as_float(skimage.data.camera())
    >>> sigma = 0.1
    >>> img = img + sigma * np.random.standard_normal(img.shape)
    >>> denoised = cycle_spin(img, func=denoise_wavelet,
    ...                       max_shifts=3)

    """
    x = np.asanyarray(x)
    multichannel = channel_axis is not None
    all_shifts = _generate_shifts(x.ndim, multichannel, max_shifts,
                                  shift_steps)
    all_shifts = list(all_shifts)
    roll_axes = tuple(range(x.ndim))

    def _run_one_shift(shift):
        # shift, apply function, inverse shift
        xs = np.roll(x, shift, axis=roll_axes)
        tmp = func(xs, **func_kw)
        return np.roll(tmp, tuple(-s for s in shift), axis=roll_axes)

    if not dask_available and (num_workers is None or num_workers > 1):
        num_workers = 1
        warn('The optional dask dependency is not installed. '
             'The number of workers is set to 1. To silence '
             'this warning, install dask or explicitly set `num_workers=1` '
             'when calling the `cycle_spin` function')
    # compute a running average across the cycle shifts
    if num_workers == 1:
        # serial processing
        mean = _run_one_shift(all_shifts[0])
        for shift in all_shifts[1:]:
            mean += _run_one_shift(shift)
        mean /= len(all_shifts)
    else:
        # multithreaded via dask
        futures = [dask.delayed(_run_one_shift)(s) for s in all_shifts]
        mean = sum(futures) / len(futures)
        mean = mean.compute(num_workers=num_workers)
    return mean
