import itertools
import functools

import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import _supported_float_type
from ..metrics import mean_squared_error
from ..util import img_as_float



def _interpolate_image(image, *, multichannel=False):
    """Replacing each pixel in ``image`` with the average of its neighbors.

    Parameters
    ----------
    image : ndarray
        Input data to be interpolated.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.

    Returns
    -------
    interp : ndarray
        Interpolated version of `image`.
    """
    spatialdims = image.ndim if not multichannel else image.ndim - 1
    conv_filter = ndi.generate_binary_structure(spatialdims, 1).astype(image.dtype)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()

    if multichannel:
        interp = np.zeros_like(image)
        for i in range(image.shape[-1]):
            interp[..., i] = ndi.convolve(image[..., i], conv_filter,
                                          mode='mirror')
    else:
        interp = ndi.convolve(image, conv_filter, mode='mirror')
    return interp


def _generate_grid_slice(shape, *, offset, stride=3):
    """Generate slices of uniformly-spaced points in an array.

    Parameters
    ----------
    shape : tuple of int
        Shape of the mask.
    offset : int
        The offset of the grid of ones. Iterating over ``offset`` will cover
        the entire array. It should be between 0 and ``stride ** ndim``, not
        inclusive, where ``ndim = len(shape)``.
    stride : int, optional
        The spacing between ones, used in each dimension.

    Returns
    -------
    mask : ndarray
        The mask.

    Examples
    --------
    >>> shape = (4, 4)
    >>> array = np.zeros(shape, dtype=int)
    >>> grid_slice = _generate_grid_slice(shape, offset=0, stride=2)
    >>> array[grid_slice] = 1
    >>> print(array)
    [[1 0 1 0]
     [0 0 0 0]
     [1 0 1 0]
     [0 0 0 0]]

    Changing the offset moves the location of the 1s:

    >>> array = np.zeros(shape, dtype=int)
    >>> grid_slice = _generate_grid_slice(shape, offset=3, stride=2)
    >>> array[grid_slice] = 1
    >>> print(array)
    [[0 0 0 0]
     [0 1 0 1]
     [0 0 0 0]
     [0 1 0 1]]
    """
    phases = np.unravel_index(offset, (stride,) * len(shape))
    mask = tuple(slice(p, None, stride) for p in phases)

    return mask


def _invariant_denoise(image, denoise_function, *, stride=4,
                       masks=None, denoiser_kwargs=None):
    """Apply a J-invariant version of `denoise_function`.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Original denoising function.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    masks : list of ndarray, optional
        Set of masks to use for computing J-invariant output. If `None`,
        a full set of masks covering the image will be used.
    denoiser_kwargs:
        Keyword arguments passed to `denoise_function`.

    Returns
    -------
    output : ndarray
        Denoised image, of same shape as `image`.
    """
    image = img_as_float(image)

    # promote float16->float32 if needed
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    if denoiser_kwargs is None:
        denoiser_kwargs = {}


    if 'multichannel' in denoiser_kwargs:
        multichannel = denoiser_kwargs['multichannel']
    else:
        multichannel = denoiser_kwargs.get('channel_axis', None) is not None

    interp = _interpolate_image(image, multichannel=multichannel)
    output = np.zeros_like(image)

    if masks is None:
        spatialdims = image.ndim if not multichannel else image.ndim - 1
        n_masks = stride ** spatialdims
        masks = (_generate_grid_slice(image.shape[:spatialdims],
                                      offset=idx, stride=stride)
                 for idx in range(n_masks))

    for mask in masks:
        input_image = image.copy()
        input_image[mask] = interp[mask]
        output[mask] = denoise_function(input_image, **denoiser_kwargs)[mask]
    return output


def _product_from_dict(dictionary):
    """Utility function to convert parameter ranges to parameter combinations.

    Converts a dict of lists into a list of dicts whose values consist of the
    cartesian product of the values in the original dict.

    Parameters
    ----------
    dictionary : dict of lists
        Dictionary of lists to be multiplied.

    Yields
    ------
    selections : dicts of values
        Dicts containing individual combinations of the values in the input
        dict.
    """
    keys = dictionary.keys()
    for element in itertools.product(*dictionary.values()):
        yield dict(zip(keys, element))


def calibrate_denoiser(image, denoise_function, denoise_parameters, *,
                       stride=4, approximate_loss=True,
                       extra_output=False):
    """Calibrate a denoising function and return optimal J-invariant version.

    The returned function is partially evaluated with optimal parameter values
    set for denoising the input image.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Denoising function to be calibrated.
    denoise_parameters : dict of list
        Ranges of parameters for `denoise_function` to be calibrated over.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    approximate_loss : bool, optional
        Whether to approximate the self-supervised loss used to evaluate the
        denoiser by only computing it on one masked version of the image.
        If False, the runtime will be a factor of `stride**image.ndim` longer.
    extra_output : bool, optional
        If True, return parameters and losses in addition to the calibrated
        denoising function

    Returns
    -------
    best_denoise_function : function
        The optimal J-invariant version of `denoise_function`.

    If `extra_output` is True, the following tuple is also returned:

    (parameters_tested, losses) : tuple (list of dict, list of int)
        List of parameters tested for `denoise_function`, as a dictionary of
        kwargs
        Self-supervised loss for each set of parameters in `parameters_tested`.


    Notes
    -----

    The calibration procedure uses a self-supervised mean-square-error loss
    to evaluate the performance of J-invariant versions of `denoise_function`.
    The minimizer of the self-supervised loss is also the minimizer of the
    ground-truth loss (i.e., the true MSE error) [1]. The returned function
    can be used on the original noisy image, or other images with similar
    characteristics.

    Increasing the stride increases the performance of `best_denoise_function`
     at the expense of increasing its runtime. It has no effect on the runtime
     of the calibration.

    References
    ----------
    .. [1] J. Batson & L. Royer. Noise2Self: Blind Denoising by Self-Supervision,
           International Conference on Machine Learning, p. 524-533 (2019).

    Examples
    --------

    >>> from skimage import color, data
    >>> from skimage.restoration import denoise_wavelet
    >>> import numpy as np
    >>> img = color.rgb2gray(data.astronaut()[:50, :50])
    >>> rng = np.random.default_rng()
    >>> noisy = img + 0.5 * img.std() * rng.standard_normal(img.shape)
    >>> parameters = {'sigma': np.arange(0.1, 0.4, 0.02)}
    >>> denoising_function = calibrate_denoiser(noisy, denoise_wavelet,
    ...                                         denoise_parameters=parameters)
    >>> denoised_img = denoising_function(img)

    """
    parameters_tested, losses = _calibrate_denoiser_search(
        image, denoise_function,
        denoise_parameters=denoise_parameters,
        stride=stride,
        approximate_loss=approximate_loss
    )

    idx = np.argmin(losses)
    best_parameters = parameters_tested[idx]

    best_denoise_function = functools.partial(
        _invariant_denoise,
        denoise_function=denoise_function,
        stride=stride,
        denoiser_kwargs=best_parameters,
    )

    if extra_output:
        return best_denoise_function, (parameters_tested, losses)
    else:
        return best_denoise_function


def _calibrate_denoiser_search(image, denoise_function, denoise_parameters, *,
                               stride=4, approximate_loss=True):
    """Return a parameter search history with losses for a denoise function.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Denoising function to be calibrated.
    denoise_parameters : dict of list
        Ranges of parameters for `denoise_function` to be calibrated over.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    approximate_loss : bool, optional
        Whether to approximate the self-supervised loss used to evaluate the
        denoiser by only computing it on one masked version of the image.
        If False, the runtime will be a factor of `stride**image.ndim` longer.

    Returns
    -------
    parameters_tested : list of dict
        List of parameters tested for `denoise_function`, as a dictionary of
        kwargs.
    losses : list of int
        Self-supervised loss for each set of parameters in `parameters_tested`.
    """
    image = img_as_float(image)
    parameters_tested = list(_product_from_dict(denoise_parameters))
    losses = []

    for denoiser_kwargs in parameters_tested:
        if 'multichannel' in denoiser_kwargs:
            multichannel = denoiser_kwargs['multichannel']
        else:
            multichannel = \
                denoiser_kwargs.get('channel_axis', None) is not None
        if not approximate_loss:
            denoised = _invariant_denoise(
                image, denoise_function,
                stride=stride,
                denoiser_kwargs=denoiser_kwargs
            )
            loss = mean_squared_error(image, denoised)
        else:
            spatialdims = image.ndim if not multichannel else image.ndim - 1
            n_masks = stride ** spatialdims
            mask = _generate_grid_slice(image.shape[:spatialdims],
                                        offset=n_masks // 2, stride=stride)

            masked_denoised = _invariant_denoise(
                image, denoise_function,
                masks=[mask],
                denoiser_kwargs=denoiser_kwargs
            )

            loss = mean_squared_error(image[mask], masked_denoised[mask])

        losses.append(loss)

    return parameters_tested, losses
