import numpy as np

from .._shared import utils
from .._shared.utils import convert_to_float
from ._nl_means_denoising import (_nl_means_denoising_2d,
                                  _nl_means_denoising_3d,
                                  _fast_nl_means_denoising_2d,
                                  _fast_nl_means_denoising_3d,
                                  _fast_nl_means_denoising_4d)


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg(multichannel_position=4)
def denoise_nl_means(image, patch_size=7, patch_distance=11, h=0.1,
                     multichannel=False, fast_mode=True, sigma=0., *,
                     preserve_range=False, channel_axis=None):
    """Perform non-local means denoising on 2D-4D grayscale or RGB images.

    Parameters
    ----------
    image : 2D or 3D ndarray
        Input image to be denoised, which can be 2D or 3D, and grayscale
        or RGB (for 2D images only, see ``multichannel`` parameter). There can
        be any number of channels (does not strictly have to be RGB).
    patch_size : int, optional
        Size of patches used for denoising.
    patch_distance : int, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : float, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches. A higher h results in a smoother image,
        at the expense of blurring features. For a Gaussian noise of standard
        deviation sigma, a rule of thumb is to choose the value of h to be
        sigma of slightly less.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. This argument is deprecated:
        specify `channel_axis` instead.
    fast_mode : bool, optional
        If True (default value), a fast version of the non-local means
        algorithm is used. If False, the original version of non-local means is
        used. See the Notes section for more details about the algorithms.
    sigma : float, optional
        The standard deviation of the (Gaussian) noise.  If provided, a more
        robust computation of patch weights is computed that takes the expected
        noise variance into account (see Notes below).
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as `image`.

    Notes
    -----

    The non-local means algorithm is well suited for denoising images with
    specific textures. The principle of the algorithm is to average the value
    of a given pixel with values of other pixels in a limited neighbourhood,
    provided that the *patches* centered on the other pixels are similar enough
    to the patch centered on the pixel of interest.

    In the original version of the algorithm [1]_, corresponding to
    ``fast=False``, the computational complexity is::

        image.size * patch_size ** image.ndim * patch_distance ** image.ndim

    Hence, changing the size of patches or their maximal distance has a
    strong effect on computing times, especially for 3-D images.

    However, the default behavior corresponds to ``fast_mode=True``, for which
    another version of non-local means [2]_ is used, corresponding to a
    complexity of::

        image.size * patch_distance ** image.ndim

    The computing time depends only weakly on the patch size, thanks to
    the computation of the integral of patches distances for a given
    shift, that reduces the number of operations [1]_. Therefore, this
    algorithm executes faster than the classic algorithm
    (``fast_mode=False``), at the expense of using twice as much memory.
    This implementation has been proven to be more efficient compared to
    other alternatives, see e.g. [3]_.

    Compared to the classic algorithm, all pixels of a patch contribute
    to the distance to another patch with the same weight, no matter
    their distance to the center of the patch. This coarser computation
    of the distance can result in a slightly poorer denoising
    performance. Moreover, for small images (images with a linear size
    that is only a few times the patch size), the classic algorithm can
    be faster due to boundary effects.

    The image is padded using the `reflect` mode of `skimage.util.pad`
    before denoising.

    If the noise standard deviation, `sigma`, is provided a more robust
    computation of patch weights is used.  Subtracting the known noise variance
    from the computed patch distances improves the estimates of patch
    similarity, giving a moderate improvement to denoising performance [4]_.
    It was also mentioned as an option for the fast variant of the algorithm in
    [3]_.

    When `sigma` is provided, a smaller `h` should typically be used to
    avoid oversmoothing.  The optimal value for `h` depends on the image
    content and noise level, but a reasonable starting point is
    ``h = 0.8 * sigma`` when `fast_mode` is `True`, or ``h = 0.6 * sigma`` when
    `fast_mode` is `False`.

    References
    ----------
    .. [1] A. Buades, B. Coll, & J-M. Morel. A non-local algorithm for image
           denoising. In CVPR 2005, Vol. 2, pp. 60-65, IEEE.
           :DOI:`10.1109/CVPR.2005.38`

    .. [2] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
           nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
           International Symposium on Biomedical Imaging: From Nano to Macro,
           2008, pp. 1331-1334.
           :DOI:`10.1109/ISBI.2008.4541250`

    .. [3] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
           Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
           :DOI:`10.5201/ipol.2014.120`

    .. [4] A. Buades, B. Coll, & J-M. Morel. Non-Local Means Denoising.
           Image Processing On Line, 2011, vol. 1, pp. 208-212.
           :DOI:`10.5201/ipol.2011.bcm_nlm`

    Examples
    --------
    >>> a = np.zeros((40, 40))
    >>> a[10:-10, 10:-10] = 1.
    >>> rng = np.random.default_rng()
    >>> a += 0.3 * rng.standard_normal(a.shape)
    >>> denoised_a = denoise_nl_means(a, 7, 5, 0.1)
    """
    if channel_axis is None:
        multichannel = False
        image = image[..., np.newaxis]
    else:
        multichannel = True

    ndim_no_channel = image.ndim - 1
    if (ndim_no_channel < 2) or (ndim_no_channel > 4):
        raise NotImplementedError(
            "Non-local means denoising is only implemented for 2D, "
            "3D or 4D grayscale or multichannel images.")

    image = convert_to_float(image, preserve_range)
    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)

    kwargs = dict(s=patch_size, d=patch_distance, h=h, var=sigma * sigma)
    if ndim_no_channel == 2:
        nlm_func = (_fast_nl_means_denoising_2d if fast_mode else
                    _nl_means_denoising_2d)
    elif ndim_no_channel == 3:
        if multichannel and not fast_mode:
            raise NotImplementedError(
                "Multichannel 3D requires fast_mode to be True.")
        if fast_mode:
            nlm_func = _fast_nl_means_denoising_3d
        else:
            # have to drop the size 1 channel axis for slow mode
            image = image[..., 0]
            nlm_func = _nl_means_denoising_3d
    elif ndim_no_channel == 4:
        if fast_mode:
            nlm_func = _fast_nl_means_denoising_4d
        else:
            raise NotImplementedError("4D requires fast_mode to be True.")
    dn = np.asarray(nlm_func(image, **kwargs))
    return dn
