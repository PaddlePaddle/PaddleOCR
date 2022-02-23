"""
Augmenters that perform contrast changes.

List of augmenters:

    * :class:`GammaContrast`
    * :class:`SigmoidContrast`
    * :class:`LogContrast`
    * :class:`LinearContrast`
    * :class:`AllChannelsHistogramEqualization`
    * :class:`HistogramEqualization`
    * :class:`AllChannelsCLAHE`
    * :class:`CLAHE`

"""
from __future__ import print_function, division, absolute_import

import numpy as np
import six.moves as sm
import skimage.exposure as ski_exposure
import cv2

import imgaug as ia
from imgaug.imgaug import _normalize_cv2_input_arr_
from . import meta
from . import color as color_lib
from .. import parameters as iap
from .. import dtypes as iadt
from ..augmentables.batches import _BatchInAugmentation


class _ContrastFuncWrapper(meta.Augmenter):
    def __init__(self, func, params1d, per_channel, dtypes_allowed=None,
                 dtypes_disallowed=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(_ContrastFuncWrapper, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.func = func
        self.params1d = params1d
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")
        self.dtypes_allowed = dtypes_allowed
        self.dtypes_disallowed = dtypes_disallowed

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        if self.dtypes_allowed is not None:
            iadt.gate_dtypes(images,
                             allowed=self.dtypes_allowed,
                             disallowed=self.dtypes_disallowed,
                             augmenter=self)

        nb_images = len(images)
        rss = random_state.duplicate(1+nb_images)
        per_channel = self.per_channel.draw_samples((nb_images,),
                                                    random_state=rss[0])

        gen = enumerate(zip(images, per_channel, rss[1:]))
        for i, (image, per_channel_i, rs) in gen:
            nb_channels = 1 if per_channel_i <= 0.5 else image.shape[2]
            # TODO improve efficiency by sampling once
            samples_i = [
                param.draw_samples((nb_channels,), random_state=rs)
                for param in self.params1d]
            if per_channel_i > 0.5:
                input_dtype = image.dtype
                # TODO This was previously a cast of image to float64. Do the
                #      adjust_* functions return float64?
                result = []
                for c in sm.xrange(nb_channels):
                    samples_i_c = [sample_i[c] for sample_i in samples_i]
                    args = tuple([image[..., c]] + samples_i_c)
                    result.append(self.func(*args))
                image_aug = np.stack(result, axis=-1)
                image_aug = image_aug.astype(input_dtype)
            else:
                # don't use something like samples_i[...][0] here, because
                # that returns python scalars and is slightly less accurate
                # than keeping the numpy values
                args = tuple([image] + samples_i)
                image_aug = self.func(*args)
            batch.images[i] = image_aug
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return self.params1d


# TODO quite similar to the other adjust_contrast_*() functions, make DRY
def adjust_contrast_gamma(arr, gamma):
    """
    Adjust image contrast by scaling pixel values to ``255*((v/255)**gamma)``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1) (2) (3)
        * ``uint16``: yes; tested (2) (3)
        * ``uint32``: yes; tested (2) (3)
        * ``uint64``: yes; tested (2) (3) (4)
        * ``int8``: limited; tested (2) (3) (5)
        * ``int16``: limited; tested (2) (3) (5)
        * ``int32``: limited; tested (2) (3) (5)
        * ``int64``: limited; tested (2) (3) (4) (5)
        * ``float16``: limited; tested (5)
        * ``float32``: limited; tested (5)
        * ``float64``: limited; tested (5)
        * ``float128``: no (6)
        * ``bool``: no (7)

        - (1) Handled by ``cv2``. Other dtypes are handled by ``skimage``.
        - (2) Normalization is done as ``I_ij/max``, where ``max`` is the
              maximum value of the dtype, e.g. 255 for ``uint8``. The
              normalization is reversed afterwards, e.g. ``result*255`` for
              ``uint8``.
        - (3) Integer-like values are not rounded after applying the contrast
              adjustment equation (before inverting the normalization to
              ``[0.0, 1.0]`` space), i.e. projection from continuous
              space to discrete happens according to floor function.
        - (4) Note that scikit-image doc says that integers are converted to
              ``float64`` values before applying the contrast normalization
              method. This might lead to inaccuracies for large 64bit integer
              values. Tests showed no indication of that happening though.
        - (5) Must not contain negative values. Values >=0 are fully supported.
        - (6) Leads to error in scikit-image.
        - (7) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    gamma : number
        Exponent for the contrast adjustment. Higher values darken the image.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    if arr.size == 0:
        return np.copy(arr)

    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT ,
    # but here it seemed like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype.name == "uint8":
        min_value, _center_value, max_value = \
            iadt.get_value_range_of_dtype(arr.dtype)
        dynamic_range = max_value - min_value

        value_range = np.linspace(0, 1.0, num=dynamic_range+1,
                                  dtype=np.float32)

        # 255 * ((I_ij/255)**gamma)
        # using np.float32(.) here still works when the input is a numpy array
        # of size 1
        table = (min_value
                 + (value_range ** np.float32(gamma))
                 * dynamic_range)
        table = np.clip(table, min_value, max_value).astype(arr.dtype)
        arr_aug = ia.apply_lut(arr, table)
        return arr_aug
    return ski_exposure.adjust_gamma(arr, gamma)


# TODO quite similar to the other adjust_contrast_*() functions, make DRY
def adjust_contrast_sigmoid(arr, gain, cutoff):
    """
    Adjust image contrast to ``255*1/(1+exp(gain*(cutoff-I_ij/255)))``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1) (2) (3)
        * ``uint16``: yes; tested (2) (3)
        * ``uint32``: yes; tested (2) (3)
        * ``uint64``: yes; tested (2) (3) (4)
        * ``int8``: limited; tested (2) (3) (5)
        * ``int16``: limited; tested (2) (3) (5)
        * ``int32``: limited; tested (2) (3) (5)
        * ``int64``: limited; tested (2) (3) (4) (5)
        * ``float16``: limited; tested (5)
        * ``float32``: limited; tested (5)
        * ``float64``: limited; tested (5)
        * ``float128``: no (6)
        * ``bool``: no (7)

        - (1) Handled by ``cv2``. Other dtypes are handled by ``skimage``.
        - (2) Normalization is done as ``I_ij/max``, where ``max`` is the
              maximum value of the dtype, e.g. 255 for ``uint8``. The
              normalization is reversed afterwards, e.g. ``result*255``
              for ``uint8``.
        - (3) Integer-like values are not rounded after applying the contrast
              adjustment equation before inverting the normalization
              to ``[0.0, 1.0]`` space), i.e. projection from continuous
              space to discrete happens according to floor function.
        - (4) Note that scikit-image doc says that integers are converted to
              ``float64`` values before applying the contrast normalization
              method. This might lead to inaccuracies for large 64bit integer
              values. Tests showed no indication of that happening though.
        - (5) Must not contain negative values. Values >=0 are fully supported.
        - (6) Leads to error in scikit-image.
        - (7) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    gain : number
        Multiplier for the sigmoid function's output.
        Higher values lead to quicker changes from dark to light pixels.

    cutoff : number
        Cutoff that shifts the sigmoid function in horizontal direction.
        Higher values mean that the switch from dark to light pixels happens
        later, i.e. the pixels will remain darker.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    if arr.size == 0:
        return np.copy(arr)

    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT ,
    # but here it seemed like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype.name == "uint8":
        min_value, _center_value, max_value = \
            iadt.get_value_range_of_dtype(arr.dtype)
        dynamic_range = max_value - min_value

        value_range = np.linspace(0, 1.0, num=dynamic_range+1,
                                  dtype=np.float32)

        # 255 * 1/(1 + exp(gain*(cutoff - I_ij/255)))
        # using np.float32(.) here still works when the input is a numpy array
        # of size 1
        gain = np.float32(gain)
        cutoff = np.float32(cutoff)
        table = (min_value
                 + dynamic_range
                 * 1/(1 + np.exp(gain * (cutoff - value_range))))
        table = np.clip(table, min_value, max_value).astype(arr.dtype)
        arr_aug = ia.apply_lut(arr, table)
        return arr_aug
    return ski_exposure.adjust_sigmoid(arr, cutoff=cutoff, gain=gain)


# TODO quite similar to the other adjust_contrast_*() functions, make DRY
# TODO add dtype gating
def adjust_contrast_log(arr, gain):
    """
    Adjust image contrast by scaling pixels to ``255*gain*log_2(1+v/255)``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1) (2) (3)
        * ``uint16``: yes; tested (2) (3)
        * ``uint32``: no; tested (2) (3) (8)
        * ``uint64``: no; tested (2) (3) (4) (8)
        * ``int8``: limited; tested (2) (3) (5)
        * ``int16``: limited; tested (2) (3) (5)
        * ``int32``: no; tested (2) (3) (5) (8)
        * ``int64``: no; tested (2) (3) (4) (5) (8)
        * ``float16``: limited; tested (5)
        * ``float32``: limited; tested (5)
        * ``float64``: limited; tested (5)
        * ``float128``: no (6)
        * ``bool``: no (7)

        - (1) Handled by ``cv2``. Other dtypes are handled by ``skimage``.
        - (2) Normalization is done as ``I_ij/max``, where ``max`` is the
              maximum value of the dtype, e.g. 255 for ``uint8``. The
              normalization is reversed afterwards, e.g. ``result*255`` for
              ``uint8``.
        - (3) Integer-like values are not rounded after applying the contrast
              adjustment equation (before inverting the normalization
              to ``[0.0, 1.0]`` space), i.e. projection from continuous
              space to discrete happens according to floor function.
        - (4) Note that scikit-image doc says that integers are converted to
              ``float64`` values before applying the contrast normalization
              method. This might lead to inaccuracies for large 64bit integer
              values. Tests showed no indication of that happening though.
        - (5) Must not contain negative values. Values >=0 are fully supported.
        - (6) Leads to error in scikit-image.
        - (7) Does not make sense for contrast adjustments.
        - (8) No longer supported since numpy 1.17. Previously: 'yes' for
              ``uint32``, ``uint64``; 'limited' for ``int32``, ``int64``.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    gain : number
        Multiplier for the logarithm result. Values around 1.0 lead to a
        contrast-adjusted images. Values above 1.0 quickly lead to partially
        broken images due to exceeding the datatype's value range.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    if arr.size == 0:
        return np.copy(arr)

    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT ,
    # but here it seemed like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype.name == "uint8":
        min_value, _center_value, max_value = \
            iadt.get_value_range_of_dtype(arr.dtype)
        dynamic_range = max_value - min_value

        value_range = np.linspace(0, 1.0, num=dynamic_range+1,
                                  dtype=np.float32)

        # 255 * 1/(1 + exp(gain*(cutoff - I_ij/255)))
        # using np.float32(.) here still works when the input is a numpy array
        # of size 1
        gain = np.float32(gain)
        table = min_value + dynamic_range * gain * np.log2(1 + value_range)
        table = np.clip(table, min_value, max_value).astype(arr.dtype)
        arr_aug = ia.apply_lut(arr, table)
        return arr_aug
    return ski_exposure.adjust_log(arr, gain=gain)


# TODO quite similar to the other adjust_contrast_*() functions, make DRY
def adjust_contrast_linear(arr, alpha):
    """Adjust contrast by scaling each pixel to ``127 + alpha*(v-127)``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1) (2)
        * ``uint16``: yes; tested (2)
        * ``uint32``: yes; tested (2)
        * ``uint64``: no (3)
        * ``int8``: yes; tested (2)
        * ``int16``: yes; tested (2)
        * ``int32``: yes; tested (2)
        * ``int64``: no (2)
        * ``float16``: yes; tested (2)
        * ``float32``: yes; tested (2)
        * ``float64``: yes; tested (2)
        * ``float128``: no (2)
        * ``bool``: no (4)

        - (1) Handled by ``cv2``. Other dtypes are handled by raw ``numpy``.
        - (2) Only tested for reasonable alphas with up to a value of
              around ``100``.
        - (3) Conversion to ``float64`` is done during augmentation, hence
              ``uint64``, ``int64``, and ``float128`` support cannot be
              guaranteed.
        - (4) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    alpha : number
        Multiplier to linearly pronounce (``>1.0``), dampen (``0.0`` to
        ``1.0``) or invert (``<0.0``) the difference between each pixel value
        and the dtype's center value, e.g. ``127`` for ``uint8``.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    # pylint: disable=no-else-return
    if arr.size == 0:
        return np.copy(arr)

    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT ,
    # but here it seemed like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype.name == "uint8":
        min_value, center_value, max_value = \
            iadt.get_value_range_of_dtype(arr.dtype)
        # TODO get rid of this int(...)
        center_value = int(center_value)

        value_range = np.arange(0, 256, dtype=np.float32)

        # 127 + alpha*(I_ij-127)
        # using np.float32(.) here still works when the input is a numpy array
        # of size 1
        alpha = np.float32(alpha)
        table = center_value + alpha * (value_range - center_value)
        table = np.clip(table, min_value, max_value).astype(arr.dtype)
        arr_aug = ia.apply_lut(arr, table)
        return arr_aug
    else:
        input_dtype = arr.dtype
        _min_value, center_value, _max_value = \
            iadt.get_value_range_of_dtype(input_dtype)
        # TODO get rid of this int(...)
        if input_dtype.kind in ["u", "i"]:
            center_value = int(center_value)
        image_aug = (center_value
                     + alpha
                     * (arr.astype(np.float64)-center_value))
        image_aug = iadt.restore_dtypes_(image_aug, input_dtype)
        return image_aug


class GammaContrast(_ContrastFuncWrapper):
    """
    Adjust image contrast by scaling pixel values to ``255*((v/255)**gamma)``.

    Values in the range ``gamma=(0.5, 2.0)`` seem to be sensible.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.contrast.adjust_contrast_gamma`.

    Parameters
    ----------
    gamma : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Exponent for the contrast adjustment. Higher values darken the image.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]``
              will be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.GammaContrast((0.5, 2.0))

    Modify the contrast of images according to ``255*((v/255)**gamma)``,
    where ``v`` is a pixel value and ``gamma`` is sampled uniformly from
    the interval ``[0.5, 2.0]`` (once per image).

    >>> aug = iaa.GammaContrast((0.5, 2.0), per_channel=True)

    Same as in the previous example, but ``gamma`` is sampled once per image
    *and* channel.

    """

    def __init__(self, gamma=(0.7, 1.7), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        params1d = [iap.handle_continuous_param(
            gamma, "gamma", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)]
        func = adjust_contrast_gamma
        super(GammaContrast, self).__init__(
            func, params1d, per_channel,
            dtypes_allowed=["uint8", "uint16", "uint32", "uint64",
                            "int8", "int16", "int32", "int64",
                            "float16", "float32", "float64"],
            dtypes_disallowed=["float96", "float128", "float256", "bool"],
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class SigmoidContrast(_ContrastFuncWrapper):
    """
    Adjust image contrast to ``255*1/(1+exp(gain*(cutoff-I_ij/255)))``.

    Values in the range ``gain=(5, 20)`` and ``cutoff=(0.25, 0.75)`` seem to
    be sensible.

    A combination of ``gain=5.5`` and ``cutof=0.45`` is fairly close to
    the identity function.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.contrast.adjust_contrast_sigmoid`.

    Parameters
    ----------
    gain : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier for the sigmoid function's output.
        Higher values lead to quicker changes from dark to light pixels.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled uniformly per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    cutoff : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Cutoff that shifts the sigmoid function in horizontal direction.
        Higher values mean that the switch from dark to light pixels happens
        later, i.e. the pixels will remain darker.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]``
              will be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))

    Modify the contrast of images according to
    ``255*1/(1+exp(gain*(cutoff-v/255)))``, where ``v`` is a pixel value,
    ``gain`` is sampled uniformly from the interval ``[3, 10]`` (once per
    image) and ``cutoff`` is sampled uniformly from the interval
    ``[0.4, 0.6]`` (also once per image).

    >>> aug = iaa.SigmoidContrast(
    >>>     gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)

    Same as in the previous example, but ``gain`` and ``cutoff`` are each
    sampled once per image *and* channel.

    """
    def __init__(self, gain=(5, 6), cutoff=(0.3, 0.6), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # TODO add inv parameter?
        params1d = [
            iap.handle_continuous_param(
                gain, "gain", value_range=(0, None), tuple_to_uniform=True,
                list_to_choice=True),
            iap.handle_continuous_param(
                cutoff, "cutoff", value_range=(0, 1.0), tuple_to_uniform=True,
                list_to_choice=True)
        ]
        func = adjust_contrast_sigmoid

        super(SigmoidContrast, self).__init__(
            func, params1d, per_channel,
            dtypes_allowed=["uint8", "uint16", "uint32", "uint64",
                            "int8", "int16", "int32", "int64",
                            "float16", "float32", "float64"],
            dtypes_disallowed=["float96", "float128", "float256", "bool"],
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class LogContrast(_ContrastFuncWrapper):
    """Adjust image contrast by scaling pixels to ``255*gain*log_2(1+v/255)``.

    This augmenter is fairly similar to
    ``imgaug.augmenters.arithmetic.Multiply``.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.contrast.adjust_contrast_log`.

    Parameters
    ----------
    gain : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier for the logarithm result. Values around ``1.0`` lead to a
        contrast-adjusted images. Values above ``1.0`` quickly lead to
        partially broken images due to exceeding the datatype's value range.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will uniformly sampled be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.LogContrast(gain=(0.6, 1.4))

    Modify the contrast of images according to ``255*gain*log_2(1+v/255)``,
    where ``v`` is a pixel value and ``gain`` is sampled uniformly from the
    interval ``[0.6, 1.4]`` (once per image).

    >>> aug = iaa.LogContrast(gain=(0.6, 1.4), per_channel=True)

    Same as in the previous example, but ``gain`` is sampled once per image
    *and* channel.

    """
    def __init__(self, gain=(0.4, 1.6), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # TODO add inv parameter?
        params1d = [iap.handle_continuous_param(
            gain, "gain", value_range=(0, None), tuple_to_uniform=True,
            list_to_choice=True)]
        func = adjust_contrast_log

        super(LogContrast, self).__init__(
            func, params1d, per_channel,
            dtypes_allowed=["uint8", "uint16", "uint32", "uint64",
                            "int8", "int16", "int32", "int64",
                            "float16", "float32", "float64"],
            dtypes_disallowed=["float96", "float128", "float256", "bool"],
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class LinearContrast(_ContrastFuncWrapper):
    """Adjust contrast by scaling each pixel to ``127 + alpha*(v-127)``.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.contrast.adjust_contrast_linear`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier to linearly pronounce (``>1.0``), dampen (``0.0`` to
        ``1.0``) or invert (``<0.0``) the difference between each pixel value
        and the dtype's center value, e.g. ``127`` for ``uint8``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.LinearContrast((0.4, 1.6))

    Modify the contrast of images according to `127 + alpha*(v-127)``,
    where ``v`` is a pixel value and ``alpha`` is sampled uniformly from the
    interval ``[0.4, 1.6]`` (once per image).

    >>> aug = iaa.LinearContrast((0.4, 1.6), per_channel=True)

    Same as in the previous example, but ``alpha`` is sampled once per image
    *and* channel.

    """
    def __init__(self, alpha=(0.6, 1.4), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        params1d = [
            iap.handle_continuous_param(
                alpha, "alpha", value_range=None, tuple_to_uniform=True,
                list_to_choice=True)
        ]
        func = adjust_contrast_linear

        super(LinearContrast, self).__init__(
            func, params1d, per_channel,
            dtypes_allowed=["uint8", "uint16", "uint32",
                            "int8", "int16", "int32",
                            "float16", "float32", "float64"],
            dtypes_disallowed=["uint64", "int64", "float96", "float128",
                               "float256", "bool"],
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# TODO maybe offer the other contrast augmenters also wrapped in this, similar
#      to CLAHE and HistogramEqualization?
#      this is essentially tested by tests for CLAHE
class _IntensityChannelBasedApplier(object):
    RGB = color_lib.CSPACE_RGB
    BGR = color_lib.CSPACE_BGR
    HSV = color_lib.CSPACE_HSV
    HLS = color_lib.CSPACE_HLS
    Lab = color_lib.CSPACE_Lab
    _CHANNEL_MAPPING = {
        HSV: 2,
        HLS: 1,
        Lab: 0
    }

    def __init__(self, from_colorspace, to_colorspace):
        super(_IntensityChannelBasedApplier, self).__init__()

        # TODO maybe add CIE, Luv?
        valid_from_colorspaces = [self.RGB, self.BGR, self.Lab, self.HLS,
                                  self.HSV]
        assert from_colorspace in valid_from_colorspaces, (
            "Expected 'from_colorspace' to be one of %s, got %s." % (
                valid_from_colorspaces, from_colorspace))

        valid_to_colorspaces = [self.Lab, self.HLS, self.HSV]
        assert to_colorspace in valid_to_colorspaces, (
            "Expected 'to_colorspace' to be one of %s, got %s." % (
                valid_to_colorspaces, to_colorspace))

        self.from_colorspace = from_colorspace
        self.to_colorspace = to_colorspace

    def apply(self, images, random_state, parents, hooks, func):
        input_was_array = ia.is_np_array(images)
        rss = random_state.duplicate(3)

        # normalize images
        # (H, W, 1)      will be used directly in AllChannelsCLAHE
        # (H, W, 3)      will be converted to target colorspace in the next
        #                block
        # (H, W, 4)      will be reduced to (H, W, 3) (remove 4th channel) and
        #                converted to target colorspace in next block
        # (H, W, <else>) will raise a warning and be treated channelwise by
        #                AllChannelsCLAHE
        images_normalized = []
        images_change_cs = []
        images_change_cs_indices = []
        for i, image in enumerate(images):
            nb_channels = image.shape[2]
            if nb_channels == 1:
                images_normalized.append(image)
            elif nb_channels == 3:
                images_normalized.append(None)
                images_change_cs.append(image)
                images_change_cs_indices.append(i)
            elif nb_channels == 4:
                # assume that 4th channel is an alpha channel, e.g. in RGBA
                images_normalized.append(None)
                images_change_cs.append(image[..., 0:3])
                images_change_cs_indices.append(i)
            else:
                ia.warn(
                    "Got image with %d channels in "
                    "_IntensityChannelBasedApplier (parents: %s), "
                    "expected 0, 1, 3 or 4 channels." % (
                        nb_channels, ", ".join(
                            parent.name for parent in parents)))
                images_normalized.append(image)

        # convert colorspaces of normalized 3-channel images
        images_after_color_conversion = [None] * len(images_normalized)
        if len(images_change_cs) > 0:
            images_new_cs = color_lib.change_colorspaces_(
                images_change_cs,
                to_colorspaces=self.to_colorspace,
                from_colorspaces=self.from_colorspace)

            for image_new_cs, target_idx in zip(images_new_cs,
                                                images_change_cs_indices):
                chan_idx = self._CHANNEL_MAPPING[self.to_colorspace]
                images_normalized[target_idx] = image_new_cs[
                    ..., chan_idx:chan_idx+1]
                images_after_color_conversion[target_idx] = image_new_cs

        # apply function channelwise
        images_aug = func(images_normalized, rss[1])

        # denormalize
        result = []
        images_change_cs = []
        images_change_cs_indices = []
        gen = enumerate(zip(images, images_after_color_conversion, images_aug))
        for i, (image, image_conv, image_aug) in gen:
            nb_channels = image.shape[2]
            if nb_channels in [3, 4]:
                chan_idx = self._CHANNEL_MAPPING[self.to_colorspace]
                image_tmp = image_conv
                image_tmp[..., chan_idx:chan_idx+1] = image_aug

                result.append(None if nb_channels == 3 else image[..., 3:4])
                images_change_cs.append(image_tmp)
                images_change_cs_indices.append(i)
            else:
                result.append(image_aug)

        # invert colorspace conversion
        if len(images_change_cs) > 0:
            images_new_cs = color_lib.change_colorspaces_(
                images_change_cs,
                to_colorspaces=self.from_colorspace,
                from_colorspaces=self.to_colorspace)
            for image_new_cs, target_idx in zip(images_new_cs,
                                                images_change_cs_indices):
                if result[target_idx] is None:
                    result[target_idx] = image_new_cs
                else:
                    # input image had four channels, 4th channel is already
                    # in result
                    result[target_idx] = np.dstack((image_new_cs,
                                                    result[target_idx]))

        # convert to array if necessary
        if input_was_array:
            result = np.array(result, dtype=result[0].dtype)

        return result

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.from_colorspace, self.to_colorspace]


# TODO add parameter `tile_grid_size_percent`
class AllChannelsCLAHE(meta.Augmenter):
    """Apply CLAHE to all channels of images in their original colorspaces.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) performs
    histogram equilization within image patches, i.e. over local
    neighbourhoods.

    In contrast to ``imgaug.augmenters.contrast.CLAHE``, this augmenter
    operates directly on all channels of the input images. It does not
    perform any colorspace transformations and does not focus on specific
    channels (e.g. ``L`` in ``Lab`` colorspace).

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: no (2)
        * ``int16``: no (2)
        * ``int32``: no (2)
        * ``int64``: no (2)
        * ``float16``: no (2)
        * ``float32``: no (2)
        * ``float64``: no (2)
        * ``float128``: no (1)
        * ``bool``: no (1)

        - (1) rejected by cv2
        - (2) results in error in cv2: ``cv2.error:
              OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion
              failed) src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
              || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in
              function 'apply'``

    Parameters
    ----------
    clip_limit : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See ``imgaug.augmenters.contrast.CLAHE``.

    tile_grid_size_px : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        See ``imgaug.augmenters.contrast.CLAHE``.

    tile_grid_size_px_min : int, optional
        See ``imgaug.augmenters.contrast.CLAHE``.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AllChannelsCLAHE()

    Create an augmenter that applies CLAHE to all channels of input images.

    >>> aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10))

    Same as in the previous example, but the `clip_limit` used by CLAHE is
    uniformly sampled per image from the interval ``[1, 10]``. Some images
    will therefore have stronger contrast than others (i.e. higher clip limit
    values).

    >>> aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)

    Same as in the previous example, but the `clip_limit` is sampled per
    image *and* channel, leading to different levels of contrast for each
    channel.

    """

    def __init__(self, clip_limit=(0.1, 8), tile_grid_size_px=(3, 12),
                 tile_grid_size_px_min=3, per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(AllChannelsCLAHE, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.clip_limit = iap.handle_continuous_param(
            clip_limit, "clip_limit", value_range=(0+1e-4, None),
            tuple_to_uniform=True, list_to_choice=True)
        self.tile_grid_size_px = iap.handle_discrete_kernel_size_param(
            tile_grid_size_px, "tile_grid_size_px", value_range=(0, None),
            allow_floats=False)
        self.tile_grid_size_px_min = tile_grid_size_px_min
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        iadt.gate_dtypes(
            images,
            allowed=["uint8", "uint16"],
            disallowed=["bool",
                        "uint32", "uint64", "uint128", "uint256",
                        "int8", "int16", "int32", "int64", "int128", "int256",
                        "float16", "float32", "float64", "float96",
                        "float128", "float256"],
            augmenter=self)

        nb_images = len(images)
        nb_channels = meta.estimate_max_number_of_channels(images)

        mode = "single" if self.tile_grid_size_px[1] is None else "two"
        rss = random_state.duplicate(3 if mode == "single" else 4)
        per_channel = self.per_channel.draw_samples((nb_images,),
                                                    random_state=rss[0])
        clip_limit = self.clip_limit.draw_samples((nb_images, nb_channels),
                                                  random_state=rss[1])
        tile_grid_size_px_h = self.tile_grid_size_px[0].draw_samples(
            (nb_images, nb_channels), random_state=rss[2])
        if mode == "single":
            tile_grid_size_px_w = tile_grid_size_px_h
        else:
            tile_grid_size_px_w = self.tile_grid_size_px[1].draw_samples(
                (nb_images, nb_channels), random_state=rss[3])

        tile_grid_size_px_w = np.maximum(tile_grid_size_px_w,
                                         self.tile_grid_size_px_min)
        tile_grid_size_px_h = np.maximum(tile_grid_size_px_h,
                                         self.tile_grid_size_px_min)

        gen = enumerate(zip(images, clip_limit, tile_grid_size_px_h,
                            tile_grid_size_px_w, per_channel))
        for i, (image, clip_limit_i, tgs_px_h_i, tgs_px_w_i, pchannel_i) in gen:
            if image.size == 0:
                continue

            nb_channels = image.shape[2]
            c_param = 0
            image_warped = []
            for c in sm.xrange(nb_channels):
                if tgs_px_w_i[c_param] > 1 or tgs_px_h_i[c_param] > 1:
                    clahe = cv2.createCLAHE(
                        clipLimit=clip_limit_i[c_param],
                        tileGridSize=(tgs_px_w_i[c_param], tgs_px_h_i[c_param])
                    )
                    channel_warped = clahe.apply(
                        _normalize_cv2_input_arr_(image[..., c])
                    )
                    image_warped.append(channel_warped)
                else:
                    image_warped.append(image[..., c])
                if pchannel_i > 0.5:
                    c_param += 1

            # combine channels to one image
            image_warped = np.stack(image_warped, axis=-1)

            batch.images[i] = image_warped
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.clip_limit, self.tile_grid_size_px,
                self.tile_grid_size_px_min, self.per_channel]


class CLAHE(meta.Augmenter):
    """Apply CLAHE to L/V/L channels in HLS/HSV/Lab colorspaces.

    This augmenter applies CLAHE (Contrast Limited Adaptive Histogram
    Equalization) to images, a form of histogram equalization that normalizes
    within local image patches.
    The augmenter transforms input images to a target colorspace (e.g.
    ``Lab``), extracts an intensity-related channel from the converted
    images (e.g. ``L`` for ``Lab``), applies CLAHE to the channel and then
    converts the resulting image back to the original colorspace.

    Grayscale images (images without channel axis or with only one channel
    axis) are automatically handled, `from_colorspace` does not have to be
    adjusted for them. For images with four channels (e.g. ``RGBA``), the
    fourth channel is ignored in the colorspace conversion (e.g. from an
    ``RGBA`` image, only the ``RGB`` part is converted, normalized, converted
    back and concatenated with the input ``A`` channel). Images with unusual
    channel numbers (2, 5 or more than 5) are normalized channel-by-channel
    (same behaviour as ``AllChannelsCLAHE``, though a warning will be raised).

    If you want to apply CLAHE to each channel of the original input image's
    colorspace (without any colorspace conversion), use
    ``imgaug.augmenters.contrast.AllChannelsCLAHE`` instead.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no (1)
        * ``uint32``: no (1)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (1)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (1)
        * ``bool``: no (1)

        - (1) This augmenter uses
              :class:`~imgaug.augmenters.color.ChangeColorspace`, which is
              currently limited to ``uint8``.

    Parameters
    ----------
    clip_limit : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Clipping limit. Higher values result in stronger contrast. OpenCV
        uses a default of ``40``, though values around ``5`` seem to already
        produce decent contrast.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]``
              will be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    tile_grid_size_px : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        Kernel size, i.e. size of each local neighbourhood in pixels.

            * If an ``int``, then that value will be used for all images for
              both kernel height and width.
            * If a tuple ``(a, b)``, then a value from the discrete interval
              ``[a..b]`` will be uniformly sampled per image.
            * If a list, then a random value will be sampled from that list
              per image and used for both kernel height and width.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter per image and used for both kernel
              height and width.
            * If a tuple of tuple of ``int`` given as ``((a, b), (c, d))``,
              then two values will be sampled independently from the discrete
              ranges ``[a..b]`` and ``[c..d]`` per image and used as the
              kernel height and width.
            * If a tuple of lists of ``int``, then two values will be sampled
              independently per image, one from the first list and one from
              the second, and used as the kernel height and width.
            * If a tuple of ``StochasticParameter``, then two values will be
              sampled indepdently per image, one from the first parameter and
              one from the second, and used as the kernel height and width.

    tile_grid_size_px_min : int, optional
        Minimum kernel size in px, per axis. If the sampling results in a
        value lower than this minimum, it will be clipped to this value.

    from_colorspace : {"RGB", "BGR", "HSV", "HLS", "Lab"}, optional
        Colorspace of the input images.
        If any input image has only one or zero channels, this setting will
        be ignored and it will be assumed that the input is grayscale.
        If a fourth channel is present in an input image, it will be removed
        before the colorspace conversion and later re-added.
        See also :func:`~imgaug.augmenters.color.change_colorspace_` for
        details.

    to_colorspace : {"Lab", "HLS", "HSV"}, optional
        Colorspace in which to perform CLAHE. For ``Lab``, CLAHE will only be
        applied to the first channel (``L``), for ``HLS`` to the
        second (``L``) and for ``HSV`` to the third (``V``). To apply CLAHE
        to all channels of an input image (without colorspace conversion),
        see ``imgaug.augmenters.contrast.AllChannelsCLAHE``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CLAHE()

    Create a standard CLAHE augmenter.

    >>> aug = iaa.CLAHE(clip_limit=(1, 10))

    Create a CLAHE augmenter with a clip limit uniformly sampled from
    ``[1..10]``, where ``1`` is rather low contrast and ``10`` is rather
    high contrast.

    >>> aug = iaa.CLAHE(tile_grid_size_px=(3, 21))

    Create a CLAHE augmenter with kernel sizes of ``SxS``, where ``S`` is
    uniformly sampled from ``[3..21]``. Sampling happens once per image.

    >>> aug = iaa.CLAHE(
    >>>     tile_grid_size_px=iap.Discretize(iap.Normal(loc=7, scale=2)),
    >>>     tile_grid_size_px_min=3)

    Create a CLAHE augmenter with kernel sizes of ``SxS``, where ``S`` is
    sampled from ``N(7, 2)``, but does not go below ``3``.

    >>> aug = iaa.CLAHE(tile_grid_size_px=((3, 21), [3, 5, 7]))

    Create a CLAHE augmenter with kernel sizes of ``HxW``, where ``H`` is
    uniformly sampled from ``[3..21]`` and ``W`` is randomly picked from the
    list ``[3, 5, 7]``.

    >>> aug = iaa.CLAHE(
    >>>     from_colorspace=iaa.CSPACE_BGR,
    >>>     to_colorspace=iaa.CSPACE_HSV)

    Create a CLAHE augmenter that converts images from BGR colorspace to
    HSV colorspace and then applies the local histogram equalization to the
    ``V`` channel of the images (before converting back to ``BGR``).
    Alternatively, ``Lab`` (default) or ``HLS`` can be used as the target
    colorspace. Grayscale images (no channels / one channel) are never
    converted and are instead directly normalized (i.e. `from_colorspace`
    does not have to be changed for them).

    """
    RGB = color_lib.CSPACE_RGB
    BGR = color_lib.CSPACE_BGR
    HSV = color_lib.CSPACE_HSV
    HLS = color_lib.CSPACE_HLS
    Lab = color_lib.CSPACE_Lab

    def __init__(self, clip_limit=(0.1, 8), tile_grid_size_px=(3, 12),
                 tile_grid_size_px_min=3,
                 from_colorspace=color_lib.CSPACE_RGB,
                 to_colorspace=color_lib.CSPACE_Lab,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(CLAHE, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.all_channel_clahe = AllChannelsCLAHE(
            clip_limit=clip_limit,
            tile_grid_size_px=tile_grid_size_px,
            tile_grid_size_px_min=tile_grid_size_px_min,
            name="%s_AllChannelsCLAHE" % (name,))

        self.intensity_channel_based_applier = _IntensityChannelBasedApplier(
            from_colorspace, to_colorspace)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        iadt.gate_dtypes(
            images,
            allowed=["uint8"],
            disallowed=["bool",
                        "uint16", "uint32", "uint64", "uint128", "uint256",
                        "int8", "int16", "int32", "int64", "int128", "int256",
                        "float16", "float32", "float64", "float96", "float128",
                        "float256"],
            augmenter=self)

        def _augment_all_channels_clahe(images_normalized,
                                        random_state_derived):
            # pylint: disable=protected-access
            # TODO would .augment_batch() be sufficient here?
            batch_imgs = _BatchInAugmentation(
                images=images_normalized)
            return self.all_channel_clahe._augment_batch_(
                batch_imgs, random_state_derived, parents + [self],
                hooks
            ).images

        batch.images = self.intensity_channel_based_applier.apply(
            images, random_state, parents + [self], hooks,
            _augment_all_channels_clahe)
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        ac_clahe = self.all_channel_clahe
        intb_applier = self.intensity_channel_based_applier
        return [
            ac_clahe.clip_limit,
            ac_clahe.tile_grid_size_px,
            ac_clahe.tile_grid_size_px_min
        ] + intb_applier.get_parameters()


class AllChannelsHistogramEqualization(meta.Augmenter):
    """
    Apply Histogram Eq. to all channels of images in their original colorspaces.

    In contrast to ``imgaug.augmenters.contrast.HistogramEqualization``, this
    augmenter operates directly on all channels of the input images. It does
    not perform any colorspace transformations and does not focus on specific
    channels (e.g. ``L`` in ``Lab`` colorspace).

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no (1)
        * ``uint32``: no (2)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (2)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (2)
        * ``bool``: no (1)

        - (1) causes cv2 error: ``cv2.error:
              OpenCV(3.4.5) (...)/histogram.cpp:3345: error: (-215:Assertion
              failed) src.type() == CV_8UC1 in function 'equalizeHist'``
        - (2) rejected by cv2

    Parameters
    ----------
    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AllChannelsHistogramEqualization()

    Create an augmenter that applies histogram equalization to all channels
    of input images in the original colorspaces.

    >>> aug = iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization())

    Same as in the previous example, but alpha-blends the contrast-enhanced
    augmented images with the original input images using random blend
    strengths. This leads to random strengths of the contrast adjustment.

    """
    def __init__(self, seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(AllChannelsHistogramEqualization, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        iadt.gate_dtypes(
            images,
            allowed=["uint8"],
            disallowed=["bool",
                        "uint16", "uint32", "uint64", "uint128", "uint256",
                        "int8", "int16", "int32", "int64", "int128", "int256",
                        "float16", "float32", "float64", "float96", "float128",
                        "float256"],
            augmenter=self)

        for i, image in enumerate(images):
            if image.size == 0:
                continue

            image_warped = [
                cv2.equalizeHist(_normalize_cv2_input_arr_(image[..., c]))
                for c in sm.xrange(image.shape[2])]
            image_warped = np.array(image_warped, dtype=image_warped[0].dtype)
            image_warped = image_warped.transpose((1, 2, 0))

            batch.images[i] = image_warped
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []


class HistogramEqualization(meta.Augmenter):
    """
    Apply Histogram Eq. to L/V/L channels of images in HLS/HSV/Lab colorspaces.

    This augmenter is similar to ``imgaug.augmenters.contrast.CLAHE``.

    The augmenter transforms input images to a target colorspace (e.g.
    ``Lab``), extracts an intensity-related channel from the converted images
    (e.g. ``L`` for ``Lab``), applies Histogram Equalization to the channel
    and then converts the resulting image back to the original colorspace.

    Grayscale images (images without channel axis or with only one channel
    axis) are automatically handled, `from_colorspace` does not have to be
    adjusted for them. For images with four channels (e.g. RGBA), the fourth
    channel is ignored in the colorspace conversion (e.g. from an ``RGBA``
    image, only the ``RGB`` part is converted, normalized, converted back and
    concatenated with the input ``A`` channel). Images with unusual channel
    numbers (2, 5 or more than 5) are normalized channel-by-channel (same
    behaviour as ``AllChannelsHistogramEqualization``, though a warning will
    be raised).

    If you want to apply HistogramEqualization to each channel of the original
    input image's colorspace (without any colorspace conversion), use
    ``imgaug.augmenters.contrast.AllChannelsHistogramEqualization`` instead.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no (1)
        * ``uint32``: no (1)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (1)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (1)
        * ``bool``: no (1)

        - (1) This augmenter uses :class:`AllChannelsHistogramEqualization`,
              which only supports ``uint8``.

    Parameters
    ----------
    from_colorspace : {"RGB", "BGR", "HSV", "HLS", "Lab"}, optional
        Colorspace of the input images.
        If any input image has only one or zero channels, this setting will be
        ignored and it will be assumed that the input is grayscale.
        If a fourth channel is present in an input image, it will be removed
        before the colorspace conversion and later re-added.
        See also :func:`~imgaug.augmenters.color.change_colorspace_` for
        details.

    to_colorspace : {"Lab", "HLS", "HSV"}, optional
        Colorspace in which to perform Histogram Equalization. For ``Lab``,
        the equalization will only be applied to the first channel (``L``),
        for ``HLS`` to the second (``L``) and for ``HSV`` to the third (``V``).
        To apply histogram equalization to all channels of an input image
        (without colorspace conversion), see
        ``imgaug.augmenters.contrast.AllChannelsHistogramEqualization``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.HistogramEqualization()

    Create an augmenter that converts images to ``HLS``/``HSV``/``Lab``
    colorspaces, extracts intensity-related channels (i.e. ``L``/``V``/``L``),
    applies histogram equalization to these channels and converts back to the
    input colorspace.

    >>> aug = iaa.Alpha((0.0, 1.0), iaa.HistogramEqualization())

    Same as in the previous example, but alpha blends the result, leading
    to various strengths of contrast normalization.

    >>> aug = iaa.HistogramEqualization(
    >>>     from_colorspace=iaa.CSPACE_BGR,
    >>>     to_colorspace=iaa.CSPACE_HSV)

    Same as in the first example, but the colorspace of input images has
    to be ``BGR`` (instead of default ``RGB``) and the histogram equalization
    is applied to the ``V`` channel in ``HSV`` colorspace.

    """
    RGB = color_lib.CSPACE_RGB
    BGR = color_lib.CSPACE_BGR
    HSV = color_lib.CSPACE_HSV
    HLS = color_lib.CSPACE_HLS
    Lab = color_lib.CSPACE_Lab

    def __init__(self, from_colorspace=color_lib.CSPACE_RGB,
                 to_colorspace=color_lib.CSPACE_Lab,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(HistogramEqualization, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.all_channel_histogram_equalization = \
            AllChannelsHistogramEqualization(
                name="%s_AllChannelsHistogramEqualization" % (name,))

        self.intensity_channel_based_applier = _IntensityChannelBasedApplier(
            from_colorspace, to_colorspace)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        iadt.gate_dtypes(
            images,
            allowed=["uint8"],
            disallowed=["bool",
                        "uint16", "uint32", "uint64", "uint128", "uint256",
                        "int8", "int16", "int32", "int64", "int128", "int256",
                        "float16", "float32", "float64", "float96", "float128",
                        "float256"],
            augmenter=self)

        def _augment_all_channels_histogram_equalization(images_normalized,
                                                         random_state_derived):
            # pylint: disable=protected-access
            # TODO would .augment_batch() be sufficient here
            batch_imgs = _BatchInAugmentation(
                images=images_normalized)
            return self.all_channel_histogram_equalization._augment_batch_(
                batch_imgs, random_state_derived, parents + [self],
                hooks
            ).images

        batch.images = self.intensity_channel_based_applier.apply(
            images, random_state, parents + [self], hooks,
            _augment_all_channels_histogram_equalization)
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        icb_applier = self.intensity_channel_based_applier
        return icb_applier.get_parameters()
