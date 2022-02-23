"""
Augmenters that are based on applying convolution kernels to images.

List of augmenters:

    * :class:`Convolve`
    * :class:`Sharpen`
    * :class:`Emboss`
    * :class:`EdgeDetect`
    * :class:`DirectedEdgeDetect`

For MotionBlur, see ``blur.py``.

"""
from __future__ import print_function, division, absolute_import

import itertools

import numpy as np
import cv2
import six.moves as sm

import imgaug as ia
from imgaug.imgaug import _normalize_cv2_input_arr_
from . import meta
from .. import parameters as iap
from .. import dtypes as iadt


# TODO allow 3d matrices as input (not only 2D)
# TODO add _augment_keypoints and other _augment funcs, as these should do
#      something for e.g. [[0, 0, 1]]
class Convolve(meta.Augmenter):
    """
    Apply a convolution to input images.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: no (2)
        * ``int64``: no (2)
        * ``float16``: yes; tested (4)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (4)

        - (1) rejected by ``cv2.filter2D()``.
        - (2) causes error: cv2.error: OpenCV(3.4.2) (...)/filter.cpp:4487:
              error: (-213:The function/feature is not implemented)
              Unsupported combination of source format (=1), and destination
              format (=1) in function 'getLinearFilter'.
        - (3) mapped internally to ``int16``.
        - (4) mapped internally to ``float32``.

    Parameters
    ----------
    matrix : None or (H, W) ndarray or imgaug.parameters.StochasticParameter or callable, optional
        The weight matrix of the convolution kernel to apply.

            * If ``None``, the input images will not be changed.
            * If a 2D numpy array, that array will always be used for all
              images and channels as the kernel.
            * If a callable, that method will be called for each image
              via ``parameter(image, C, random_state)``. The function must
              either return a list of ``C`` matrices (i.e. one per channel)
              or a 2D numpy array (will be used for all channels) or a
              3D ``HxWxC`` numpy array. If a list is returned, each entry may
              be ``None``, which will result in no changes to the respective
              channel.

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
    >>> matrix = np.array([[0, -1, 0],
    >>>                    [-1, 4, -1],
    >>>                    [0, -1, 0]])
    >>> aug = iaa.Convolve(matrix=matrix)

    Convolves all input images with the kernel shown in the ``matrix``
    variable.

    >>> def gen_matrix(image, nb_channels, random_state):
    >>>     matrix_A = np.array([[0, -1, 0],
    >>>                          [-1, 4, -1],
    >>>                          [0, -1, 0]])
    >>>     matrix_B = np.array([[0, 1, 0],
    >>>                          [1, -4, 1],
    >>>                          [0, 1, 0]])
    >>>     if image.shape[0] % 2 == 0:
    >>>         return [matrix_A] * nb_channels
    >>>     else:
    >>>         return [matrix_B] * nb_channels
    >>> aug = iaa.Convolve(matrix=gen_matrix)

    Convolves images that have an even height with matrix A and images
    having an odd height with matrix B.

    """

    def __init__(self, matrix=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Convolve, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        if matrix is None:
            self.matrix = None
            self.matrix_type = "None"
        elif ia.is_np_array(matrix):
            assert matrix.ndim == 2, (
                "Expected convolution matrix to have exactly two dimensions, "
                "got %d (shape %s)." % (matrix.ndim, matrix.shape))
            self.matrix = matrix
            self.matrix_type = "constant"
        elif ia.is_callable(matrix):
            self.matrix = matrix
            self.matrix_type = "function"
        else:
            raise Exception(
                "Expected float, int, tuple/list with 2 entries or "
                "StochasticParameter. Got %s." % (
                    type(matrix),))

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        iadt.gate_dtypes(images,
                         allowed=["bool",
                                  "uint8", "uint16",
                                  "int8", "int16",
                                  "float16", "float32", "float64"],
                         disallowed=["uint32", "uint64", "uint128", "uint256",
                                     "int32", "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=self)
        rss = random_state.duplicate(len(images))

        for i, image in enumerate(images):
            _height, _width, nb_channels = image.shape

            # currently we don't have to worry here about alignemnt with
            # non-image data and therefore can just place this before any
            # sampling
            if image.size == 0:
                continue

            input_dtype = image.dtype
            if image.dtype.name in ["bool", "float16"]:
                image = image.astype(np.float32, copy=False)
            elif image.dtype.name == "int8":
                image = image.astype(np.int16, copy=False)

            if self.matrix_type == "None":
                matrices = [None] * nb_channels
            elif self.matrix_type == "constant":
                matrices = [self.matrix] * nb_channels
            elif self.matrix_type == "function":
                matrices = self.matrix(images[i], nb_channels, rss[i])
                if ia.is_np_array(matrices) and matrices.ndim == 2:
                    matrices = np.tile(
                        matrices[..., np.newaxis],
                        (1, 1, nb_channels))

                is_valid_list = (isinstance(matrices, list)
                                 and len(matrices) == nb_channels)
                is_valid_array = (ia.is_np_array(matrices)
                                  and matrices.ndim == 3
                                  and matrices.shape[2] == nb_channels)
                assert is_valid_list or is_valid_array, (
                    "Callable provided to Convole must return either a "
                    "list of 2D matrices (one per image channel) "
                    "or a 2D numpy array "
                    "or a 3D numpy array where the last dimension's size "
                    "matches the number of image channels. "
                    "Got type %s." % (type(matrices),))

                if ia.is_np_array(matrices):
                    # Shape of matrices is currently (H, W, C), but in the
                    # loop below we need the first axis to be the channel
                    # index to unify handling of lists of arrays and arrays.
                    # So we move the channel axis here to the start.
                    matrices = matrices.transpose((2, 0, 1))
            else:
                raise Exception("Invalid matrix type")

            # TODO check if sampled matrices are identical over channels
            #      and then just apply once. (does that really help wrt speed?)
            image_aug = image
            for channel in sm.xrange(nb_channels):
                if matrices[channel] is not None:
                    # ndimage.convolve caused problems here cv2.filter2D()
                    # always returns same output dtype as input dtype
                    image_aug[..., channel] = cv2.filter2D(
                        _normalize_cv2_input_arr_(image_aug[..., channel]),
                        -1,
                        matrices[channel]
                    )

            if input_dtype.name == "bool":
                image_aug = image_aug > 0.5
            elif input_dtype.name in ["int8", "float16"]:
                image_aug = iadt.restore_dtypes_(image_aug, input_dtype)

            batch.images[i] = image_aug

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.matrix, self.matrix_type]


class Sharpen(Convolve):
    """
    Sharpen images and alpha-blend the result with the original input images.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.convolutional.Convolve`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Blending factor of the sharpened image. At ``0.0``, only the original
        image is visible, at ``1.0`` only its sharpened version is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

    lightness : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Lightness/brightness of the sharped image.
        Sane values are somewhere in the interval ``[0.5, 2.0]``.
        The value ``0.0`` results in an edge map. Values higher than ``1.0``
        create bright images. Default value is ``1.0``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

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
    >>> aug = iaa.Sharpen(alpha=(0.0, 1.0))

    Sharpens input images and blends the sharpened image with the input image
    using a random blending factor between ``0%`` and ``100%`` (uniformly
    sampled).

    >>> aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))

    Sharpens input images with a variable `lightness` sampled uniformly from
    the interval ``[0.75, 2.0]`` and with a fully random blending factor
    (as in the above example).

    """
    def __init__(self, alpha=(0.0, 0.2), lightness=(0.8, 1.2),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        alpha_param = iap.handle_continuous_param(
            alpha, "alpha",
            value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True)
        lightness_param = iap.handle_continuous_param(
            lightness, "lightness",
            value_range=(0, None), tuple_to_uniform=True, list_to_choice=True)

        matrix_gen = _SharpeningMatrixGenerator(alpha_param, lightness_param)

        super(Sharpen, self).__init__(
            matrix=matrix_gen,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class _SharpeningMatrixGenerator(object):
    def __init__(self, alpha, lightness):
        self.alpha = alpha
        self.lightness = lightness

    def __call__(self, _image, nb_channels, random_state):
        alpha_sample = self.alpha.draw_sample(random_state=random_state)
        assert 0 <= alpha_sample <= 1.0, (
            "Expected 'alpha' to be in the interval [0.0, 1.0], "
            "got %.4f." % (alpha_sample,))
        lightness_sample = self.lightness.draw_sample(random_state=random_state)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [-1, -1, -1],
            [-1, 8+lightness_sample, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        matrix = (
            (1-alpha_sample) * matrix_nochange
            + alpha_sample * matrix_effect
        )
        return [matrix] * nb_channels


class Emboss(Convolve):
    """
    Emboss images and alpha-blend the result with the original input images.

    The embossed version pronounces highlights and shadows,
    letting the image look as if it was recreated on a metal plate ("embossed").

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.convolutional.Convolve`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Blending factor of the embossed image. At ``0.0``, only the original
        image is visible, at ``1.0`` only its embossed version is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

    strength : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Parameter that controls the strength of the embossing.
        Sane values are somewhere in the interval ``[0.0, 2.0]`` with ``1.0``
        being the standard embossing effect. Default value is ``1.0``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

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
    >>> aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))

    Emboss an image with a strength sampled uniformly from the interval
    ``[0.5, 1.5]`` and alpha-blend the result with the original input image
    using a random blending factor between ``0%`` and ``100%``.

    """
    def __init__(self, alpha=(0.0, 1.0), strength=(0.25, 1.0),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        alpha_param = iap.handle_continuous_param(
            alpha, "alpha",
            value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True)
        strength_param = iap.handle_continuous_param(
            strength, "strength",
            value_range=(0, None), tuple_to_uniform=True, list_to_choice=True)

        matrix_gen = _EmbossMatrixGenerator(alpha_param, strength_param)

        super(Emboss, self).__init__(
            matrix=matrix_gen,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class _EmbossMatrixGenerator(object):
    def __init__(self, alpha, strength):
        self.alpha = alpha
        self.strength = strength

    def __call__(self, _image, nb_channels, random_state):
        alpha_sample = self.alpha.draw_sample(random_state=random_state)
        assert 0 <= alpha_sample <= 1.0, (
            "Expected 'alpha' to be in the interval [0.0, 1.0], "
            "got %.4f." % (alpha_sample,))
        strength_sample = self.strength.draw_sample(random_state=random_state)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [-1-strength_sample, 0-strength_sample, 0],
            [0-strength_sample, 1, 0+strength_sample],
            [0, 0+strength_sample, 1+strength_sample]
        ], dtype=np.float32)
        matrix = (
            (1-alpha_sample) * matrix_nochange
            + alpha_sample * matrix_effect
        )
        return [matrix] * nb_channels


# TODO add tests
# TODO move this to edges.py?
class EdgeDetect(Convolve):
    """
    Generate a black & white edge image and alpha-blend it with the input image.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.convolutional.Convolve`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Blending factor of the edge image. At ``0.0``, only the original
        image is visible, at ``1.0`` only the edge image is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

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
    >>> aug = iaa.EdgeDetect(alpha=(0.0, 1.0))

    Detect edges in an image, mark them as black (non-edge) and white (edges)
    and alpha-blend the result with the original input image using a random
    blending factor between ``0%`` and ``100%``.

    """
    def __init__(self, alpha=(0.0, 0.75),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        alpha_param = iap.handle_continuous_param(
            alpha, "alpha",
            value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True)

        matrix_gen = _EdgeDetectMatrixGenerator(alpha_param)

        super(EdgeDetect, self).__init__(
            matrix=matrix_gen,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class _EdgeDetectMatrixGenerator(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, _image, nb_channels, random_state):
        alpha_sample = self.alpha.draw_sample(random_state=random_state)
        assert 0 <= alpha_sample <= 1.0, (
            "Expected 'alpha' to be in the interval [0.0, 1.0], "
            "got %.4f." % (alpha_sample,))
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        matrix = (
            (1-alpha_sample) * matrix_nochange
            + alpha_sample * matrix_effect
        )
        return [matrix] * nb_channels


# TODO add tests
# TODO merge EdgeDetect and DirectedEdgeDetect?
# TODO deprecate and rename to AngledEdgeDetect
# TODO rename arg "direction" to "angle"
# TODO change direction/angle value range to (0, 360)
# TODO move this to edges.py?
class DirectedEdgeDetect(Convolve):
    """
    Detect edges from specified angles and alpha-blend with the input image.

    This augmenter first detects edges along a certain angle.
    Usually, edges are detected in x- or y-direction, while here the edge
    detection kernel is rotated to match a specified angle.
    The result of applying the kernel is a black (non-edges) and white (edges)
    image. That image is alpha-blended with the input image.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.convolutional.Convolve`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Blending factor of the edge image. At ``0.0``, only the original
        image is visible, at ``1.0`` only the edge image is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

    direction : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Angle (in degrees) of edges to pronounce, where ``0`` represents
        ``0`` degrees and ``1.0`` represents 360 degrees (both clockwise,
        starting at the top). Default value is ``(0.0, 1.0)``, i.e. pick a
        random angle per image.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

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
    >>> aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=0)

    Turn input images into edge images in which edges are detected from
    the top side of the image (i.e. the top sides of horizontal edges are
    part of the edge image, while vertical edges are ignored).

    >>> aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=90/360)

    Same as before, but edges are detected from the right. Horizontal edges
    are now ignored.

    >>> aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=(0.0, 1.0))

    Same as before, but edges are detected from a random angle sampled
    uniformly from the interval ``[0deg, 360deg]``.

    >>> aug = iaa.DirectedEdgeDetect(alpha=(0.0, 0.3), direction=0)

    Similar to the previous examples, but here the edge image is alpha-blended
    with the input image. The result is a mixture between the edge image and
    the input image. The blending factor is randomly sampled between ``0%``
    and ``30%``.

    """
    def __init__(self, alpha=(0.0, 0.75), direction=(0.0, 1.0),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        alpha_param = iap.handle_continuous_param(
            alpha, "alpha",
            value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True)
        direction_param = iap.handle_continuous_param(
            direction, "direction",
            value_range=None, tuple_to_uniform=True, list_to_choice=True)

        matrix_gen = _DirectedEdgeDetectMatrixGenerator(alpha_param,
                                                        direction_param)

        super(DirectedEdgeDetect, self).__init__(
            matrix=matrix_gen,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class _DirectedEdgeDetectMatrixGenerator(object):
    def __init__(self, alpha, direction):
        self.alpha = alpha
        self.direction = direction

    def __call__(self, _image, nb_channels, random_state):
        alpha_sample = self.alpha.draw_sample(random_state=random_state)
        assert 0 <= alpha_sample <= 1.0, (
            "Expected 'alpha' to be in the interval [0.0, 1.0], "
            "got %.4f." % (alpha_sample,))
        direction_sample = self.direction.draw_sample(random_state=random_state)

        deg = int(direction_sample * 360) % 360
        rad = np.deg2rad(deg)
        x = np.cos(rad - 0.5*np.pi)
        y = np.sin(rad - 0.5*np.pi)
        direction_vector = np.array([x, y])

        matrix_effect = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        for x, y in itertools.product([-1, 0, 1], [-1, 0, 1]):
            if (x, y) != (0, 0):
                cell_vector = np.array([x, y])
                distance_deg = np.rad2deg(
                    ia.angle_between_vectors(cell_vector,
                                             direction_vector))
                distance = distance_deg / 180
                similarity = (1 - distance)**4
                matrix_effect[y+1, x+1] = similarity
        matrix_effect = matrix_effect / np.sum(matrix_effect)
        matrix_effect = matrix_effect * (-1)
        matrix_effect[1, 1] = 1

        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)

        matrix = (
            (1-alpha_sample) * matrix_nochange
            + alpha_sample * matrix_effect
        )

        return [matrix] * nb_channels
