"""Collection of basic functions used throughout imgaug."""
from __future__ import print_function, division, absolute_import

import math
import numbers
import sys
import os
import json
import types
import functools
# collections.abc exists since 3.3 and is expected to be used for 3.8+
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy as np
import cv2
import imageio
import six
import six.moves as sm
import skimage.draw
import skimage.measure


ALL = "ALL"

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# filepath to the quokka image, its annotations and depth map
QUOKKA_FP = os.path.join(FILE_DIR, "quokka.jpg")
QUOKKA_ANNOTATIONS_FP = os.path.join(FILE_DIR, "quokka_annotations.json")
QUOKKA_DEPTH_MAP_HALFRES_FP = os.path.join(
    FILE_DIR, "quokka_depth_map_halfres.png")

DEFAULT_FONT_FP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "DejaVuSans.ttf"
)


# to check if a dtype instance is among these dtypes, use e.g.
# `dtype.type in  NP_FLOAT_TYPES` do not just use `dtype in NP_FLOAT_TYPES` as
# that would fail
NP_FLOAT_TYPES = set(np.sctypes["float"])
NP_INT_TYPES = set(np.sctypes["int"])
NP_UINT_TYPES = set(np.sctypes["uint"])

IMSHOW_BACKEND_DEFAULT = "matplotlib"

IMRESIZE_VALID_INTERPOLATIONS = [
    "nearest", "linear", "area", "cubic",
    cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]


###############################################################################
# Helpers for deprecation
###############################################################################

class DeprecationWarning(Warning):  # pylint: disable=redefined-builtin
    """Warning for deprecated calls.

    Since python 2.7 DeprecatedWarning is silent by default. So we define
    our own DeprecatedWarning here so that it is not silent by default.

    """


def warn(msg, category=UserWarning, stacklevel=2):
    """Generate a a warning with stacktrace.

    Parameters
    ----------
    msg : str
        The message of the warning.

    category : class
        The class of the warning to produce.

    stacklevel : int, optional
        How many steps above this function to "jump" in the stacktrace when
        displaying file and line number of the error message.
        Usually ``2``.

    """
    import warnings
    warnings.warn(msg, category=category, stacklevel=stacklevel)


def warn_deprecated(msg, stacklevel=2):
    """Generate a non-silent deprecation warning with stacktrace.

    The used warning is ``imgaug.imgaug.DeprecationWarning``.

    Parameters
    ----------
    msg : str
        The message of the warning.

    stacklevel : int, optional
        How many steps above this function to "jump" in the stacktrace when
        displaying file and line number of the error message.
        Usually ``2``

    """
    warn(msg, category=DeprecationWarning, stacklevel=stacklevel)


class deprecated(object):  # pylint: disable=invalid-name
    """Decorator to mark deprecated functions with warning.

    Adapted from
    <https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/utils.py>.

    Parameters
    ----------
    alt_func : None or str, optional
        If given, tell user what function to use instead.

    behavior : {'warn', 'raise'}, optional
        Behavior during call to deprecated function: ``warn`` means that the
        user is warned that the function is deprecated; ``raise`` means that
        an error is raised.

    removed_version : None or str, optional
        The package version in which the deprecated function will be removed.

    comment : None or str, optional
        An optional comment that will be appended to the warning message.

    """

    def __init__(self, alt_func=None, behavior="warn", removed_version=None,
                 comment=None):
        self.alt_func = alt_func
        self.behavior = behavior
        self.removed_version = removed_version
        self.comment = comment

    def __call__(self, func):
        alt_msg = None
        if self.alt_func is not None:
            alt_msg = "Use ``%s`` instead." % (self.alt_func,)

        rmv_msg = None
        if self.removed_version is not None:
            rmv_msg = "It will be removed in version %s." % (
                self.removed_version,)

        comment_msg = None
        if self.comment is not None and len(self.comment) > 0:
            comment_msg = "%s." % (self.comment.rstrip(". "),)

        addendum = " ".join([submsg
                             for submsg
                             in [alt_msg, rmv_msg, comment_msg]
                             if submsg is not None])

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # getargpec() is deprecated
            # pylint: disable=deprecated-method

            # TODO add class name if class method
            import inspect
            # arg_names = func.__code__.co_varnames

            # getargspec() was deprecated in py3, but doesn't exist in py2
            if hasattr(inspect, "getfullargspec"):
                arg_names = inspect.getfullargspec(func)[0]
            else:
                arg_names = inspect.getargspec(func)[0]

            if "self" in arg_names or "cls" in arg_names:
                main_msg = "Method ``%s.%s()`` is deprecated." % (
                    args[0].__class__.__name__, func.__name__)
            else:
                main_msg = "Function ``%s()`` is deprecated." % (
                    func.__name__,)

            msg = (main_msg + " " + addendum).rstrip(" ").replace("``", "`")

            if self.behavior == "warn":
                warn_deprecated(msg, stacklevel=3)
            elif self.behavior == "raise":
                raise DeprecationWarning(msg)
            return func(*args, **kwargs)

        # modify doc string to display deprecation warning
        doc = "**Deprecated**. " + addendum
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + "\n\n    " + wrapped.__doc__

        return wrapped

###############################################################################


def is_np_array(val):
    """Check whether a variable is a numpy array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a numpy array. Otherwise ``False``.

    """
    # using np.generic here via isinstance(val, (np.ndarray, np.generic))
    # seems to also fire for scalar numpy values even though those are not
    # arrays
    return isinstance(val, np.ndarray)


def is_np_scalar(val):
    """Check whether a variable is a numpy scalar.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a numpy scalar. Otherwise ``False``.

    """
    # Note that isscalar() alone also fires for thinks like python strings
    # or booleans.
    # The isscalar() was added to make this function not fire for non-scalar
    # numpy types. Not sure if it is necessary.
    return isinstance(val, np.generic) and np.isscalar(val)


def is_single_integer(val):
    """Check whether a variable is an ``int``.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is an ``int``. Otherwise ``False``.

    """
    return isinstance(val, numbers.Integral) and not isinstance(val, bool)


def is_single_float(val):
    """Check whether a variable is a ``float``.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a ``float``. Otherwise ``False``.

    """
    return (
        isinstance(val, numbers.Real)
        and not is_single_integer(val)
        and not isinstance(val, bool)
    )


def is_single_number(val):
    """Check whether a variable is a ``number``, i.e. an ``int`` or ``float``.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a ``number``. Otherwise ``False``.

    """
    return is_single_integer(val) or is_single_float(val)


def is_iterable(val):
    """
    Checks whether a variable is iterable.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is an iterable. Otherwise ``False``.

    """
    return isinstance(val, Iterable)


# TODO convert to is_single_string() or rename is_single_integer/float/number()
def is_string(val):
    """Check whether a variable is a string.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a string. Otherwise ``False``.

    """
    return isinstance(val, six.string_types)


def is_single_bool(val):
    """Check whether a variable is a ``bool``.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a ``bool``. Otherwise ``False``.

    """
    # pylint: disable=unidiomatic-typecheck
    return type(val) == type(True)


def is_integer_array(val):
    """Check whether a variable is a numpy integer array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a numpy integer array. Otherwise ``False``.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.integer)


def is_float_array(val):
    """Check whether a variable is a numpy float array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a numpy float array. Otherwise ``False``.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.floating)


def is_callable(val):
    """Check whether a variable is a callable, e.g. a function.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a callable. Otherwise ``False``.

    """
    # python 3.x with x <= 2 does not support callable(), apparently
    if sys.version_info[0] == 3 and sys.version_info[1] <= 2:
        return hasattr(val, '__call__')
    return callable(val)


def is_generator(val):
    """Check whether a variable is a generator.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` is the variable is a generator. Otherwise ``False``.

    """
    return isinstance(val, types.GeneratorType)


def flatten(nested_iterable):
    """Flatten arbitrarily nested lists/tuples.

    Code partially taken from https://stackoverflow.com/a/10824420.

    Parameters
    ----------
    nested_iterable
        A ``list`` or ``tuple`` of arbitrarily nested values.

    Yields
    ------
    any
        All values in `nested_iterable`, flattened.

    """
    # don't just check if something is iterable here, because then strings
    # and arrays will be split into their characters and components
    if not isinstance(nested_iterable, (list, tuple)):
        yield nested_iterable
    else:
        for i in nested_iterable:
            if isinstance(i, (list, tuple)):
                for j in flatten(i):
                    yield j
            else:
                yield i


# TODO no longer used anywhere. deprecate?
def caller_name():
    """Return the name of the caller, e.g. a function.

    Returns
    -------
    str
        The name of the caller as a string

    """
    # pylint: disable=protected-access
    return sys._getframe(1).f_code.co_name


def seed(entropy=None, seedval=None):
    """Set the seed of imgaug's global RNG.

    The global RNG controls most of the "randomness" in imgaug.

    The global RNG is the default one used by all augmenters. Under special
    circumstances (e.g. when an augmenter is switched to deterministic mode),
    the global RNG is replaced with a local one. The state of that replacement
    may be dependent on the global RNG's state at the time of creating the
    child RNG.

    .. note::

        This function is not yet marked as deprecated, but might be in the
        future. The preferred way to seed `imgaug` is via
        :func:`~imgaug.random.seed`.

    Parameters
    ----------
    entropy : int
        The seed value to use.

    seedval : None or int, optional
        Deprecated since 0.4.0.

    """
    assert entropy is not None or seedval is not None, (
        "Expected argument 'entropy' or 'seedval' to be not-None, but both"
        "were None.")

    if seedval is not None:
        assert entropy is None, (
            "Argument 'seedval' is the outdated name for 'entropy'. Hence, "
            "if it is provided, 'entropy' must be None. Got 'entropy' value "
            "of type %s." % (type(entropy),))

        warn_deprecated("Parameter 'seedval' is deprecated. Use "
                        "'entropy' instead.")
        entropy = seedval

    import imgaug.random
    imgaug.random.seed(entropy)


@deprecated("imgaug.random.normalize_generator")
def normalize_random_state(random_state):
    """Normalize various inputs to a numpy random generator.

    Parameters
    ----------
    random_state : None or int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.bit_generator.SeedSequence or numpy.random.RandomState
        See :func:`~imgaug.random.normalize_generator`.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator`` (even if
        the input was a ``RandomState``).

    """
    import imgaug.random
    return imgaug.random.normalize_generator_(random_state)


@deprecated("imgaug.random.get_global_rng")
def current_random_state():
    """Get or create the current global RNG of imgaug.

    Note that the first call to this function will create a global RNG.

    Returns
    -------
    imgaug.random.RNG
        The global RNG to use.

    """
    import imgaug.random
    return imgaug.random.get_global_rng()


@deprecated("imgaug.random.convert_seed_to_rng")
def new_random_state(seed=None, fully_random=False):
    """Create a new numpy random number generator.

    Parameters
    ----------
    seed : None or int, optional
        The seed value to use. If ``None`` and `fully_random` is ``False``,
        the seed will be derived from the global RNG. If `fully_random` is
        ``True``, the seed will be provided by the OS.

    fully_random : bool, optional
        Whether the seed will be provided by the OS.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are initialized with the provided seed.

    """
    # pylint: disable=redefined-outer-name
    import imgaug.random
    if seed is None:
        if fully_random:
            return imgaug.random.RNG.create_fully_random()
        return imgaug.random.RNG.create_pseudo_random_()
    return imgaug.random.RNG(seed)


# TODO seems to not be used anywhere anymore
@deprecated("imgaug.random.convert_seed_to_rng")
def dummy_random_state():
    """Create a dummy random state using a seed of ``1``.

    Returns
    -------
    imgaug.random.RNG
        The new random state.

    """
    import imgaug.random
    return imgaug.random.RNG(1)


@deprecated("imgaug.random.copy_generator_unless_global_rng")
def copy_random_state(random_state, force_copy=False):
    """Copy an existing numpy (random number) generator.

    Parameters
    ----------
    random_state : numpy.random.Generator or numpy.random.RandomState
        The generator to copy.

    force_copy : bool, optional
        If ``True``, this function will always create a copy of every random
        state. If ``False``, it will not copy numpy's default random state,
        but all other random states.

    Returns
    -------
    rs_copy : numpy.random.RandomState
        The copied random state.

    """
    import imgaug.random
    if force_copy:
        return imgaug.random.copy_generator(random_state)
    return imgaug.random.copy_generator_unless_global_generator(random_state)


@deprecated("imgaug.random.derive_generator_")
def derive_random_state(random_state):
    """Derive a child numpy random generator from another one.

    Parameters
    ----------
    random_state : numpy.random.Generator or numpy.random.RandomState
        The generator from which to derive a new child generator.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        In both cases a derived child generator.

    """
    import imgaug.random
    return imgaug.random.derive_generator_(random_state)


@deprecated("imgaug.random.derive_generators_")
def derive_random_states(random_state, n=1):
    """Derive child numpy random generators from another one.

    Parameters
    ----------
    random_state : numpy.random.Generator or numpy.random.RandomState
        The generator from which to derive new child generators.

    n : int, optional
        Number of child generators to derive.

    Returns
    -------
    list of numpy.random.Generator or list of numpy.random.RandomState
        In numpy <=1.16 a ``list`` of  ``RandomState`` s,
        in 1.17+ a ``list`` of ``Generator`` s.
        In both cases lists of derived child generators.

    """
    import imgaug.random
    return imgaug.random.derive_generators_(random_state, n=n)


@deprecated("imgaug.random.advance_generator_")
def forward_random_state(random_state):
    """Advance a numpy random generator's internal state.

    Parameters
    ----------
    random_state : numpy.random.Generator or numpy.random.RandomState
        Generator of which to advance the internal state.

    """
    import imgaug.random
    imgaug.random.advance_generator_(random_state)


def _quokka_normalize_extract(extract):
    """Generate a normalized rectangle for the standard quokka image.

    Parameters
    ----------
    extract : 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Unnormalized representation of the image subarea to be extracted.

            * If ``str`` ``square``, then a squared area
              ``(x: 0 to max 643, y: 0 to max 643)`` will be extracted from
              the image.
            * If a ``tuple``, then expected to contain four ``number`` s
              denoting ``(x1, y1, x2, y2)``.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBox`, then that
              bounding box's area will be extracted from the image.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBoxesOnImage`,
              then expected to contain exactly one bounding box and a shape
              matching the full image dimensions (i.e. ``(643, 960, *)``).
              Then the one bounding box will be used similar to
              ``BoundingBox`` above.

    Returns
    -------
    imgaug.augmentables.bbs.BoundingBox
        Normalized representation of the area to extract from the standard
        quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

    if extract == "square":
        bb = BoundingBox(x1=0, y1=0, x2=643, y2=643)
    elif isinstance(extract, tuple) and len(extract) == 4:
        bb = BoundingBox(x1=extract[0], y1=extract[1],
                         x2=extract[2], y2=extract[3])
    elif isinstance(extract, BoundingBox):
        bb = extract
    elif isinstance(extract, BoundingBoxesOnImage):
        assert len(extract.bounding_boxes) == 1, (
            "Provided BoundingBoxesOnImage instance may currently only "
            "contain a single bounding box.")
        assert extract.shape[0:2] == (643, 960), (
            "Expected BoundingBoxesOnImage instance on an image of shape "
            "(643, 960, ?). Got shape %s." % (extract.shape,))
        bb = extract.bounding_boxes[0]
    else:
        raise Exception(
            "Expected 'square' or tuple of four entries or BoundingBox or "
            "BoundingBoxesOnImage for parameter 'extract', "
            "got %s." % (type(extract),)
        )
    return bb


# TODO is this the same as the project functions in augmentables?
def _compute_resized_shape(from_shape, to_shape):
    """Compute the intended new shape of an image-like array after resizing.

    Parameters
    ----------
    from_shape : tuple or ndarray
        Old shape of the array. Usually expected to be a ``tuple`` of form
        ``(H, W)`` or ``(H, W, C)`` or alternatively an array with two or
        three dimensions.

    to_shape : None or tuple of ints or tuple of floats or int or float or ndarray
        New shape of the array.

            * If ``None``, then `from_shape` will be used as the new shape.
            * If an ``int`` ``V``, then the new shape will be ``(V, V, [C])``,
              where ``C`` will be added if it is part of `from_shape`.
            * If a ``float`` ``V``, then the new shape will be
              ``(H*V, W*V, [C])``, where ``H`` and ``W`` are the old
              height/width.
            * If a ``tuple`` ``(H', W', [C'])`` of ints, then ``H'`` and ``W'``
              will be used as the new height and width.
            * If a ``tuple`` ``(H', W', [C'])`` of floats (except ``C``), then
              ``H'`` and ``W'`` will be used as the new height and width.
            * If a numpy array, then the array's shape will be used.

    Returns
    -------
    tuple of int
        New shape.

    """
    if is_np_array(from_shape):
        from_shape = from_shape.shape
    if is_np_array(to_shape):
        to_shape = to_shape.shape

    to_shape_computed = list(from_shape)

    if to_shape is None:
        pass
    elif isinstance(to_shape, tuple):
        assert len(from_shape) in [2, 3]
        assert len(to_shape) in [2, 3]

        if len(from_shape) == 3 and len(to_shape) == 3:
            assert from_shape[2] == to_shape[2]
        elif len(to_shape) == 3:
            to_shape_computed.append(to_shape[2])

        is_to_s_valid_values = all(
            [v is None or is_single_number(v) for v in to_shape[0:2]])
        assert is_to_s_valid_values, (
            "Expected the first two entries in to_shape to be None or "
            "numbers, got types %s." % (
                str([type(v) for v in to_shape[0:2]]),))

        for i, from_shape_i in enumerate(from_shape[0:2]):
            if to_shape[i] is None:
                to_shape_computed[i] = from_shape_i
            elif is_single_integer(to_shape[i]):
                to_shape_computed[i] = to_shape[i]
            else:  # float
                to_shape_computed[i] = int(np.round(from_shape_i * to_shape[i]))
    elif is_single_integer(to_shape) or is_single_float(to_shape):
        to_shape_computed = _compute_resized_shape(
            from_shape, (to_shape, to_shape))
    else:
        raise Exception(
            "Expected to_shape to be None or ndarray or tuple of floats or "
            "tuple of ints or single int or single float, "
            "got %s." % (type(to_shape),))

    return tuple(to_shape_computed)


def quokka(size=None, extract=None):
    """Return an image of a quokka as a numpy array.

    Parameters
    ----------
    size : None or float or tuple of int, optional
        Size of the output image. Input into
        :func:`~imgaug.imgaug.imresize_single_image`. Usually expected to be a
        ``tuple`` ``(H, W)``, where ``H`` is the desired height and ``W`` is
        the width. If ``None``, then the image will not be resized.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea of the quokka image to extract:

            * If ``None``, then the whole image will be used.
            * If ``str`` ``square``, then a squared area
              ``(x: 0 to max 643, y: 0 to max 643)`` will be extracted from
              the image.
            * If a ``tuple``, then expected to contain four ``number`` s
              denoting ``(x1, y1, x2, y2)``.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBox`, then that
              bounding box's area will be extracted from the image.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBoxesOnImage`,
              then expected to contain exactly one bounding box and a shape
              matching the full image dimensions (i.e. ``(643, 960, *)``).
              Then the one bounding box will be used similar to
              ``BoundingBox`` above.

    Returns
    -------
    (H,W,3) ndarray
        The image array of dtype ``uint8``.

    """
    img = imageio.imread(QUOKKA_FP, pilmode="RGB")
    if extract is not None:
        bb = _quokka_normalize_extract(extract)
        img = bb.extract_from_image(img)
    if size is not None:
        shape_resized = _compute_resized_shape(img.shape, size)
        img = imresize_single_image(img, shape_resized[0:2])
    return img


def quokka_square(size=None):
    """Return an (square) image of a quokka as a numpy array.

    Parameters
    ----------
    size : None or float or tuple of int, optional
        Size of the output image. Input into
        :func:`~imgaug.imgaug.imresize_single_image`. Usually expected to be a
        ``tuple`` ``(H, W)``, where ``H`` is the desired height and ``W`` is
        the width. If ``None``, then the image will not be resized.

    Returns
    -------
    (H,W,3) ndarray
        The image array of dtype ``uint8``.

    """
    return quokka(size=size, extract="square")


def quokka_heatmap(size=None, extract=None):
    """Return a heatmap (here: depth map) for the standard example quokka image.

    Parameters
    ----------
    size : None or float or tuple of int, optional
        See :func:`~imgaug.imgaug.quokka`.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.heatmaps.HeatmapsOnImage
        Depth map as an heatmap object. Values close to ``0.0`` denote objects
        that are close to the camera. Values close to ``1.0`` denote objects
        that are furthest away (among all shown objects).

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

    img = imageio.imread(QUOKKA_DEPTH_MAP_HALFRES_FP, pilmode="RGB")
    img = imresize_single_image(img, (643, 960), interpolation="cubic")

    if extract is not None:
        bb = _quokka_normalize_extract(extract)
        img = bb.extract_from_image(img)
    if size is None:
        size = img.shape[0:2]

    shape_resized = _compute_resized_shape(img.shape, size)
    img = imresize_single_image(img, shape_resized[0:2])
    img_0to1 = img[..., 0]  # depth map was saved as 3-channel RGB
    img_0to1 = img_0to1.astype(np.float32) / 255.0
    img_0to1 = 1 - img_0to1  # depth map was saved as 0 being furthest away

    return HeatmapsOnImage(img_0to1, shape=img_0to1.shape[0:2] + (3,))


def quokka_segmentation_map(size=None, extract=None):
    """Return a segmentation map for the standard example quokka image.

    Parameters
    ----------
    size : None or float or tuple of int, optional
        See :func:`~imgaug.imgaug.quokka`.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.segmaps.SegmentationMapsOnImage
        Segmentation map object.

    """
    # pylint: disable=invalid-name
    # TODO get rid of this deferred import
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    with open(QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)

    xx = []
    yy = []
    for kp_dict in json_dict["polygons"][0]["keypoints"]:
        x = kp_dict["x"]
        y = kp_dict["y"]
        xx.append(x)
        yy.append(y)

    img_seg = np.zeros((643, 960, 1), dtype=np.int32)
    rr, cc = skimage.draw.polygon(
        np.array(yy), np.array(xx), shape=img_seg.shape)
    img_seg[rr, cc, 0] = 1

    if extract is not None:
        bb = _quokka_normalize_extract(extract)
        img_seg = bb.extract_from_image(img_seg)

    segmap = SegmentationMapsOnImage(img_seg, shape=img_seg.shape[0:2] + (3,))

    if size is not None:
        shape_resized = _compute_resized_shape(img_seg.shape, size)
        segmap = segmap.resize(shape_resized[0:2])
        segmap.shape = tuple(shape_resized[0:2]) + (3,)

    return segmap


def quokka_keypoints(size=None, extract=None):
    """Return example keypoints on the standard example quokke image.

    The keypoints cover the eyes, ears, nose and paws.

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the keypoints are placed. If
        ``None``, then the keypoints are not projected to any new size
        (positions on the original image are used). ``float`` s lead to
        relative size changes, ``int`` s to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.kps.KeypointsOnImage
        Example keypoints on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

    left, top = 0, 0
    if extract is not None:
        bb_extract = _quokka_normalize_extract(extract)
        left = bb_extract.x1
        top = bb_extract.y1
    with open(QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    keypoints = []
    for kp_dict in json_dict["keypoints"]:
        keypoints.append(Keypoint(x=kp_dict["x"] - left, y=kp_dict["y"] - top))
    if extract is not None:
        shape = (bb_extract.height, bb_extract.width, 3)
    else:
        shape = (643, 960, 3)
    kpsoi = KeypointsOnImage(keypoints, shape=shape)
    if size is not None:
        shape_resized = _compute_resized_shape(shape, size)
        kpsoi = kpsoi.on(shape_resized)
    return kpsoi


def quokka_bounding_boxes(size=None, extract=None):
    """Return example bounding boxes on the standard example quokke image.

    Currently only a single bounding box is returned that covers the quokka.

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the BBs are placed. If ``None``, then
        the BBs are not projected to any new size (positions on the original
        image are used). ``float`` s lead to relative size changes, ``int`` s
        to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.bbs.BoundingBoxesOnImage
        Example BBs on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

    left, top = 0, 0
    if extract is not None:
        bb_extract = _quokka_normalize_extract(extract)
        left = bb_extract.x1
        top = bb_extract.y1
    with open(QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    bbs = []
    for bb_dict in json_dict["bounding_boxes"]:
        bbs.append(
            BoundingBox(
                x1=bb_dict["x1"] - left,
                y1=bb_dict["y1"] - top,
                x2=bb_dict["x2"] - left,
                y2=bb_dict["y2"] - top
            )
        )
    if extract is not None:
        shape = (bb_extract.height, bb_extract.width, 3)
    else:
        shape = (643, 960, 3)
    bbsoi = BoundingBoxesOnImage(bbs, shape=shape)
    if size is not None:
        shape_resized = _compute_resized_shape(shape, size)
        bbsoi = bbsoi.on(shape_resized)
    return bbsoi


def quokka_polygons(size=None, extract=None):
    """
    Returns example polygons on the standard example quokke image.

    The result contains one polygon, covering the quokka's outline.

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the polygons are placed. If ``None``,
        then the polygons are not projected to any new size (positions on the
        original image are used). ``float`` s lead to relative size changes,
        ``int`` s to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.polys.PolygonsOnImage
        Example polygons on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.polys import Polygon, PolygonsOnImage

    left, top = 0, 0
    if extract is not None:
        bb_extract = _quokka_normalize_extract(extract)
        left = bb_extract.x1
        top = bb_extract.y1
    with open(QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    polygons = []
    for poly_json in json_dict["polygons"]:
        polygons.append(
            Polygon([(point["x"] - left, point["y"] - top)
                     for point in poly_json["keypoints"]])
        )
    if extract is not None:
        shape = (bb_extract.height, bb_extract.width, 3)
    else:
        shape = (643, 960, 3)
    psoi = PolygonsOnImage(polygons, shape=shape)
    if size is not None:
        shape_resized = _compute_resized_shape(shape, size)
        psoi = psoi.on(shape_resized)
    return psoi


# TODO change this to some atan2 stuff?
def angle_between_vectors(v1, v2):
    """Calculcate the angle in radians between vectors `v1` and `v2`.

    From
    http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

    Parameters
    ----------
    v1 : (N,) ndarray
        First vector.

    v2 : (N,) ndarray
        Second vector.

    Returns
    -------
    float
        Angle in radians.

    Examples
    --------
    >>> angle_between_vectors(np.float32([1, 0, 0]), np.float32([0, 1, 0]))
    1.570796...

    >>> angle_between_vectors(np.float32([1, 0, 0]), np.float32([1, 0, 0]))
    0.0

    >>> angle_between_vectors(np.float32([1, 0, 0]), np.float32([-1, 0, 0]))
    3.141592...

    """
    # pylint: disable=invalid-name
    length1 = np.linalg.norm(v1)
    length2 = np.linalg.norm(v2)
    v1_unit = (v1 / length1) if length1 > 0 else np.float32(v1) * 0
    v2_unit = (v2 / length2) if length2 > 0 else np.float32(v2) * 0
    return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))


# TODO is this used anywhere?
# TODO this might also be covered by augmentables.utils or
#      augmentables.polys/lines
def compute_line_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4):
    """Compute the intersection point of two lines.

    Taken from https://stackoverflow.com/a/20679579 .

    Parameters
    ----------
    x1 : number
        x coordinate of the first point on line 1.
        (The lines extends beyond this point.)

    y1 : number
        y coordinate of the first point on line 1.
        (The lines extends beyond this point.)

    x2 : number
        x coordinate of the second point on line 1.
        (The lines extends beyond this point.)

    y2 : number
        y coordinate of the second point on line 1.
        (The lines extends beyond this point.)

    x3 : number
        x coordinate of the first point on line 2.
        (The lines extends beyond this point.)

    y3 : number
        y coordinate of the first point on line 2.
        (The lines extends beyond this point.)

    x4 : number
        x coordinate of the second point on line 2.
        (The lines extends beyond this point.)

    y4 : number
        y coordinate of the second point on line 2.
        (The lines extends beyond this point.)

    Returns
    -------
    tuple of number or bool
        The coordinate of the intersection point as a ``tuple`` ``(x, y)``.
        If the lines are parallel (no intersection point or an infinite number
        of them), the result is ``False``.

    """
    # pylint: disable=invalid-name
    def _make_line(point1, point2):
        line_y = (point1[1] - point2[1])
        line_x = (point2[0] - point1[0])
        slope = (point1[0] * point2[1] - point2[0] * point1[1])
        return line_y, line_x, -slope

    line1 = _make_line((x1, y1), (x2, y2))
    line2 = _make_line((x3, y3), (x4, y4))

    D = line1[0] * line2[1] - line1[1] * line2[0]
    Dx = line1[2] * line2[1] - line1[1] * line2[2]
    Dy = line1[0] * line2[2] - line1[2] * line2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    return False


# TODO replace by cv2.putText()?
def draw_text(img, y, x, text, color=(0, 255, 0), size=25):
    """Draw text on an image.

    This uses by default DejaVuSans as its font, which is included in this
    library.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: yes; not tested
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

        TODO check if other dtypes could be enabled

    Parameters
    ----------
    img : (H,W,3) ndarray
        The image array to draw text on.
        Expected to be of dtype ``uint8`` or ``float32`` (expected value
        range is ``[0.0, 255.0]``).

    y : int
        x-coordinate of the top left corner of the text.

    x : int
        y- coordinate of the top left corner of the text.

    text : str
        The text to draw.

    color : iterable of int, optional
        Color of the text to draw. For RGB-images this is expected to be an
        RGB color.

    size : int, optional
        Font size of the text to draw.

    Returns
    -------
    (H,W,3) ndarray
        Input image with text drawn on it.

    """
    from PIL import (
        Image as PIL_Image,
        ImageDraw as PIL_ImageDraw,
        ImageFont as PIL_ImageFont
    )

    assert img.dtype.name in ["uint8", "float32"], (
        "Can currently draw text only on images of dtype 'uint8' or "
        "'float32'. Got dtype %s." % (img.dtype.name,))

    input_dtype = img.dtype
    if img.dtype == np.float32:
        img = img.astype(np.uint8)

    img = PIL_Image.fromarray(img)
    font = PIL_ImageFont.truetype(DEFAULT_FONT_FP, size)
    context = PIL_ImageDraw.Draw(img)
    context.text((x, y), text, fill=tuple(color), font=font)
    img_np = np.asarray(img)

    # PIL/asarray returns read only array
    if not img_np.flags["WRITEABLE"]:
        try:
            # this seems to no longer work with np 1.16 (or was pillow
            # updated?)
            img_np.setflags(write=True)
        except ValueError as ex:
            if "cannot set WRITEABLE flag to True of this array" in str(ex):
                img_np = np.copy(img_np)

    if img_np.dtype != input_dtype:
        img_np = img_np.astype(input_dtype)

    return img_np


# TODO rename sizes to size?
def imresize_many_images(images, sizes=None, interpolation=None):
    """Resize each image in a list or array to a specified size.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: limited; tested (4)
        * ``int64``: no (2)
        * ``float16``: yes; tested (5)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (6)

        - (1) rejected by ``cv2.imresize``
        - (2) results too inaccurate
        - (3) mapped internally to ``int16`` when interpolation!="nearest"
        - (4) only supported for interpolation="nearest", other interpolations
              lead to cv2 error
        - (5) mapped internally to ``float32``
        - (6) mapped internally to ``uint8``

    Parameters
    ----------
    images : (N,H,W,[C]) ndarray or list of (H,W,[C]) ndarray
        Array of the images to resize.
        Usually recommended to be of dtype ``uint8``.

    sizes : float or iterable of int or iterable of float
        The new size of the images, given either as a fraction (a single
        float) or as a ``(height, width)`` ``tuple`` of two integers or as a
        ``(height fraction, width fraction)`` ``tuple`` of two floats.

    interpolation : None or str or int, optional
        The interpolation to use during resize.
        If ``int``, then expected to be one of:

            * ``cv2.INTER_NEAREST`` (nearest neighbour interpolation)
            * ``cv2.INTER_LINEAR`` (linear interpolation)
            * ``cv2.INTER_AREA`` (area interpolation)
            * ``cv2.INTER_CUBIC`` (cubic interpolation)

        If ``str``, then expected to be one of:

            * ``nearest`` (identical to ``cv2.INTER_NEAREST``)
            * ``linear`` (identical to ``cv2.INTER_LINEAR``)
            * ``area`` (identical to ``cv2.INTER_AREA``)
            * ``cubic`` (identical to ``cv2.INTER_CUBIC``)

        If ``None``, the interpolation will be chosen automatically. For size
        increases, ``area`` interpolation will be picked and for size
        decreases, ``linear`` interpolation will be picked.

    Returns
    -------
    (N,H',W',[C]) ndarray
        Array of the resized images.

    Examples
    --------
    >>> import imgaug as ia
    >>> images = np.zeros((2, 8, 16, 3), dtype=np.uint8)
    >>> images_resized = ia.imresize_many_images(images, 2.0)
    >>> images_resized.shape
    (2, 16, 32, 3)

    Convert two RGB images of height ``8`` and width ``16`` to images of
    height ``2*8=16`` and width ``2*16=32``.

    >>> images_resized = ia.imresize_many_images(images, (2.0, 4.0))
    >>> images_resized.shape
    (2, 16, 64, 3)

    Convert two RGB images of height ``8`` and width ``16`` to images of
    height ``2*8=16`` and width ``4*16=64``.

    >>> images_resized = ia.imresize_many_images(images, (16, 32))
    >>> images_resized.shape
    (2, 16, 32, 3)

    Converts two RGB images of height ``8`` and width ``16`` to images of
    height ``16`` and width ``32``.

    """
    # pylint: disable=too-many-statements

    # we just do nothing if the input contains zero images
    # one could also argue that an exception would be appropriate here
    if len(images) == 0:
        return images

    # verify that sizes contains only values >0
    if is_single_number(sizes) and sizes <= 0:
        raise ValueError(
            "If 'sizes' is given as a single number, it is expected to "
            "be >= 0, got %.8f." % (sizes,))

    # change after the validation to make the above error messages match the
    # original input
    if is_single_number(sizes):
        sizes = (sizes, sizes)
    else:
        assert len(sizes) == 2, (
            "If 'sizes' is given as a tuple, it is expected be a tuple of two "
            "entries, got %d entries." % (len(sizes),))
        assert all([is_single_number(val) and val >= 0 for val in sizes]), (
            "If 'sizes' is given as a tuple, it is expected be a tuple of two "
            "ints or two floats, each >= 0, got types %s with values %s." % (
                str([type(val) for val in sizes]), str(sizes)))

    # if input is a list, call this function N times for N images
    # but check beforehand if all images have the same shape, then just
    # convert to a single array and de-convert afterwards
    if isinstance(images, list):
        nb_shapes = len({image.shape for image in images})
        if nb_shapes == 1:
            return list(imresize_many_images(
                np.array(images), sizes=sizes, interpolation=interpolation))

        return [
            imresize_many_images(
                image[np.newaxis, ...],
                sizes=sizes,
                interpolation=interpolation)[0, ...]
            for image in images]

    shape = images.shape
    assert images.ndim in [3, 4], "Expected array of shape (N, H, W, [C]), " \
                                  "got shape %s" % (str(shape),)
    nb_images = shape[0]
    height_image, width_image = shape[1], shape[2]
    nb_channels = shape[3] if images.ndim > 3 else None

    height_target, width_target = sizes[0], sizes[1]
    height_target = (int(np.round(height_image * height_target))
                     if is_single_float(height_target)
                     else height_target)
    width_target = (int(np.round(width_image * width_target))
                    if is_single_float(width_target)
                    else width_target)

    if height_target == height_image and width_target == width_image:
        return np.copy(images)

    # return empty array if input array contains zero-sized axes
    # note that None==0 is not True (for case nb_channels=None)
    if 0 in [height_target, width_target, nb_channels]:
        shape_out = tuple([shape[0], height_target, width_target]
                          + list(shape[3:]))
        return np.zeros(shape_out, dtype=images.dtype)

    # place this after the (h==h' and w==w') check so that images with
    # zero-sized don't result in errors if the aren't actually resized
    # verify that all input images have height/width > 0
    has_zero_size_axes = any([axis == 0 for axis in images.shape[1:]])
    assert not has_zero_size_axes, (
        "Cannot resize images, because at least one image has a height and/or "
        "width and/or number of channels of zero. "
        "Observed shapes were: %s." % (
            str([image.shape for image in images]),))

    inter = interpolation
    assert inter is None or inter in IMRESIZE_VALID_INTERPOLATIONS, (
        "Expected 'interpolation' to be None or one of %s. Got %s." % (
            ", ".join(
                [str(valid_ip) for valid_ip in IMRESIZE_VALID_INTERPOLATIONS]
            ),
            str(inter)
        )
    )
    if inter is None:
        if height_target > height_image or width_target > width_image:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
    elif inter in ["nearest", cv2.INTER_NEAREST]:
        inter = cv2.INTER_NEAREST
    elif inter in ["linear", cv2.INTER_LINEAR]:
        inter = cv2.INTER_LINEAR
    elif inter in ["area", cv2.INTER_AREA]:
        inter = cv2.INTER_AREA
    else:  # if ip in ["cubic", cv2.INTER_CUBIC]:
        inter = cv2.INTER_CUBIC

    # TODO find more beautiful way to avoid circular imports
    from . import dtypes as iadt
    if inter == cv2.INTER_NEAREST:
        iadt.gate_dtypes(
            images,
            allowed=["bool",
                     "uint8", "uint16",
                     "int8", "int16", "int32",
                     "float16", "float32", "float64"],
            disallowed=["uint32", "uint64", "uint128", "uint256",
                        "int64", "int128", "int256",
                        "float96", "float128", "float256"],
            augmenter=None)
    else:
        iadt.gate_dtypes(
            images,
            allowed=["bool",
                     "uint8", "uint16",
                     "int8", "int16",
                     "float16", "float32", "float64"],
            disallowed=["uint32", "uint64", "uint128", "uint256",
                        "int32", "int64", "int128", "int256",
                        "float96", "float128", "float256"],
            augmenter=None)

    result_shape = (nb_images, height_target, width_target)
    if nb_channels is not None:
        result_shape = result_shape + (nb_channels,)
    result = np.zeros(result_shape, dtype=images.dtype)
    for i, image in enumerate(images):
        input_dtype = image.dtype
        input_dtype_name = input_dtype.name

        if input_dtype_name == "bool":
            image = image.astype(np.uint8) * 255
        elif input_dtype_name == "int8" and inter != cv2.INTER_NEAREST:
            image = image.astype(np.int16)
        elif input_dtype_name == "float16":
            image = image.astype(np.float32)

        if nb_channels is not None and nb_channels > 512:
            channels = [
                cv2.resize(image[..., c], (width_target, height_target),
                           interpolation=inter) for c in sm.xrange(nb_channels)]
            result_img = np.stack(channels, axis=-1)
        else:
            result_img = cv2.resize(
                image, (width_target, height_target), interpolation=inter)

        assert result_img.dtype.name == image.dtype.name, (
            "Expected cv2.resize() to keep the input dtype '%s', but got "
            "'%s'. This is an internal error. Please report." % (
                image.dtype.name, result_img.dtype.name
            )
        )

        # cv2 removes the channel axis if input was (H, W, 1)
        # we re-add it (but only if input was not (H, W))
        if (len(result_img.shape) == 2 and nb_channels is not None
                and nb_channels == 1):
            result_img = result_img[:, :, np.newaxis]

        if input_dtype_name == "bool":
            result_img = result_img > 127
        elif input_dtype_name == "int8" and inter != cv2.INTER_NEAREST:
            # TODO somehow better avoid circular imports here
            from . import dtypes as iadt
            result_img = iadt.restore_dtypes_(result_img, np.int8)
        elif input_dtype_name == "float16":
            # TODO see above
            from . import dtypes as iadt
            result_img = iadt.restore_dtypes_(result_img, np.float16)
        result[i] = result_img
    return result


def _assert_two_or_three_dims(shape):
    if hasattr(shape, "shape"):
        shape = shape.shape
    assert len(shape) in [2, 3], (
        "Expected image with two or three dimensions, but got %d dimensions "
        "and shape %s." % (len(shape), shape))


def imresize_single_image(image, sizes, interpolation=None):
    """Resize a single image.

    **Supported dtypes**:

        See :func:`~imgaug.imgaug.imresize_many_images`.

    Parameters
    ----------
    image : (H,W,C) ndarray or (H,W) ndarray
        Array of the image to resize.
        Usually recommended to be of dtype ``uint8``.

    sizes : float or iterable of int or iterable of float
        See :func:`~imgaug.imgaug.imresize_many_images`.

    interpolation : None or str or int, optional
        See :func:`~imgaug.imgaug.imresize_many_images`.

    Returns
    -------
    (H',W',C) ndarray or (H',W') ndarray
        The resized image.

    """
    _assert_two_or_three_dims(image)

    grayscale = False
    if image.ndim == 2:
        grayscale = True
        image = image[:, :, np.newaxis]

    rs = imresize_many_images(
        image[np.newaxis, :, :, :], sizes, interpolation=interpolation)
    if grayscale:
        return rs[0, :, :, 0]
    return rs[0, ...]


def pool(arr, block_size, func, pad_mode="constant", pad_cval=0,
         preserve_dtype=True, cval=None):
    """Resize an array by pooling values within blocks.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested (2)
        * ``uint64``: no (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested (2)
        * ``int64``: no (1)
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested (2)
        * ``bool``: yes; tested

        - (1) results too inaccurate (at least when using np.average as func)
        - (2) Note that scikit-image documentation says that the wrapped
              pooling function converts inputs to ``float64``. Actual tests
              showed no indication of that happening (at least when using
              preserve_dtype=True).

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool. Ideally of datatype ``float64``.

    block_size : int or tuple of int
        Spatial size of each group of values to pool, aka kernel size.

          * If a single ``int``, then a symmetric block of that size along
            height and width will be used.
          * If a ``tuple`` of two values, it is assumed to be the block size
            along height and width of the image-like, with pooling happening
            per channel.
          * If a ``tuple`` of three values, it is assumed to be the block size
            along height, width and channels.

    func : callable
        Function to apply to a given block in order to convert it to a single
        number, e.g. :func:`numpy.average`, :func:`numpy.min`,
        :func:`numpy.max`.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder. See :func:`~imgaug.imgaug.pad` for details.

    pad_cval : number, optional
        Value to use for padding if `mode` is ``constant``.
        See :func:`numpy.pad` for details.

    preserve_dtype : bool, optional
        Whether to convert the array back to the input datatype if it is
        changed away from that in the pooling process.

    cval : None or number, optional
        Deprecated. Old name for `pad_cval`.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after pooling.

    """
    # TODO find better way to avoid circular import
    from . import dtypes as iadt
    from .augmenters import size as iasize

    if arr.size == 0:
        return np.copy(arr)

    iadt.gate_dtypes(arr,
                     allowed=["bool",
                              "uint8", "uint16", "uint32",
                              "int8", "int16", "int32",
                              "float16", "float32", "float64", "float128"],
                     disallowed=["uint64", "uint128", "uint256",
                                 "int64", "int128", "int256",
                                 "float256"],
                     augmenter=None)

    if cval is not None:
        warn_deprecated("`cval` is a deprecated argument in pool(). "
                        "Use `pad_cval` instead.")
        pad_cval = cval

    _assert_two_or_three_dims(arr)

    is_valid_int = is_single_integer(block_size) and block_size >= 1
    is_valid_tuple = is_iterable(block_size) and len(block_size) in [2, 3] \
        and [is_single_integer(val) and val >= 1 for val in block_size]
    assert is_valid_int or is_valid_tuple, (
        "Expected argument 'block_size' to be a single integer >0 or "
        "a tuple of 2 or 3 values with each one being >0. Got %s." % (
            str(block_size)))

    if is_single_integer(block_size):
        block_size = [block_size, block_size]
    if len(block_size) < arr.ndim:
        block_size = list(block_size) + [1]

    # We use custom padding here instead of the one from block_reduce(),
    # because (1) it is expected to be faster and (2) it allows us more
    # flexibility wrt to padding modes.
    arr = iasize.pad_to_multiples_of(
        arr,
        height_multiple=block_size[0],
        width_multiple=block_size[1],
        mode=pad_mode,
        cval=pad_cval
    )

    input_dtype = arr.dtype

    arr_reduced = skimage.measure.block_reduce(arr, tuple(block_size), func,
                                               cval=cval)
    if preserve_dtype and arr_reduced.dtype.name != input_dtype.name:
        arr_reduced = arr_reduced.astype(input_dtype)
    return arr_reduced


# TODO does OpenCV have a faster avg pooling method?
def avg_pool(arr, block_size, pad_mode="reflect", pad_cval=128,
             preserve_dtype=True, cval=None):
    """Resize an array using average pooling.

    Defaults to ``pad_mode="reflect"`` to ensure that padded values do not
    affect the average.

    **Supported dtypes**:

        See :func:`~imgaug.imgaug.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        See :func:`~imgaug.imgaug.pool` for details.

    block_size : int or tuple of int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug.imgaug.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug.imgaug.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug.imgaug.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug.imgaug.pool` for details.

    cval : None or number, optional
        Deprecated. Old name for `pad_cval`.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after average pooling.

    """
    return pool(arr, block_size, np.average, pad_mode=pad_mode,
                pad_cval=pad_cval, preserve_dtype=preserve_dtype, cval=cval)


def max_pool(arr, block_size, pad_mode="edge", pad_cval=0,
             preserve_dtype=True, cval=None):
    """Resize an array using max-pooling.

    Defaults to ``pad_mode="edge"`` to ensure that padded values do not affect
    the maximum, even if the dtype was something else than ``uint8``.

    **Supported dtypes**:

        See :func:`~imgaug.imgaug.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        See :func:`~imgaug.imgaug.pool` for details.

    block_size : int or tuple of int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug.imgaug.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug.imgaug.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug.imgaug.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug.imgaug.pool` for details.

    cval : None or number, optional
        Deprecated. Old name for `pad_cval`.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after max-pooling.

    """
    return pool(arr, block_size, np.max, pad_mode=pad_mode,
                pad_cval=pad_cval, preserve_dtype=preserve_dtype, cval=cval)


def min_pool(arr, block_size, pad_mode="edge", pad_cval=255,
             preserve_dtype=True):
    """Resize an array using min-pooling.

    Defaults to ``pad_mode="edge"`` to ensure that padded values do not affect
    the minimum, even if the dtype was something else than ``uint8``.

    **Supported dtypes**:

        See :func:`~imgaug.imgaug.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        See :func:`~imgaug.imgaug.pool` for details.

    block_size : int or tuple of int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug.imgaug.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug.imgaug.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug.imgaug.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug.imgaug.pool` for details.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after min-pooling.

    """
    return pool(arr, block_size, np.min, pad_mode=pad_mode, pad_cval=pad_cval,
                preserve_dtype=preserve_dtype)


def median_pool(arr, block_size, pad_mode="reflect", pad_cval=128,
                preserve_dtype=True):
    """Resize an array using median-pooling.

    Defaults to ``pad_mode="reflect"`` to ensure that padded values do not
    affect the average.

    **Supported dtypes**:

        See :func:`~imgaug.imgaug.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        See :func:`~imgaug.imgaug.pool` for details.

    block_size : int or tuple of int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug.imgaug.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug.imgaug.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug.imgaug.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug.imgaug.pool` for details.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after min-pooling.

    """
    return pool(arr, block_size, np.median, pad_mode=pad_mode,
                pad_cval=pad_cval, preserve_dtype=preserve_dtype)


def draw_grid(images, rows=None, cols=None):
    """Combine multiple images into a single grid-like image.

    Calling this function with four images of the same shape and ``rows=2``,
    ``cols=2`` will combine the four images to a single image array of shape
    ``(2*H, 2*W, C)``, where ``H`` is the height of any of the images
    (analogous ``W``) and ``C`` is the number of channels of any image.

    Calling this function with four images of the same shape and ``rows=4``,
    ``cols=1`` is analogous to calling :func:`numpy.vstack` on the images.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested
        * ``float128``: yes; fully tested
        * ``bool``: yes; fully tested

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        The input images to convert to a grid.

    rows : None or int, optional
        The number of rows to show in the grid.
        If ``None``, it will be automatically derived.

    cols : None or int, optional
        The number of cols to show in the grid.
        If ``None``, it will be automatically derived.

    Returns
    -------
    (H',W',3) ndarray
        Image of the generated grid.

    """
    nb_images = len(images)
    assert nb_images > 0, "Expected to get at least one image, got none."

    if is_np_array(images):
        assert images.ndim == 4, (
            "Expected to get an array of four dimensions denoting "
            "(N, H, W, C), got %d dimensions and shape %s." % (
                images.ndim, images.shape))
    else:
        assert is_iterable(images), (
            "Expected to get an iterable of ndarrays, "
            "got %s." % (type(images),))
        assert all([is_np_array(image) for image in images]), (
            "Expected to get an iterable of ndarrays, "
            "got types %s." % (
                ", ".join([str(type(image)) for image in images],)))
        assert all([image.ndim == 3 for image in images]), (
            "Expected to get images with three dimensions. Got shapes %s." % (
                ", ".join([str(image.shape) for image in images])))
        assert len({image.dtype.name for image in images}) == 1, (
            "Expected to get images with the same dtypes, got dtypes %s." % (
                ", ".join([image.dtype.name for image in images])))
        assert len({image.shape[-1] for image in images}) == 1, (
            "Expected to get images with the same number of channels, "
            "got shapes %s." % (
                ", ".join([str(image.shape) for image in images])))

    cell_height = max([image.shape[0] for image in images])
    cell_width = max([image.shape[1] for image in images])
    nb_channels = images[0].shape[2]

    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(nb_images)))
    elif rows is not None:
        cols = int(math.ceil(nb_images / rows))
    elif cols is not None:
        rows = int(math.ceil(nb_images / cols))
    assert rows * cols >= nb_images, (
        "Expected rows*cols to lead to at least as many cells as there were "
        "images provided, but got %d rows, %d cols (=%d cells) for %d "
        "images. " % (rows, cols, rows*cols, nb_images))

    width = cell_width * cols
    height = cell_height * rows
    dtype = images.dtype if is_np_array(images) else images[0].dtype
    grid = np.zeros((height, width, nb_channels), dtype=dtype)
    cell_idx = 0
    for row_idx in sm.xrange(rows):
        for col_idx in sm.xrange(cols):
            if cell_idx < nb_images:
                image = images[cell_idx]
                cell_y1 = cell_height * row_idx
                cell_y2 = cell_y1 + image.shape[0]
                cell_x1 = cell_width * col_idx
                cell_x2 = cell_x1 + image.shape[1]
                grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image
            cell_idx += 1

    return grid


def show_grid(images, rows=None, cols=None):
    """Combine multiple images into a single image and plot the result.

    This will show a window of the results of :func:`~imgaug.imgaug.draw_grid`.

    **Supported dtypes**:

        minimum of (
            :func:`~imgaug.imgaug.draw_grid`,
            :func:`~imgaug.imgaug.imshow`
        )

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        See :func:`~imgaug.imgaug.draw_grid`.

    rows : None or int, optional
        See :func:`~imgaug.imgaug.draw_grid`.

    cols : None or int, optional
        See :func:`~imgaug.imgaug.draw_grid`.

    """
    grid = draw_grid(images, rows=rows, cols=cols)
    imshow(grid)


def imshow(image, backend=IMSHOW_BACKEND_DEFAULT):
    """Show an image in a window.

    **Supported dtypes**:

        * ``uint8``: yes; not tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    image : (H,W,3) ndarray
        Image to show.

    backend : {'matplotlib', 'cv2'}, optional
        Library to use to show the image. May be either matplotlib or
        OpenCV ('cv2'). OpenCV tends to be faster, but apparently causes more
        technical issues.

    """
    assert backend in ["matplotlib", "cv2"], (
        "Expected backend 'matplotlib' or 'cv2', got %s." % (backend,))

    if backend == "cv2":
        image_bgr = image
        if image.ndim == 3 and image.shape[2] in [3, 4]:
            image_bgr = image[..., 0:3][..., ::-1]

        win_name = "imgaug-default-window"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, image_bgr)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)
    else:
        # import only when necessary (faster startup; optional dependency;
        # less fragile -- see issue #225)
        import matplotlib.pyplot as plt

        dpi = 96
        h, w = image.shape[0] / dpi, image.shape[1] / dpi
        # if the figure is too narrow, the footer may appear and make the fig
        # suddenly wider (ugly)
        w = max(w, 6)

        fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
        fig.canvas.set_window_title("imgaug.imshow(%s)" % (image.shape,))
        # cmap=gray is automatically only activate for grayscale images
        ax.imshow(image, cmap="gray")
        plt.show()


def do_assert(condition, message="Assertion failed."):
    """Assert that a ``condition`` holds or raise an ``Exception`` otherwise.

    This was added because `assert` statements are removed in optimized code.
    It replaced `assert` statements throughout the library, but that was
    reverted again for readability and performance reasons.

    Parameters
    ----------
    condition : bool
        If ``False``, an exception is raised.

    message : str, optional
        Error message.

    """
    if not condition:
        raise AssertionError(str(message))


# Added in 0.4.0.
def _normalize_cv2_input_arr_(arr):
    flags = arr.flags
    if not flags["OWNDATA"]:
        arr = np.copy(arr)
        flags = arr.flags
    if not flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def apply_lut(image, table):
    """Map an input image to a new one using a lookup table.

    Added in 0.4.0.

    **Supported dtypes**:

        See :func:`~imgaug.imgaug.apply_lut_`.

    Parameters
    ----------
    image : ndarray
        See :func:`~imgaug.imgaug.apply_lut_`.

    table : ndarray or list of ndarray
        See :func:`~imgaug.imgaug.apply_lut_`.

    Returns
    -------
    ndarray
        Image after mapping via lookup table.

    """
    return apply_lut_(np.copy(image), table)


# TODO make this function compatible with short max sized images, probably
#      isn't right now
def apply_lut_(image, table):
    """Map an input image in-place to a new one using a lookup table.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        Image of dtype ``uint8`` and shape ``(H,W)`` or ``(H,W,C)``.

    table : ndarray or list of ndarray
        Table of dtype ``uint8`` containing the mapping from old to new
        values. Either a ``list`` of ``C`` ``(256,)`` arrays or a single
        array of shape ``(256,)`` or ``(256, C)`` or ``(1, 256, C)``.
        In case of ``(256,)`` the same table is used for all channels,
        otherwise a channelwise table is used and ``C`` is expected to match
        the number of channels.

    Returns
    -------
    ndarray
        Image after mapping via lookup table.
        This *might* be the same array instance as provided via `image`.

    """

    image_shape_orig = image.shape
    nb_channels = 1 if len(image_shape_orig) == 2 else image_shape_orig[-1]

    if 0 in image_shape_orig:
        return image

    image = _normalize_cv2_input_arr_(image)

    # [(256,), (256,), ...] => (256, C)
    if isinstance(table, list):
        assert len(table) == nb_channels, (
            "Expected to get %d tables (one per channel), got %d instead." % (
                nb_channels, len(table)))
        table = np.stack(table, axis=-1)

    # (256, C) => (1, 256, C)
    if table.shape == (256, nb_channels):
        table = table[np.newaxis, :, :]

    assert table.shape == (256,) or table.shape == (1, 256, nb_channels), (
        "Expected 'table' to be any of the following: "
        "A list of C (256,) arrays, an array of shape (256,), an array of "
        "shape (256, C), an array of shape (1, 256, C). Transformed 'table' "
        "up to shape %s for image with shape %s (C=%d)." % (
            table.shape, image_shape_orig, nb_channels))

    if nb_channels > 512:
        if table.shape == (256,):
            table = np.tile(table[np.newaxis, :, np.newaxis],
                            (1, 1, nb_channels))

        subluts = []
        for group_idx in np.arange(int(np.ceil(nb_channels / 512))):
            c_start = group_idx * 512
            c_end = c_start + 512
            subluts.append(apply_lut_(image[:, :, c_start:c_end],
                                      table[:, :, c_start:c_end]))

        return np.concatenate(subluts, axis=2)

    assert image.dtype.name == "uint8", (
        "Expected uint8 image, got dtype %s." % (image.dtype.name,))
    assert table.dtype.name == "uint8", (
        "Expected uint8 table, got dtype %s." % (table.dtype.name,))

    image = cv2.LUT(image, table, dst=image)
    return image


class HooksImages(object):
    """Class to intervene with image augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    Parameters
    ----------
    activator : None or callable, optional
        A function that gives permission to execute an augmenter.
        The expected interface is::

            ``f(images, augmenter, parents, default)``

        where ``images`` are the input images to augment, ``augmenter`` is the
        instance of the augmenter to execute, ``parents`` are previously
        executed augmenters and ``default`` is an expected default value to be
        returned if the activator function does not plan to make a decision
        for the given inputs.

    propagator : None or callable, optional
        A function that gives permission to propagate the augmentation further
        to the children of an augmenter. This happens after the activator.
        In theory, an augmenter may augment images itself (if allowed by the
        activator) and then execute child augmenters afterwards (if allowed by
        the propagator). If the activator returned ``False``, the propagation
        step will never be executed.
        The expected interface is::

            ``f(images, augmenter, parents, default)``

        with all arguments having identical meaning to the activator.

    preprocessor : None or callable, optional
        A function to call before an augmenter performed any augmentations.
        The interface is:

            ``f(images, augmenter, parents)``

        with all arguments having identical meaning to the activator.
        It is expected to return the input images, optionally modified.

    postprocessor : None or callable, optional
        A function to call after an augmenter performed augmentations.
        The interface is the same as for the `preprocessor`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug as ia
    >>> import imgaug.augmenters as iaa
    >>> seq = iaa.Sequential([
    >>>     iaa.GaussianBlur(3.0, name="blur"),
    >>>     iaa.Dropout(0.05, name="dropout"),
    >>>     iaa.Affine(translate_px=-5, name="affine")
    >>> ])
    >>> images = [np.zeros((10, 10), dtype=np.uint8)]
    >>>
    >>> def activator(images, augmenter, parents, default):
    >>>     return False if augmenter.name in ["blur", "dropout"] else default
    >>>
    >>> seq_det = seq.to_deterministic()
    >>> images_aug = seq_det.augment_images(images)
    >>> heatmaps = [np.random.rand(*(3, 10, 10))]
    >>> heatmaps_aug = seq_det.augment_images(
    >>>     heatmaps,
    >>>     hooks=ia.HooksImages(activator=activator)
    >>> )

    This augments images and their respective heatmaps in the same way.
    The heatmaps however are only modified by ``Affine``, not by
    ``GaussianBlur`` or ``Dropout``.

    """

    def __init__(self, activator=None, propagator=None, preprocessor=None,
                 postprocessor=None):
        self.activator = activator
        self.propagator = propagator
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def is_activated(self, images, augmenter, parents, default):
        """Estimate whether an augmenter may be executed.

        This also affects propagation of data to child augmenters.

        Returns
        -------
        bool
            If ``True``, the augmenter may be executed.
            Otherwise ``False``.

        """
        if self.activator is None:
            return default
        return self.activator(images, augmenter, parents, default)

    def is_propagating(self, images, augmenter, parents, default):
        """Estimate whether an augmenter may call its children.

        This function decides whether an augmenter with children is allowed
        to call these in order to further augment the inputs.
        Note that if the augmenter itself performs augmentations (before/after
        calling its children), these may still be executed, even if this
        method returns ``False``.

        Returns
        -------
        bool
            If ``True``, the augmenter may propagate data to its children.
            Otherwise ``False``.

        """
        if self.propagator is None:
            return default
        return self.propagator(images, augmenter, parents, default)

    def preprocess(self, images, augmenter, parents):
        """Preprocess input data per augmenter before augmentation.

        Returns
        -------
        (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            The input images, optionally modified.

        """
        if self.preprocessor is None:
            return images
        return self.preprocessor(images, augmenter, parents)

    def postprocess(self, images, augmenter, parents):
        """Postprocess input data per augmenter after augmentation.

        Returns
        -------
        (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            The input images, optionally modified.

        """
        if self.postprocessor is None:
            return images
        return self.postprocessor(images, augmenter, parents)


class HooksHeatmaps(HooksImages):
    """Class to intervene with heatmap augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    This class is currently the same as the one for images. This may or may
    not change in the future.

    """


class HooksKeypoints(HooksImages):
    """Class to intervene with keypoint augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    This class is currently the same as the one for images. This may or may
    not change in the future.

    """


#####################################################################
# Create classes/functions that were moved to other files and create
# DeprecatedWarnings when they are called.
#####################################################################

def _mark_moved_class_or_function(class_name_old, module_name_new,
                                  class_name_new):
    # pylint: disable=redefined-outer-name
    class_name_new = (class_name_new
                      if class_name_new is not None
                      else class_name_old)

    def _func(*args, **kwargs):
        import importlib
        warn_deprecated(
            "Using imgaug.imgaug.%s is deprecated. Use %s.%s instead." % (
                class_name_old, module_name_new, class_name_new
            ))
        module = importlib.import_module(module_name_new)
        return getattr(module, class_name_new)(*args, **kwargs)

    return _func


MOVED = [
    ("Keypoint", "imgaug.augmentables.kps", None),
    ("KeypointsOnImage", "imgaug.augmentables.kps", None),
    ("BoundingBox", "imgaug.augmentables.bbs", None),
    ("BoundingBoxesOnImage", "imgaug.augmentables.bbs", None),
    ("Polygon", "imgaug.augmentables.polys", None),
    ("PolygonsOnImage", "imgaug.augmentables.polys", None),
    ("MultiPolygon", "imgaug.augmentables.polys", None),
    ("_ConcavePolygonRecoverer", "imgaug.augmentables.polys", None),
    ("HeatmapsOnImage", "imgaug.augmentables.heatmaps", None),
    ("SegmentationMapsOnImage", "imgaug.augmentables.segmaps", None),
    ("Batch", "imgaug.augmentables.batches", None),
    ("BatchLoader", "imgaug.multicore", None),
    ("BackgroundAugmenter", "imgaug.multicore", None),
    ("compute_geometric_median", "imgaug.augmentables.kps", None),
    ("_convert_points_to_shapely_line_string", "imgaug.augmentables.polys",
     None),
    ("_interpolate_point_pair", "imgaug.augmentables.polys", None),
    ("_interpolate_points", "imgaug.augmentables.polys", None),
    ("_interpolate_points_by_max_distance", "imgaug.augmentables.polys", None),
    ("pad", "imgaug.augmenters.size", None),
    ("pad_to_aspect_ratio", "imgaug.augmenters.size", None),
    ("pad_to_multiples_of", "imgaug.augmenters.size", None),
    ("compute_paddings_for_aspect_ratio", "imgaug.augmenters.size",
     "compute_paddings_to_reach_aspect_ratio"),
    ("compute_paddings_to_reach_multiples_of", "imgaug.augmenters.size", None),
    ("compute_paddings_to_reach_exponents_of", "imgaug.augmenters.size", None)
]

for class_name_old, module_name_new, class_name_new in MOVED:
    locals()[class_name_old] = _mark_moved_class_or_function(
        class_name_old, module_name_new, class_name_new)
