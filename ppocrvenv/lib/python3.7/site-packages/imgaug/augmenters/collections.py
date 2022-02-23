"""Augmenters that are collections of other augmenters.

List of augmenters:

    * :class:`RandAugment`

Added in 0.4.0.

"""
from __future__ import print_function, division, absolute_import

import numpy as np

from .. import parameters as iap
from .. import random as iarandom
from . import meta
from . import arithmetic
from . import flip
from . import pillike
from . import size as sizelib


class RandAugment(meta.Sequential):
    """Apply RandAugment to inputs as described in the corresponding paper.

    See paper::

        Cubuk et al.

        RandAugment: Practical automated data augmentation with a reduced
        search space

    .. note::

        The paper contains essentially no hyperparameters for the individual
        augmentation techniques. The hyperparameters used here come mostly
        from the official code repository, which however seems to only contain
        code for CIFAR10 and SVHN, not for ImageNet. So some guesswork was
        involved and a few of the hyperparameters were also taken from
        https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py .

        This implementation deviates from the code repository for all PIL
        enhance operations. In the repository these use a factor of
        ``0.1 + M*1.8/M_max``, which would lead to a factor of ``0.1`` for the
        weakest ``M`` of ``M=0``. For e.g. ``Brightness`` that would result in
        a basically black image. This definition is fine for AutoAugment (from
        where the code and hyperparameters are copied), which optimizes
        each transformation's ``M`` individually, but not for RandAugment,
        which uses a single fixed ``M``. We hence redefine these
        hyperparameters to ``1.0 + S * M * 0.9/M_max``, where ``S`` is
        randomly either ``1`` or ``-1``.

        We also note that it is not entirely clear which transformations
        were used in the ImageNet experiments. The paper lists some
        transformations in Figure 2, but names others in the text too (e.g.
        crops, flips, cutout). While Figure 2 lists the Identity function,
        this transformation seems to not appear in the repository (and in fact,
        the function ``randaugment(N, M)`` doesn't seem to exist in the
        repository either). So we also make a best guess here about what
        transformations might have been used.

    .. warning::

        This augmenter only works with image data, not e.g. bounding boxes.
        The used PIL-based affine transformations are not yet able to
        process non-image data. (This augmenter uses PIL-based affine
        transformations to ensure that outputs are as similar as possible
        to the paper's implementation.)

    Added in 0.4.0.

    **Supported dtypes**:

    minimum of (
        :class:`~imgaug.augmenters.flip.Fliplr`,
        :class:`~imgaug.augmenters.size.KeepSizeByResize`,
        :class:`~imgaug.augmenters.size.Crop`,
        :class:`~imgaug.augmenters.meta.Sequential`,
        :class:`~imgaug.augmenters.meta.SomeOf`,
        :class:`~imgaug.augmenters.meta.Identity`,
        :class:`~imgaug.augmenters.pillike.Autocontrast`,
        :class:`~imgaug.augmenters.pillike.Equalize`,
        :class:`~imgaug.augmenters.arithmetic.Invert`,
        :class:`~imgaug.augmenters.pillike.Affine`,
        :class:`~imgaug.augmenters.pillike.Posterize`,
        :class:`~imgaug.augmenters.pillike.Solarize`,
        :class:`~imgaug.augmenters.pillike.EnhanceColor`,
        :class:`~imgaug.augmenters.pillike.EnhanceContrast`,
        :class:`~imgaug.augmenters.pillike.EnhanceBrightness`,
        :class:`~imgaug.augmenters.pillike.EnhanceSharpness`,
        :class:`~imgaug.augmenters.arithmetic.Cutout`,
        :class:`~imgaug.augmenters.pillike.FilterBlur`,
        :class:`~imgaug.augmenters.pillike.FilterSmooth`
    )

    Parameters
    ----------
    n : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or None, optional
        Parameter ``N`` in the paper, i.e. number of transformations to apply.
        The paper suggests ``N=2`` for ImageNet.
        See also parameter ``n`` in :class:`~imgaug.augmenters.meta.SomeOf`
        for more details.

        Note that horizontal flips (p=50%) and crops are always applied. This
        parameter only determines how many of the other transformations
        are applied per image.


    m : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or None, optional
        Parameter ``M`` in the paper, i.e. magnitude/severity/strength of the
        applied transformations in interval ``[0 .. 30]`` with ``M=0`` being
        the weakest. The paper suggests for ImageNet ``M=9`` in case of
        ResNet-50 and ``M=28`` in case of EfficientNet-B7.
        This implementation uses a default value of ``(6, 12)``, i.e. the
        value is uniformly sampled per image from the interval ``[6 .. 12]``.
        This ensures greater diversity of transformations than using a single
        fixed value.

        * If ``int``: That value will always be used.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled per
          image from the discrete interval ``[a .. b]``.
        * If ``list``: A random value will be picked from the list per image.
        * If ``StochasticParameter``: For ``B`` images in a batch, ``B`` values
          will be sampled per augmenter (provided the augmenter is dependent
          on the magnitude).

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value to use when filling in newly created pixels.
        See parameter `fillcolor` in
        :class:`~imgaug.augmenters.pillike.Affine` for details.

        The paper's repository uses an RGB value of ``125, 122, 113``.
        This implementation uses a single intensity value of ``128``, which
        should work better for cases where input images don't have exactly
        ``3`` channels or come from a different dataset than used by the
        paper.

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
    >>> aug = iaa.RandAugment(n=2, m=9)

    Create a RandAugment augmenter similar to the suggested hyperparameters
    in the paper.

    >>> aug = iaa.RandAugment(m=30)

    Create a RandAugment augmenter with maximum magnitude/strength.

    >>> aug = iaa.RandAugment(m=(0, 9))

    Create a RandAugment augmenter that applies its transformations with a
    random magnitude between ``0`` (very weak) and ``9`` (recommended for
    ImageNet and ResNet-50). ``m`` is sampled per transformation.

    >>> aug = iaa.RandAugment(n=(0, 3))

    Create a RandAugment augmenter that applies ``0`` to ``3`` of its
    child transformations to images. Horizontal flips (p=50%) and crops are
    always applied.

    """

    _M_MAX = 30

    # according to paper:
    # N=2, M=9 is optimal for ImageNet with ResNet-50
    # N=2, M=28 is optimal for ImageNet with EfficientNet-B7
    # for cval they use [125, 122, 113]
    # Added in 0.4.0.
    def __init__(self, n=2, m=(6, 12), cval=128,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=invalid-name
        seed = seed if random_state == "deprecated" else random_state
        rng = iarandom.RNG(seed)

        # we don't limit the value range to 10 here, because the paper
        # gives several examples of using more than 10 for M
        m = iap.handle_discrete_param(
            m, "m", value_range=(0, None),
            tuple_to_uniform=True, list_to_choice=True,
            allow_floats=False)
        self._m = m
        self._cval = cval

        # The paper says in Appendix A.2.3 "ImageNet", that they actually
        # always execute Horizontal Flips and Crops first and only then a
        # random selection of the other transformations.
        # Hence, we split here into two groups.
        # It's not really clear what crop parameters they use, so we
        # choose [0..M] here.
        initial_augs = self._create_initial_augmenters_list(m)
        main_augs = self._create_main_augmenters_list(m, cval)

        # assign random state to all child augmenters
        for lst in [initial_augs, main_augs]:
            for augmenter in lst:
                augmenter.random_state = rng

        super(RandAugment, self).__init__(
            [
                meta.Sequential(initial_augs,
                                seed=rng.derive_rng_()),
                meta.SomeOf(n, main_augs, random_order=True,
                            seed=rng.derive_rng_())
            ],
            seed=rng, name=name,
            random_state=random_state, deterministic=deterministic
        )

    # Added in 0.4.0.
    @classmethod
    def _create_initial_augmenters_list(cls, m):
        # pylint: disable=invalid-name
        return [
            flip.Fliplr(0.5),
            sizelib.KeepSizeByResize(
                # assuming that the paper implementation crops M pixels from
                # 224px ImageNet images, we crop here a fraction of
                # M*(M_max/224)
                sizelib.Crop(
                    percent=iap.Divide(
                        iap.Uniform(0, m),
                        224,
                        elementwise=True),
                    sample_independently=True,
                    keep_size=False),
                interpolation="linear"
            )
        ]

    # Added in 0.4.0.
    @classmethod
    def _create_main_augmenters_list(cls, m, cval):
        # pylint: disable=invalid-name
        m_max = cls._M_MAX

        def _float_parameter(level, maxval):
            maxval_norm = maxval / m_max
            return iap.Multiply(level, maxval_norm, elementwise=True)

        def _int_parameter(level, maxval):
            # paper applies just int(), so we don't round here
            return iap.Discretize(_float_parameter(level, maxval),
                                  round=False)

        # In the paper's code they use the definition from AutoAugment,
        # which is 0.1 + M*1.8/10. But that results in 0.1 for M=0, i.e. for
        # Brightness an almost black image, while M=5 would result in an
        # unaltered image. For AutoAugment that may be fine, as M is optimized
        # for each operation individually, but here we have only one fixed M
        # for all operations. Hence, we rather set this to 1.0 +/- M*0.9/10,
        # so that M=10 would result in 0.1 or 1.9.
        def _enhance_parameter(level):
            fparam = _float_parameter(level, 0.9)
            return iap.Clip(
                iap.Add(1.0, iap.RandomSign(fparam), elementwise=True),
                0.1, 1.9
            )

        def _subtract(a, b):
            return iap.Subtract(a, b, elementwise=True)

        def _affine(*args, **kwargs):
            kwargs["fillcolor"] = cval
            if "center" not in kwargs:
                kwargs["center"] = (0.0, 0.0)
            return pillike.Affine(*args, **kwargs)

        _rnd_s = iap.RandomSign
        shear_max = np.rad2deg(0.3)

        # we don't add vertical flips here, paper is not really clear about
        # whether they used them or not
        return [
            meta.Identity(),
            pillike.Autocontrast(cutoff=0),
            pillike.Equalize(),
            arithmetic.Invert(p=1.0),
            # they use Image.rotate() for the rotation, which uses
            # the image center as the rotation center
            _affine(rotate=_rnd_s(_float_parameter(m, 30)),
                    center=(0.5, 0.5)),
            # paper uses 4 - int_parameter(M, 4)
            pillike.Posterize(
                nb_bits=_subtract(
                    8,
                    iap.Clip(_int_parameter(m, 6), 0, 6)
                )
            ),
            # paper uses 256 - int_parameter(M, 256)
            pillike.Solarize(
                p=1.0,
                threshold=iap.Clip(
                    _subtract(256, _int_parameter(m, 256)),
                    0, 256
                )
            ),
            pillike.EnhanceColor(_enhance_parameter(m)),
            pillike.EnhanceContrast(_enhance_parameter(m)),
            pillike.EnhanceBrightness(_enhance_parameter(m)),
            pillike.EnhanceSharpness(_enhance_parameter(m)),
            _affine(shear={"x": _rnd_s(_float_parameter(m, shear_max))}),
            _affine(shear={"y": _rnd_s(_float_parameter(m, shear_max))}),
            _affine(translate_percent={"x": _rnd_s(_float_parameter(m, 0.33))}),
            _affine(translate_percent={"y": _rnd_s(_float_parameter(m, 0.33))}),
            # paper code uses 20px on CIFAR (i.e. size 20/32), no information
            # on ImageNet values so we just use the same values
            arithmetic.Cutout(1,
                              size=iap.Clip(
                                  _float_parameter(m, 20 / 32), 0, 20 / 32),
                              squared=True,
                              fill_mode="constant",
                              cval=cval),
            pillike.FilterBlur(),
            pillike.FilterSmooth()
        ]

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        someof = self[1]
        return [someof.n, self._m, self._cval]
