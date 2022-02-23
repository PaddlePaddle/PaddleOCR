"""
Augmenters that create weather effects.

List of augmenters:

    * :class:`FastSnowyLandscape`
    * :class:`CloudLayer`
    * :class:`Clouds`
    * :class:`Fog`
    * :class:`SnowflakesLayer`
    * :class:`Snowflakes`
    * :class:`RainLayer`
    * :class:`Rain`

"""
from __future__ import print_function, division, absolute_import

import numpy as np

import imgaug as ia
from . import meta, arithmetic, blur, contrast, color as colorlib
from .. import parameters as iap
from .. import dtypes as iadt


class FastSnowyLandscape(meta.Augmenter):
    """Convert non-snowy landscapes to snowy ones.

    This augmenter expects to get an image that roughly shows a landscape.

    This augmenter is based on the method proposed in
    https://medium.freecodecamp.org/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f?gi=bca4a13e634c

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

        - (1) This augmenter is based on a colorspace conversion to HLS.
              Hence, only RGB ``uint8`` inputs are sensible.

    Parameters
    ----------
    lightness_threshold : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        All pixels with lightness in HLS colorspace that is below this value
        will have their lightness increased by `lightness_multiplier`.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    lightness_multiplier : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier for pixel's lightness value in HLS colorspace.
        Affects all pixels selected via `lightness_threshold`.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    from_colorspace : str, optional
        The source colorspace of the input images.
        See :func:`~imgaug.augmenters.color.ChangeColorspace.__init__`.

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
    >>> aug = iaa.FastSnowyLandscape(
    >>>     lightness_threshold=140,
    >>>     lightness_multiplier=2.5
    >>> )

    Search for all pixels in the image with a lightness value in HLS
    colorspace of less than ``140`` and increase their lightness by a factor
    of ``2.5``.

    >>> aug = iaa.FastSnowyLandscape(
    >>>     lightness_threshold=[128, 200],
    >>>     lightness_multiplier=(1.5, 3.5)
    >>> )

    Search for all pixels in the image with a lightness value in HLS
    colorspace of less than ``128`` or less than ``200`` (one of these
    values is picked per image) and multiply their lightness by a factor
    of ``x`` with ``x`` being sampled from ``uniform(1.5, 3.5)`` (once per
    image).

    >>> aug = iaa.FastSnowyLandscape(
    >>>     lightness_threshold=(100, 255),
    >>>     lightness_multiplier=(1.0, 4.0)
    >>> )

    Similar to the previous example, but the lightness threshold is sampled
    from ``uniform(100, 255)`` (per image) and the multiplier
    from ``uniform(1.0, 4.0)`` (per image). This seems to produce good and
    varied results.

    """

    def __init__(self, lightness_threshold=(100, 255),
                 lightness_multiplier=(1.0, 4.0),
                 from_colorspace=colorlib.CSPACE_RGB,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FastSnowyLandscape, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.lightness_threshold = iap.handle_continuous_param(
            lightness_threshold, "lightness_threshold",
            value_range=(0, 255), tuple_to_uniform=True, list_to_choice=True)
        self.lightness_multiplier = iap.handle_continuous_param(
            lightness_multiplier, "lightness_multiplier",
            value_range=(0, None), tuple_to_uniform=True, list_to_choice=True)
        self.from_colorspace = from_colorspace

    def _draw_samples(self, augmentables, random_state):
        nb_augmentables = len(augmentables)
        rss = random_state.duplicate(2)
        thresh_samples = self.lightness_threshold.draw_samples(
            (nb_augmentables,), rss[1])
        lmul_samples = self.lightness_multiplier.draw_samples(
            (nb_augmentables,), rss[0])
        return thresh_samples, lmul_samples

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        thresh_samples, lmul_samples = self._draw_samples(images, random_state)

        gen = enumerate(zip(images, thresh_samples, lmul_samples))
        for i, (image, thresh, lmul) in gen:
            image_hls = colorlib.change_colorspace_(
                image, colorlib.CSPACE_HLS, self.from_colorspace)
            cvt_dtype = image_hls.dtype
            image_hls = image_hls.astype(np.float64)
            lightness = image_hls[..., 1]

            lightness[lightness < thresh] *= lmul

            image_hls = iadt.restore_dtypes_(image_hls, cvt_dtype)
            image_rgb = colorlib.change_colorspace_(
                image_hls, self.from_colorspace, colorlib.CSPACE_HLS)

            batch.images[i] = image_rgb

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.lightness_threshold, self.lightness_multiplier]


# TODO add examples and add these to the overview docs
# TODO add perspective transform to each cloud layer to make them look more
#      distant?
# TODO alpha_mean and density overlap - remove one of them
class CloudLayer(meta.Augmenter):
    """Add a single layer of clouds to an image.

    **Supported dtypes**:

        * ``uint8``: yes; indirectly tested (1)
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: yes; not tested
        * ``float32``: yes; not tested
        * ``float64``: yes; not tested
        * ``float128``: yes; not tested (2)
        * ``bool``: no

        - (1) Indirectly tested via tests for :class:`Clouds`` and :class:`Fog`
        - (2) Note that random values are usually sampled as ``int64`` or
              ``float64``, which ``float128`` images would exceed. Note also
              that random values might have to upscaled, which is done
              via :func:`~imgaug.imgaug.imresize_many_images` and has its own
              limited dtype support (includes however floats up to ``64bit``).

    Parameters
    ----------
    intensity_mean : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Mean intensity of the clouds (i.e. mean color).
        Recommended to be in the interval ``[190, 255]``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    intensity_freq_exponent : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Exponent of the frequency noise used to add fine intensity to the
        mean intensity.
        Recommended to be in the interval ``[-2.5, -1.5]``.
        See :func:`~imgaug.parameters.FrequencyNoise.__init__` for details.

    intensity_coarse_scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Standard deviation of the gaussian distribution used to add more
        localized intensity to the mean intensity. Sampled in low resolution
        space, i.e. affects final intensity on a coarse level.
        Recommended to be in the interval ``(0, 10]``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    alpha_min : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Minimum alpha when blending cloud noise with the image.
        High values will lead to clouds being "everywhere".
        Recommended to usually be at around ``0.0`` for clouds and ``>0`` for
        fog.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    alpha_multiplier : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Multiplier for the sampled alpha values. High values will lead to
        denser clouds wherever they are visible.
        Recommended to be in the interval ``[0.3, 1.0]``.
        Note that this parameter currently overlaps with `density_multiplier`,
        which is applied a bit later to the alpha mask.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    alpha_size_px_max : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Controls the image size at which the alpha mask is sampled.
        Lower values will lead to coarser alpha masks and hence larger
        clouds (and empty areas).
        See :func:`~imgaug.parameters.FrequencyNoise.__init__` for details.

    alpha_freq_exponent : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Exponent of the frequency noise used to sample the alpha mask.
        Similarly to `alpha_size_max_px`, lower values will lead to coarser
        alpha patterns.
        Recommended to be in the interval ``[-4.0, -1.5]``.
        See :func:`~imgaug.parameters.FrequencyNoise.__init__` for details.

    sparsity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Exponent applied late to the alpha mask. Lower values will lead to
        coarser cloud patterns, higher values to finer patterns.
        Recommended to be somewhere around ``1.0``.
        Do not deviate far from that value, otherwise the alpha mask might
        get weird patterns with sudden fall-offs to zero that look very
        unnatural.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    density_multiplier : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Late multiplier for the alpha mask, similar to `alpha_multiplier`.
        Set this higher to get "denser" clouds wherever they are visible.
        Recommended to be around ``[0.5, 1.5]``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

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

    """

    def __init__(self, intensity_mean, intensity_freq_exponent,
                 intensity_coarse_scale, alpha_min, alpha_multiplier,
                 alpha_size_px_max, alpha_freq_exponent, sparsity,
                 density_multiplier,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(CloudLayer, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.intensity_mean = iap.handle_continuous_param(
            intensity_mean, "intensity_mean")
        self.intensity_freq_exponent = intensity_freq_exponent
        self.intensity_coarse_scale = intensity_coarse_scale
        self.alpha_min = iap.handle_continuous_param(alpha_min, "alpha_min")
        self.alpha_multiplier = iap.handle_continuous_param(
            alpha_multiplier, "alpha_multiplier")
        self.alpha_size_px_max = alpha_size_px_max
        self.alpha_freq_exponent = alpha_freq_exponent
        self.sparsity = iap.handle_continuous_param(sparsity, "sparsity")
        self.density_multiplier = iap.handle_continuous_param(
            density_multiplier, "density_multiplier")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        rss = random_state.duplicate(len(images))
        for i, (image, rs) in enumerate(zip(images, rss)):
            batch.images[i] = self.draw_on_image(image, rs)
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.intensity_mean,
                self.alpha_min,
                self.alpha_multiplier,
                self.alpha_size_px_max,
                self.alpha_freq_exponent,
                self.intensity_freq_exponent,
                self.sparsity,
                self.density_multiplier,
                self.intensity_coarse_scale]

    def draw_on_image(self, image, random_state):
        iadt.gate_dtypes(image,
                         allowed=["uint8",
                                  "float16", "float32", "float64",
                                  "float96", "float128", "float256"],
                         disallowed=["bool",
                                     "uint16", "uint32", "uint64", "uint128",
                                     "uint256",
                                     "int8", "int16", "int32", "int64",
                                     "int128", "int256"])

        alpha, intensity = self.generate_maps(image, random_state)
        alpha = alpha[..., np.newaxis]
        intensity = intensity[..., np.newaxis]

        if image.dtype.kind == "f":
            intensity = intensity.astype(image.dtype)
            return (1 - alpha) * image + alpha * intensity

        intensity = np.clip(intensity, 0, 255)
        # TODO use blend_alpha_() here
        return np.clip(
            (1 - alpha) * image.astype(alpha.dtype)
            + alpha * intensity.astype(alpha.dtype),
            0,
            255
        ).astype(np.uint8)

    def generate_maps(self, image, random_state):
        intensity_mean_sample = self.intensity_mean.draw_sample(random_state)
        alpha_min_sample = self.alpha_min.draw_sample(random_state)
        alpha_multiplier_sample = \
            self.alpha_multiplier.draw_sample(random_state)
        alpha_size_px_max = self.alpha_size_px_max
        intensity_freq_exponent = self.intensity_freq_exponent
        alpha_freq_exponent = self.alpha_freq_exponent
        sparsity_sample = self.sparsity.draw_sample(random_state)
        density_multiplier_sample = \
            self.density_multiplier.draw_sample(random_state)

        height, width = image.shape[0:2]
        rss_alpha, rss_intensity = random_state.duplicate(2)

        intensity_coarse = self._generate_intensity_map_coarse(
            height, width, intensity_mean_sample,
            iap.Normal(0, scale=self.intensity_coarse_scale),
            rss_intensity
        )
        intensity_fine = self._generate_intensity_map_fine(
            height, width, intensity_mean_sample, intensity_freq_exponent,
            rss_intensity)
        intensity = intensity_coarse + intensity_fine

        alpha = self._generate_alpha_mask(
            height, width, alpha_min_sample, alpha_multiplier_sample,
            alpha_freq_exponent, alpha_size_px_max, sparsity_sample,
            density_multiplier_sample, rss_alpha)

        return alpha, intensity

    @classmethod
    def _generate_intensity_map_coarse(cls, height, width, intensity_mean,
                                       intensity_local_offset, random_state):
        # TODO (8, 8) might be too simplistic for some image sizes
        height_intensity, width_intensity = (8, 8)
        intensity = (
            intensity_mean
            + intensity_local_offset.draw_samples(
                (height_intensity, width_intensity), random_state)
        )
        intensity = ia.imresize_single_image(
            intensity, (height, width), interpolation="cubic")

        return intensity

    @classmethod
    def _generate_intensity_map_fine(cls, height, width, intensity_mean,
                                     exponent, random_state):
        intensity_details_generator = iap.FrequencyNoise(
            exponent=exponent,
            size_px_max=max(height, width, 1),  # 1 here for case H, W being 0
            upscale_method="cubic"
        )
        intensity_details = intensity_details_generator.draw_samples(
            (height, width), random_state)
        return intensity_mean * ((2*intensity_details - 1.0)/5.0)

    @classmethod
    def _generate_alpha_mask(cls, height, width, alpha_min, alpha_multiplier,
                             exponent, alpha_size_px_max, sparsity,
                             density_multiplier, random_state):
        alpha_generator = iap.FrequencyNoise(
            exponent=exponent,
            size_px_max=alpha_size_px_max,
            upscale_method="cubic"
        )
        alpha_local = alpha_generator.draw_samples(
            (height, width), random_state)
        alpha = alpha_min + (alpha_multiplier * alpha_local)
        alpha = (alpha ** sparsity) * density_multiplier
        alpha = np.clip(alpha, 0.0, 1.0)

        return alpha


# TODO add vertical gradient alpha to have clouds only at skylevel/groundlevel
# TODO add configurable parameters
class Clouds(meta.SomeOf):
    """
    Add clouds to images.

    This is a wrapper around :class:`~imgaug.augmenters.weather.CloudLayer`.
    It executes 1 to 2 layers per image, leading to varying densities and
    frequency patterns of clouds.

    This augmenter seems to be fairly robust w.r.t. the image size. Tested
    with ``96x128``, ``192x256`` and ``960x1280``.

    **Supported dtypes**:

        * ``uint8``: yes; tested
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

        - (1) Parameters of this augmenter are optimized for the value range
              of ``uint8``. While other dtypes may be accepted, they will lead
              to images augmented in ways inappropriate for the respective
              dtype.

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
    >>> aug = iaa.Clouds()

    Create an augmenter that adds clouds to images.

    """

    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        layers = [
            CloudLayer(
                intensity_mean=(196, 255),
                intensity_freq_exponent=(-2.5, -2.0),
                intensity_coarse_scale=10,
                alpha_min=0,
                alpha_multiplier=(0.25, 0.75),
                alpha_size_px_max=(2, 8),
                alpha_freq_exponent=(-2.5, -2.0),
                sparsity=(0.8, 1.0),
                density_multiplier=(0.5, 1.0),
                seed=seed,
                random_state=random_state,
                deterministic=deterministic
            ),
            CloudLayer(
                intensity_mean=(196, 255),
                intensity_freq_exponent=(-2.0, -1.0),
                intensity_coarse_scale=10,
                alpha_min=0,
                alpha_multiplier=(0.5, 1.0),
                alpha_size_px_max=(64, 128),
                alpha_freq_exponent=(-2.0, -1.0),
                sparsity=(1.0, 1.4),
                density_multiplier=(0.8, 1.5),
                seed=seed,
                random_state=random_state,
                deterministic=deterministic
            )
        ]

        super(Clouds, self).__init__(
            (1, 2),
            children=layers,
            random_order=False,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# TODO add vertical gradient alpha to have fog only at skylevel/groundlevel
# TODO add configurable parameters
class Fog(CloudLayer):
    """Add fog to images.

    This is a wrapper around :class:`~imgaug.augmenters.weather.CloudLayer`.
    It executes a single layer per image with a configuration leading to
    fairly dense clouds with low-frequency patterns.

    This augmenter seems to be fairly robust w.r.t. the image size. Tested
    with ``96x128``, ``192x256`` and ``960x1280``.

    **Supported dtypes**:

        * ``uint8``: yes; tested
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

        - (1) Parameters of this augmenter are optimized for the value range
              of ``uint8``. While other dtypes may be accepted, they will lead
              to images augmented in ways inappropriate for the respective
              dtype.

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
    >>> aug = iaa.Fog()

    Create an augmenter that adds fog to images.

    """

    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Fog, self).__init__(
            intensity_mean=(220, 255),
            intensity_freq_exponent=(-2.0, -1.5),
            intensity_coarse_scale=2,
            alpha_min=(0.7, 0.9),
            alpha_multiplier=0.3,
            alpha_size_px_max=(2, 8),
            alpha_freq_exponent=(-4.0, -2.0),
            sparsity=0.9,
            density_multiplier=(0.4, 0.9),
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# TODO add examples and add these to the overview docs
# TODO snowflakes are all almost 100% white, add some grayish tones and
#      maybe color to them
class SnowflakesLayer(meta.Augmenter):
    """Add a single layer of falling snowflakes to images.

    **Supported dtypes**:

        * ``uint8``: yes; indirectly tested (1)
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

        - (1) indirectly tested via tests for :class:`Snowflakes`

    Parameters
    ----------
    density : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Density of the snowflake layer, as a probability of each pixel in
        low resolution space to be a snowflake.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be in the interval ``[0.01, 0.075]``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    density_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Size uniformity of the snowflakes. Higher values denote more
        similarly sized snowflakes.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be around ``0.5``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    flake_size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Size of the snowflakes. This parameter controls the resolution at
        which snowflakes are sampled. Higher values mean that the resolution
        is closer to the input image's resolution and hence each sampled
        snowflake will be smaller (because of the smaller pixel size).

        Valid values are in the interval ``(0.0, 1.0]``.
        Recommended values:

            * On 96x128 a value of ``(0.1, 0.4)`` worked well.
            * On 192x256 a value of ``(0.2, 0.7)`` worked well.
            * On 960x1280 a value of ``(0.7, 0.95)`` worked well.

        Datatype behaviour:

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    flake_size_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Controls the size uniformity of the snowflakes. Higher values mean
        that the snowflakes are more similarly sized.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be around ``0.5``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    angle : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Angle in degrees of motion blur applied to the snowflakes, where
        ``0.0`` is motion blur that points straight upwards.
        Recommended to be in the interval ``[-30, 30]``.
        See also :func:`~imgaug.augmenters.blur.MotionBlur.__init__`.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    speed : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Perceived falling speed of the snowflakes. This parameter controls the
        motion blur's kernel size. It follows roughly the form
        ``kernel_size = image_size * speed``. Hence, values around ``1.0``
        denote that the motion blur should "stretch" each snowflake over the
        whole image.

        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended values:

            * On 96x128 a value of ``(0.01, 0.05)`` worked well.
            * On 192x256 a value of ``(0.007, 0.03)`` worked well.
            * On 960x1280 a value of ``(0.001, 0.03)`` worked well.

        Datatype behaviour:

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    blur_sigma_fraction : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Standard deviation (as a fraction of the image size) of gaussian blur
        applied to the snowflakes.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be in the interval ``[0.0001, 0.001]``. May still
        require tinkering based on image size.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    blur_sigma_limits : tuple of float, optional
        Controls allowed min and max values of `blur_sigma_fraction`
        after(!) multiplication with the image size. First value is the
        minimum, second value is the maximum. Values outside of that range
        will be clipped to be within that range. This prevents extreme
        values for very small or large images.

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

    """

    def __init__(self, density, density_uniformity, flake_size,
                 flake_size_uniformity, angle, speed, blur_sigma_fraction,
                 blur_sigma_limits=(0.5, 3.75),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(SnowflakesLayer, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.density = density
        self.density_uniformity = iap.handle_continuous_param(
            density_uniformity, "density_uniformity", value_range=(0.0, 1.0))
        self.flake_size = iap.handle_continuous_param(
            flake_size, "flake_size", value_range=(0.0+1e-4, 1.0))
        self.flake_size_uniformity = iap.handle_continuous_param(
            flake_size_uniformity, "flake_size_uniformity",
            value_range=(0.0, 1.0))
        self.angle = iap.handle_continuous_param(angle, "angle")
        self.speed = iap.handle_continuous_param(
            speed, "speed", value_range=(0.0, 1.0))
        self.blur_sigma_fraction = iap.handle_continuous_param(
            blur_sigma_fraction, "blur_sigma_fraction", value_range=(0.0, 1.0))

        # (min, max), same for all images
        self.blur_sigma_limits = blur_sigma_limits

        # (height, width), same for all images
        self.gate_noise_size = (8, 8)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        rss = random_state.duplicate(len(images))
        for i, (image, rs) in enumerate(zip(images, rss)):
            batch.images[i] = self.draw_on_image(image, rs)
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.density,
                self.density_uniformity,
                self.flake_size,
                self.flake_size_uniformity,
                self.angle,
                self.speed,
                self.blur_sigma_fraction,
                self.blur_sigma_limits,
                self.gate_noise_size]

    def draw_on_image(self, image, random_state):
        assert image.ndim == 3, (
            "Expected input image to be three-dimensional, "
            "got %d dimensions." % (image.ndim,))
        assert image.shape[2] in [1, 3], (
            "Expected to get image with a channel axis of size 1 or 3, "
            "got %d (shape: %s)" % (image.shape[2], image.shape))

        rss = random_state.duplicate(2)

        flake_size_sample = self.flake_size.draw_sample(random_state)
        flake_size_uniformity_sample = self.flake_size_uniformity.draw_sample(
            random_state)
        angle_sample = self.angle.draw_sample(random_state)
        speed_sample = self.speed.draw_sample(random_state)
        blur_sigma_fraction_sample = self.blur_sigma_fraction.draw_sample(
            random_state)

        height, width, nb_channels = image.shape
        downscale_factor = np.clip(1.0 - flake_size_sample, 0.001, 1.0)
        height_down = max(1, int(height*downscale_factor))
        width_down = max(1, int(width*downscale_factor))
        noise = self._generate_noise(
            height_down,
            width_down,
            self.density,
            rss[0]
        )

        # gate the sampled noise via noise in range [0.0, 1.0]
        # this leads to less flakes in some areas of the image and more in
        # other areas
        gate_noise = iap.Beta(1.0, 1.0 - self.density_uniformity)
        noise = self._gate(noise, gate_noise, self.gate_noise_size, rss[1])
        noise = ia.imresize_single_image(noise, (height, width),
                                         interpolation="cubic")

        # apply a bit of gaussian blur and then motion blur according to
        # angle and speed
        sigma = max(height, width) * blur_sigma_fraction_sample
        sigma = np.clip(sigma,
                        self.blur_sigma_limits[0], self.blur_sigma_limits[1])
        noise_small_blur = self._blur(noise, sigma)
        noise_small_blur = self._motion_blur(noise_small_blur,
                                             angle=angle_sample,
                                             speed=speed_sample,
                                             random_state=random_state)

        noise_small_blur_rgb = self._postprocess_noise(
            noise_small_blur, flake_size_uniformity_sample, nb_channels)

        return self._blend(image, speed_sample, noise_small_blur_rgb)

    @classmethod
    def _generate_noise(cls, height, width, density, random_state):
        noise = arithmetic.Salt(p=density, random_state=random_state)
        return noise.augment_image(np.zeros((height, width), dtype=np.uint8))

    @classmethod
    def _gate(cls, noise, gate_noise, gate_size, random_state):
        # the beta distribution here has most of its weight around 1.0 and
        # will only rarely sample values around 0.0 the average of the
        # sampled values seems to be at around 0.6-0.75
        gate_noise = gate_noise.draw_samples(gate_size, random_state)
        gate_noise_up = ia.imresize_single_image(gate_noise, noise.shape[0:2],
                                                 interpolation="cubic")
        gate_noise_up = np.clip(gate_noise_up, 0.0, 1.0)
        return np.clip(
            noise.astype(np.float32) * gate_noise_up, 0, 255
        ).astype(np.uint8)

    @classmethod
    def _blur(cls, noise, sigma):
        return blur.blur_gaussian_(noise, sigma=sigma)

    @classmethod
    def _motion_blur(cls, noise, angle, speed, random_state):
        size = max(noise.shape[0:2])
        k = int(speed * size)
        if k <= 1:
            return noise

        # we use max(k, 3) here because MotionBlur errors for anything less
        # than 3
        blurer = blur.MotionBlur(
            k=max(k, 3), angle=angle, direction=1.0, random_state=random_state)
        return blurer.augment_image(noise)

    # Added in 0.4.0.
    @classmethod
    def _postprocess_noise(cls, noise_small_blur,
                           flake_size_uniformity_sample, nb_channels):
        # use contrast adjustment of noise to make the flake size a bit less
        # uniform then readjust the noise values to make them more visible
        # again
        gain = 1.0 + 2*(1 - flake_size_uniformity_sample)
        gain_adj = 1.0 + 5*(1 - flake_size_uniformity_sample)
        noise_small_blur = contrast.GammaContrast(gain).augment_image(
            noise_small_blur)
        noise_small_blur = noise_small_blur.astype(np.float32) * gain_adj
        noise_small_blur_rgb = np.tile(
            noise_small_blur[..., np.newaxis], (1, 1, nb_channels))
        return noise_small_blur_rgb

    # Added in 0.4.0.
    @classmethod
    def _blend(cls, image, speed_sample, noise_small_blur_rgb):
        # blend:
        # sum for a bit of glowy, hardly visible flakes
        # max for the main flakes
        image_f32 = image.astype(np.float32)
        image_f32 = cls._blend_by_sum(
            image_f32, (0.1 + 20*speed_sample) * noise_small_blur_rgb)
        image_f32 = cls._blend_by_max(
            image_f32, (1.0 + 20*speed_sample) * noise_small_blur_rgb)
        return image_f32

    # TODO replace this by a function from module blend.py
    @classmethod
    def _blend_by_sum(cls, image_f32, noise_small_blur_rgb):
        image_f32 = image_f32 + noise_small_blur_rgb
        return np.clip(image_f32, 0, 255).astype(np.uint8)

    # TODO replace this by a function from module blend.py
    @classmethod
    def _blend_by_max(cls, image_f32, noise_small_blur_rgb):
        image_f32 = np.maximum(image_f32, noise_small_blur_rgb)
        return np.clip(image_f32, 0, 255).astype(np.uint8)


class Snowflakes(meta.SomeOf):
    """Add falling snowflakes to images.

    This is a wrapper around
    :class:`~imgaug.augmenters.weather.SnowflakesLayer`. It executes 1 to 3
    layers per image.

    **Supported dtypes**:

        * ``uint8``: yes; tested
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

        - (1) Parameters of this augmenter are optimized for the value range
              of ``uint8``. While other dtypes may be accepted, they will lead
              to images augmented in ways inappropriate for the respective
              dtype.

    Parameters
    ----------
    density : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Density of the snowflake layer, as a probability of each pixel in
        low resolution space to be a snowflake.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be in the interval ``[0.01, 0.075]``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    density_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Size uniformity of the snowflakes. Higher values denote more
        similarly sized snowflakes.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be around ``0.5``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    flake_size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Size of the snowflakes. This parameter controls the resolution at
        which snowflakes are sampled. Higher values mean that the resolution
        is closer to the input image's resolution and hence each sampled
        snowflake will be smaller (because of the smaller pixel size).

        Valid values are in the interval ``(0.0, 1.0]``.
        Recommended values:

            * On ``96x128`` a value of ``(0.1, 0.4)`` worked well.
            * On ``192x256`` a value of ``(0.2, 0.7)`` worked well.
            * On ``960x1280`` a value of ``(0.7, 0.95)`` worked well.

        Datatype behaviour:

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    flake_size_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Controls the size uniformity of the snowflakes. Higher values mean
        that the snowflakes are more similarly sized.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be around ``0.5``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    angle : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Angle in degrees of motion blur applied to the snowflakes, where
        ``0.0`` is motion blur that points straight upwards.
        Recommended to be in the interval ``[-30, 30]``.
        See also :func:`~imgaug.augmenters.blur.MotionBlur.__init__`.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    speed : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Perceived falling speed of the snowflakes. This parameter controls the
        motion blur's kernel size. It follows roughly the form
        ``kernel_size = image_size * speed``. Hence, values around ``1.0``
        denote that the motion blur should "stretch" each snowflake over
        the whole image.

        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended values:

            * On ``96x128`` a value of ``(0.01, 0.05)`` worked well.
            * On ``192x256`` a value of ``(0.007, 0.03)`` worked well.
            * On ``960x1280`` a value of ``(0.001, 0.03)`` worked well.

        Datatype behaviour:

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

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
    >>> aug = iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))

    Add snowflakes to small images (around ``96x128``).

    >>> aug = iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03))

    Add snowflakes to medium-sized images (around ``192x256``).

    >>> aug = iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))

    Add snowflakes to large images (around ``960x1280``).

    """

    def __init__(self, density=(0.005, 0.075), density_uniformity=(0.3, 0.9),
                 flake_size=(0.2, 0.7), flake_size_uniformity=(0.4, 0.8),
                 angle=(-30, 30), speed=(0.007, 0.03),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        layer = SnowflakesLayer(
            density=density,
            density_uniformity=density_uniformity,
            flake_size=flake_size,
            flake_size_uniformity=flake_size_uniformity,
            angle=angle,
            speed=speed,
            blur_sigma_fraction=(0.0001, 0.001),
            seed=seed,
            random_state=random_state,
            deterministic=deterministic
        )

        super(Snowflakes, self).__init__(
            (1, 3),
            children=[layer.deepcopy() for _ in range(3)],
            random_order=False,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class RainLayer(SnowflakesLayer):
    """Add a single layer of falling raindrops to images.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; indirectly tested (1)
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

        - (1) indirectly tested via tests for :class:`Rain`

    Parameters
    ----------
    density : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Same as in :class:`~imgaug.augmenters.weather.SnowflakesLayer`.

    density_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Same as in :class:`~imgaug.augmenters.weather.SnowflakesLayer`.

    drop_size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Same as `flake_size` in
        :class:`~imgaug.augmenters.weather.SnowflakesLayer`.

    drop_size_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Same as `flake_size_uniformity` in
        :class:`~imgaug.augmenters.weather.SnowflakesLayer`.

    angle : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Same as in :class:`~imgaug.augmenters.weather.SnowflakesLayer`.

    speed : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Same as in :class:`~imgaug.augmenters.weather.SnowflakesLayer`.

    blur_sigma_fraction : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Same as in :class:`~imgaug.augmenters.weather.SnowflakesLayer`.

    blur_sigma_limits : tuple of float, optional
        Same as in :class:`~imgaug.augmenters.weather.SnowflakesLayer`.

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

    """

    # Added in 0.4.0.
    def __init__(self, density, density_uniformity, drop_size,
                 drop_size_uniformity, angle, speed, blur_sigma_fraction,
                 blur_sigma_limits=(0.5, 3.75),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(RainLayer, self).__init__(
            density, density_uniformity, drop_size,
            drop_size_uniformity, angle, speed, blur_sigma_fraction,
            blur_sigma_limits=blur_sigma_limits,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    # Added in 0.4.0.
    @classmethod
    def _blur(cls, noise, sigma):
        return noise

    # Added in 0.4.0.
    @classmethod
    def _postprocess_noise(cls, noise_small_blur,
                           flake_size_uniformity_sample, nb_channels):
        noise_small_blur_rgb = np.tile(
            noise_small_blur[..., np.newaxis], (1, 1, nb_channels))
        return noise_small_blur_rgb

    # Added in 0.4.0.
    @classmethod
    def _blend(cls, image, speed_sample, noise_small_blur_rgb):
        # We set the mean color based on the noise here. That's a pseudo-random
        # approach that saves us from adding the random state as a parameter.
        # Note that the sum of noise_small_blur_rgb can be 0 when at least one
        # image axis size is 0.
        noise_sum = np.sum(noise_small_blur_rgb.flat[0:1000])
        noise_sum = noise_sum if noise_sum > 0 else 1
        drop_mean_color = 110 + (240 - 110) % noise_sum
        noise_small_blur_rgb = noise_small_blur_rgb / 255.0
        # The 1.3 multiplier increases the visibility of drops a bit.
        noise_small_blur_rgb = np.clip(1.3 * noise_small_blur_rgb, 0, 1.0)
        image_f32 = image.astype(np.float32)
        image_f32 = (
            (1 - noise_small_blur_rgb) * image_f32
            + noise_small_blur_rgb * drop_mean_color
        )
        return np.clip(image_f32, 0, 255).astype(np.uint8)


class Rain(meta.SomeOf):
    """Add falling snowflakes to images.

    This is a wrapper around
    :class:`~imgaug.augmenters.weather.RainLayer`. It executes 1 to 3
    layers per image.

    .. note::

        This augmenter currently seems to work best for medium-sized images
        around ``192x256``. For smaller images, you may want to increase the
        `speed` value to e.g. ``(0.1, 0.3)``, otherwise the drops tend to
        look like snowflakes. For larger images, you may want to increase
        the `drop_size` to e.g. ``(0.10, 0.20)``.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; tested
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

        - (1) Parameters of this augmenter are optimized for the value range
              of ``uint8``. While other dtypes may be accepted, they will lead
              to images augmented in ways inappropriate for the respective
              dtype.

    Parameters
    ----------
    drop_size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        See :class:`~imgaug.augmenters.weather.RainLayer`.

    speed : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        See :class:`~imgaug.augmenters.weather.RainLayer`.

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
    >>> aug = iaa.Rain(speed=(0.1, 0.3))

    Add rain to small images (around ``96x128``).

    >>> aug = iaa.Rain()

    Add rain to medium sized images (around ``192x256``).

    >>> aug = iaa.Rain(drop_size=(0.10, 0.20))

    Add rain to large images (around ``960x1280``).

    """

    # Added in 0.4.0.
    def __init__(self, nb_iterations=(1, 3),
                 drop_size=(0.01, 0.02),
                 speed=(0.04, 0.20),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        layer = RainLayer(
            density=(0.03, 0.14),
            density_uniformity=(0.8, 1.0),
            drop_size=drop_size,
            drop_size_uniformity=(0.2, 0.5),
            angle=(-15, 15),
            speed=speed,
            blur_sigma_fraction=(0.001, 0.001),
            seed=seed,
            random_state=random_state,
            deterministic=deterministic
        )

        super(Rain, self).__init__(
            nb_iterations,
            children=[layer.deepcopy() for _ in range(3)],
            random_order=False,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
