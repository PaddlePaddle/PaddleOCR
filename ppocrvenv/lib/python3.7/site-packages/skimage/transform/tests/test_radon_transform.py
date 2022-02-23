import itertools

import numpy as np
import pytest

from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import test_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale


PHANTOM = shepp_logan_phantom()[::2, ::2]
PHANTOM = rescale(PHANTOM, 0.5, order=1,
                  mode='constant', anti_aliasing=False, channel_axis=None)


def _debug_plot(original, result, sinogram=None):
    from matplotlib import pyplot as plt
    imkwargs = dict(cmap='gray', interpolation='nearest')
    if sinogram is None:
        plt.figure(figsize=(15, 6))
        sp = 130
    else:
        plt.figure(figsize=(11, 11))
        sp = 221
        plt.subplot(sp + 0)
        plt.imshow(sinogram, aspect='auto', **imkwargs)
    plt.subplot(sp + 1)
    plt.imshow(original, **imkwargs)
    plt.subplot(sp + 2)
    plt.imshow(result, vmin=original.min(), vmax=original.max(), **imkwargs)
    plt.subplot(sp + 3)
    plt.imshow(result - original, **imkwargs)
    plt.colorbar()
    plt.show()


def _rescale_intensity(x):
    x = x.astype(float)
    x -= x.min()
    x /= x.max()
    return x


def test_iradon_bias_circular_phantom():
    """
    test that a uniform circular phantom has a small reconstruction bias
    """
    pixels = 128
    xy = np.arange(-pixels / 2, pixels / 2) + 0.5
    x, y = np.meshgrid(xy, xy)
    image = x**2 + y**2 <= (pixels/4)**2

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)

    reconstruction_fbp = iradon(sinogram, theta=theta)
    error = reconstruction_fbp - image

    tol = 5e-5
    roi_err = np.abs(np.mean(error))
    assert roi_err < tol


def check_radon_center(shape, circle, dtype, preserve_range):
    # Create a test image with only a single non-zero pixel at the origin
    image = np.zeros(shape, dtype=dtype)
    image[(shape[0] // 2, shape[1] // 2)] = 1.
    # Calculate the sinogram
    theta = np.linspace(0., 180., max(shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=circle,
                     preserve_range=preserve_range)
    assert sinogram.dtype == _supported_float_type(sinogram.dtype)
    # The sinogram should be a straight, horizontal line
    sinogram_max = np.argmax(sinogram, axis=0)
    print(sinogram_max)
    assert np.std(sinogram_max) < 1e-6


@pytest.mark.parametrize("shape", [(16, 16), (17, 17)])
@pytest.mark.parametrize("circle", [False, True])
@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.float16, np.uint8, bool]
)
@pytest.mark.parametrize("preserve_range", [False, True])
def test_radon_center(shape, circle, dtype, preserve_range):
    check_radon_center(shape, circle, dtype, preserve_range)


@pytest.mark.parametrize("shape", [(32, 16), (33, 17)])
@pytest.mark.parametrize("circle", [False])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint8, bool])
@pytest.mark.parametrize("preserve_range", [False, True])
def test_radon_center_rectangular(shape, circle, dtype, preserve_range):
    check_radon_center(shape, circle, dtype, preserve_range)


def check_iradon_center(size, theta, circle):
    debug = False
    # Create a test sinogram corresponding to a single projection
    # with a single non-zero pixel at the rotation center
    if circle:
        sinogram = np.zeros((size, 1), dtype=float)
        sinogram[size // 2, 0] = 1.
    else:
        diagonal = int(np.ceil(np.sqrt(2) * size))
        sinogram = np.zeros((diagonal, 1), dtype=float)
        sinogram[sinogram.shape[0] // 2, 0] = 1.
    maxpoint = np.unravel_index(np.argmax(sinogram), sinogram.shape)
    print('shape of generated sinogram', sinogram.shape)
    print('maximum in generated sinogram', maxpoint)
    # Compare reconstructions for theta=angle and theta=angle + 180;
    # these should be exactly equal
    reconstruction = iradon(sinogram, theta=[theta], circle=circle)
    reconstruction_opposite = iradon(sinogram, theta=[theta + 180],
                                     circle=circle)
    print('rms deviance:',
          np.sqrt(np.mean((reconstruction_opposite - reconstruction)**2)))
    if debug and has_mpl:
        import matplotlib.pyplot as plt
        imkwargs = dict(cmap='gray', interpolation='nearest')
        plt.figure()
        plt.subplot(221)
        plt.imshow(sinogram, **imkwargs)
        plt.subplot(222)
        plt.imshow(reconstruction_opposite - reconstruction, **imkwargs)
        plt.subplot(223)
        plt.imshow(reconstruction, **imkwargs)
        plt.subplot(224)
        plt.imshow(reconstruction_opposite, **imkwargs)
        plt.show()

    assert np.allclose(reconstruction, reconstruction_opposite)


sizes_for_test_iradon_center = [16, 17]
thetas_for_test_iradon_center = [0, 90]
circles_for_test_iradon_center = [False, True]


@pytest.mark.parametrize(
    "size, theta, circle",
    itertools.product(sizes_for_test_iradon_center,
                      thetas_for_test_iradon_center,
                      circles_for_test_iradon_center)
)
def test_iradon_center(size, theta, circle):
    check_iradon_center(size, theta, circle)


def check_radon_iradon(interpolation_type, filter_type):
    debug = False
    image = PHANTOM
    reconstructed = iradon(radon(image, circle=False), filter_name=filter_type,
                           interpolation=interpolation_type, circle=False)
    delta = np.mean(np.abs(image - reconstructed))
    print('\n\tmean error:', delta)
    if debug and has_mpl:
        _debug_plot(image, reconstructed)
    if filter_type in ('ramp', 'shepp-logan'):
        if interpolation_type == 'nearest':
            allowed_delta = 0.03
        else:
            allowed_delta = 0.025
    else:
        allowed_delta = 0.05
    assert delta < allowed_delta


filter_types = ["ramp", "shepp-logan", "cosine", "hamming", "hann"]
interpolation_types = ['linear', 'nearest']
radon_iradon_inputs = list(itertools.product(interpolation_types,
                                             filter_types))
# cubic interpolation is slow; only run one test for it
radon_iradon_inputs.append(('cubic', 'shepp-logan'))


@pytest.mark.parametrize(
    "interpolation_type, filter_type", radon_iradon_inputs
)
def test_radon_iradon(interpolation_type, filter_type):
    check_radon_iradon(interpolation_type, filter_type)


def test_iradon_angles():
    """
    Test with different number of projections
    """
    size = 100
    # Synthetic data
    image = np.tri(size) + np.tri(size)[::-1]
    # Large number of projections: a good quality is expected
    nb_angles = 200
    theta = np.linspace(0, 180, nb_angles, endpoint=False)
    radon_image_200 = radon(image, theta=theta, circle=False)
    reconstructed = iradon(radon_image_200, circle=False)
    delta_200 = np.mean(abs(_rescale_intensity(image) -
                            _rescale_intensity(reconstructed)))
    assert delta_200 < 0.03
    # Lower number of projections
    nb_angles = 80
    radon_image_80 = radon(image, theta=theta, circle=False)
    # Test whether the sum of all projections is approximately the same
    s = radon_image_80.sum(axis=0)
    assert np.allclose(s, s[0], rtol=0.01)
    reconstructed = iradon(radon_image_80, circle=False)
    delta_80 = np.mean(abs(image / np.max(image) -
                           reconstructed / np.max(reconstructed)))
    # Loss of quality when the number of projections is reduced
    assert delta_80 > delta_200


def check_radon_iradon_minimal(shape, slices):
    debug = False
    theta = np.arange(180)
    image = np.zeros(shape, dtype=float)
    image[slices] = 1.
    sinogram = radon(image, theta, circle=False)
    reconstructed = iradon(sinogram, theta, circle=False)
    print('\n\tMaximum deviation:', np.max(np.abs(image - reconstructed)))
    if debug and has_mpl:
        _debug_plot(image, reconstructed, sinogram)
    if image.sum() == 1:
        assert (np.unravel_index(np.argmax(reconstructed), image.shape)
                == np.unravel_index(np.argmax(image), image.shape))


shapes = [(3, 3), (4, 4), (5, 5)]


def generate_test_data_for_radon_iradon_minimal(shapes):
    def shape2coordinates(shape):
        c0, c1 = shape[0] // 2, shape[1] // 2
        coordinates = itertools.product((c0 - 1, c0, c0 + 1),
                                        (c1 - 1, c1, c1 + 1))
        return coordinates

    def shape2shapeandcoordinates(shape):
        return itertools.product([shape], shape2coordinates(shape))

    return itertools.chain.from_iterable([shape2shapeandcoordinates(shape)
                                          for shape in shapes])


@pytest.mark.parametrize(
    "shape, coordinate",
    generate_test_data_for_radon_iradon_minimal(shapes)
)
def test_radon_iradon_minimal(shape, coordinate):
    check_radon_iradon_minimal(shape, coordinate)


def test_reconstruct_with_wrong_angles():
    a = np.zeros((3, 3))
    p = radon(a, theta=[0, 1, 2], circle=False)
    iradon(p, theta=[0, 1, 2], circle=False)
    with pytest.raises(ValueError):
        iradon(p, theta=[0, 1, 2, 3])


def _random_circle(shape):
    # Synthetic random data, zero outside reconstruction circle
    np.random.seed(98312871)
    image = np.random.rand(*shape)
    c0, c1 = np.ogrid[0:shape[0], 0:shape[1]]
    r = np.sqrt((c0 - shape[0] // 2)**2 + (c1 - shape[1] // 2)**2)
    radius = min(shape) // 2
    image[r > radius] = 0.
    return image


def test_radon_circle():
    a = np.ones((10, 10))
    with expected_warnings(['reconstruction circle']):
        radon(a, circle=True)

    # Synthetic data, circular symmetry
    shape = (61, 79)
    c0, c1 = np.ogrid[0:shape[0], 0:shape[1]]
    r = np.sqrt((c0 - shape[0] // 2)**2 + (c1 - shape[1] // 2)**2)
    radius = min(shape) // 2
    image = np.clip(radius - r, 0, np.inf)
    image = _rescale_intensity(image)
    angles = np.linspace(0, 180, min(shape), endpoint=False)
    sinogram = radon(image, theta=angles, circle=True)
    assert np.all(sinogram.std(axis=1) < 1e-2)

    # Synthetic data, random
    image = _random_circle(shape)
    sinogram = radon(image, theta=angles, circle=True)
    mass = sinogram.sum(axis=0)
    average_mass = mass.mean()
    relative_error = np.abs(mass - average_mass) / average_mass
    print(relative_error.max(), relative_error.mean())
    assert np.all(relative_error < 3.2e-3)


def check_sinogram_circle_to_square(size):
    from skimage.transform.radon_transform import _sinogram_circle_to_square
    image = _random_circle((size, size))
    theta = np.linspace(0., 180., size, False)
    sinogram_circle = radon(image, theta, circle=True)

    def argmax_shape(a):
        return np.unravel_index(np.argmax(a), a.shape)

    print('\n\targmax of circle:', argmax_shape(sinogram_circle))
    sinogram_square = radon(image, theta, circle=False)
    print('\targmax of square:', argmax_shape(sinogram_square))
    sinogram_circle_to_square = _sinogram_circle_to_square(sinogram_circle)
    print('\targmax of circle to square:',
          argmax_shape(sinogram_circle_to_square))
    error = abs(sinogram_square - sinogram_circle_to_square)
    print(np.mean(error), np.max(error))
    assert (argmax_shape(sinogram_square) ==
            argmax_shape(sinogram_circle_to_square))


@pytest.mark.parametrize("size", (50, 51))
def test_sinogram_circle_to_square(size):
    check_sinogram_circle_to_square(size)


def check_radon_iradon_circle(interpolation, shape, output_size):
    # Forward and inverse radon on synthetic data
    image = _random_circle(shape)
    radius = min(shape) // 2
    sinogram_rectangle = radon(image, circle=False)
    reconstruction_rectangle = iradon(sinogram_rectangle,
                                      output_size=output_size,
                                      interpolation=interpolation,
                                      circle=False)
    sinogram_circle = radon(image, circle=True)
    reconstruction_circle = iradon(sinogram_circle,
                                   output_size=output_size,
                                   interpolation=interpolation,
                                   circle=True)
    # Crop rectangular reconstruction to match circle=True reconstruction
    width = reconstruction_circle.shape[0]
    excess = int(np.ceil((reconstruction_rectangle.shape[0] - width) / 2))
    s = np.s_[excess:width + excess, excess:width + excess]
    reconstruction_rectangle = reconstruction_rectangle[s]
    # Find the reconstruction circle, set reconstruction to zero outside
    c0, c1 = np.ogrid[0:width, 0:width]
    r = np.sqrt((c0 - width // 2)**2 + (c1 - width // 2)**2)
    reconstruction_rectangle[r > radius] = 0.
    print(reconstruction_circle.shape)
    print(reconstruction_rectangle.shape)
    np.allclose(reconstruction_rectangle, reconstruction_circle)


# if adding more shapes to test data, you might want to look at commit d0f2bac3f
shapes_radon_iradon_circle = ((61, 79), )
interpolations = ('nearest', 'linear')
output_sizes = (None,
                min(shapes_radon_iradon_circle[0]),
                max(shapes_radon_iradon_circle[0]),
                97)


@pytest.mark.parametrize(
    "shape, interpolation, output_size",
    itertools.product(shapes_radon_iradon_circle, interpolations, output_sizes)
)
def test_radon_iradon_circle(shape, interpolation, output_size):
    check_radon_iradon_circle(interpolation, shape, output_size)


def test_order_angles_golden_ratio():
    from skimage.transform.radon_transform import order_angles_golden_ratio
    np.random.seed(1231)
    lengths = [1, 4, 10, 180]
    for l in lengths:
        theta_ordered = np.linspace(0, 180, l, endpoint=False)
        theta_random = np.random.uniform(0, 180, l)
        for theta in (theta_random, theta_ordered):
            indices = [x for x in order_angles_golden_ratio(theta)]
            # no duplicate indices allowed
            assert len(indices) == len(set(indices))


@test_parallel()
def test_iradon_sart():
    debug = False

    image = rescale(PHANTOM, 0.8, mode='reflect',
                    channel_axis=None, anti_aliasing=False)
    theta_ordered = np.linspace(0., 180., image.shape[0], endpoint=False)
    theta_missing_wedge = np.linspace(0., 150., image.shape[0], endpoint=True)
    for theta, error_factor in ((theta_ordered, 1.),
                                (theta_missing_wedge, 2.)):
        sinogram = radon(image, theta, circle=True)
        reconstructed = iradon_sart(sinogram, theta)

        if debug and has_mpl:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.subplot(221)
            plt.imshow(image, interpolation='nearest')
            plt.subplot(222)
            plt.imshow(sinogram, interpolation='nearest')
            plt.subplot(223)
            plt.imshow(reconstructed, interpolation='nearest')
            plt.subplot(224)
            plt.imshow(reconstructed - image, interpolation='nearest')
            plt.show()

        delta = np.mean(np.abs(reconstructed - image))
        print('delta (1 iteration) =', delta)
        assert delta < 0.02 * error_factor
        reconstructed = iradon_sart(sinogram, theta, reconstructed)
        delta = np.mean(np.abs(reconstructed - image))
        print('delta (2 iterations) =', delta)
        assert delta < 0.014 * error_factor
        reconstructed = iradon_sart(sinogram, theta, clip=(0, 1))
        delta = np.mean(np.abs(reconstructed - image))
        print('delta (1 iteration, clip) =', delta)
        assert delta < 0.018 * error_factor

        np.random.seed(1239867)
        shifts = np.random.uniform(-3, 3, sinogram.shape[1])
        x = np.arange(sinogram.shape[0])
        sinogram_shifted = np.vstack([np.interp(x + shifts[i], x,
                                                sinogram[:, i])
                                      for i in range(sinogram.shape[1])]).T
        reconstructed = iradon_sart(sinogram_shifted, theta,
                                    projection_shifts=shifts)
        if debug and has_mpl:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.subplot(221)
            plt.imshow(image, interpolation='nearest')
            plt.subplot(222)
            plt.imshow(sinogram_shifted, interpolation='nearest')
            plt.subplot(223)
            plt.imshow(reconstructed, interpolation='nearest')
            plt.subplot(224)
            plt.imshow(reconstructed - image, interpolation='nearest')
            plt.show()

        delta = np.mean(np.abs(reconstructed - image))
        print('delta (1 iteration, shifted sinogram) =', delta)
        assert delta < 0.022 * error_factor


@pytest.mark.parametrize("preserve_range", [True, False])
def test_iradon_dtype(preserve_range):
    sinogram = np.zeros((16, 1), dtype=int)
    sinogram[8, 0] = 1.
    sinogram64 = sinogram.astype('float64')
    sinogram32 = sinogram.astype('float32')

    assert iradon(sinogram, theta=[0],
                  preserve_range=preserve_range).dtype == 'float64'
    assert iradon(sinogram64, theta=[0],
                  preserve_range=preserve_range).dtype == sinogram64.dtype
    assert iradon(sinogram32, theta=[0],
                  preserve_range=preserve_range).dtype == sinogram32.dtype


def test_radon_dtype():
    img = convert_to_float(PHANTOM, False)
    img32 = img.astype(np.float32)

    assert radon(img).dtype == img.dtype
    assert radon(img32).dtype == img32.dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_iradon_sart_dtype(dtype):
    sinogram = np.zeros((16, 1), dtype=int)
    sinogram[8, 0] = 1.
    sinogram64 = sinogram.astype('float64')
    sinogram32 = sinogram.astype('float32')

    with expected_warnings(['Input data is cast to float']):
        assert iradon_sart(sinogram, theta=[0]).dtype == 'float64'

    assert iradon_sart(sinogram64, theta=[0]).dtype == sinogram64.dtype
    assert iradon_sart(sinogram32, theta=[0]).dtype == sinogram32.dtype

    assert iradon_sart(sinogram, theta=[0], dtype=dtype).dtype == dtype
    assert iradon_sart(sinogram32, theta=[0], dtype=dtype).dtype == dtype
    assert iradon_sart(sinogram64, theta=[0], dtype=dtype).dtype == dtype


def test_iradon_sart_wrong_dtype():
    sinogram = np.zeros((16, 1))

    with pytest.raises(ValueError):
        iradon_sart(sinogram, dtype=int)
