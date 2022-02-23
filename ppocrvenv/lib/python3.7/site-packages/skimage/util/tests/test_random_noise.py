from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose

import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float


def test_set_seed():
    seed = 42
    cam = camera()
    test = random_noise(cam, seed=seed)
    assert_array_equal(test, random_noise(cam, seed=seed))


def test_salt():
    seed = 42
    cam = img_as_float(camera())
    amount = 0.15
    cam_noisy = random_noise(cam, seed=seed, mode='salt', amount=amount)
    saltmask = cam != cam_noisy

    # Ensure all changes are to 1.0
    assert_allclose(cam_noisy[saltmask], np.ones(saltmask.sum()))

    # Ensure approximately correct amount of noise was added
    proportion = float(saltmask.sum()) / (cam.shape[0] * cam.shape[1])
    tolerance = 1e-2
    assert abs(amount - proportion) <= tolerance


def test_salt_p1():
    image = np.random.rand(2, 3)
    noisy = random_noise(image, mode='salt', amount=1)
    assert_array_equal(noisy, [[1, 1, 1], [1, 1, 1]])


def test_singleton_dim():
    """Ensure images where size of a given dimension is 1 work correctly."""
    image = np.random.rand(1, 1000)
    noisy = random_noise(image, mode='salt', amount=0.1, seed=42)
    tolerance = 5e-2
    assert abs(np.average(noisy == 1) - 0.1) <= tolerance


def test_pepper():
    seed = 42
    cam = img_as_float(camera())
    data_signed = cam * 2. - 1.   # Same image, on range [-1, 1]

    amount = 0.15
    cam_noisy = random_noise(cam, seed=seed, mode='pepper', amount=amount)
    peppermask = cam != cam_noisy

    # Ensure all changes are to 1.0
    assert_allclose(cam_noisy[peppermask], np.zeros(peppermask.sum()))

    # Ensure approximately correct amount of noise was added
    proportion = float(peppermask.sum()) / (cam.shape[0] * cam.shape[1])
    tolerance = 1e-2
    assert abs(amount - proportion) <= tolerance

    # Check to make sure pepper gets added properly to signed images
    orig_zeros = (data_signed == -1).sum()
    cam_noisy_signed = random_noise(data_signed, seed=seed, mode='pepper',
                                    amount=.15)

    proportion = (float((cam_noisy_signed == -1).sum() - orig_zeros) /
                  (cam.shape[0] * cam.shape[1]))
    assert abs(amount - proportion) <= tolerance


def test_salt_and_pepper():
    seed = 42
    cam = img_as_float(camera())
    amount = 0.15
    cam_noisy = random_noise(cam, seed=seed, mode='s&p', amount=amount,
                             salt_vs_pepper=0.25)
    saltmask = np.logical_and(cam != cam_noisy, cam_noisy == 1.)
    peppermask = np.logical_and(cam != cam_noisy, cam_noisy == 0.)

    # Ensure all changes are to 0. or 1.
    assert_allclose(cam_noisy[saltmask], np.ones(saltmask.sum()))
    assert_allclose(cam_noisy[peppermask], np.zeros(peppermask.sum()))

    # Ensure approximately correct amount of noise was added
    proportion = float(
        saltmask.sum() + peppermask.sum()) / (cam.shape[0] * cam.shape[1])
    tolerance = 1e-2
    assert abs(amount - proportion) <= tolerance

    # Verify the relative amount of salt vs. pepper is close to expected
    assert 0.18 < saltmask.sum() / peppermask.sum() < 0.35


def test_gaussian():
    seed = 42
    data = np.zeros((128, 128)) + 0.5
    data_gaussian = random_noise(data, seed=seed, var=0.01)
    assert 0.008 < data_gaussian.var() < 0.012

    data_gaussian = random_noise(data, seed=seed, mean=0.3, var=0.015)
    assert 0.28 < data_gaussian.mean() - 0.5 < 0.32
    assert 0.012 < data_gaussian.var() < 0.018


def test_localvar():
    seed = 23703
    data = np.zeros((128, 128)) + 0.5
    local_vars = np.zeros((128, 128)) + 0.001
    local_vars[:64, 64:] = 0.1
    local_vars[64:, :64] = 0.25
    local_vars[64:, 64:] = 0.45

    data_gaussian = random_noise(data, mode='localvar', seed=seed,
                                 local_vars=local_vars, clip=False)
    assert 0. < data_gaussian[:64, :64].var() < 0.002
    assert 0.095 < data_gaussian[:64, 64:].var() < 0.105
    assert 0.245 < data_gaussian[64:, :64].var() < 0.255
    assert 0.445 < data_gaussian[64:, 64:].var() < 0.455

    # Ensure local variance bounds checking works properly
    bad_local_vars = np.zeros_like(data)
    with testing.raises(ValueError):
        random_noise(data, mode='localvar', seed=seed,
                     local_vars=bad_local_vars)
    bad_local_vars += 0.1
    bad_local_vars[0, 0] = -1
    with testing.raises(ValueError):
        random_noise(data, mode='localvar', seed=seed,
                     local_vars=bad_local_vars)


def test_speckle():
    seed = 42
    data = np.zeros((128, 128)) + 0.1
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.1, 0.02 ** 0.5, (128, 128))
    expected = np.clip(data + data * noise, 0, 1)

    data_speckle = random_noise(data, mode='speckle', seed=seed, mean=0.1,
                                var=0.02)
    assert_allclose(expected, data_speckle)


def test_poisson():
    seed = 42
    data = camera()  # 512x512 grayscale uint8
    cam_noisy = random_noise(data, mode='poisson', seed=seed)
    cam_noisy2 = random_noise(data, mode='poisson', seed=seed, clip=False)

    rng = np.random.default_rng(seed)
    expected = rng.poisson(img_as_float(data) * 256) / 256.
    assert_allclose(cam_noisy, np.clip(expected, 0., 1.))
    assert_allclose(cam_noisy2, expected)


def test_clip_poisson():
    seed = 42
    data = camera()  # 512x512 grayscale uint8
    data_signed = img_as_float(data) * 2. - 1.  # Same image, on range [-1, 1]

    # Signed and unsigned, clipped
    cam_poisson = random_noise(data, mode='poisson', seed=seed, clip=True)
    cam_poisson2 = random_noise(data_signed, mode='poisson', seed=seed,
                                clip=True)
    assert (cam_poisson.max() == 1.) and (cam_poisson.min() == 0.)
    assert (cam_poisson2.max() == 1.) and (cam_poisson2.min() == -1.)

    # Signed and unsigned, unclipped
    cam_poisson = random_noise(data, mode='poisson', seed=seed, clip=False)
    cam_poisson2 = random_noise(data_signed, mode='poisson', seed=seed,
                                clip=False)
    assert (cam_poisson.max() > 1.15) and (cam_poisson.min() == 0.)
    assert (cam_poisson2.max() > 1.3) and (cam_poisson2.min() == -1.)


def test_clip_gaussian():
    seed = 42
    data = camera()  # 512x512 grayscale uint8
    data_signed = img_as_float(data) * 2. - 1.  # Same image, on range [-1, 1]

    # Signed and unsigned, clipped
    cam_gauss = random_noise(data, mode='gaussian', seed=seed, clip=True)
    cam_gauss2 = random_noise(data_signed, mode='gaussian', seed=seed,
                              clip=True)
    assert (cam_gauss.max() == 1.) and (cam_gauss.min() == 0.)
    assert (cam_gauss2.max() == 1.) and (cam_gauss2.min() == -1.)

    # Signed and unsigned, unclipped
    cam_gauss = random_noise(data, mode='gaussian', seed=seed, clip=False)
    cam_gauss2 = random_noise(data_signed, mode='gaussian', seed=seed,
                              clip=False)
    assert (cam_gauss.max() > 1.22) and (cam_gauss.min() < -0.35)
    assert (cam_gauss2.max() > 1.219) and (cam_gauss2.min() < -1.219)


def test_clip_speckle():
    seed = 42
    data = camera()  # 512x512 grayscale uint8
    data_signed = img_as_float(data) * 2. - 1.  # Same image, on range [-1, 1]

    # Signed and unsigned, clipped
    cam_speckle = random_noise(data, mode='speckle', seed=seed, clip=True)
    cam_speckle_sig = random_noise(data_signed, mode='speckle', seed=seed,
                                   clip=True)
    assert (cam_speckle.max() == 1.) and (cam_speckle.min() == 0.)
    assert (cam_speckle_sig.max() == 1.) and (cam_speckle_sig.min() == -1.)

    # Signed and unsigned, unclipped
    cam_speckle = random_noise(data, mode='speckle', seed=seed, clip=False)
    cam_speckle_sig = random_noise(data_signed, mode='speckle', seed=seed,
                                clip=False)
    assert (cam_speckle.max() > 1.219) and (cam_speckle.min() == 0.)
    assert (cam_speckle_sig.max() > 1.219) and (cam_speckle_sig.min() < -1.219)


def test_bad_mode():
    data = np.zeros((64, 64))
    with testing.raises(KeyError):
        random_noise(data, 'perlin')
