"""Tests for color conversion functions.

Authors
-------
- the rgb2hsv test was written by Nicolas Pinto, 2009
- other tests written by Ralf Gommers, 2009

:license: modified BSD
"""

import colorsys
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_equal)

from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.color import (rgb2hsv, hsv2rgb,
                           rgb2xyz, xyz2rgb,
                           rgb2hed, hed2rgb,
                           separate_stains,
                           combine_stains,
                           rgb2rgbcie, rgbcie2rgb,
                           convert_colorspace,
                           rgb2gray, gray2rgb,
                           xyz2lab, lab2xyz,
                           lab2rgb, rgb2lab,
                           xyz2luv, luv2xyz,
                           luv2rgb, rgb2luv,
                           lab2lch, lch2lab,
                           rgb2yuv, yuv2rgb,
                           rgb2yiq, yiq2rgb,
                           rgb2ypbpr, ypbpr2rgb,
                           rgb2ycbcr, ycbcr2rgb,
                           rgb2ydbdr, ydbdr2rgb,
                           rgba2rgb, gray2rgba)
from skimage.util import img_as_float, img_as_ubyte, img_as_float32


class TestColorconv():

    img_rgb = data.colorwheel()
    img_grayscale = data.camera()
    img_rgba = np.array([[[0, 0.5, 1, 0],
                          [0, 0.5, 1, 1],
                          [0, 0.5, 1, 0.5]]]).astype(float)
    img_stains = img_as_float(img_rgb) * 0.3

    colbars = np.array([[1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 0, 1, 0, 1, 0, 1, 0]]).astype(float)
    colbars_array = np.swapaxes(colbars.reshape(3, 4, 2), 0, 2)
    colbars_point75 = colbars * 0.75
    colbars_point75_array = np.swapaxes(colbars_point75.reshape(3, 4, 2), 0, 2)

    xyz_array = np.array([[[0.4124, 0.21260, 0.01930]],    # red
                          [[0, 0, 0]],    # black
                          [[.9505, 1., 1.089]],    # white
                          [[.1805, .0722, .9505]],    # blue
                          [[.07719, .15438, .02573]],    # green
                          ])
    lab_array = np.array([[[53.233, 80.109, 67.220]],    # red
                          [[0., 0., 0.]],    # black
                          [[100.0, 0.005, -0.010]],    # white
                          [[32.303, 79.197, -107.864]],    # blue
                          [[46.229, -51.7, 49.898]],    # green
                          ])

    luv_array = np.array([[[53.233, 175.053, 37.751]],   # red
                          [[0., 0., 0.]],   # black
                          [[100., 0.001, -0.017]],   # white
                          [[32.303, -9.400, -130.358]],   # blue
                          [[46.228, -43.774, 56.589]],   # green
                          ])

    # RGBA to RGB
    @pytest.mark.parametrize("channel_axis", [0, 1, 2, -1, -2, -3])
    def test_rgba2rgb_conversion(self, channel_axis):
        rgba = self.img_rgba

        rgba = np.moveaxis(rgba, source=-1, destination=channel_axis)
        rgb = rgba2rgb(rgba, channel_axis=channel_axis)
        rgb = np.moveaxis(rgb, source=channel_axis, destination=-1)

        expected = np.array([[[1, 1, 1],
                              [0, 0.5, 1],
                              [0.5, 0.75, 1]]]).astype(float)
        assert_equal(rgb.shape, expected.shape)
        assert_almost_equal(rgb, expected)

    def test_rgba2rgb_error_grayscale(self):
        with pytest.raises(ValueError):
            rgba2rgb(self.img_grayscale)

    @pytest.mark.parametrize("channel_axis", [None, 1.5])
    def test_rgba2rgb_error_channel_axis_invalid(self, channel_axis):
        with pytest.raises(TypeError):
            rgba2rgb(self.img_rgba, channel_axis=channel_axis)

    @pytest.mark.parametrize("channel_axis", [-4, 3])
    def test_rgba2rgb_error_channel_axis_out_of_range(self, channel_axis):
        with pytest.raises(np.AxisError):
            rgba2rgb(self.img_rgba, channel_axis=channel_axis)

    def test_rgba2rgb_error_rgb(self):
        with pytest.raises(ValueError):
            rgba2rgb(self.img_rgb)

    def test_rgba2rgb_dtype(self):
        rgba = self.img_rgba.astype('float64')
        rgba32 = img_as_float32(rgba)

        assert rgba2rgb(rgba).dtype == rgba.dtype
        assert rgba2rgb(rgba32).dtype == rgba32.dtype

    # RGB to HSV
    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_rgb2hsv_conversion(self, channel_axis):
        rgb = img_as_float(self.img_rgb)[::16, ::16]

        _rgb = np.moveaxis(rgb, source=-1, destination=channel_axis)
        hsv = rgb2hsv(_rgb, channel_axis=channel_axis)
        hsv = np.moveaxis(hsv, source=channel_axis, destination=-1)
        hsv = hsv.reshape(-1, 3)

        # ground truth from colorsys
        gt = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2])
                       for pt in rgb.reshape(-1, 3)]
                      )
        assert_almost_equal(hsv, gt)

    def test_rgb2hsv_error_grayscale(self):
        with pytest.raises(ValueError):
            rgb2hsv(self.img_grayscale)

    def test_rgb2hsv_dtype(self):
        rgb = img_as_float(self.img_rgb)
        rgb32 = img_as_float32(self.img_rgb)

        assert rgb2hsv(rgb).dtype == rgb.dtype
        assert rgb2hsv(rgb32).dtype == rgb32.dtype

    # HSV to RGB
    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_hsv2rgb_conversion(self, channel_axis):
        rgb = self.img_rgb.astype("float32")[::16, ::16]
        # create HSV image with colorsys
        hsv = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2])
                        for pt in rgb.reshape(-1, 3)]).reshape(rgb.shape)

        hsv = np.moveaxis(hsv, source=-1, destination=channel_axis)
        _rgb = hsv2rgb(hsv, channel_axis=channel_axis)
        _rgb = np.moveaxis(_rgb, source=channel_axis, destination=-1)

        # convert back to RGB and compare with original.
        # relative precision for RGB -> HSV roundtrip is about 1e-6
        assert_almost_equal(rgb, _rgb, decimal=4)

    def test_hsv2rgb_error_grayscale(self):
        with pytest.raises(ValueError):
            hsv2rgb(self.img_grayscale)

    def test_hsv2rgb_dtype(self):
        rgb = self.img_rgb.astype("float32")[::16, ::16]
        # create HSV image with colorsys
        hsv = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2])
                        for pt in rgb.reshape(-1, 3)],
                       dtype='float64').reshape(rgb.shape)
        hsv32 = hsv.astype('float32')

        assert hsv2rgb(hsv).dtype == hsv.dtype
        assert hsv2rgb(hsv32).dtype == hsv32.dtype

    # RGB to XYZ
    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_rgb2xyz_conversion(self, channel_axis):
        gt = np.array([[[0.950456, 1.      , 1.088754],
                        [0.538003, 0.787329, 1.06942 ],
                        [0.592876, 0.28484 , 0.969561],
                        [0.180423, 0.072169, 0.950227]],
                       [[0.770033, 0.927831, 0.138527],
                        [0.35758 , 0.71516 , 0.119193],
                        [0.412453, 0.212671, 0.019334],
                        [0.      , 0.      , 0.      ]]])

        img = np.moveaxis(
            self.colbars_array, source=-1, destination=channel_axis
        )
        out = rgb2xyz(img, channel_axis=channel_axis)
        out = np.moveaxis(out, source=channel_axis, destination=-1)

        assert_almost_equal(out, gt)

    # stop repeating the "raises" checks for all other functions that are
    # implemented with color._convert()
    def test_rgb2xyz_error_grayscale(self):
        with pytest.raises(ValueError):
            rgb2xyz(self.img_grayscale)

    def test_rgb2xyz_dtype(self):
        img = self.colbars_array
        img32 = img.astype('float32')

        assert rgb2xyz(img).dtype == img.dtype
        assert rgb2xyz(img32).dtype == img32.dtype

    # XYZ to RGB
    def test_xyz2rgb_conversion(self):
        assert_almost_equal(xyz2rgb(rgb2xyz(self.colbars_array)),
                            self.colbars_array)

    def test_xyz2rgb_dtype(self):
        img = rgb2xyz(self.colbars_array)
        img32 = img.astype('float32')

        assert xyz2rgb(img).dtype == img.dtype
        assert xyz2rgb(img32).dtype == img32.dtype

    # RGB<->XYZ roundtrip on another image
    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_xyz_rgb_roundtrip(self, channel_axis):
        img_rgb = img_as_float(self.img_rgb)

        img_rgb = np.moveaxis(img_rgb, source=-1, destination=channel_axis)
        round_trip = xyz2rgb(rgb2xyz(img_rgb, channel_axis=channel_axis),
                             channel_axis=channel_axis)

        assert_array_almost_equal(round_trip, img_rgb)

    # HED<->RGB roundtrip with ubyte image
    def test_hed_rgb_roundtrip(self):
        img_in = img_as_ubyte(self.img_stains)
        img_out = rgb2hed(hed2rgb(img_in))
        assert_equal(img_as_ubyte(img_out), img_in)

    # HED<->RGB roundtrip with float image
    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_hed_rgb_float_roundtrip(self, channel_axis):
        img_in = self.img_stains
        img_in = np.moveaxis(img_in, source=-1, destination=channel_axis)
        img_out = rgb2hed(
            hed2rgb(img_in, channel_axis=channel_axis),
            channel_axis=channel_axis
        )
        assert_array_almost_equal(img_out, img_in)

    # BRO<->RGB roundtrip with ubyte image
    def test_bro_rgb_roundtrip(self):
        from skimage.color.colorconv import bro_from_rgb, rgb_from_bro
        img_in = img_as_ubyte(self.img_stains)
        img_out = combine_stains(img_in, rgb_from_bro)
        img_out = separate_stains(img_out, bro_from_rgb)
        assert_equal(img_as_ubyte(img_out), img_in)

    # BRO<->RGB roundtrip with float image
    @pytest.mark.parametrize("channel_axis", [0, 1, -1])
    def test_bro_rgb_roundtrip_float(self, channel_axis):
        from skimage.color.colorconv import bro_from_rgb, rgb_from_bro
        img_in = self.img_stains
        img_in = np.moveaxis(img_in, source=-1, destination=channel_axis)
        img_out = combine_stains(
            img_in, rgb_from_bro, channel_axis=channel_axis
        )
        img_out = separate_stains(
            img_out, bro_from_rgb, channel_axis=channel_axis
        )
        assert_array_almost_equal(img_out, img_in)

    # RGB to RGB CIE
    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_rgb2rgbcie_conversion(self, channel_axis):
        gt = np.array([[[ 0.1488856 ,  0.18288098,  0.19277574],
                        [ 0.01163224,  0.16649536,  0.18948516],
                        [ 0.12259182,  0.03308008,  0.17298223],
                        [-0.01466154,  0.01669446,  0.16969164]],
                       [[ 0.16354714,  0.16618652,  0.0230841 ],
                        [ 0.02629378,  0.1498009 ,  0.01979351],
                        [ 0.13725336,  0.01638562,  0.00329059],
                        [ 0.        ,  0.        ,  0.        ]]])

        img = np.moveaxis(
            self.colbars_array, source=-1, destination=channel_axis
        )
        out = rgb2rgbcie(img, channel_axis=channel_axis)

        out = np.moveaxis(out, source=channel_axis, destination=-1)

        assert_almost_equal(out, gt)

    def test_rgb2rgbcie_dtype(self):
        img = self.colbars_array.astype('float64')
        img32 = img.astype('float32')

        assert rgb2rgbcie(img).dtype == img.dtype
        assert rgb2rgbcie(img32).dtype == img32.dtype

    # RGB CIE to RGB
    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_rgbcie2rgb_conversion(self, channel_axis):
        rgb = np.moveaxis(
            self.colbars_array, source=-1, destination=channel_axis
        )
        round_trip = rgbcie2rgb(rgb2rgbcie(rgb, channel_axis=channel_axis),
                                channel_axis=channel_axis)
        # only roundtrip test, we checked rgb2rgbcie above already
        assert_almost_equal(round_trip, rgb)

    def test_rgbcie2rgb_dtype(self):
        img = rgb2rgbcie(self.colbars_array).astype('float64')
        img32 = img.astype('float32')

        assert rgbcie2rgb(img).dtype == img.dtype
        assert rgbcie2rgb(img32).dtype == img32.dtype

    @pytest.mark.parametrize("channel_axis", [0, -1])
    def test_convert_colorspace(self, channel_axis):
        colspaces = ['HSV', 'RGB CIE', 'XYZ', 'YCbCr', 'YPbPr', 'YDbDr']
        colfuncs_from = [
            hsv2rgb, rgbcie2rgb, xyz2rgb,
            ycbcr2rgb, ypbpr2rgb, ydbdr2rgb
        ]
        colfuncs_to = [
            rgb2hsv, rgb2rgbcie, rgb2xyz,
            rgb2ycbcr, rgb2ypbpr, rgb2ydbdr
        ]

        colbars_array = np.moveaxis(
            self.colbars_array, source=-1, destination=channel_axis
        )

        kw = dict(channel_axis=channel_axis)

        assert_almost_equal(
            convert_colorspace(colbars_array, 'RGB', 'RGB', **kw),
            colbars_array)

        for i, space in enumerate(colspaces):
            gt = colfuncs_from[i](colbars_array, **kw)
            assert_almost_equal(
                convert_colorspace(colbars_array, space, 'RGB', **kw), gt)
            gt = colfuncs_to[i](colbars_array, **kw)
            assert_almost_equal(
                convert_colorspace(colbars_array, 'RGB', space, **kw), gt)

        with pytest.raises(ValueError):
            convert_colorspace(self.colbars_array, 'nokey', 'XYZ')
        with pytest.raises(ValueError):
            convert_colorspace(self.colbars_array, 'RGB', 'nokey')

    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_rgb2gray(self, channel_axis):
        x = np.array([1, 1, 1]).reshape((1, 1, 3)).astype(float)
        x = np.moveaxis(x, source=-1, destination=channel_axis)
        g = rgb2gray(x, channel_axis=channel_axis)
        assert_array_almost_equal(g, 1)

        assert_equal(g.shape, (1, 1))

    def test_rgb2gray_contiguous(self):
        x = np.random.rand(10, 10, 3)
        assert rgb2gray(x).flags["C_CONTIGUOUS"]
        assert rgb2gray(x[:5, :5]).flags["C_CONTIGUOUS"]

    def test_rgb2gray_alpha(self):
        x = np.empty((10, 10, 4))
        with pytest.raises(ValueError):
            rgb2gray(x)

    def test_rgb2gray_on_gray(self):
        with pytest.raises(ValueError):
            rgb2gray(np.empty((5, 5)))

    def test_rgb2gray_dtype(self):
        img = np.random.rand(10, 10, 3).astype('float64')
        img32 = img.astype('float32')

        assert rgb2gray(img).dtype == img.dtype
        assert rgb2gray(img32).dtype == img32.dtype

    # test matrices for xyz2lab and lab2xyz generated using
    # http://www.easyrgb.com/index.php?X=CALC
    # Note: easyrgb website displays xyz*100
    def test_xyz2lab(self):
        assert_array_almost_equal(xyz2lab(self.xyz_array),
                                  self.lab_array, decimal=3)

        # Test the conversion with the rest of the illuminants.
        for I in ["A", "B", "C", "d50", "d55", "d65"]:
            I = I.lower()
            for obs in ["2", "10", "R"]:
                obs = obs.lower()
                fname = f'color/tests/data/lab_array_{I}_{obs}.npy'
                lab_array_I_obs = np.load(fetch(fname))
                assert_array_almost_equal(lab_array_I_obs,
                                          xyz2lab(self.xyz_array, I, obs),
                                          decimal=2)
        for I in ["d75", "e"]:
            fname = f'color/tests/data/lab_array_{I}_2.npy'
            lab_array_I_obs = np.load(fetch(fname))
            assert_array_almost_equal(lab_array_I_obs,
                                      xyz2lab(self.xyz_array, I, "2"),
                                      decimal=2)

    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_xyz2lab_channel_axis(self, channel_axis):
        # test conversion with channels along a specified axis
        xyz = np.moveaxis(self.xyz_array, source=-1, destination=channel_axis)
        lab = xyz2lab(xyz, channel_axis=channel_axis)
        lab = np.moveaxis(lab, source=channel_axis, destination=-1)
        assert_array_almost_equal(lab, self.lab_array, decimal=3)

    def test_xyz2lab_dtype(self):
        img = self.xyz_array.astype('float64')
        img32 = img.astype('float32')

        assert xyz2lab(img).dtype == img.dtype
        assert xyz2lab(img32).dtype == img32.dtype

    def test_lab2xyz(self):
        assert_array_almost_equal(lab2xyz(self.lab_array),
                                  self.xyz_array, decimal=3)

        # Test the conversion with the rest of the illuminants.
        for I in ["A", "B", "C", "d50", "d55", "d65"]:
            I = I.lower()
            for obs in ["2", "10", "R"]:
                obs = obs.lower()
                fname = f'color/tests/data/lab_array_{I}_{obs}.npy'
                lab_array_I_obs = np.load(fetch(fname))
                assert_array_almost_equal(lab2xyz(lab_array_I_obs, I, obs),
                                          self.xyz_array, decimal=3)
        for I in ["d75", "e"]:
            fname = f'color/tests/data/lab_array_{I}_2.npy'
            lab_array_I_obs = np.load(fetch(fname))
            assert_array_almost_equal(lab2xyz(lab_array_I_obs, I, "2"),
                                      self.xyz_array, decimal=3)

        # And we include a call to test the exception handling in the code.
        with pytest.raises(ValueError):
            lab2xyz(lab_array_I_obs, "NaI", "2")   # Not an illuminant

        with pytest.raises(ValueError):
            lab2xyz(lab_array_I_obs, "d50", "42")   # Not a degree

    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_lab2xyz_channel_axis(self, channel_axis):
        # test conversion with channels along a specified axis
        lab = np.moveaxis(self.lab_array, source=-1, destination=channel_axis)
        xyz = lab2xyz(lab, channel_axis=channel_axis)
        xyz = np.moveaxis(xyz, source=channel_axis, destination=-1)
        assert_array_almost_equal(xyz, self.xyz_array, decimal=3)

    def test_lab2xyz_dtype(self):
        img = self.lab_array.astype('float64')
        img32 = img.astype('float32')

        assert lab2xyz(img).dtype == img.dtype
        assert lab2xyz(img32).dtype == img32.dtype

    def test_rgb2lab_brucelindbloom(self):
        """
        Test the RGB->Lab conversion by comparing to the calculator on the
        authoritative Bruce Lindbloom
        [website](http://brucelindbloom.com/index.html?ColorCalculator.html).
        """
        # Obtained with D65 white point, sRGB model and gamma
        gt_for_colbars = np.array([
            [100, 0, 0],
            [97.1393, -21.5537, 94.4780],
            [91.1132, -48.0875, -14.1312],
            [87.7347, -86.1827, 83.1793],
            [60.3242, 98.2343, -60.8249],
            [53.2408, 80.0925, 67.2032],
            [32.2970, 79.1875, -107.8602],
            [0, 0, 0]]).T
        gt_array = np.swapaxes(gt_for_colbars.reshape(3, 4, 2), 0, 2)
        assert_array_almost_equal(
            rgb2lab(self.colbars_array), gt_array, decimal=2
        )

    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_lab_rgb_roundtrip(self, channel_axis):
        img_rgb = img_as_float(self.img_rgb)
        img_rgb = np.moveaxis(img_rgb, source=-1, destination=channel_axis)
        assert_array_almost_equal(
            lab2rgb(
                rgb2lab(img_rgb, channel_axis=channel_axis),
                channel_axis=channel_axis
            ),
            img_rgb,
        )

    def test_rgb2lab_dtype(self):
        img = self.colbars_array.astype('float64')
        img32 = img.astype('float32')

        assert rgb2lab(img).dtype == img.dtype
        assert rgb2lab(img32).dtype == img32.dtype

    def test_lab2rgb_dtype(self):
        img = self.lab_array.astype('float64')
        img32 = img.astype('float32')

        assert lab2rgb(img).dtype == img.dtype
        assert lab2rgb(img32).dtype == img32.dtype

    # test matrices for xyz2luv and luv2xyz generated using
    # http://www.easyrgb.com/index.php?X=CALC
    # Note: easyrgb website displays xyz*100
    def test_xyz2luv(self):
        assert_array_almost_equal(xyz2luv(self.xyz_array),
                                  self.luv_array, decimal=3)

        # Test the conversion with the rest of the illuminants.
        for I in ["A", "B", "C", "d50", "d55", "d65"]:
            I = I.lower()
            for obs in ["2", "10", "R"]:
                obs = obs.lower()
                fname = f'color/tests/data/luv_array_{I}_{obs}.npy'
                luv_array_I_obs = np.load(fetch(fname))
                assert_array_almost_equal(luv_array_I_obs,
                                          xyz2luv(self.xyz_array, I, obs),
                                          decimal=2)
        for I in ["d75", "e"]:
            fname = f'color/tests/data/luv_array_{I}_2.npy'
            luv_array_I_obs = np.load(fetch(fname))
            assert_array_almost_equal(luv_array_I_obs,
                                      xyz2luv(self.xyz_array, I, "2"),
                                      decimal=2)

    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_xyz2luv_channel_axis(self, channel_axis):
        # test conversion with channels along a specified axis
        xyz = np.moveaxis(self.xyz_array, source=-1, destination=channel_axis)
        luv = xyz2luv(xyz, channel_axis=channel_axis)
        luv = np.moveaxis(luv, source=channel_axis, destination=-1)
        assert_array_almost_equal(luv, self.luv_array, decimal=3)

    def test_xyz2luv_dtype(self):
        img = self.xyz_array.astype('float64')
        img32 = img.astype('float32')

        assert xyz2luv(img).dtype == img.dtype
        assert xyz2luv(img32).dtype == img32.dtype

    def test_luv2xyz(self):
        assert_array_almost_equal(luv2xyz(self.luv_array),
                                  self.xyz_array, decimal=3)

        # Test the conversion with the rest of the illuminants.
        for I in ["A", "B", "C", "d50", "d55", "d65"]:
            I = I.lower()
            for obs in ["2", "10", "R"]:
                obs = obs.lower()
                fname = f'color/tests/data/luv_array_{I}_{obs}.npy'
                luv_array_I_obs = np.load(fetch(fname))
                assert_array_almost_equal(luv2xyz(luv_array_I_obs, I, obs),
                                          self.xyz_array, decimal=3)
        for I in ["d75", "e"]:
            fname = f'color/tests/data/luv_array_{I}_2.npy'
            luv_array_I_obs = np.load(fetch(fname))
            assert_array_almost_equal(luv2xyz(luv_array_I_obs, I, "2"),
                                      self.xyz_array, decimal=3)

    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_luv2xyz_channel_axis(self, channel_axis):
        # test conversion with channels along a specified axis
        luv = np.moveaxis(self.luv_array, source=-1, destination=channel_axis)
        xyz = luv2xyz(luv, channel_axis=channel_axis)
        xyz = np.moveaxis(xyz, source=channel_axis, destination=-1)
        assert_array_almost_equal(xyz, self.xyz_array, decimal=3)

    def test_luv2xyz_dtype(self):
        img = self.luv_array.astype('float64')
        img32 = img.astype('float32')

        assert luv2xyz(img).dtype == img.dtype
        assert luv2xyz(img32).dtype == img32.dtype

    def test_rgb2luv_brucelindbloom(self):
        """
        Test the RGB->Lab conversion by comparing to the calculator on the
        authoritative Bruce Lindbloom
        [website](http://brucelindbloom.com/index.html?ColorCalculator.html).
        """
        # Obtained with D65 white point, sRGB model and gamma
        gt_for_colbars = np.array([
            [100, 0, 0],
            [97.1393, 7.7056, 106.7866],
            [91.1132, -70.4773, -15.2042],
            [87.7347, -83.0776, 107.3985],
            [60.3242, 84.0714, -108.6834],
            [53.2408, 175.0151, 37.7564],
            [32.2970, -9.4054, -130.3423],
            [0, 0, 0]]).T
        gt_array = np.swapaxes(gt_for_colbars.reshape(3, 4, 2), 0, 2)
        assert_array_almost_equal(rgb2luv(self.colbars_array),
                                  gt_array, decimal=2)

    def test_rgb2luv_dtype(self):
        img = self.colbars_array.astype('float64')
        img32 = img.astype('float32')

        assert rgb2luv(img).dtype == img.dtype
        assert rgb2luv(img32).dtype == img32.dtype

    def test_luv2rgb_dtype(self):
        img = self.luv_array.astype('float64')
        img32 = img.astype('float32')

        assert luv2rgb(img).dtype == img.dtype
        assert luv2rgb(img32).dtype == img32.dtype

    @pytest.mark.parametrize("channel_axis", [0, 1, -1 -2])
    def test_luv_rgb_roundtrip(self, channel_axis):
        img_rgb = img_as_float(self.img_rgb)
        img_rgb = np.moveaxis(img_rgb, source=-1, destination=channel_axis)
        assert_array_almost_equal(
            luv2rgb(
                rgb2luv(img_rgb, channel_axis=channel_axis),
                channel_axis=channel_axis
            ),
            img_rgb,
        )

    def test_lab_rgb_outlier(self):
        lab_array = np.ones((3, 1, 3))
        lab_array[0] = [50, -12, 85]
        lab_array[1] = [50, 12, -85]
        lab_array[2] = [90, -4, -47]
        rgb_array = np.array([[[0.501, 0.481, 0]],
                              [[0, 0.482, 1.]],
                              [[0.578, 0.914, 1.]],
                              ])
        assert_almost_equal(lab2rgb(lab_array), rgb_array, decimal=3)

    def test_lab_full_gamut(self):
        a, b = np.meshgrid(np.arange(-100, 100), np.arange(-100, 100))
        L = np.ones(a.shape)
        lab = np.dstack((L, a, b))
        for value in [0, 10, 20]:
            lab[:, :, 0] = value
            with expected_warnings(['Color data out of range']):
                lab2xyz(lab)

    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_lab_lch_roundtrip(self, channel_axis):
        rgb = img_as_float(self.img_rgb)
        rgb = np.moveaxis(rgb, source=-1, destination=channel_axis)
        lab = rgb2lab(rgb, channel_axis=channel_axis)
        lab2 = lch2lab(
            lab2lch(lab, channel_axis=channel_axis),
            channel_axis=channel_axis,
        )
        assert_array_almost_equal(lab2, lab)

    def test_rgb_lch_roundtrip(self):
        rgb = img_as_float(self.img_rgb)
        lab = rgb2lab(rgb)
        lch = lab2lch(lab)
        lab2 = lch2lab(lch)
        rgb2 = lab2rgb(lab2)
        assert_array_almost_equal(rgb, rgb2)

    def test_lab_lch_0d(self):
        lab0 = self._get_lab0()
        lch0 = lab2lch(lab0)
        lch2 = lab2lch(lab0[None, None, :])
        assert_array_almost_equal(lch0, lch2[0, 0, :])

    def test_lab_lch_1d(self):
        lab0 = self._get_lab0()
        lch0 = lab2lch(lab0)
        lch1 = lab2lch(lab0[None, :])
        assert_array_almost_equal(lch0, lch1[0, :])

    def test_lab_lch_3d(self):
        lab0 = self._get_lab0()
        lch0 = lab2lch(lab0)
        lch3 = lab2lch(lab0[None, None, None, :])
        assert_array_almost_equal(lch0, lch3[0, 0, 0, :])

    def _get_lab0(self):
        rgb = img_as_float(self.img_rgb[:1, :1, :])
        return rgb2lab(rgb)[0, 0, :]

    def test_yuv(self):
        rgb = np.array([[[1.0, 1.0, 1.0]]])
        assert_array_almost_equal(rgb2yuv(rgb), np.array([[[1, 0, 0]]]))
        assert_array_almost_equal(rgb2yiq(rgb), np.array([[[1, 0, 0]]]))
        assert_array_almost_equal(rgb2ypbpr(rgb), np.array([[[1, 0, 0]]]))
        assert_array_almost_equal(
            rgb2ycbcr(rgb), np.array([[[235, 128, 128]]])
        )
        assert_array_almost_equal(rgb2ydbdr(rgb), np.array([[[1, 0, 0]]]))
        rgb = np.array([[[0.0, 1.0, 0.0]]])
        assert_array_almost_equal(
            rgb2yuv(rgb), np.array([[[0.587, -0.28886916, -0.51496512]]])
        )
        assert_array_almost_equal(
            rgb2yiq(rgb), np.array([[[0.587, -0.27455667, -0.52273617]]])
        )
        assert_array_almost_equal(
            rgb2ypbpr(rgb), np.array([[[0.587, -0.331264, -0.418688]]])
        )
        assert_array_almost_equal(
            rgb2ycbcr(rgb), np.array([[[144.553, 53.797, 34.214]]])
        )
        assert_array_almost_equal(
            rgb2ydbdr(rgb), np.array([[[0.587, -0.883, 1.116]]])
        )

    @pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
    def test_yuv_roundtrip(self, channel_axis):
        img_rgb = img_as_float(self.img_rgb)[::16, ::16]
        img_rgb = np.moveaxis(img_rgb, source=-1, destination=channel_axis)
        assert_array_almost_equal(
            yuv2rgb(rgb2yuv(img_rgb, channel_axis=channel_axis),
                    channel_axis=channel_axis),
            img_rgb)
        assert_array_almost_equal(
            yiq2rgb(rgb2yiq(img_rgb, channel_axis=channel_axis),
                    channel_axis=channel_axis),
            img_rgb)
        assert_array_almost_equal(
            ypbpr2rgb(rgb2ypbpr(img_rgb, channel_axis=channel_axis),
                      channel_axis=channel_axis),
            img_rgb)
        assert_array_almost_equal(
            ycbcr2rgb(rgb2ycbcr(img_rgb, channel_axis=channel_axis),
                      channel_axis=channel_axis),
            img_rgb)
        assert_array_almost_equal(
            ydbdr2rgb(rgb2ydbdr(img_rgb, channel_axis=channel_axis),
                      channel_axis=channel_axis),
            img_rgb)

    def test_rgb2yuv_dtype(self):
        img = self.colbars_array.astype('float64')
        img32 = img.astype('float32')

        assert rgb2yuv(img).dtype == img.dtype
        assert rgb2yuv(img32).dtype == img32.dtype

    def test_yuv2rgb_dtype(self):
        img = rgb2yuv(self.colbars_array).astype('float64')
        img32 = img.astype('float32')

        assert yuv2rgb(img).dtype == img.dtype
        assert yuv2rgb(img32).dtype == img32.dtype

    def test_rgb2yiq_conversion(self):
        rgb = img_as_float(self.img_rgb)[::16, ::16]
        yiq = rgb2yiq(rgb).reshape(-1, 3)
        gt = np.array([colorsys.rgb_to_yiq(pt[0], pt[1], pt[2])
                       for pt in rgb.reshape(-1, 3)]
                      )
        assert_almost_equal(yiq, gt, decimal=2)


def test_gray2rgb():
    x = np.array([0, 0.5, 1])
    w = gray2rgb(x)
    expected_output = np.array([[ 0, 0, 0 ],
                                [ 0.5, 0.5, 0.5, ],
                                [ 1, 1, 1 ]])

    assert_equal(w, expected_output)

    x = x.reshape((3, 1))
    y = gray2rgb(x)

    assert_equal(y.shape, (3, 1, 3))
    assert_equal(y.dtype, x.dtype)
    assert_equal(y[..., 0], x)
    assert_equal(y[0, 0, :], [0, 0, 0])

    x = np.array([[0, 128, 255]], dtype=np.uint8)
    z = gray2rgb(x)

    assert_equal(z.shape, (1, 3, 3))
    assert_equal(z[..., 0], x)
    assert_equal(z[0, 1, :], [128, 128, 128])


def test_gray2rgb_rgb():
    x = np.random.rand(5, 5, 4)
    y = gray2rgb(x)
    assert y.shape == (x.shape + (3,))
    for i in range(3):
        assert_equal(x, y[..., i])


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 4), (5, 4, 5, 4)])
@pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
def test_gray2rgba(shape, channel_axis):
    # nD case
    img = np.random.random(shape)
    rgba = gray2rgba(img, channel_axis=channel_axis)
    assert rgba.ndim == img.ndim + 1

    # Shape check
    new_axis_loc = channel_axis % rgba.ndim
    assert_equal(rgba.shape,
                 shape[:new_axis_loc] + (4, ) + shape[new_axis_loc:])

    # dtype check
    assert rgba.dtype == img.dtype

    # RGB channels check
    for channel in range(3):
        assert_equal(rgba[slice_at_axis(channel, axis=new_axis_loc)], img)

    # Alpha channel check
    assert_equal(rgba[slice_at_axis(3, axis=new_axis_loc)], 1.0)


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 4), (5, 4, 5, 4)])
@pytest.mark.parametrize("channel_axis", [0, 1, -1, -2])
def test_gray2rgb_channel_axis(shape, channel_axis):
    # nD case
    img = np.random.random(shape)
    rgb = gray2rgb(img, channel_axis=channel_axis)
    assert rgb.ndim == img.ndim + 1

    # Shape check
    new_axis_loc = channel_axis % rgb.ndim
    assert_equal(rgb.shape,
                 shape[:new_axis_loc] + (3, ) + shape[new_axis_loc:])

    # dtype check
    assert rgb.dtype == img.dtype


def test_gray2rgba_dtype():
    img_f64 = np.random.random((5, 5))
    img_f32 = img_f64.astype('float32')
    img_u8 = img_as_ubyte(img_f64)
    img_int = img_u8.astype(int)

    for img in [img_f64, img_f32, img_u8, img_int]:
        assert gray2rgba(img).dtype == img.dtype


def test_gray2rgba_alpha():
    img = np.random.random((5, 5))
    img_u8 = img_as_ubyte(img)

    # Default
    alpha = None
    rgba = gray2rgba(img, alpha)

    assert_equal(rgba[..., :3], gray2rgb(img))
    assert_equal(rgba[..., 3], 1.0)

    # Scalar
    alpha = 0.5
    rgba = gray2rgba(img, alpha)

    assert_equal(rgba[..., :3], gray2rgb(img))
    assert_equal(rgba[..., 3], alpha)

    # Array
    alpha = np.random.random((5, 5))
    rgba = gray2rgba(img, alpha)

    assert_equal(rgba[..., :3], gray2rgb(img))
    assert_equal(rgba[..., 3], alpha)

    # Warning about alpha cast
    alpha = 0.5
    with expected_warnings(["alpha cannot be safely cast to image dtype"]):
        rgba = gray2rgba(img_u8, alpha)
        assert_equal(rgba[..., :3], gray2rgb(img_u8))

    # Invalid shape
    alpha = np.random.random((5, 5, 1))
    expected_err_msg = ("alpha.shape must match image.shape")

    with pytest.raises(ValueError) as err:
        rgba = gray2rgba(img, alpha)
    assert expected_err_msg == str(err.value)


@pytest.mark.parametrize("func", [rgb2gray, gray2rgb, gray2rgba])
@pytest.mark.parametrize("shape", ([(3, ), (2, 3), (4, 5, 3), (5, 4, 5, 3),
                                    (4, 5, 4, 5, 3)]))
def test_nD_gray_conversion(func, shape):
    img = np.random.rand(*shape)
    out = func(img)
    common_ndim = min(out.ndim, len(shape))
    assert out.shape[:common_ndim] == shape[:common_ndim]


@pytest.mark.parametrize("func", [rgb2hsv, hsv2rgb,
                                  rgb2xyz, xyz2rgb,
                                  rgb2hed, hed2rgb,
                                  rgb2rgbcie, rgbcie2rgb,
                                  xyz2lab, lab2xyz,
                                  lab2rgb, rgb2lab,
                                  xyz2luv, luv2xyz,
                                  luv2rgb, rgb2luv,
                                  lab2lch, lch2lab,
                                  rgb2yuv, yuv2rgb,
                                  rgb2yiq, yiq2rgb,
                                  rgb2ypbpr, ypbpr2rgb,
                                  rgb2ycbcr, ycbcr2rgb,
                                  rgb2ydbdr, ydbdr2rgb])
@pytest.mark.parametrize("shape", ([(3, ), (2, 3), (4, 5, 3), (5, 4, 5, 3),
                                    (4, 5, 4, 5, 3)]))
def test_nD_color_conversion(func, shape):
    img = np.random.rand(*shape)
    out = func(img)

    assert out.shape == img.shape


@pytest.mark.parametrize("shape", ([(4, ), (2, 4), (4, 5, 4), (5, 4, 5, 4),
                                    (4, 5, 4, 5, 4)]))
def test_rgba2rgb_nD(shape):
    img = np.random.rand(*shape)
    out = rgba2rgb(img)

    expected_shape = shape[:-1] + (3, )

    assert out.shape == expected_shape


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_rgba2rgb_dtypes(dtype):
    rgba = np.array([[[0, 0.5, 1, 0],
                      [0, 0.5, 1, 1],
                      [0, 0.5, 1, 0.5]]]).astype(dtype=dtype)
    rgb = rgba2rgb(rgba)
    float_dtype = _supported_float_type(rgba.dtype)
    assert rgb.dtype == float_dtype
    expected = np.array([[[1, 1, 1],
                          [0, 0.5, 1],
                          [0.5, 0.75, 1]]]).astype(float)
    assert rgb.shape == expected.shape
    assert_almost_equal(rgb, expected)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_lab_lch_roundtrip_dtypes(dtype):
    rgb = img_as_float(data.colorwheel()).astype(dtype=dtype, copy=False)
    lab = rgb2lab(rgb)
    float_dtype = _supported_float_type(dtype)
    assert lab.dtype == float_dtype
    lab2 = lch2lab(lab2lch(lab))
    decimal = 4 if float_dtype == np.float32 else 7
    assert_array_almost_equal(lab2, lab, decimal=decimal)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_rgb2hsv_dtypes(dtype):
    rgb = img_as_float(data.colorwheel())[::16, ::16]
    rgb = rgb.astype(dtype=dtype, copy=False)
    hsv = rgb2hsv(rgb).reshape(-1, 3)
    float_dtype = _supported_float_type(dtype)
    assert hsv.dtype == float_dtype
    # ground truth from colorsys
    gt = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2])
                   for pt in rgb.reshape(-1, 3)]
                  )
    decimal = 3 if float_dtype == np.float32 else 7
    assert_array_almost_equal(hsv, gt, decimal=decimal)
