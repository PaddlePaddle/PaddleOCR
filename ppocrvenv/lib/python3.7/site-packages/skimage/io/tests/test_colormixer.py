import numpy as np
import skimage.io._plugins._colormixer as cm

from skimage._shared.testing import (assert_array_equal, assert_almost_equal,
                                     assert_equal, assert_array_almost_equal)


class ColorMixerTest(object):
    def setup(self):
        self.state = np.full((18, 33, 3), 200, dtype=np.uint8)
        self.img = np.zeros_like(self.state)

    def test_basic(self):
        self.op(self.img, self.state, 0, self.positive)
        assert_array_equal(self.img[..., 0],
                           self.py_op(self.state[..., 0], self.positive))

    def test_clip(self):
        self.op(self.img, self.state, 0, self.positive_clip)
        assert_array_equal(self.img[..., 0],
                           np.full_like(self.img[..., 0], 255))

    def test_negative(self):
        self.op(self.img, self.state, 0, self.negative)
        assert_array_equal(self.img[..., 0],
                           self.py_op(self.state[..., 0], self.negative))

    def test_negative_clip(self):
        self.op(self.img, self.state, 0, self.negative_clip)
        assert_array_equal(self.img[..., 0],
                           np.zeros_like(self.img[..., 0]))


class TestColorMixerAdd(ColorMixerTest):
    op = staticmethod(cm.add)
    py_op = np.add
    positive = 50
    positive_clip = 56
    negative = -50
    negative_clip = -220


class TestColorMixerMul(ColorMixerTest):
    op = staticmethod(cm.multiply)
    py_op = np.multiply
    positive = 1.2
    positive_clip = 2
    negative = 0.5
    negative_clip = -0.5


class TestColorMixerBright(object):

    def setup(self):
        self.state = np.ones((18, 33, 3), dtype=np.uint8) * 200
        self.img = np.zeros_like(self.state)

    def test_brightness_pos(self):
        cm.brightness(self.img, self.state, 1.25, 1)
        assert_array_equal(self.img, np.ones_like(self.img) * 251)

    def test_brightness_neg(self):
        cm.brightness(self.img, self.state, 0.5, -50)
        assert_array_equal(self.img, np.ones_like(self.img) * 50)

    def test_brightness_pos_clip(self):
        cm.brightness(self.img, self.state, 2, 0)
        assert_array_equal(self.img, np.ones_like(self.img) * 255)

    def test_brightness_neg_clip(self):
        cm.brightness(self.img, self.state, 0, 0)
        assert_array_equal(self.img, np.zeros_like(self.img))


class TestColorMixer(object):

    def setup(self):
        self.state = np.ones((18, 33, 3), dtype=np.uint8) * 50
        self.img = np.zeros_like(self.state)

    def test_sigmoid(self):
        import math
        alpha = 1.5
        beta = 1.5
        c1 = 1 / (1 + math.exp(beta))
        c2 = 1 / (1 + math.exp(beta - alpha)) - c1
        state = self.state / 255.
        cm.sigmoid_gamma(self.img, self.state, alpha, beta)
        img = 1 / (1 + np.exp(beta - state * alpha))
        img = np.asarray((img - c1) / c2 * 255, dtype='uint8')
        assert_almost_equal(img, self.img)

    def test_gamma(self):
        gamma = 1.5
        cm.gamma(self.img, self.state, gamma)
        img = np.asarray(((self.state / 255.)**(1 / gamma)) * 255,
                         dtype='uint8')
        assert_array_almost_equal(img, self.img)

    def test_rgb_2_hsv(self):
        r = 255
        g = 0
        b = 0
        h, s, v = cm.py_rgb_2_hsv(r, g, b)
        assert_almost_equal(np.array([h]), np.array([0]))
        assert_almost_equal(np.array([s]), np.array([1]))
        assert_almost_equal(np.array([v]), np.array([1]))

    def test_hsv_2_rgb(self):
        h = 0
        s = 1
        v = 1
        r, g, b = cm.py_hsv_2_rgb(h, s, v)
        assert_almost_equal(np.array([r]), np.array([255]))
        assert_almost_equal(np.array([g]), np.array([0]))
        assert_almost_equal(np.array([b]), np.array([0]))

    def test_hsv_add(self):
        cm.hsv_add(self.img, self.state, 360, 0, 0)
        assert_almost_equal(self.img, self.state)

    def test_hsv_add_clip_neg(self):
        cm.hsv_add(self.img, self.state, 0, 0, -1)
        assert_equal(self.img, np.zeros_like(self.state))

    def test_hsv_add_clip_pos(self):
        cm.hsv_add(self.img, self.state, 0, 0, 1)
        assert_equal(self.img, np.ones_like(self.state) * 255)

    def test_hsv_mul(self):
        cm.hsv_multiply(self.img, self.state, 360, 1, 1)
        assert_almost_equal(self.img, self.state)

    def test_hsv_mul_clip_neg(self):
        cm.hsv_multiply(self.img, self.state, 0, 0, 0)
        assert_equal(self.img, np.zeros_like(self.state))
