"""The vectorised WarpMLS (TIA distort / stretch / perspective) must match the original loop version."""

import cv2
import numpy as np
import pytest

from ppocr.data.imaug.text_image_aug import augment
from ppocr.data.imaug.text_image_aug.warp_mls import WarpMLS


class LoopWarpMLS:
    """The original implementation (PaddleOCR 3.7, from RubanSeven/Text-Image-Augmentation-python), kept as reference.

    FLOAT is float32 as in the original; Loop64 below is the same loop in float64 -- the exact
    reference for the displacement field (the float32 loop is off by up to ~0.5 px at 4000 px).
    """

    FLOAT = np.float32

    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.0):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def _bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=self.FLOAT)
        if self.pt_count < 2:
            return
        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break
            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break
                sw = 0
                swp = np.zeros(2, dtype=self.FLOAT)
                swq = np.zeros(2, dtype=self.FLOAT)
                new_pt = np.zeros(2, dtype=self.FLOAT)
                cur_pt = np.array([i, j], dtype=self.FLOAT)
                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break
                    w[k] = 1.0 / (
                        (i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0])
                        + (j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1])
                    )
                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])
                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq
                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)
                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])
                        tmp_pt = np.zeros(2, dtype=self.FLOAT)
                        tmp_pt[0] = (
                            np.sum(pt_i * cur_pt) * self.src_pts[k][0]
                            - np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        )
                        tmp_pt[1] = (
                            -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0]
                            + np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        )
                        tmp_pt *= w[k] / miu_s
                        new_pt += tmp_pt
                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]
                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j
                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)
        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1
                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self._bilinear_interp(
                    di / h,
                    dj / w,
                    self.rdx[i, j],
                    self.rdx[i, nj],
                    self.rdx[ni, j],
                    self.rdx[ni, nj],
                )
                delta_y = self._bilinear_interp(
                    di / h,
                    dj / w,
                    self.rdy[i, j],
                    self.rdy[i, nj],
                    self.rdy[ni, j],
                    self.rdy[ni, nj],
                )
                nx = np.clip(j + dj + delta_x * self.trans_ratio, 0, src_w - 1)
                ny = np.clip(i + di + delta_y * self.trans_ratio, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)
                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i : i + h, j : j + w] = self._bilinear_interp(
                    x,
                    y,
                    self.src[nyi, nxi],
                    self.src[nyi, nxi1],
                    self.src[nyi1, nxi],
                    self.src[nyi1, nxi1],
                )
        dst = np.clip(dst, 0, 255)
        return np.array(dst, dtype=np.uint8)


# The loop version keeps weights and partial sums in float32; with coordinates up to ~1600 its
# node displacements are off by up to a few hundredths of a pixel (the float64 vectorised
# version is the accurate one). Hence the tolerance on the field and smooth test images:
# on text-like images such a shift moves a pixel by a level or two at most.
FIELD_ATOL_PX = 0.05


class Loop64(LoopWarpMLS):
    FLOAT = np.float64


def _image(h, w, channels, seed):
    """Blurred noise: text crops change smoothly from pixel to pixel, unlike raw noise."""
    rng = np.random.RandomState(seed)
    shape = (h, w, channels) if channels == 3 else (h, w)
    noise = rng.randint(0, 256, size=shape).astype(np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 1.5)


def _assert_same_image(new, old):
    """Sampling is cv2.remap now: it rounds to the nearest level where the loop version
    truncated (which darkened images by half a level on average) and quantises the sub-pixel
    position to 1/32 px -- about half of the values move by one level, none by more than two.
    """
    assert new.shape == old.shape and new.dtype == old.dtype
    diff = np.abs(new.astype(np.int16) - old.astype(np.int16))
    assert diff.max() <= 2
    assert diff.mean() < 0.75


@pytest.mark.parametrize(
    "h,w",
    [(16, 30), (48, 100), (48, 101), (32, 250), (64, 1600), (150, 333), (44, 4000)],
)
def test_displacement_field_matches_loop_version(h, w):
    rng = np.random.RandomState(h * 7 + w)
    # corner points coincide with grid nodes -- the special case of the original loop
    dst = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]] + rng.randint(
        0, max(w, h), size=(6, 2)
    ).tolist()
    src = (np.array(dst) + rng.randint(-5, 6, size=(len(dst), 2))).tolist()
    img = _image(h, w, 3, 0)
    old, new = LoopWarpMLS(img, src, dst, w, h), WarpMLS(img, src, dst, w, h)
    old.calc_delta()
    new.calc_delta()
    exact = Loop64(img, src, dst, w, h)
    exact.calc_delta()
    np.testing.assert_allclose(new.rdx, exact.rdx, atol=1e-6)
    np.testing.assert_allclose(new.rdy, exact.rdy, atol=1e-6)
    # the float32 loop drifts with the coordinate size; keep its error bounded where it is used
    if w <= 1600:
        np.testing.assert_allclose(new.rdx, old.rdx, atol=FIELD_ATOL_PX)
        np.testing.assert_allclose(new.rdy, old.rdy, atol=FIELD_ATOL_PX)


@pytest.mark.parametrize(
    "func", [augment.tia_distort, augment.tia_stretch, augment.tia_perspective]
)
@pytest.mark.parametrize(
    "h,w,channels",
    [(32, 120, 3), (48, 960, 3), (48, 97, 1), (64, 1600, 3), (20, 40, 3)],
)
def test_tia_output_matches_loop_version(monkeypatch, func, h, w, channels):
    img = _image(h, w, channels, h + w)
    args = (img, 4) if func is not augment.tia_perspective else (img,)
    outputs = []
    for impl in (LoopWarpMLS, WarpMLS):
        monkeypatch.setattr(augment, "WarpMLS", impl)
        np.random.seed(123)
        outputs.append(func(*args))
    _assert_same_image(outputs[1], outputs[0])


def test_fewer_than_two_points_leaves_image_unwarped():
    img = _image(20, 40, 3, 1)
    out = WarpMLS(img, [[3, 3]], [[3, 3]], 40, 20).generate()
    _assert_same_image(out, LoopWarpMLS(img, [[3, 3]], [[3, 3]], 40, 20).generate())
