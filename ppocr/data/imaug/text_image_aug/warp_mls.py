# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/RubanSeven/Text-Image-Augmentation-python/blob/master/warp_mls.py
"""

import cv2
import numpy as np


def _grid_nodes(size, step):
    """Grid coordinates visited by the original loop: 0, step, 2*step, ... and the last pixel."""
    nodes, t = [], 0
    while True:
        if size <= t < size + step - 1:
            t = size - 1
        elif t >= size:
            break
        nodes.append(t)
        t += step
    return np.array(nodes)


def _cells(size, step):
    """Per pixel: start and end node of its grid cell and the position inside it (as in gen_img)."""
    t = np.arange(size)
    start = (t // step) * step
    end = start + step
    last = end >= size
    length = np.where(last, size - start, step)
    end = np.where(last, size - 1, end)
    return start, end, (t - start) / length


class WarpMLS:
    """Moving-least-squares warp of text images (TIA distort / stretch / perspective).

    Vectorised over the grid nodes and the pixels (the original looped over both in
    Python and took most of the data-loading time); results match the loop version
    within a level or two (tests/ppocr/test_warp_mls.py).
    """

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

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        """Displacement (rdx, rdy) at the grid nodes by affine-free similarity MLS."""
        if self.pt_count < 2:
            return
        p = np.asarray(self.dst_pts, dtype=np.float64)
        q = np.asarray(self.src_pts, dtype=np.float64)
        ys, xs = np.meshgrid(
            _grid_nodes(self.dst_h, self.grid_size),
            _grid_nodes(self.dst_w, self.grid_size),
            indexing="ij",
        )
        node = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

        d2 = ((node[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
        same = d2 == 0
        # the original loop stops at the first control point equal to the node: before the
        # last point the node takes that point's source position; equal to the last point only,
        # the node is solved from the other points
        first = np.where(same.any(axis=1), same.argmax(axis=1), self.pt_count)
        snap = first < self.pt_count - 1

        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(same, 0.0, 1.0 / np.where(same, 1.0, d2))
            sw = w.sum(axis=1, keepdims=True)
            pstar = (w @ p) / sw
            qstar = (w @ q) / sw
            pt_i = p[None, :, :] - pstar[:, None, :]
            miu = (w * (pt_i**2).sum(axis=2)).sum(axis=1, keepdims=True)
            cur = node - pstar
            cur_j = np.stack([-cur[:, 1], cur[:, 0]], axis=1)
            pt_j = np.stack([-pt_i[..., 1], pt_i[..., 0]], axis=2)
            a = (pt_i * cur[:, None, :]).sum(axis=2)
            b = (pt_j * cur[:, None, :]).sum(axis=2)
            c = (pt_i * cur_j[:, None, :]).sum(axis=2)
            e = (pt_j * cur_j[:, None, :]).sum(axis=2)
            scale = w / miu
            new_x = (scale * (a * q[None, :, 0] - b * q[None, :, 1])).sum(
                axis=1
            ) + qstar[:, 0]
            new_y = (scale * (-c * q[None, :, 0] + e * q[None, :, 1])).sum(
                axis=1
            ) + qstar[:, 1]

        if snap.any():
            new_x[snap] = q[first[snap], 0]
            new_y[snap] = q[first[snap], 1]
        rows, cols = ys.ravel(), xs.ravel()
        self.rdx[rows, cols] = new_x - cols
        self.rdy[rows, cols] = new_y - rows

    def gen_img(self):
        """Bilinear interpolation of the node displacements to every pixel, then bilinear
        sampling of the source (cv2.remap; positions clipped to the image as before)."""
        src_h, src_w = self.src.shape[:2]
        y0, y1, fy = _cells(self.dst_h, self.grid_size)
        x0, x1, fx = _cells(self.dst_w, self.grid_size)
        fy = fy[:, None]
        nx = (
            np.arange(self.dst_w)[None, :]
            + self._interp(self.rdx, y0, y1, fy, x0, x1, fx) * self.trans_ratio
        )
        ny = (
            np.arange(self.dst_h)[:, None]
            + self._interp(self.rdy, y0, y1, fy, x0, x1, fx) * self.trans_ratio
        )
        map_x = np.clip(nx, 0, src_w - 1).astype(np.float32)
        map_y = np.clip(ny, 0, src_h - 1).astype(np.float32)
        dst = cv2.remap(
            self.src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return np.array(dst, dtype=np.uint8)

    @classmethod
    def _interp(cls, field, y0, y1, fy, x0, x1, fx):
        """field known at the grid nodes -> every pixel: along x on the node rows, then along y."""
        rows = np.unique(np.concatenate([y0, y1]))
        along_x = field[rows][:, x0] * (1 - fx) + field[rows][:, x1] * fx
        index = np.searchsorted(rows, np.arange(rows.max() + 1))
        return along_x[index[y0]] * (1 - fy) + along_x[index[y1]] * fy
