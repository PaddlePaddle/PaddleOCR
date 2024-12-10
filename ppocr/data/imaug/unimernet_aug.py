# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import math
import numpy as np
from io import BytesIO
import albumentations as A
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import zoom as scizoom


class Erosion(A.ImageOnlyTransform):
    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.erode(img, kernel, iterations=1)
        return img


class Dilation(A.ImageOnlyTransform):
    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.dilate(img, kernel, iterations=1)
        return img


class Bitmap(A.ImageOnlyTransform):

    def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.lower = lower
        self.value = value

    def apply(self, img, **params):
        img = img.copy()
        img[img < self.lower] = self.value
        return img


def clipped_zoom(img, zoom_factor):
    h = img.shape[1]
    ch = int(np.ceil(h / float(zoom_factor)))
    top = (h - ch) // 2
    img = scizoom(
        img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor, 1), order=1
    )
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top : trim_top + h, trim_top : trim_top + h]


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        coords = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        coords = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(coords, coords)
    aliased_disk = np.asarray((x**2 + y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(mapsize=256, wibbledecay=3, rng=None):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100
    if rng is None:
        rng = np.random.default_rng()

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


class Fog(A.ImageOnlyTransform):
    def __init__(self, mag=-1, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.rng = np.random.default_rng()
        self.mag = mag

    def apply(self, img, **params):
        img = Image.fromarray(img.astype(np.uint8))
        w, h = img.size
        c = [(1.5, 2), (2.0, 2), (2.5, 1.7)]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img) / 255.0
        max_val = img.max()
        max_size = 2 ** math.ceil(math.log2(max(w, h)) + 1)
        fog = (
            c[0]
            * plasma_fractal(mapsize=max_size, wibbledecay=c[1], rng=self.rng)[:h, :w][
                ..., np.newaxis
            ]
        )
        if isgray:
            fog = np.squeeze(fog)
        else:
            fog = np.repeat(fog, 3, axis=2)

        img += fog
        img = np.clip(img * max_val / (max_val + c[0]), 0, 1) * 255
        return img.astype(np.uint8)


class Frost(A.ImageOnlyTransform):
    def __init__(self, mag=-1, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.rng = np.random.default_rng()
        self.mag = mag

    def apply(self, img, **params):
        img = Image.fromarray(img.astype(np.uint8))
        w, h = img.size
        c = [(0.78, 0.22), (0.64, 0.36), (0.5, 0.5)]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        file_dir = os.path.dirname(__file__)
        filename = [
            os.path.join(file_dir, "frost_img", "frost1.jpg"),
            os.path.join(file_dir, "frost_img", "frost2.png"),
            os.path.join(file_dir, "frost_img", "frost3.png"),
            os.path.join(file_dir, "frost_img", "frost4.jpg"),
            os.path.join(file_dir, "frost_img", "frost5.jpg"),
            os.path.join(file_dir, "frost_img", "frost6.jpg"),
        ]
        index = self.rng.integers(0, len(filename))
        filename = filename[index]
        frost = Image.open(filename).convert("RGB")

        f_w, f_h = frost.size
        if w / h > f_w / f_h:
            f_h = round(f_h * w / f_w)
            f_w = w
        else:
            f_w = round(f_w * h / f_h)
            f_h = h
        frost = np.asarray(frost.resize((f_w, f_h)))

        # randomly crop
        y_start, x_start = self.rng.integers(0, f_h - h + 1), self.rng.integers(
            0, f_w - w + 1
        )
        frost = frost[y_start : y_start + h, x_start : x_start + w]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img)

        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = np.clip(np.round(c[0] * img + c[1] * frost), 0, 255)
        img = img.astype(np.uint8)
        if isgray:
            img = np.squeeze(img)
        return img


class Snow(A.ImageOnlyTransform):
    def __init__(self, mag=-1, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.rng = np.random.default_rng()
        self.mag = mag

    def apply(self, img, **params):
        from wand.image import Image as WandImage

        img = Image.fromarray(img.astype(np.uint8))
        w, h = img.size
        c = [
            (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        ]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img, dtype=np.float32) / 255.0
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        snow_layer = self.rng.normal(size=img.shape[:2], loc=c[0], scale=c[1])

        snow_layer[snow_layer < c[3]] = 0

        snow_layer = Image.fromarray(
            (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode="L"
        )
        output = BytesIO()
        snow_layer.save(output, format="PNG")
        snow_layer = WandImage(blob=output.getvalue())

        snow_layer.motion_blur(
            radius=c[4], sigma=c[5], angle=self.rng.uniform(-135, -45)
        )

        snow_layer = (
            cv2.imdecode(
                np.frombuffer(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
            )
            / 255.0
        )

        snow_layer = snow_layer[..., np.newaxis]

        img = c[6] * img
        gray_img = (1 - c[6]) * np.maximum(
            img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5
        )
        img += gray_img
        img = np.clip(img + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
        img = img.astype(np.uint8)
        if isgray:
            img = np.squeeze(img)
        return img


class Rain(A.ImageOnlyTransform):
    def __init__(self, mag=-1, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.rng = np.random.default_rng()
        self.mag = mag

    def apply(self, img, **params):
        img = Image.fromarray(img.astype(np.uint8))
        img = img.copy()
        w, h = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1
        line_width = self.rng.integers(1, 2)

        c = [50, 70, 90]
        if self.mag < 0 or self.mag >= len(c):
            index = 0
        else:
            index = self.mag
        c = c[index]

        n_rains = self.rng.integers(c, c + 20)
        slant = self.rng.integers(-60, 60)
        fillcolor = 200 if isgray else (200, 200, 200)

        draw = ImageDraw.Draw(img)
        max_length = min(w, h, 10)
        for i in range(1, n_rains):
            length = self.rng.integers(5, max_length)
            x1 = self.rng.integers(0, w - length)
            y1 = self.rng.integers(0, h - length)
            x2 = x1 + length * math.sin(slant * math.pi / 180.0)
            y2 = y1 + length * math.cos(slant * math.pi / 180.0)
            x2 = int(x2)
            y2 = int(y2)
            draw.line([(x1, y1), (x2, y2)], width=line_width, fill=fillcolor)
        img = np.asarray(img).astype(np.uint8)
        return img


class Shadow(A.ImageOnlyTransform):
    def __init__(self, mag=-1, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.rng = np.random.default_rng()
        self.mag = mag

    def apply(self, img, **params):
        img = Image.fromarray(img.astype(np.uint8))
        w, h = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1

        c = [64, 96, 128]
        if self.mag < 0 or self.mag >= len(c):
            index = 0
        else:
            index = self.mag
        c = c[index]

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        transparency = self.rng.integers(c, c + 32)
        x1 = self.rng.integers(0, w // 2)
        y1 = 0

        x2 = self.rng.integers(w // 2, w)
        y2 = 0

        x3 = self.rng.integers(w // 2, w)
        y3 = h - 1

        x4 = self.rng.integers(0, w // 2)
        y4 = h - 1

        draw.polygon(
            [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=(0, 0, 0, transparency)
        )

        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        if isgray:
            img = ImageOps.grayscale(img)
        img = np.asarray(img).astype(np.uint8)
        return img


class UniMERNetTrainTransform:
    def __init__(self, bitmap_prob=0.04, **kwargs):
        self.bitmap_prob = bitmap_prob
        self.train_transform = A.Compose(
            [
                A.Compose(
                    [
                        Bitmap(p=0.05),
                        A.OneOf([Fog(), Frost(), Snow(), Rain(), Shadow()], p=0.2),
                        A.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.2),
                        A.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=(-0.15, 0),
                            rotate_limit=1,
                            border_mode=0,
                            interpolation=3,
                            value=[255, 255, 255],
                            p=1,
                        ),
                        A.GridDistortion(
                            distort_limit=0.1,
                            border_mode=0,
                            interpolation=3,
                            value=[255, 255, 255],
                            p=0.5,
                        ),
                    ],
                    p=0.15,
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                A.GaussNoise(10, p=0.2),
                A.RandomBrightnessContrast(0.05, (-0.2, 0), True, p=0.2),
                A.ImageCompression(95, p=0.3),
                A.ToGray(always_apply=True),
                A.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
            ]
        )

    def __call__(self, data):
        img = data["image"]
        if np.random.random() < self.bitmap_prob:
            img[img != 255] = 0
        img = self.train_transform(image=img)["image"]
        data["image"] = img
        return data


class UniMERNetTestTransform:
    def __init__(self, **kwargs):
        self.test_transform = A.Compose(
            [
                A.ToGray(always_apply=True),
                A.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
            ]
        )

    def __call__(self, data):
        img = data["image"]
        img = self.test_transform(image=img)["image"]
        data["image"] = img
        return data


class GoTImgDecode:
    def __init__(self, input_size, random_padding=False, **kwargs):
        self.input_size = input_size
        self.random_padding = random_padding

    def crop_margin(self, img):
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)
        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    def get_dimensions(self, img):
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]

    def _compute_resized_output_size(self, image_size, size, max_size=None):
        if len(size) == 1:  # specified size only for the smallest edge
            h, w = image_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = size if isinstance(size, int) else size[0]

            new_short, new_long = requested_new_short, int(
                requested_new_short * long / short
            )

            if max_size is not None:
                if max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge size = {size}"
                    )
                if new_long > max_size:
                    new_short, new_long = int(max_size * new_short / new_long), max_size

            new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]
        return [new_h, new_w]

    def resize(self, img, size):
        _, image_height, image_width = self.get_dimensions(img)
        if isinstance(size, int):
            size = [size]
        max_size = None
        output_size = self._compute_resized_output_size(
            (image_height, image_width), size, max_size
        )
        img = img.resize(tuple(output_size[::-1]), resample=2)
        return img

    def __call__(self, data):
        filename = data["filename"]
        img = Image.open(filename)
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            return
        if img.height == 0 or img.width == 0:
            return
        img = self.resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if self.random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

        data["image"] = np.array(ImageOps.expand(img, padding))
        return data


class UniMERNetImgDecode:
    def __init__(self, input_size, random_padding=False, **kwargs):
        self.input_size = input_size
        self.random_padding = random_padding

    def crop_margin(self, img):
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)
        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    def get_dimensions(self, img):
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]

    def _compute_resized_output_size(self, image_size, size, max_size=None):
        if len(size) == 1:  # specified size only for the smallest edge
            h, w = image_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = size if isinstance(size, int) else size[0]

            new_short, new_long = requested_new_short, int(
                requested_new_short * long / short
            )

            if max_size is not None:
                if max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge size = {size}"
                    )
                if new_long > max_size:
                    new_short, new_long = int(max_size * new_short / new_long), max_size

            new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]
        return [new_h, new_w]

    def resize(self, img, size):
        _, image_height, image_width = self.get_dimensions(img)
        if isinstance(size, int):
            size = [size]
        max_size = None
        output_size = self._compute_resized_output_size(
            (image_height, image_width), size, max_size
        )
        img = img.resize(tuple(output_size[::-1]), resample=2)
        return img

    def __call__(self, data):
        filename = data["filename"]
        img = Image.open(filename)
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            return
        if img.height == 0 or img.width == 0:
            return
        img = self.resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if self.random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

        data["image"] = np.array(ImageOps.expand(img, padding))
        return data


class UniMERNetResize:
    def __init__(self, input_size, random_padding=False, **kwargs):
        self.input_size = input_size
        self.random_padding = random_padding

    def crop_margin(self, img):
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    def get_dimensions(self, img):
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]

    def _compute_resized_output_size(self, image_size, size, max_size=None):
        if len(size) == 1:  # specified size only for the smallest edge
            h, w = image_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = size if isinstance(size, int) else size[0]

            new_short, new_long = requested_new_short, int(
                requested_new_short * long / short
            )

            if max_size is not None:
                if max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge size = {size}"
                    )
                if new_long > max_size:
                    new_short, new_long = int(max_size * new_short / new_long), max_size

            new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]
        return [new_h, new_w]

    def resize(self, img, size):
        _, image_height, image_width = self.get_dimensions(img)
        if isinstance(size, int):
            size = [size]
        max_size = None
        output_size = self._compute_resized_output_size(
            (image_height, image_width), size, max_size
        )
        img.resize(tuple(output_size[::-1]), resample=2)
        return img

    def __call__(self, data):
        img = data["image"]
        img = Image.fromarray(img)
        try:
            img = self.crop_margin(img)
        except OSError:
            return
        if img.height == 0 or img.width == 0:
            return
        img = self.resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if self.random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        data["image"] = np.array(ImageOps.expand(img, padding))
        return data


class UniMERNetImageFormat:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data["image"]
        im_h, im_w = img.shape[:2]
        divide_h = math.ceil(im_h / 32) * 32
        divide_w = math.ceil(im_w / 32) * 32
        img = img[:, :, 0]
        img = np.pad(
            img, ((0, divide_h - im_h), (0, divide_w - im_w)), constant_values=(1, 1)
        )
        img_expanded = img[:, :, np.newaxis].transpose(2, 0, 1)
        data["image"] = img_expanded
        return data
