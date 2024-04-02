import PIL.Image
import numpy as np


def rgb2hsv(rgb):
    # type: (np.ndarray) -> np.ndarray
    """Convert rgb to hsv.

    Parameters
    ----------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Input rgb image.

    Returns
    -------
    hsv: numpy.ndarray, (H, W, 3), np.uint8
        Output hsv image.

    """
    hsv = PIL.Image.fromarray(rgb, mode="RGB")
    hsv = hsv.convert("HSV")
    hsv = np.array(hsv)
    return hsv


def hsv2rgb(hsv):
    # type: (np.ndarray) -> np.ndarray
    """Convert hsv to rgb.

    Parameters
    ----------
    hsv: numpy.ndarray, (H, W, 3), np.uint8
        Input hsv image.

    Returns
    -------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Output rgb image.

    """
    rgb = PIL.Image.fromarray(hsv, mode="HSV")
    rgb = rgb.convert("RGB")
    rgb = np.array(rgb)
    return rgb


def label_colormap(n_label=256, value=None):
    """Label colormap.

    Parameters
    ----------
    n_label: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        hsv = rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = hsv2rgb(hsv).reshape(-1, 3)
    return cmap
