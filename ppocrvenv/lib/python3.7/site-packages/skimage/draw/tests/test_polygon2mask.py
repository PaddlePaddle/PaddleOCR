import numpy as np

from skimage import draw


image_shape = (512, 512)
polygon = np.array([[80, 111, 146, 234, 407, 300, 187, 45],
                    [465, 438, 499, 380, 450, 287, 210, 167]]).T


def test_polygon2mask():
    mask = draw.polygon2mask(image_shape, polygon)
    assert mask.shape == image_shape
    assert mask.sum() == 57653
