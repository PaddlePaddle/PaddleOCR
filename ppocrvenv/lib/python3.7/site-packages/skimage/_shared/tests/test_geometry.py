import pytest
from skimage._shared._geometry import polygon_clip, polygon_area

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

pytest.importorskip("matplotlib")


hand = np.array(
    [[ 1.64516129,  1.16145833 ],
     [ 1.64516129,  1.59375    ],
     [ 1.35080645,  1.921875   ],
     [ 1.375     ,  2.18229167 ],
     [ 1.68548387,  1.9375     ],
     [ 1.60887097,  2.55208333 ],
     [ 1.68548387,  2.69791667 ],
     [ 1.76209677,  2.56770833 ],
     [ 1.83064516,  1.97395833 ],
     [ 1.89516129,  2.75       ],
     [ 1.9516129 ,  2.84895833 ],
     [ 2.01209677,  2.76041667 ],
     [ 1.99193548,  1.99479167 ],
     [ 2.11290323,  2.63020833 ],
     [ 2.2016129 ,  2.734375   ],
     [ 2.25403226,  2.60416667 ],
     [ 2.14919355,  1.953125   ],
     [ 2.30645161,  2.36979167 ],
     [ 2.39112903,  2.36979167 ],
     [ 2.41532258,  2.1875     ],
     [ 2.1733871 ,  1.703125   ],
     [ 2.07782258,  1.16666667 ]])


def test_polygon_area():
    x = [0, 0, 1, 1]
    y = [0, 1, 1, 0]

    assert_almost_equal(polygon_area(y, x), 1)

    x = [0, 0, 1]
    y = [0, 1, 1]

    assert_almost_equal(polygon_area(y, x), 0.5)

    x = [0, 0, 0.5, 1, 1, 0.5]
    y = [0, 1, 0.5, 1, 0, 0.5]

    assert_almost_equal(polygon_area(y, x), 0.5)


def test_poly_clip():
    x = [0,  1, 2, 1]
    y = [0, -1, 0, 1]

    yc, xc = polygon_clip(y, x, 0, 0, 1, 1)
    assert_equal(polygon_area(yc, xc), 0.5)

    x = [-1, 1.5, 1.5, -1]
    y = [.5, 0.5, 1.5, 1.5]
    yc, xc = polygon_clip(y, x, 0, 0, 1, 1)
    assert_equal(polygon_area(yc, xc), 0.5)


def test_hand_clip():
    (r0, c0, r1, c1) = (1.0, 1.5, 2.1, 2.5)
    clip_r, clip_c = polygon_clip(hand[:, 1], hand[:, 0], r0, c0, r1, c1)
    assert_equal(clip_r.size, 19)
    assert_equal(clip_r[0], clip_r[-1])
    assert_equal(clip_c[0], clip_c[-1])

    (r0, c0, r1, c1) = (1.0, 1.5, 1.7, 2.5)
    clip_r, clip_c = polygon_clip(hand[:, 1], hand[:, 0], r0, c0, r1, c1)
    assert_equal(clip_r.size, 6)

    (r0, c0, r1, c1) = (1.0, 1.5, 1.5, 2.5)
    clip_r, clip_c = polygon_clip(hand[:, 1], hand[:, 0], r0, c0, r1, c1)
    assert_equal(clip_r.size, 5)
