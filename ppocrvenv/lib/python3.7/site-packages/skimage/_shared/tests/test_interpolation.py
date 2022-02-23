from skimage._shared.interpolation import coord_map_py
from skimage._shared.testing import assert_array_equal

def test_coord_map():
    symmetric = [coord_map_py(4, n, 'S') for n in range(-6, 6)]
    expected_symmetric = [2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2]
    assert_array_equal(symmetric, expected_symmetric)

    wrap = [coord_map_py(4, n, 'W') for n in range(-6, 6)]
    expected_wrap = [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert_array_equal(wrap, expected_wrap)

    edge = [coord_map_py(4, n, 'E') for n in range(-6, 6)]
    expected_edge = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3]
    assert_array_equal(edge, expected_edge)

    reflect = [coord_map_py(4, n, 'R') for n in range(-6, 6)]
    expected_reflect = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1]
    assert_array_equal(reflect, expected_reflect)

    reflect = [coord_map_py(1, n, 'R') for n in range(-6, 6)]
    expected_reflect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert_array_equal(reflect, expected_reflect)

    other = [coord_map_py(4, n, 'undefined') for n in range(-6, 6)]
    expected_other = list(range(-6, 6))
    assert_array_equal(other, expected_other)
