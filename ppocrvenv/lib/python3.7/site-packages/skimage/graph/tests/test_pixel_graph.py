import numpy as np
from skimage.graph._graph import pixel_graph, central_pixel

mask = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=bool)
image = np.random.default_rng().random(mask.shape)


def test_small_graph():
    g, n = pixel_graph(mask, connectivity=2)
    assert g.shape == (4, 4)
    assert len(g.data) == 8
    np.testing.assert_allclose(np.unique(g.data), [1, np.sqrt(2)])
    np.testing.assert_array_equal(n, [0, 4, 5, 7])


def test_central_pixel():
    g, n = pixel_graph(mask, connectivity=2)
    px, ds = central_pixel(g, n, shape=mask.shape)
    np.testing.assert_array_equal(px, (1, 1))
    s2 = np.sqrt(2)
    np.testing.assert_allclose(ds, [s2*3 + 2, s2 + 2, s2*2 + 2, s2*2 + 2])

    # test raveled coordinate
    px, _ = central_pixel(g, n)
    assert px == 4

    # test no nodes given
    px, _ = central_pixel(g)
    assert px == 1


def test_edge_function():
    def edge_func(values_src, values_dst, distances):
        return np.abs(values_src - values_dst) + distances

    g, n = pixel_graph(
            image, mask=mask, connectivity=2, edge_function=edge_func
            )
    s2 = np.sqrt(2)
    np.testing.assert_allclose(g[0, 1], np.abs(image[0, 0] - image[1, 1]) + s2)
    np.testing.assert_allclose(g[1, 2], np.abs(image[1, 1] - image[1, 2]) + 1)
    np.testing.assert_array_equal(n, [0, 4, 5, 7])


def test_default_edge_func():
    g, n = pixel_graph(image, spacing=np.array([0.78, 0.78]))
    num_edges = len(g.data) // 2  # each edge appears in both directions
    assert num_edges == 12  # lattice in a (3, 3) grid
    np.testing.assert_almost_equal(
            g[0, 1], 0.78 * np.abs(image[0, 0] - image[0, 1])
            )
    np.testing.assert_array_equal(n, np.arange(image.size))


if __name__ == '__main__':
    test_edge_function()
