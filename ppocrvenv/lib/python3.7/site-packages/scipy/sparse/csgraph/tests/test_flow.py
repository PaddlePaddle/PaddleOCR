import numpy as np
from numpy.testing import assert_array_equal
import pytest

from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow


def test_raises_on_dense_input():
    with pytest.raises(TypeError):
        graph = np.array([[0, 1], [0, 0]])
        maximum_flow(graph, 0, 1)


def test_raises_on_csc_input():
    with pytest.raises(TypeError):
        graph = csc_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, 1)


def test_raises_on_floating_point_input():
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1.5], [0, 0]], dtype=np.float64)
        maximum_flow(graph, 0, 1)


def test_raises_when_source_is_sink():
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, 0)


@pytest.mark.parametrize('source', [-1, 2, 3])
def test_raises_when_source_is_out_of_bounds(source):
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, source, 1)


@pytest.mark.parametrize('sink', [-1, 2, 3])
def test_raises_when_sink_is_out_of_bounds(sink):
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, sink)


def test_simple_graph():
    # This graph looks as follows:
    #     (0) --5--> (1)
    graph = csr_matrix([[0, 5], [0, 0]])
    res = maximum_flow(graph, 0, 1)
    assert res.flow_value == 5
    expected_residual = np.array([[0, 5], [-5, 0]])
    assert_array_equal(res.residual.toarray(), expected_residual)


def test_bottle_neck_graph():
    # This graph cannot use the full capacity between 0 and 1:
    #     (0) --5--> (1) --3--> (2)
    graph = csr_matrix([[0, 5, 0], [0, 0, 3], [0, 0, 0]])
    res = maximum_flow(graph, 0, 2)
    assert res.flow_value == 3
    expected_residual = np.array([[0, 3, 0], [-3, 0, 3], [0, -3, 0]])
    assert_array_equal(res.residual.toarray(), expected_residual)


def test_backwards_flow():
    # This example causes backwards flow between vertices 3 and 4,
    # and so this test ensures that we handle that accordingly. See
    #     https://stackoverflow.com/q/38843963/5085211
    # for more information.
    graph = csr_matrix([[0, 10, 0, 0, 10, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 10],
                        [0, 0, 0, 10, 0, 10, 0, 0],
                        [0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 10],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
    res = maximum_flow(graph, 0, 7)
    assert res.flow_value == 20
    expected_residual = np.array([[0, 10, 0, 0, 10, 0, 0, 0],
                                  [-10, 0, 10, 0, 0, 0, 0, 0],
                                  [0, -10, 0, 10, 0, 0, 0, 0],
                                  [0, 0, -10, 0, 0, 0, 0, 10],
                                  [-10, 0, 0, 0, 0, 10, 0, 0],
                                  [0, 0, 0, 0, -10, 0, 10, 0],
                                  [0, 0, 0, 0, 0, -10, 0, 10],
                                  [0, 0, 0, -10, 0, 0, -10, 0]])
    assert_array_equal(res.residual.toarray(), expected_residual)


def test_example_from_clrs_chapter_26_1():
    # See page 659 in CLRS second edition, but note that the maximum flow
    # we find is slightly different than the one in CLRS; we push a flow of
    # 12 to v_1 instead of v_2.
    graph = csr_matrix([[0, 16, 13, 0, 0, 0],
                        [0, 0, 10, 12, 0, 0],
                        [0, 4, 0, 0, 14, 0],
                        [0, 0, 9, 0, 0, 20],
                        [0, 0, 0, 7, 0, 4],
                        [0, 0, 0, 0, 0, 0]])
    res = maximum_flow(graph, 0, 5)
    assert res.flow_value == 23
    expected_residual = np.array([[0, 12, 11, 0, 0, 0],
                                  [-12, 0, 0, 12, 0, 0],
                                  [-11, 0, 0, 0, 11, 0],
                                  [0, -12, 0, 0, -7, 19],
                                  [0, 0, -11, 7, 0, 4],
                                  [0, 0, 0, -19, -4, 0]])
    assert_array_equal(res.residual.toarray(), expected_residual)


def test_disconnected_graph():
    # This tests the following disconnected graph:
    #     (0) --5--> (1)    (2) --3--> (3)
    graph = csr_matrix([[0, 5, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 9, 3],
                        [0, 0, 0, 0]])
    res = maximum_flow(graph, 0, 3)
    assert res.flow_value == 0
    expected_residual = np.zeros((4, 4), dtype=np.int32)
    assert_array_equal(res.residual.toarray(), expected_residual)
