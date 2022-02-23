from copy import copy

import pytest
import networkx as nx
import random
from networkx.utils import (
    arbitrary_element,
    create_py_random_state,
    create_random_state,
    discrete_sequence,
    dict_to_numpy_array,
    dict_to_numpy_array1,
    dict_to_numpy_array2,
    flatten,
    is_string_like,
    iterable,
    groups,
    make_list_of_ints,
    make_str,
    pairwise,
    powerlaw_sequence,
    PythonRandomInterface,
    to_tuple,
)

nested_depth = (
    1,
    2,
    (
        3,
        4,
        (
            (5, 6, (7,), (8, (9, 10), 11), (12, 13, (14, 15)), 16),
            17,
        ),
        18,
        19,
    ),
    20,
)

nested_set = {
    (1, 2, 3, 4),
    (5, 6, 7, 8, 9),
    (10, 11, (12, 13, 14), (15, 16, 17, 18)),
    19,
    20,
}

nested_mixed = [
    1,
    (2, 3, {4, (5, 6), 7}, [8, 9]),
    {10: "foo", 11: "bar", (12, 13): "baz"},
    {(14, 15): "qwe", 16: "asd"},
    (17, (18, "19"), 20),
]


@pytest.mark.parametrize("result", [None, [], ["existing"], ["existing1", "existing2"]])
@pytest.mark.parametrize(
    "nested",
    [nested_depth, nested_mixed, nested_set],
)
def test_flatten(nested, result):
    if result is None:
        val = flatten(nested, result)
        assert len(val) == 20
    else:
        _result = copy(result)  # because pytest passes parameters as is
        nexisting = len(_result)
        val = flatten(nested, _result)
        assert len(val) == len(_result) == 20 + nexisting

    assert issubclass(type(val), tuple)


def test_is_string_like():
    assert is_string_like("aaaa")
    assert not is_string_like(None)
    assert not is_string_like(123)


def test_iterable():
    assert not iterable(None)
    assert not iterable(10)
    assert iterable([1, 2, 3])
    assert iterable((1, 2, 3))
    assert iterable({1: "A", 2: "X"})
    assert iterable("ABC")


def test_graph_iterable():
    K = nx.complete_graph(10)
    assert iterable(K)
    assert iterable(K.nodes())
    assert iterable(K.edges())


def test_make_list_of_ints():
    mylist = [1, 2, 3.0, 42, -2]
    assert make_list_of_ints(mylist) is mylist
    assert make_list_of_ints(mylist) == mylist
    assert type(make_list_of_ints(mylist)[2]) is int
    pytest.raises(nx.NetworkXError, make_list_of_ints, [1, 2, 3, "kermit"])
    pytest.raises(nx.NetworkXError, make_list_of_ints, [1, 2, 3.1])


def test_random_number_distribution():
    # smoke test only
    z = powerlaw_sequence(20, exponent=2.5)
    z = discrete_sequence(20, distribution=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3])


def test_make_str_with_bytes():
    x = "qualité"
    y = make_str(x)
    assert isinstance(y, str)
    assert len(y) == 7


def test_make_str_with_unicode():
    x = "qualité"
    y = make_str(x)
    assert isinstance(y, str)
    assert len(y) == 7


class TestNumpyArray:
    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip("numpy")

    def test_numpy_to_list_of_ints(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([1.0, 2, 3])
        c = np.array([1.1, 2, 3])
        assert type(make_list_of_ints(a)) == list
        assert make_list_of_ints(b) == list(b)
        B = make_list_of_ints(b)
        assert type(B[0]) == int
        pytest.raises(nx.NetworkXError, make_list_of_ints, c)

    def test_dict_to_numpy_array1(self):
        d = {"a": 1, "b": 2}
        a = dict_to_numpy_array1(d, mapping={"a": 0, "b": 1})
        np.testing.assert_allclose(a, np.array([1, 2]))
        a = dict_to_numpy_array1(d, mapping={"b": 0, "a": 1})
        np.testing.assert_allclose(a, np.array([2, 1]))

        a = dict_to_numpy_array1(d)
        np.testing.assert_allclose(a.sum(), 3)

    def test_dict_to_numpy_array2(self):
        d = {"a": {"a": 1, "b": 2}, "b": {"a": 10, "b": 20}}

        mapping = {"a": 1, "b": 0}
        a = dict_to_numpy_array2(d, mapping=mapping)
        np.testing.assert_allclose(a, np.array([[20, 10], [2, 1]]))

        a = dict_to_numpy_array2(d)
        np.testing.assert_allclose(a.sum(), 33)

    def test_dict_to_numpy_array_a(self):
        d = {"a": {"a": 1, "b": 2}, "b": {"a": 10, "b": 20}}

        mapping = {"a": 0, "b": 1}
        a = dict_to_numpy_array(d, mapping=mapping)
        np.testing.assert_allclose(a, np.array([[1, 2], [10, 20]]))

        mapping = {"a": 1, "b": 0}
        a = dict_to_numpy_array(d, mapping=mapping)
        np.testing.assert_allclose(a, np.array([[20, 10], [2, 1]]))

        a = dict_to_numpy_array2(d)
        np.testing.assert_allclose(a.sum(), 33)

    def test_dict_to_numpy_array_b(self):
        d = {"a": 1, "b": 2}

        mapping = {"a": 0, "b": 1}
        a = dict_to_numpy_array(d, mapping=mapping)
        np.testing.assert_allclose(a, np.array([1, 2]))

        a = dict_to_numpy_array1(d)
        np.testing.assert_allclose(a.sum(), 3)


def test_pairwise():
    nodes = range(4)
    node_pairs = [(0, 1), (1, 2), (2, 3)]
    node_pairs_cycle = node_pairs + [(3, 0)]
    assert list(pairwise(nodes)) == node_pairs
    assert list(pairwise(iter(nodes))) == node_pairs
    assert list(pairwise(nodes, cyclic=True)) == node_pairs_cycle
    empty_iter = iter(())
    assert list(pairwise(empty_iter)) == []
    empty_iter = iter(())
    assert list(pairwise(empty_iter, cyclic=True)) == []


def test_groups():
    many_to_one = dict(zip("abcde", [0, 0, 1, 1, 2]))
    actual = groups(many_to_one)
    expected = {0: {"a", "b"}, 1: {"c", "d"}, 2: {"e"}}
    assert actual == expected
    assert {} == groups({})


def test_to_tuple():
    a_list = [1, 2, [1, 3]]
    actual = to_tuple(a_list)
    expected = (1, 2, (1, 3))
    assert actual == expected

    a_tuple = (1, 2)
    actual = to_tuple(a_tuple)
    expected = a_tuple
    assert actual == expected

    a_mix = (1, 2, [1, 3])
    actual = to_tuple(a_mix)
    expected = (1, 2, (1, 3))
    assert actual == expected


def test_create_random_state():
    np = pytest.importorskip("numpy")
    rs = np.random.RandomState

    assert isinstance(create_random_state(1), rs)
    assert isinstance(create_random_state(None), rs)
    assert isinstance(create_random_state(np.random), rs)
    assert isinstance(create_random_state(rs(1)), rs)
    pytest.raises(ValueError, create_random_state, "a")

    assert np.all(rs(1).rand(10) == create_random_state(1).rand(10))


def test_create_py_random_state():
    pyrs = random.Random

    assert isinstance(create_py_random_state(1), pyrs)
    assert isinstance(create_py_random_state(None), pyrs)
    assert isinstance(create_py_random_state(pyrs(1)), pyrs)
    pytest.raises(ValueError, create_py_random_state, "a")

    np = pytest.importorskip("numpy")

    rs = np.random.RandomState
    nprs = PythonRandomInterface
    assert isinstance(create_py_random_state(np.random), nprs)
    assert isinstance(create_py_random_state(rs(1)), nprs)
    # test default rng input
    assert isinstance(PythonRandomInterface(), nprs)


def test_PythonRandomInterface():
    np = pytest.importorskip("numpy")
    rs = np.random.RandomState
    rng = PythonRandomInterface(rs(42))
    rs42 = rs(42)

    # make sure these functions are same as expected outcome
    assert rng.randrange(3, 5) == rs42.randint(3, 5)
    assert np.all(rng.choice([1, 2, 3]) == rs42.choice([1, 2, 3]))
    assert rng.gauss(0, 1) == rs42.normal(0, 1)
    assert rng.expovariate(1.5) == rs42.exponential(1 / 1.5)
    assert np.all(rng.shuffle([1, 2, 3]) == rs42.shuffle([1, 2, 3]))
    assert np.all(
        rng.sample([1, 2, 3], 2) == rs42.choice([1, 2, 3], (2,), replace=False)
    )
    assert rng.randint(3, 5) == rs42.randint(3, 6)
    assert rng.random() == rs42.random_sample()


@pytest.mark.parametrize(
    ("iterable_type", "expected"),
    (
        (list, 1),
        (tuple, 1),
        (str, "["),
        (set, 1),
    ),
)
def test_arbitrary_element(iterable_type, expected):
    iterable = iterable_type([1, 2, 3])
    assert arbitrary_element(iterable) == expected


@pytest.mark.parametrize(
    "iterator",
    (
        (i for i in range(3)),  # generator
        iter([1, 2, 3]),
    ),
)
def test_arbitrary_element_raises(iterator):
    """Value error is raised when input is an iterator."""
    with pytest.raises(ValueError, match="from an iterator"):
        arbitrary_element(iterator)
