import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
import string


def generate_random_token():
    k = len(string.ascii_letters)
    tokens = list(np.arange(k, dtype=int))
    tokens += list(np.arange(k, dtype=float))
    tokens += list(string.ascii_letters)
    tokens += [None for i in range(k)]
    tokens = np.array(tokens, dtype=object)
    rng = np.random.RandomState(seed=0)

    while 1:
        size = rng.randint(1, 3)
        element = rng.choice(tokens, size)
        if size == 1:
            yield element[0]
        else:
            yield tuple(element)


def get_elements(n):
    # dict is deterministic without difficulty of comparing numpy ints
    elements = {}
    for element in generate_random_token():
        if element not in elements:
            elements[element] = len(elements)
            if len(elements) >= n:
                break
    return list(elements.keys())


def test_init():
    n = 10
    elements = get_elements(n)
    dis = DisjointSet(elements)
    assert dis.n_subsets == n
    assert list(dis) == elements


def test_len():
    n = 10
    elements = get_elements(n)
    dis = DisjointSet(elements)
    assert len(dis) == n

    dis.add("dummy")
    assert len(dis) == n + 1


@pytest.mark.parametrize("n", [10, 100])
def test_contains(n):
    elements = get_elements(n)
    dis = DisjointSet(elements)
    for x in elements:
        assert x in dis

    assert "dummy" not in dis


@pytest.mark.parametrize("n", [10, 100])
def test_add(n):
    elements = get_elements(n)
    dis1 = DisjointSet(elements)

    dis2 = DisjointSet()
    for i, x in enumerate(elements):
        dis2.add(x)
        assert len(dis2) == i + 1

        # test idempotency by adding element again
        dis2.add(x)
        assert len(dis2) == i + 1

    assert list(dis1) == list(dis2)


def test_element_not_present():
    elements = get_elements(n=10)
    dis = DisjointSet(elements)

    with assert_raises(KeyError):
        dis["dummy"]

    with assert_raises(KeyError):
        dis.merge(elements[0], "dummy")

    with assert_raises(KeyError):
        dis.connected(elements[0], "dummy")


@pytest.mark.parametrize("direction", ["forwards", "backwards"])
@pytest.mark.parametrize("n", [10, 100])
def test_linear_union_sequence(n, direction):
    elements = get_elements(n)
    dis = DisjointSet(elements)
    assert elements == list(dis)

    indices = list(range(n - 1))
    if direction == "backwards":
        indices = indices[::-1]

    for it, i in enumerate(indices):
        assert not dis.connected(elements[i], elements[i + 1])
        assert dis.merge(elements[i], elements[i + 1])
        assert dis.connected(elements[i], elements[i + 1])
        assert dis.n_subsets == n - 1 - it

    roots = [dis[i] for i in elements]
    if direction == "forwards":
        assert all(elements[0] == r for r in roots)
    else:
        assert all(elements[-2] == r for r in roots)
    assert not dis.merge(elements[0], elements[-1])


@pytest.mark.parametrize("n", [10, 100])
def test_self_unions(n):
    elements = get_elements(n)
    dis = DisjointSet(elements)

    for x in elements:
        assert dis.connected(x, x)
        assert not dis.merge(x, x)
        assert dis.connected(x, x)
    assert dis.n_subsets == len(elements)

    assert elements == list(dis)
    roots = [dis[x] for x in elements]
    assert elements == roots


@pytest.mark.parametrize("order", ["ab", "ba"])
@pytest.mark.parametrize("n", [10, 100])
def test_equal_size_ordering(n, order):
    elements = get_elements(n)
    dis = DisjointSet(elements)

    rng = np.random.RandomState(seed=0)
    indices = np.arange(n)
    rng.shuffle(indices)

    for i in range(0, len(indices), 2):
        a, b = elements[indices[i]], elements[indices[i + 1]]
        if order == "ab":
            assert dis.merge(a, b)
        else:
            assert dis.merge(b, a)

        expected = elements[min(indices[i], indices[i + 1])]
        assert dis[a] == expected
        assert dis[b] == expected


@pytest.mark.parametrize("kmax", [5, 10])
def test_binary_tree(kmax):
    n = 2**kmax
    elements = get_elements(n)
    dis = DisjointSet(elements)
    rng = np.random.RandomState(seed=0)

    for k in 2**np.arange(kmax):
        for i in range(0, n, 2 * k):
            r1, r2 = rng.randint(0, k, size=2)
            a, b = elements[i + r1], elements[i + k + r2]
            assert not dis.connected(a, b)
            assert dis.merge(a, b)
            assert dis.connected(a, b)

        assert elements == list(dis)
        roots = [dis[i] for i in elements]
        expected_indices = np.arange(n) - np.arange(n) % (2 * k)
        expected = [elements[i] for i in expected_indices]
        assert roots == expected


@pytest.mark.parametrize("n", [10, 100])
def test_subsets(n):
    elements = get_elements(n)
    dis = DisjointSet(elements)

    rng = np.random.RandomState(seed=0)
    for i, j in rng.randint(0, n, (n, 2)):
        x = elements[i]
        y = elements[j]

        expected = {element for element in dis if {dis[element]} == {dis[x]}}
        assert expected == dis.subset(x)

        expected = {dis[element]: set() for element in dis}
        for element in dis:
            expected[dis[element]].add(element)
        expected = list(expected.values())
        assert expected == dis.subsets()

        dis.merge(x, y)
        assert dis.subset(x) == dis.subset(y)
