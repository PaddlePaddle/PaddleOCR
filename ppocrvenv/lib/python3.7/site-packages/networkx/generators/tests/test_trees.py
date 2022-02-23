import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal


@pytest.mark.parametrize("prefix_tree_fn", (nx.prefix_tree, nx.prefix_tree_recursive))
def test_basic_prefix_tree(prefix_tree_fn):
    # This example is from the Wikipedia article "Trie"
    # <https://en.wikipedia.org/wiki/Trie>.
    strings = ["a", "to", "tea", "ted", "ten", "i", "in", "inn"]
    T = prefix_tree_fn(strings)
    root, NIL = 0, -1

    def source_label(v):
        return T.nodes[v]["source"]

    # First, we check that the tree has the expected
    # structure. Recall that each node that corresponds to one of
    # the input strings has an edge to the NIL node.
    #
    # Consider the three children at level 1 in the trie.
    a, i, t = sorted(T[root], key=source_label)
    # Check the 'a' branch.
    assert len(T[a]) == 1
    nil = arbitrary_element(T[a])
    assert len(T[nil]) == 0
    # Check the 'i' branch.
    assert len(T[i]) == 2
    nil, in_ = sorted(T[i], key=source_label)
    assert len(T[nil]) == 0
    assert len(T[in_]) == 2
    nil, inn = sorted(T[in_], key=source_label)
    assert len(T[nil]) == 0
    assert len(T[inn]) == 1
    nil = arbitrary_element(T[inn])
    assert len(T[nil]) == 0
    # Check the 't' branch.
    te, to = sorted(T[t], key=source_label)
    assert len(T[to]) == 1
    nil = arbitrary_element(T[to])
    assert len(T[nil]) == 0
    tea, ted, ten = sorted(T[te], key=source_label)
    assert len(T[tea]) == 1
    assert len(T[ted]) == 1
    assert len(T[ten]) == 1
    nil = arbitrary_element(T[tea])
    assert len(T[nil]) == 0
    nil = arbitrary_element(T[ted])
    assert len(T[nil]) == 0
    nil = arbitrary_element(T[ten])
    assert len(T[nil]) == 0

    # Next, we check that the "sources" of each of the nodes is the
    # rightmost letter in the string corresponding to the path to
    # that node.
    assert source_label(root) is None
    assert source_label(a) == "a"
    assert source_label(i) == "i"
    assert source_label(t) == "t"
    assert source_label(in_) == "n"
    assert source_label(inn) == "n"
    assert source_label(to) == "o"
    assert source_label(te) == "e"
    assert source_label(tea) == "a"
    assert source_label(ted) == "d"
    assert source_label(ten) == "n"
    assert source_label(NIL) == "NIL"


@pytest.mark.parametrize(
    "strings",
    (
        ["a", "to", "tea", "ted", "ten", "i", "in", "inn"],
        ["ab", "abs", "ad"],
        ["ab", "abs", "ad", ""],
        ["distant", "disparaging", "distant", "diamond", "ruby"],
    ),
)
def test_implementations_consistent(strings):
    """Ensure results are consistent between prefix_tree implementations."""
    assert graphs_equal(
        nx.prefix_tree(strings),
        nx.prefix_tree_recursive(strings),
    )


def test_random_tree():
    """Tests that a random tree is in fact a tree."""
    T = nx.random_tree(10, seed=1234)
    assert nx.is_tree(T)
