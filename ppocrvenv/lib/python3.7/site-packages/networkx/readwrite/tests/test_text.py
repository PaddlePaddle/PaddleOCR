import pytest
import networkx as nx
from textwrap import dedent


def test_directed_tree_str():
    # Create a directed forest with labels
    graph = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    for node in graph.nodes:
        graph.nodes[node]["label"] = "node_" + chr(ord("a") + node)

    node_target = dedent(
        """
        ╙── 0
            ├─╼ 1
            │   ├─╼ 3
            │   └─╼ 4
            └─╼ 2
                ├─╼ 5
                └─╼ 6
        """
    ).strip()

    label_target = dedent(
        """
        ╙── node_a
            ├─╼ node_b
            │   ├─╼ node_d
            │   └─╼ node_e
            └─╼ node_c
                ├─╼ node_f
                └─╼ node_g
        """
    ).strip()

    # Basic node case
    ret = nx.forest_str(graph, with_labels=False)
    print(ret)
    assert ret == node_target

    # Basic label case
    ret = nx.forest_str(graph, with_labels=True)
    print(ret)
    assert ret == label_target

    # Custom write function case
    lines = []
    ret = nx.forest_str(graph, write=lines.append, with_labels=False)
    assert ret is None
    assert lines == node_target.split("\n")

    # Smoke test to ensure passing the print function works. To properly test
    # this case we would need to capture stdout. (for potential reference
    # implementation see :class:`ubelt.util_stream.CaptureStdout`)
    ret = nx.forest_str(graph, write=print)
    assert ret is None


def test_empty_graph():
    assert nx.forest_str(nx.DiGraph()) == "╙"
    assert nx.forest_str(nx.Graph()) == "╙"


def test_directed_multi_tree_forest():
    tree1 = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    tree2 = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    forest = nx.disjoint_union_all([tree1, tree2])
    ret = nx.forest_str(forest)
    print(ret)

    target = dedent(
        """
        ╟── 0
        ╎   ├─╼ 1
        ╎   │   ├─╼ 3
        ╎   │   └─╼ 4
        ╎   └─╼ 2
        ╎       ├─╼ 5
        ╎       └─╼ 6
        ╙── 7
            ├─╼ 8
            │   ├─╼ 10
            │   └─╼ 11
            └─╼ 9
                ├─╼ 12
                └─╼ 13
        """
    ).strip()
    assert ret == target

    tree3 = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    forest = nx.disjoint_union_all([tree1, tree2, tree3])
    ret = nx.forest_str(forest, sources=[0, 14, 7])
    print(ret)

    target = dedent(
        """
        ╟── 0
        ╎   ├─╼ 1
        ╎   │   ├─╼ 3
        ╎   │   └─╼ 4
        ╎   └─╼ 2
        ╎       ├─╼ 5
        ╎       └─╼ 6
        ╟── 14
        ╎   ├─╼ 15
        ╎   │   ├─╼ 17
        ╎   │   └─╼ 18
        ╎   └─╼ 16
        ╎       ├─╼ 19
        ╎       └─╼ 20
        ╙── 7
            ├─╼ 8
            │   ├─╼ 10
            │   └─╼ 11
            └─╼ 9
                ├─╼ 12
                └─╼ 13
        """
    ).strip()
    assert ret == target

    ret = nx.forest_str(forest, sources=[0, 14, 7], ascii_only=True)
    print(ret)

    target = dedent(
        """
        +-- 0
        :   |-> 1
        :   |   |-> 3
        :   |   L-> 4
        :   L-> 2
        :       |-> 5
        :       L-> 6
        +-- 14
        :   |-> 15
        :   |   |-> 17
        :   |   L-> 18
        :   L-> 16
        :       |-> 19
        :       L-> 20
        +-- 7
            |-> 8
            |   |-> 10
            |   L-> 11
            L-> 9
                |-> 12
                L-> 13
        """
    ).strip()
    assert ret == target


def test_undirected_multi_tree_forest():
    tree1 = nx.balanced_tree(r=2, h=2, create_using=nx.Graph)
    tree2 = nx.balanced_tree(r=2, h=2, create_using=nx.Graph)
    tree2 = nx.relabel_nodes(tree2, {n: n + len(tree1) for n in tree2.nodes})
    forest = nx.union(tree1, tree2)
    ret = nx.forest_str(forest, sources=[0, 7])
    print(ret)

    target = dedent(
        """
        ╟── 0
        ╎   ├── 1
        ╎   │   ├── 3
        ╎   │   └── 4
        ╎   └── 2
        ╎       ├── 5
        ╎       └── 6
        ╙── 7
            ├── 8
            │   ├── 10
            │   └── 11
            └── 9
                ├── 12
                └── 13
        """
    ).strip()
    assert ret == target

    ret = nx.forest_str(forest, sources=[0, 7], ascii_only=True)
    print(ret)

    target = dedent(
        """
        +-- 0
        :   |-- 1
        :   |   |-- 3
        :   |   L-- 4
        :   L-- 2
        :       |-- 5
        :       L-- 6
        +-- 7
            |-- 8
            |   |-- 10
            |   L-- 11
            L-- 9
                |-- 12
                L-- 13
        """
    ).strip()
    assert ret == target


def test_undirected_tree_str():
    # Create a directed forest with labels
    graph = nx.balanced_tree(r=2, h=2, create_using=nx.Graph)

    # arbitrary starting point
    nx.forest_str(graph)

    node_target0 = dedent(
        """
        ╙── 0
            ├── 1
            │   ├── 3
            │   └── 4
            └── 2
                ├── 5
                └── 6
        """
    ).strip()

    # defined starting point
    ret = nx.forest_str(graph, sources=[0])
    print(ret)
    assert ret == node_target0

    # defined starting point
    node_target2 = dedent(
        """
        ╙── 2
            ├── 0
            │   └── 1
            │       ├── 3
            │       └── 4
            ├── 5
            └── 6
        """
    ).strip()
    ret = nx.forest_str(graph, sources=[2])
    print(ret)
    assert ret == node_target2


def test_forest_str_errors():
    ugraph = nx.complete_graph(3, create_using=nx.Graph)

    with pytest.raises(nx.NetworkXNotImplemented):
        nx.forest_str(ugraph)

    dgraph = nx.complete_graph(3, create_using=nx.DiGraph)

    with pytest.raises(nx.NetworkXNotImplemented):
        nx.forest_str(dgraph)


def test_overspecified_sources():
    """
    When sources are directly specified, we wont be able to determine when we
    are in the last component, so there will always be a trailing, leftmost
    pipe.
    """
    graph = nx.disjoint_union_all(
        [
            nx.balanced_tree(r=2, h=1, create_using=nx.DiGraph),
            nx.balanced_tree(r=1, h=2, create_using=nx.DiGraph),
            nx.balanced_tree(r=2, h=1, create_using=nx.DiGraph),
        ]
    )

    # defined starting point
    target1 = dedent(
        """
        ╟── 0
        ╎   ├─╼ 1
        ╎   └─╼ 2
        ╟── 3
        ╎   └─╼ 4
        ╎       └─╼ 5
        ╟── 6
        ╎   ├─╼ 7
        ╎   └─╼ 8
        """
    ).strip()

    target2 = dedent(
        """
        ╟── 0
        ╎   ├─╼ 1
        ╎   └─╼ 2
        ╟── 3
        ╎   └─╼ 4
        ╎       └─╼ 5
        ╙── 6
            ├─╼ 7
            └─╼ 8
        """
    ).strip()

    lines = []
    nx.forest_str(graph, write=lines.append, sources=graph.nodes)
    got1 = chr(10).join(lines)
    print("got1: ")
    print(got1)

    lines = []
    nx.forest_str(graph, write=lines.append)
    got2 = chr(10).join(lines)
    print("got2: ")
    print(got2)

    assert got1 == target1
    assert got2 == target2
