"""Operations on trees."""
from functools import partial
from itertools import chain

import networkx as nx
from itertools import accumulate

__all__ = ["join"]


def join(rooted_trees, label_attribute=None):
    """Returns a new rooted tree with a root node joined with the roots
    of each of the given rooted trees.

    Parameters
    ----------
    rooted_trees : list
        A list of pairs in which each left element is a NetworkX graph
        object representing a tree and each right element is the root
        node of that tree. The nodes of these trees will be relabeled to
        integers.

    label_attribute : str
        If provided, the old node labels will be stored in the new tree
        under this node attribute. If not provided, the node attribute
        ``'_old'`` will store the original label of the node in the
        rooted trees given in the input.

    Returns
    -------
    NetworkX graph
        The rooted tree whose subtrees are the given rooted trees. The
        new root node is labeled 0. Each non-root node has an attribute,
        as described under the keyword argument ``label_attribute``,
        that indicates the label of the original node in the input tree.

    Notes
    -----
    Graph, edge, and node attributes are propagated from the given
    rooted trees to the created tree. If there are any overlapping graph
    attributes, those from later trees will overwrite those from earlier
    trees in the tuple of positional arguments.

    Examples
    --------
    Join two full balanced binary trees of height *h* to get a full
    balanced binary tree of depth *h* + 1::

        >>> h = 4
        >>> left = nx.balanced_tree(2, h)
        >>> right = nx.balanced_tree(2, h)
        >>> joined_tree = nx.join([(left, 0), (right, 0)])
        >>> nx.is_isomorphic(joined_tree, nx.balanced_tree(2, h + 1))
        True

    """
    if len(rooted_trees) == 0:
        return nx.empty_graph(1)

    # Unzip the zipped list of (tree, root) pairs.
    trees, roots = zip(*rooted_trees)

    # The join of the trees has the same type as the type of the first
    # tree.
    R = type(trees[0])()

    # Relabel the nodes so that their union is the integers starting at 1.
    if label_attribute is None:
        label_attribute = "_old"
    relabel = partial(
        nx.convert_node_labels_to_integers, label_attribute=label_attribute
    )
    lengths = (len(tree) for tree in trees[:-1])
    first_labels = chain([0], accumulate(lengths))
    trees = [
        relabel(tree, first_label=first_label + 1)
        for tree, first_label in zip(trees, first_labels)
    ]

    # Get the relabeled roots.
    roots = [
        next(v for v, d in tree.nodes(data=True) if d.get("_old") == root)
        for tree, root in zip(trees, roots)
    ]

    # Remove the old node labels.
    for tree in trees:
        for v in tree:
            tree.nodes[v].pop("_old")

    # Add all sets of nodes and edges, with data.
    nodes = (tree.nodes(data=True) for tree in trees)
    edges = (tree.edges(data=True) for tree in trees)
    R.add_nodes_from(chain.from_iterable(nodes))
    R.add_edges_from(chain.from_iterable(edges))

    # Add graph attributes; later attributes take precedent over earlier
    # attributes.
    for tree in trees:
        R.graph.update(tree.graph)

    # Finally, join the subtrees at the root. We know 0 is unused by the
    # way we relabeled the subtrees.
    R.add_node(0)
    R.add_edges_from((0, root) for root in roots)

    return R
