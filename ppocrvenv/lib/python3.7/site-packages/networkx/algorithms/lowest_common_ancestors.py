"""Algorithms for finding the lowest common ancestor of trees and DAGs."""
from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import chain, count

import networkx as nx
from networkx.utils import (
    arbitrary_element,
    not_implemented_for,
    UnionFind,
)

__all__ = [
    "all_pairs_lowest_common_ancestor",
    "tree_all_pairs_lowest_common_ancestor",
    "lowest_common_ancestor",
]


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def tree_all_pairs_lowest_common_ancestor(G, root=None, pairs=None):
    r"""Yield the lowest common ancestor for sets of pairs in a tree.

    Parameters
    ----------
    G : NetworkX directed graph (must be a tree)

    root : node, optional (default: None)
        The root of the subtree to operate on.
        If None, assume the entire graph has exactly one source and use that.

    pairs : iterable or iterator of pairs of nodes, optional (default: None)
        The pairs of interest. If None, Defaults to all pairs of nodes
        under `root` that have a lowest common ancestor.

    Returns
    -------
    lcas : generator of tuples `((u, v), lca)` where `u` and `v` are nodes
        in `pairs` and `lca` is their lowest common ancestor.

    Notes
    -----
    Only defined on non-null trees represented with directed edges from
    parents to children. Uses Tarjan's off-line lowest-common-ancestors
    algorithm. Runs in time $O(4 \times (V + E + P))$ time, where 4 is the largest
    value of the inverse Ackermann function likely to ever come up in actual
    use, and $P$ is the number of pairs requested (or $V^2$ if all are needed).

    Tarjan, R. E. (1979), "Applications of path compression on balanced trees",
    Journal of the ACM 26 (4): 690-715, doi:10.1145/322154.322161.

    See Also
    --------
    all_pairs_lowest_common_ancestor: similar routine for general DAGs
    lowest_common_ancestor: just a single pair for general DAGs
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")
    elif None in G:
        raise nx.NetworkXError("None is not a valid node.")

    # Index pairs of interest for efficient lookup from either side.
    if pairs is not None:
        pair_dict = defaultdict(set)
        # See note on all_pairs_lowest_common_ancestor.
        if not isinstance(pairs, (Mapping, Set)):
            pairs = set(pairs)
        for u, v in pairs:
            for n in (u, v):
                if n not in G:
                    msg = f"The node {str(n)} is not in the digraph."
                    raise nx.NodeNotFound(msg)
            pair_dict[u].add(v)
            pair_dict[v].add(u)

    # If root is not specified, find the exactly one node with in degree 0 and
    # use it. Raise an error if none are found, or more than one is. Also check
    # for any nodes with in degree larger than 1, which would imply G is not a
    # tree.
    if root is None:
        for n, deg in G.in_degree:
            if deg == 0:
                if root is not None:
                    msg = "No root specified and tree has multiple sources."
                    raise nx.NetworkXError(msg)
                root = n
            elif deg > 1:
                msg = "Tree LCA only defined on trees; use DAG routine."
                raise nx.NetworkXError(msg)
    if root is None:
        raise nx.NetworkXError("Graph contains a cycle.")

    # Iterative implementation of Tarjan's offline lca algorithm
    # as described in CLRS on page 521 (2nd edition)/page 584 (3rd edition)
    uf = UnionFind()
    ancestors = {}
    for node in G:
        ancestors[node] = uf[node]

    colors = defaultdict(bool)
    for node in nx.dfs_postorder_nodes(G, root):
        colors[node] = True
        for v in pair_dict[node] if pairs is not None else G:
            if colors[v]:
                # If the user requested both directions of a pair, give it.
                # Otherwise, just give one.
                if pairs is not None and (node, v) in pairs:
                    yield (node, v), ancestors[uf[v]]
                if pairs is None or (v, node) in pairs:
                    yield (v, node), ancestors[uf[v]]
        if node != root:
            parent = arbitrary_element(G.pred[node])
            uf.union(parent, node)
            ancestors[uf[parent]] = parent


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def lowest_common_ancestor(G, node1, node2, default=None):
    """Compute the lowest common ancestor of the given pair of nodes.

    Parameters
    ----------
    G : NetworkX directed graph

    node1, node2 : nodes in the graph.

    default : object
        Returned if no common ancestor between `node1` and `node2`

    Returns
    -------
    The lowest common ancestor of node1 and node2,
    or default if they have no common ancestors.

    Notes
    -----
    Only defined on non-null directed acyclic graphs.
    Takes n log(n) time in the size of the graph.
    See `all_pairs_lowest_common_ancestor` when you have
    more than one pair of nodes of interest.

    See Also
    --------
    tree_all_pairs_lowest_common_ancestor
    all_pairs_lowest_common_ancestor
    """
    ans = list(all_pairs_lowest_common_ancestor(G, pairs=[(node1, node2)]))
    if ans:
        assert len(ans) == 1
        return ans[0][1]
    else:
        return default


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def all_pairs_lowest_common_ancestor(G, pairs=None):
    """Compute the lowest common ancestor for pairs of nodes.

    Parameters
    ----------
    G : NetworkX directed graph

    pairs : iterable of pairs of nodes, optional (default: all pairs)
        The pairs of nodes of interest.
        If None, will find the LCA of all pairs of nodes.

    Returns
    -------
    An iterator over ((node1, node2), lca) where (node1, node2) are
    the pairs specified and lca is a lowest common ancestor of the pair.
    Note that for the default of all pairs in G, we consider
    unordered pairs, e.g. you will not get both (b, a) and (a, b).

    Notes
    -----
    Only defined on non-null directed acyclic graphs.

    Uses the $O(n^3)$ ancestor-list algorithm from:
    M. A. Bender, M. Farach-Colton, G. Pemmasani, S. Skiena, P. Sumazin.
    "Lowest common ancestors in trees and directed acyclic graphs."
    Journal of Algorithms, 57(2): 75-94, 2005.

    See Also
    --------
    tree_all_pairs_lowest_common_ancestor
    lowest_common_ancestor
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("LCA only defined on directed acyclic graphs.")
    elif len(G) == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")
    elif None in G:
        raise nx.NetworkXError("None is not a valid node.")

    # The copy isn't ideal, neither is the switch-on-type, but without it users
    # passing an iterable will encounter confusing errors, and itertools.tee
    # does not appear to handle builtin types efficiently (IE, it materializes
    # another buffer rather than just creating listoperators at the same
    # offset). The Python documentation notes use of tee is unadvised when one
    # is consumed before the other.
    #
    # This will always produce correct results and avoid unnecessary
    # copies in many common cases.
    #
    if not isinstance(pairs, (Mapping, Set)) and pairs is not None:
        pairs = set(pairs)

    # Convert G into a dag with a single root by adding a node with edges to
    # all sources iff necessary.
    sources = [n for n, deg in G.in_degree if deg == 0]
    if len(sources) == 1:
        root = sources[0]
        super_root = None
    else:
        G = G.copy()
        # find unused node
        root = -1
        while root in G:
            root -= 1
        # use that as the super_root below all sources
        super_root = root
        for source in sources:
            G.add_edge(root, source)

    # Start by computing a spanning tree, and the DAG of all edges not in it.
    # We will then use the tree lca algorithm on the spanning tree, and use
    # the DAG to figure out the set of tree queries necessary.
    spanning_tree = nx.dfs_tree(G, root)
    dag = nx.DiGraph(
        (u, v)
        for u, v in G.edges
        if u not in spanning_tree or v not in spanning_tree[u]
    )

    # Ensure that both the dag and the spanning tree contains all nodes in G,
    # even nodes that are disconnected in the dag.
    spanning_tree.add_nodes_from(G)
    dag.add_nodes_from(G)

    counter = count()

    # Necessary to handle graphs consisting of a single node and no edges.
    root_distance = {root: next(counter)}

    for edge in nx.bfs_edges(spanning_tree, root):
        for node in edge:
            if node not in root_distance:
                root_distance[node] = next(counter)

    # Index the position of all nodes in the Euler tour so we can efficiently
    # sort lists and merge in tour order.
    euler_tour_pos = {}
    for node in nx.depth_first_search.dfs_preorder_nodes(G, root):
        if node not in euler_tour_pos:
            euler_tour_pos[node] = next(counter)

    # Generate the set of all nodes of interest in the pairs.
    pairset = set()
    if pairs is not None:
        pairset = set(chain.from_iterable(pairs))

    for n in pairset:
        if n not in G:
            msg = f"The node {str(n)} is not in the digraph."
            raise nx.NodeNotFound(msg)

    # Generate the transitive closure over the dag (not G) of all nodes, and
    # sort each node's closure set by order of first appearance in the Euler
    # tour.
    ancestors = {}
    for v in dag:
        if pairs is None or v in pairset:
            my_ancestors = nx.dag.ancestors(dag, v)
            my_ancestors.add(v)
            ancestors[v] = sorted(my_ancestors, key=euler_tour_pos.get)

    def _compute_dag_lca_from_tree_values(tree_lca, dry_run):
        """Iterate through the in-order merge for each pair of interest.

        We do this to answer the user's query, but it is also used to
        avoid generating unnecessary tree entries when the user only
        needs some pairs.
        """
        for (node1, node2) in pairs if pairs is not None else tree_lca:
            best_root_distance = None
            best = None

            indices = [0, 0]
            ancestors_by_index = [ancestors[node1], ancestors[node2]]

            def get_next_in_merged_lists(indices):
                """Returns index of the list containing the next item

                Next order refers to the merged order.
                Index can be 0 or 1 (or None if exhausted).
                """
                index1, index2 = indices
                if index1 >= len(ancestors[node1]) and index2 >= len(ancestors[node2]):
                    return None
                elif index1 >= len(ancestors[node1]):
                    return 1
                elif index2 >= len(ancestors[node2]):
                    return 0
                elif (
                    euler_tour_pos[ancestors[node1][index1]]
                    < euler_tour_pos[ancestors[node2][index2]]
                ):
                    return 0
                else:
                    return 1

            # Find the LCA by iterating through the in-order merge of the two
            # nodes of interests' ancestor sets. In principle, we need to
            # consider all pairs in the Cartesian product of the ancestor sets,
            # but by the restricted min range query reduction we are guaranteed
            # that one of the pairs of interest is adjacent in the merged list
            # iff one came from each list.
            i = get_next_in_merged_lists(indices)
            cur = ancestors_by_index[i][indices[i]], i
            while i is not None:
                prev = cur
                indices[i] += 1
                i = get_next_in_merged_lists(indices)
                if i is not None:
                    cur = ancestors_by_index[i][indices[i]], i

                    # Two adjacent entries must not be from the same list
                    # in order for their tree LCA to be considered.
                    if cur[1] != prev[1]:
                        tree_node1, tree_node2 = prev[0], cur[0]
                        if (tree_node1, tree_node2) in tree_lca:
                            ans = tree_lca[tree_node1, tree_node2]
                        else:
                            ans = tree_lca[tree_node2, tree_node1]
                        if not dry_run and (
                            best is None or root_distance[ans] > best_root_distance
                        ):
                            best_root_distance = root_distance[ans]
                            best = ans

            # If the LCA is super_root, there is no LCA in the user's graph.
            if not dry_run and (super_root is None or best != super_root):
                yield (node1, node2), best

    # Generate the spanning tree lca for all pairs. This doesn't make sense to
    # do incrementally since we are using a linear time offline algorithm for
    # tree lca.
    if pairs is None:
        # We want all pairs so we'll need the entire tree.
        tree_lca = dict(tree_all_pairs_lowest_common_ancestor(spanning_tree, root))
    else:
        # We only need the merged adjacent pairs by seeing which queries the
        # algorithm needs then generating them in a single pass.
        tree_lca = defaultdict(int)
        for _ in _compute_dag_lca_from_tree_values(tree_lca, True):
            pass

        # Replace the bogus default tree values with the real ones.
        for (pair, lca) in tree_all_pairs_lowest_common_ancestor(
            spanning_tree, root, tree_lca
        ):
            tree_lca[pair] = lca

    # All precomputations complete. Now we just need to give the user the pairs
    # they asked for, or all pairs if they want them all.
    return _compute_dag_lca_from_tree_values(tree_lca, False)
