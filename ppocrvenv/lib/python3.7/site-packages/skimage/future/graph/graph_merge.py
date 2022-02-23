import numpy as np
import heapq


def _revalidate_node_edges(rag, node, heap_list):
    """Handles validation and invalidation of edges incident to a node.

    This function invalidates all existing edges incident on `node` and inserts
    new items in `heap_list` updated with the valid weights.

    rag : RAG
        The Region Adjacency Graph.
    node : int
        The id of the node whose incident edges are to be validated/invalidated
        .
    heap_list : list
        The list containing the existing heap of edges.
    """
    # networkx updates data dictionary if edge exists
    # this would mean we have to reposition these edges in
    # heap if their weight is updated.
    # instead we invalidate them

    for nbr in rag.neighbors(node):
        data = rag[node][nbr]
        try:
            # invalidate edges incident on `dst`, they have new weights
            data['heap item'][3] = False
            _invalidate_edge(rag, node, nbr)
        except KeyError:
            # will handle the case where the edge did not exist in the existing
            # graph
            pass

        wt = data['weight']
        heap_item = [wt, node, nbr, True]
        data['heap item'] = heap_item
        heapq.heappush(heap_list, heap_item)


def _rename_node(graph, node_id, copy_id):
    """ Rename `node_id` in `graph` to `copy_id`. """

    graph._add_node_silent(copy_id)
    graph.nodes[copy_id].update(graph.nodes[node_id])

    for nbr in graph.neighbors(node_id):
        wt = graph[node_id][nbr]['weight']
        graph.add_edge(nbr, copy_id, {'weight': wt})

    graph.remove_node(node_id)


def _invalidate_edge(graph, n1, n2):
    """ Invalidates the edge (n1, n2) in the heap. """
    graph[n1][n2]['heap item'][3] = False


def merge_hierarchical(labels, rag, thresh, rag_copy, in_place_merge,
                       merge_func, weight_func):
    """Perform hierarchical merging of a RAG.

    Greedily merges the most similar pair of nodes until no edges lower than
    `thresh` remain.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The Region Adjacency Graph.
    thresh : float
        Regions connected by an edge with weight smaller than `thresh` are
        merged.
    rag_copy : bool
        If set, the RAG copied before modifying.
    in_place_merge : bool
        If set, the nodes are merged in place. Otherwise, a new node is
        created for each merge..
    merge_func : callable
        This function is called before merging two nodes. For the RAG `graph`
        while merging `src` and `dst`, it is called as follows
        ``merge_func(graph, src, dst)``.
    weight_func : callable
        The function to compute the new weights of the nodes adjacent to the
        merged node. This is directly supplied as the argument `weight_func`
        to `merge_nodes`.

    Returns
    -------
    out : ndarray
        The new labeled array.

    """
    if rag_copy:
        rag = rag.copy()

    edge_heap = []
    for n1, n2, data in rag.edges(data=True):
        # Push a valid edge in the heap
        wt = data['weight']
        heap_item = [wt, n1, n2, True]
        heapq.heappush(edge_heap, heap_item)

        # Reference to the heap item in the graph
        data['heap item'] = heap_item

    while len(edge_heap) > 0 and edge_heap[0][0] < thresh:
        _, n1, n2, valid = heapq.heappop(edge_heap)

        # Ensure popped edge is valid, if not, the edge is discarded
        if valid:
            # Invalidate all neigbors of `src` before its deleted

            for nbr in rag.neighbors(n1):
                _invalidate_edge(rag, n1, nbr)

            for nbr in rag.neighbors(n2):
                _invalidate_edge(rag, n2, nbr)

            if not in_place_merge:
                next_id = rag.next_id()
                _rename_node(rag, n2, next_id)
                src, dst = n1, next_id
            else:
                src, dst = n1, n2

            merge_func(rag, src, dst)
            new_id = rag.merge_nodes(src, dst, weight_func)
            _revalidate_node_edges(rag, new_id, edge_heap)

    label_map = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for label in d['labels']:
            label_map[label] = ix

    return label_map[labels]
