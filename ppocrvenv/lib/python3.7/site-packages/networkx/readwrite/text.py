"""
Text-based visual representations of graphs
"""

__all__ = ["forest_str"]


def forest_str(graph, with_labels=True, sources=None, write=None, ascii_only=False):
    """
    Creates a nice utf8 representation of a directed forest

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent (must be a tree, forest, or the empty graph)

    with_labels : bool
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. Defaults to True.

    sources : List
        Mainly relevant for undirected forests, specifies which nodes to list
        first. If unspecified the root nodes of each tree will be used for
        directed forests; for undirected forests this defaults to the nodes
        with the smallest degree.

    write : callable
        Function to use to write to, if None new lines are appended to
        a list and returned. If set to the `print` function, lines will
        be written to stdout as they are generated. If specified,
        this function will return None. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    Returns
    -------
    str | None :
        utf8 representation of the tree / forest

    Example
    -------
    >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    >>> print(nx.forest_str(graph))
    ╙── 0
        ├─╼ 1
        │   ├─╼ 3
        │   │   ├─╼ 7
        │   │   └─╼ 8
        │   └─╼ 4
        │       ├─╼ 9
        │       └─╼ 10
        └─╼ 2
            ├─╼ 5
            │   ├─╼ 11
            │   └─╼ 12
            └─╼ 6
                ├─╼ 13
                └─╼ 14


    >>> graph = nx.balanced_tree(r=1, h=2, create_using=nx.Graph)
    >>> print(nx.forest_str(graph))
    ╙── 0
        └── 1
            └── 2

    >>> print(nx.forest_str(graph, ascii_only=True))
    +-- 0
        L-- 1
            L-- 2
    """
    import networkx as nx

    printbuf = []
    if write is None:
        _write = printbuf.append
    else:
        _write = write

    # Define glphys
    # Notes on available box and arrow characters
    # https://en.wikipedia.org/wiki/Box-drawing_character
    # https://stackoverflow.com/questions/2701192/triangle-arrow
    if ascii_only:
        glyph_empty = "+"
        glyph_newtree_last = "+-- "
        glyph_newtree_mid = "+-- "
        glyph_endof_forest = "    "
        glyph_within_forest = ":   "
        glyph_within_tree = "|   "

        glyph_directed_last = "L-> "
        glyph_directed_mid = "|-> "

        glyph_undirected_last = "L-- "
        glyph_undirected_mid = "|-- "
    else:
        glyph_empty = "╙"
        glyph_newtree_last = "╙── "
        glyph_newtree_mid = "╟── "
        glyph_endof_forest = "    "
        glyph_within_forest = "╎   "
        glyph_within_tree = "│   "

        glyph_directed_last = "└─╼ "
        glyph_directed_mid = "├─╼ "

        glyph_undirected_last = "└── "
        glyph_undirected_mid = "├── "

    if len(graph.nodes) == 0:
        _write(glyph_empty)
    else:
        if not nx.is_forest(graph):
            raise nx.NetworkXNotImplemented("input must be a forest or the empty graph")

        is_directed = graph.is_directed()
        succ = graph.succ if is_directed else graph.adj

        if sources is None:
            if is_directed:
                # use real source nodes for directed trees
                sources = [n for n in graph.nodes if graph.in_degree[n] == 0]
            else:
                # use arbitrary sources for undirected trees
                sources = [
                    min(cc, key=lambda n: graph.degree[n])
                    for cc in nx.connected_components(graph)
                ]

        # Populate the stack with each source node, empty indentation, and mark
        # the final node. Reverse the stack so sources are popped in the
        # correct order.
        last_idx = len(sources) - 1
        stack = [(node, "", (idx == last_idx)) for idx, node in enumerate(sources)][
            ::-1
        ]

        seen = set()
        while stack:
            node, indent, islast = stack.pop()
            if node in seen:
                continue
            seen.add(node)

            if not indent:
                # Top level items (i.e. trees in the forest) get different
                # glyphs to indicate they are not actually connected
                if islast:
                    this_prefix = indent + glyph_newtree_last
                    next_prefix = indent + glyph_endof_forest
                else:
                    this_prefix = indent + glyph_newtree_mid
                    next_prefix = indent + glyph_within_forest

            else:
                # For individual tree edges distinguish between directed and
                # undirected cases
                if is_directed:
                    if islast:
                        this_prefix = indent + glyph_directed_last
                        next_prefix = indent + glyph_endof_forest
                    else:
                        this_prefix = indent + glyph_directed_mid
                        next_prefix = indent + glyph_within_tree
                else:
                    if islast:
                        this_prefix = indent + glyph_undirected_last
                        next_prefix = indent + glyph_endof_forest
                    else:
                        this_prefix = indent + glyph_undirected_mid
                        next_prefix = indent + glyph_within_tree

            if with_labels:
                label = graph.nodes[node].get("label", node)
            else:
                label = node

            _write(this_prefix + str(label))

            # Push children on the stack in reverse order so they are popped in
            # the original order.
            children = [child for child in succ[node] if child not in seen]
            for idx, child in enumerate(children[::-1], start=1):
                islast_next = idx <= 1
                try_frame = (child, next_prefix, islast_next)
                stack.append(try_frame)

    if write is None:
        # Only return a string if the custom write function was not specified
        return "\n".join(printbuf)
