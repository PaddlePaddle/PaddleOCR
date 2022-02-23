import itertools

__all__ = ["greedy_coloring_with_interchange"]


class Node:

    __slots__ = ["node_id", "color", "adj_list", "adj_color"]

    def __init__(self, node_id, n):
        self.node_id = node_id
        self.color = -1
        self.adj_list = None
        self.adj_color = [None for _ in range(n)]

    def __repr__(self):
        return (
            f"Node_id: {self.node_id}, Color: {self.color}, "
            f"Adj_list: ({self.adj_list}), adj_color: ({self.adj_color})"
        )

    def assign_color(self, adj_entry, color):
        adj_entry.col_prev = None
        adj_entry.col_next = self.adj_color[color]
        self.adj_color[color] = adj_entry
        if adj_entry.col_next is not None:
            adj_entry.col_next.col_prev = adj_entry

    def clear_color(self, adj_entry, color):
        if adj_entry.col_prev is None:
            self.adj_color[color] = adj_entry.col_next
        else:
            adj_entry.col_prev.col_next = adj_entry.col_next
        if adj_entry.col_next is not None:
            adj_entry.col_next.col_prev = adj_entry.col_prev

    def iter_neighbors(self):
        adj_node = self.adj_list
        while adj_node is not None:
            yield adj_node
            adj_node = adj_node.next

    def iter_neighbors_color(self, color):
        adj_color_node = self.adj_color[color]
        while adj_color_node is not None:
            yield adj_color_node.node_id
            adj_color_node = adj_color_node.col_next


class AdjEntry:

    __slots__ = ["node_id", "next", "mate", "col_next", "col_prev"]

    def __init__(self, node_id):
        self.node_id = node_id
        self.next = None
        self.mate = None
        self.col_next = None
        self.col_prev = None

    def __repr__(self):
        col_next = None if self.col_next is None else self.col_next.node_id
        col_prev = None if self.col_prev is None else self.col_prev.node_id
        return (
            f"Node_id: {self.node_id}, Next: ({self.next}), "
            f"Mate: ({self.mate.node_id}), "
            f"col_next: ({col_next}), col_prev: ({col_prev})"
        )


def greedy_coloring_with_interchange(original_graph, nodes):
    """
        This procedure is an adaption of the algorithm described by [1]_,
        and is an implementation of coloring with interchange. Please be
        advised, that the datastructures used are rather complex because
        they are optimized to minimize the time spent identifying
        subcomponents of the graph, which are possible candidates for color
        interchange.

    References
    ----------
    .. [1] Maciej M. Syslo, Marsingh Deo, Janusz S. Kowalik,
       Discrete Optimization Algorithms with Pascal Programs, 415-424, 1983.
       ISBN 0-486-45353-7.
    """
    n = len(original_graph)

    graph = {node_id: Node(node_id, n) for node_id in original_graph}

    for (node1, node2) in original_graph.edges():
        adj_entry1 = AdjEntry(node2)
        adj_entry2 = AdjEntry(node1)
        adj_entry1.mate = adj_entry2
        adj_entry2.mate = adj_entry1
        node1_head = graph[node1].adj_list
        adj_entry1.next = node1_head
        graph[node1].adj_list = adj_entry1
        node2_head = graph[node2].adj_list
        adj_entry2.next = node2_head
        graph[node2].adj_list = adj_entry2

    k = 0
    for node in nodes:
        # Find the smallest possible, unused color
        neighbors = graph[node].iter_neighbors()
        col_used = {graph[adj_node.node_id].color for adj_node in neighbors}
        col_used.discard(-1)
        k1 = next(itertools.dropwhile(lambda x: x in col_used, itertools.count()))

        # k1 is now the lowest available color
        if k1 > k:
            connected = True
            visited = set()
            col1 = -1
            col2 = -1
            while connected and col1 < k:
                col1 += 1
                neighbor_cols = graph[node].iter_neighbors_color(col1)
                col1_adj = [it for it in neighbor_cols]

                col2 = col1
                while connected and col2 < k:
                    col2 += 1
                    visited = set(col1_adj)
                    frontier = list(col1_adj)
                    i = 0
                    while i < len(frontier):
                        search_node = frontier[i]
                        i += 1
                        col_opp = col2 if graph[search_node].color == col1 else col1
                        neighbor_cols = graph[search_node].iter_neighbors_color(col_opp)

                        for neighbor in neighbor_cols:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                frontier.append(neighbor)

                    # Search if node is not adj to any col2 vertex
                    connected = (
                        len(
                            visited.intersection(graph[node].iter_neighbors_color(col2))
                        )
                        > 0
                    )

            # If connected is false then we can swap !!!
            if not connected:
                # Update all the nodes in the component
                for search_node in visited:
                    graph[search_node].color = (
                        col2 if graph[search_node].color == col1 else col1
                    )
                    col2_adj = graph[search_node].adj_color[col2]
                    graph[search_node].adj_color[col2] = graph[search_node].adj_color[
                        col1
                    ]
                    graph[search_node].adj_color[col1] = col2_adj

                # Update all the neighboring nodes
                for search_node in visited:
                    col = graph[search_node].color
                    col_opp = col1 if col == col2 else col2
                    for adj_node in graph[search_node].iter_neighbors():
                        if graph[adj_node.node_id].color != col_opp:
                            # Direct reference to entry
                            adj_mate = adj_node.mate
                            graph[adj_node.node_id].clear_color(adj_mate, col_opp)
                            graph[adj_node.node_id].assign_color(adj_mate, col)
                k1 = col1

        # We can color this node color k1
        graph[node].color = k1
        k = max(k1, k)

        # Update the neighbors of this node
        for adj_node in graph[node].iter_neighbors():
            adj_mate = adj_node.mate
            graph[adj_node.node_id].assign_color(adj_mate, k1)

    return {node.node_id: node.color for node in graph.values()}
