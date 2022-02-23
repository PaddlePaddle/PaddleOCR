"""Algorithm to select influential nodes in a graph using VoteRank."""

__all__ = ["voterank"]


def voterank(G, number_of_nodes=None):
    """Select a list of influential nodes in a graph using VoteRank algorithm

    VoteRank [1]_ computes a ranking of the nodes in a graph G based on a
    voting scheme. With VoteRank, all nodes vote for each of its in-neighbours
    and the node with the highest votes is elected iteratively. The voting
    ability of out-neighbors of elected nodes is decreased in subsequent turns.

    Note: We treat each edge independently in case of multigraphs.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    number_of_nodes : integer, optional
        Number of ranked nodes to extract (default all nodes).

    Returns
    -------
    voterank : list
        Ordered list of computed seeds.
        Only nodes with positive number of votes are returned.

    References
    ----------
    .. [1] Zhang, J.-X. et al. (2016).
        Identifying a set of influential spreaders in complex networks.
        Sci. Rep. 6, 27823; doi: 10.1038/srep27823.
    """
    influential_nodes = []
    voterank = {}
    if len(G) == 0:
        return influential_nodes
    if number_of_nodes is None or number_of_nodes > len(G):
        number_of_nodes = len(G)
    if G.is_directed():
        # For directed graphs compute average out-degree
        avgDegree = sum(deg for _, deg in G.out_degree()) / len(G)
    else:
        # For undirected graphs compute average degree
        avgDegree = sum(deg for _, deg in G.degree()) / len(G)
    # step 1 - initiate all nodes to (0,1) (score, voting ability)
    for n in G.nodes():
        voterank[n] = [0, 1]
    # Repeat steps 1b to 4 until num_seeds are elected.
    for _ in range(number_of_nodes):
        # step 1b - reset rank
        for n in G.nodes():
            voterank[n][0] = 0
        # step 2 - vote
        for n, nbr in G.edges():
            # In directed graphs nodes only vote for their in-neighbors
            voterank[n][0] += voterank[nbr][1]
            if not G.is_directed():
                voterank[nbr][0] += voterank[n][1]
        for n in influential_nodes:
            voterank[n][0] = 0
        # step 3 - select top node
        n = max(G.nodes, key=lambda x: voterank[x][0])
        if voterank[n][0] == 0:
            return influential_nodes
        influential_nodes.append(n)
        # weaken the selected node
        voterank[n] = [0, 0]
        # step 4 - update voterank properties
        for _, nbr in G.edges(n):
            voterank[nbr][1] -= 1 / avgDegree
            voterank[nbr][1] = max(voterank[nbr][1], 0)
    return influential_nodes
