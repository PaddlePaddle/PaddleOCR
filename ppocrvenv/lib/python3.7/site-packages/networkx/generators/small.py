"""
Various small and named graphs, together with some compact generators.

"""

__all__ = [
    "make_small_graph",
    "LCF_graph",
    "bull_graph",
    "chvatal_graph",
    "cubical_graph",
    "desargues_graph",
    "diamond_graph",
    "dodecahedral_graph",
    "frucht_graph",
    "heawood_graph",
    "hoffman_singleton_graph",
    "house_graph",
    "house_x_graph",
    "icosahedral_graph",
    "krackhardt_kite_graph",
    "moebius_kantor_graph",
    "octahedral_graph",
    "pappus_graph",
    "petersen_graph",
    "sedgewick_maze_graph",
    "tetrahedral_graph",
    "truncated_cube_graph",
    "truncated_tetrahedron_graph",
    "tutte_graph",
]

import networkx as nx
from networkx.generators.classic import (
    empty_graph,
    cycle_graph,
    path_graph,
    complete_graph,
)
from networkx.exception import NetworkXError


def make_small_undirected_graph(graph_description, create_using=None):
    """
    Return a small undirected graph described by graph_description.

    See make_small_graph.
    """
    G = empty_graph(0, create_using)
    if G.is_directed():
        raise NetworkXError("Directed Graph not supported")
    return make_small_graph(graph_description, G)


def make_small_graph(graph_description, create_using=None):
    """
    Return the small graph described by graph_description.

    graph_description is a list of the form [ltype,name,n,xlist]

    Here ltype is one of "adjacencylist" or "edgelist",
    name is the name of the graph and n the number of nodes.
    This constructs a graph of n nodes with integer labels 0,..,n-1.

    If ltype="adjacencylist"  then xlist is an adjacency list
    with exactly n entries, in with the j'th entry (which can be empty)
    specifies the nodes connected to vertex j.
    e.g. the "square" graph C_4 can be obtained by

    >>> G = nx.make_small_graph(
    ...     ["adjacencylist", "C_4", 4, [[2, 4], [1, 3], [2, 4], [1, 3]]]
    ... )

    or, since we do not need to add edges twice,

    >>> G = nx.make_small_graph(["adjacencylist", "C_4", 4, [[2, 4], [3], [4], []]])

    If ltype="edgelist" then xlist is an edge list
    written as [[v1,w2],[v2,w2],...,[vk,wk]],
    where vj and wj integers in the range 1,..,n
    e.g. the "square" graph C_4 can be obtained by

    >>> G = nx.make_small_graph(
    ...     ["edgelist", "C_4", 4, [[1, 2], [3, 4], [2, 3], [4, 1]]]
    ... )

    Use the create_using argument to choose the graph class/type.
    """

    if graph_description[0] not in ("adjacencylist", "edgelist"):
        raise NetworkXError("ltype must be either adjacencylist or edgelist")

    ltype = graph_description[0]
    name = graph_description[1]
    n = graph_description[2]

    G = empty_graph(n, create_using)
    nodes = G.nodes()

    if ltype == "adjacencylist":
        adjlist = graph_description[3]
        if len(adjlist) != n:
            raise NetworkXError("invalid graph_description")
        G.add_edges_from([(u - 1, v) for v in nodes for u in adjlist[v]])
    elif ltype == "edgelist":
        edgelist = graph_description[3]
        for e in edgelist:
            v1 = e[0] - 1
            v2 = e[1] - 1
            if v1 < 0 or v1 > n - 1 or v2 < 0 or v2 > n - 1:
                raise NetworkXError("invalid graph_description")
            else:
                G.add_edge(v1, v2)
    G.name = name
    return G


def LCF_graph(n, shift_list, repeats, create_using=None):
    """
    Return the cubic graph specified in LCF notation.

    LCF notation (LCF=Lederberg-Coxeter-Fruchte) is a compressed
    notation used in the generation of various cubic Hamiltonian
    graphs of high symmetry. See, for example, dodecahedral_graph,
    desargues_graph, heawood_graph and pappus_graph below.

    n (number of nodes)
      The starting graph is the n-cycle with nodes 0,...,n-1.
      (The null graph is returned if n < 0.)

    shift_list = [s1,s2,..,sk], a list of integer shifts mod n,

    repeats
      integer specifying the number of times that shifts in shift_list
      are successively applied to each v_current in the n-cycle
      to generate an edge between v_current and v_current+shift mod n.

    For v1 cycling through the n-cycle a total of k*repeats
    with shift cycling through shiftlist repeats times connect
    v1 with v1+shift mod n

    The utility graph $K_{3,3}$

    >>> G = nx.LCF_graph(6, [3, -3], 3)

    The Heawood graph

    >>> G = nx.LCF_graph(14, [5, -5], 7)

    See http://mathworld.wolfram.com/LCFNotation.html for a description
    and references.

    """
    if n <= 0:
        return empty_graph(0, create_using)

    # start with the n-cycle
    G = cycle_graph(n, create_using)
    if G.is_directed():
        raise NetworkXError("Directed Graph not supported")
    G.name = "LCF_graph"
    nodes = sorted(list(G))

    n_extra_edges = repeats * len(shift_list)
    # edges are added n_extra_edges times
    # (not all of these need be new)
    if n_extra_edges < 1:
        return G

    for i in range(n_extra_edges):
        shift = shift_list[i % len(shift_list)]  # cycle through shift_list
        v1 = nodes[i % n]  # cycle repeatedly through nodes
        v2 = nodes[(i + shift) % n]
        G.add_edge(v1, v2)
    return G


# -------------------------------------------------------------------------------
#   Various small and named graphs
# -------------------------------------------------------------------------------


def bull_graph(create_using=None):
    """Returns the Bull graph."""
    description = [
        "adjacencylist",
        "Bull Graph",
        5,
        [[2, 3], [1, 3, 4], [1, 2, 5], [2], [3]],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def chvatal_graph(create_using=None):
    """Returns the Chvátal graph."""
    description = [
        "adjacencylist",
        "Chvatal Graph",
        12,
        [
            [2, 5, 7, 10],
            [3, 6, 8],
            [4, 7, 9],
            [5, 8, 10],
            [6, 9],
            [11, 12],
            [11, 12],
            [9, 12],
            [11],
            [11, 12],
            [],
            [],
        ],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def cubical_graph(create_using=None):
    """Returns the 3-regular Platonic Cubical graph."""
    description = [
        "adjacencylist",
        "Platonic Cubical Graph",
        8,
        [
            [2, 4, 5],
            [1, 3, 8],
            [2, 4, 7],
            [1, 3, 6],
            [1, 6, 8],
            [4, 5, 7],
            [3, 6, 8],
            [2, 5, 7],
        ],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def desargues_graph(create_using=None):
    """Return the Desargues graph."""
    G = LCF_graph(20, [5, -5, 9, -9], 5, create_using)
    G.name = "Desargues Graph"
    return G


def diamond_graph(create_using=None):
    """Returns the Diamond graph."""
    description = [
        "adjacencylist",
        "Diamond Graph",
        4,
        [[2, 3], [1, 3, 4], [1, 2, 4], [2, 3]],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def dodecahedral_graph(create_using=None):
    """Return the Platonic Dodecahedral graph."""
    G = LCF_graph(20, [10, 7, 4, -4, -7, 10, -4, 7, -7, 4], 2, create_using)
    G.name = "Dodecahedral Graph"
    return G


def frucht_graph(create_using=None):
    """Returns the Frucht Graph.

    The Frucht Graph is the smallest cubical graph whose
    automorphism group consists only of the identity element.

    """
    G = cycle_graph(7, create_using)
    G.add_edges_from(
        [
            [0, 7],
            [1, 7],
            [2, 8],
            [3, 9],
            [4, 9],
            [5, 10],
            [6, 10],
            [7, 11],
            [8, 11],
            [8, 9],
            [10, 11],
        ]
    )

    G.name = "Frucht Graph"
    return G


def heawood_graph(create_using=None):
    """Return the Heawood graph, a (3,6) cage."""
    G = LCF_graph(14, [5, -5], 7, create_using)
    G.name = "Heawood Graph"
    return G


def hoffman_singleton_graph():
    """Return the Hoffman-Singleton Graph."""
    G = nx.Graph()
    for i in range(5):
        for j in range(5):
            G.add_edge(("pentagon", i, j), ("pentagon", i, (j - 1) % 5))
            G.add_edge(("pentagon", i, j), ("pentagon", i, (j + 1) % 5))
            G.add_edge(("pentagram", i, j), ("pentagram", i, (j - 2) % 5))
            G.add_edge(("pentagram", i, j), ("pentagram", i, (j + 2) % 5))
            for k in range(5):
                G.add_edge(("pentagon", i, j), ("pentagram", k, (i * k + j) % 5))
    G = nx.convert_node_labels_to_integers(G)
    G.name = "Hoffman-Singleton Graph"
    return G


def house_graph(create_using=None):
    """Returns the House graph (square with triangle on top)."""
    description = [
        "adjacencylist",
        "House Graph",
        5,
        [[2, 3], [1, 4], [1, 4, 5], [2, 3, 5], [3, 4]],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def house_x_graph(create_using=None):
    """Returns the House graph with a cross inside the house square."""
    description = [
        "adjacencylist",
        "House-with-X-inside Graph",
        5,
        [[2, 3, 4], [1, 3, 4], [1, 2, 4, 5], [1, 2, 3, 5], [3, 4]],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def icosahedral_graph(create_using=None):
    """Returns the Platonic Icosahedral graph."""
    description = [
        "adjacencylist",
        "Platonic Icosahedral Graph",
        12,
        [
            [2, 6, 8, 9, 12],
            [3, 6, 7, 9],
            [4, 7, 9, 10],
            [5, 7, 10, 11],
            [6, 7, 11, 12],
            [7, 12],
            [],
            [9, 10, 11, 12],
            [10],
            [11],
            [12],
            [],
        ],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def krackhardt_kite_graph(create_using=None):
    """
    Return the Krackhardt Kite Social Network.

    A 10 actor social network introduced by David Krackhardt
    to illustrate: degree, betweenness, centrality, closeness, etc.
    The traditional labeling is:
    Andre=1, Beverley=2, Carol=3, Diane=4,
    Ed=5, Fernando=6, Garth=7, Heather=8, Ike=9, Jane=10.

    """
    description = [
        "adjacencylist",
        "Krackhardt Kite Social Network",
        10,
        [
            [2, 3, 4, 6],
            [1, 4, 5, 7],
            [1, 4, 6],
            [1, 2, 3, 5, 6, 7],
            [2, 4, 7],
            [1, 3, 4, 7, 8],
            [2, 4, 5, 6, 8],
            [6, 7, 9],
            [8, 10],
            [9],
        ],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def moebius_kantor_graph(create_using=None):
    """Returns the Moebius-Kantor graph."""
    G = LCF_graph(16, [5, -5], 8, create_using)
    G.name = "Moebius-Kantor Graph"
    return G


def octahedral_graph(create_using=None):
    """Returns the Platonic Octahedral graph."""
    description = [
        "adjacencylist",
        "Platonic Octahedral Graph",
        6,
        [[2, 3, 4, 5], [3, 4, 6], [5, 6], [5, 6], [6], []],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def pappus_graph():
    """Return the Pappus graph."""
    G = LCF_graph(18, [5, 7, -7, 7, -7, -5], 3)
    G.name = "Pappus Graph"
    return G


def petersen_graph(create_using=None):
    """Returns the Petersen graph."""
    description = [
        "adjacencylist",
        "Petersen Graph",
        10,
        [
            [2, 5, 6],
            [1, 3, 7],
            [2, 4, 8],
            [3, 5, 9],
            [4, 1, 10],
            [1, 8, 9],
            [2, 9, 10],
            [3, 6, 10],
            [4, 6, 7],
            [5, 7, 8],
        ],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def sedgewick_maze_graph(create_using=None):
    """
    Return a small maze with a cycle.

    This is the maze used in Sedgewick,3rd Edition, Part 5, Graph
    Algorithms, Chapter 18, e.g. Figure 18.2 and following.
    Nodes are numbered 0,..,7
    """
    G = empty_graph(0, create_using)
    G.add_nodes_from(range(8))
    G.add_edges_from([[0, 2], [0, 7], [0, 5]])
    G.add_edges_from([[1, 7], [2, 6]])
    G.add_edges_from([[3, 4], [3, 5]])
    G.add_edges_from([[4, 5], [4, 7], [4, 6]])
    G.name = "Sedgewick Maze"
    return G


def tetrahedral_graph(create_using=None):
    """Return the 3-regular Platonic Tetrahedral graph."""
    G = complete_graph(4, create_using)
    G.name = "Platonic Tetrahedral graph"
    return G


def truncated_cube_graph(create_using=None):
    """Returns the skeleton of the truncated cube."""
    description = [
        "adjacencylist",
        "Truncated Cube Graph",
        24,
        [
            [2, 3, 5],
            [12, 15],
            [4, 5],
            [7, 9],
            [6],
            [17, 19],
            [8, 9],
            [11, 13],
            [10],
            [18, 21],
            [12, 13],
            [15],
            [14],
            [22, 23],
            [16],
            [20, 24],
            [18, 19],
            [21],
            [20],
            [24],
            [22],
            [23],
            [24],
            [],
        ],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G


def truncated_tetrahedron_graph(create_using=None):
    """Returns the skeleton of the truncated Platonic tetrahedron."""
    G = path_graph(12, create_using)
    #    G.add_edges_from([(1,3),(1,10),(2,7),(4,12),(5,12),(6,8),(9,11)])
    G.add_edges_from([(0, 2), (0, 9), (1, 6), (3, 11), (4, 11), (5, 7), (8, 10)])
    G.name = "Truncated Tetrahedron Graph"
    return G


def tutte_graph(create_using=None):
    """Returns the Tutte graph."""
    description = [
        "adjacencylist",
        "Tutte's Graph",
        46,
        [
            [2, 3, 4],
            [5, 27],
            [11, 12],
            [19, 20],
            [6, 34],
            [7, 30],
            [8, 28],
            [9, 15],
            [10, 39],
            [11, 38],
            [40],
            [13, 40],
            [14, 36],
            [15, 16],
            [35],
            [17, 23],
            [18, 45],
            [19, 44],
            [46],
            [21, 46],
            [22, 42],
            [23, 24],
            [41],
            [25, 28],
            [26, 33],
            [27, 32],
            [34],
            [29],
            [30, 33],
            [31],
            [32, 34],
            [33],
            [],
            [],
            [36, 39],
            [37],
            [38, 40],
            [39],
            [],
            [],
            [42, 45],
            [43],
            [44, 46],
            [45],
            [],
            [],
        ],
    ]
    G = make_small_undirected_graph(description, create_using)
    return G
