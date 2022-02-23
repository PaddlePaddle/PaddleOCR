"""Function for computing Local and global consistency algorithm by Zhou et al.

References
----------
Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
Learning with local and global consistency.
Advances in neural information processing systems, 16(16), 321-328.
"""
import networkx as nx

from networkx.utils.decorators import not_implemented_for
from networkx.algorithms.node_classification.utils import (
    _get_label_info,
    _init_label_matrix,
    _propagate,
    _predict,
)

__all__ = ["local_and_global_consistency"]


@not_implemented_for("directed")
def local_and_global_consistency(G, alpha=0.99, max_iter=30, label_name="label"):
    """Node classification by Local and Global Consistency

    Parameters
    ----------
    G : NetworkX Graph
    alpha : float
        Clamping factor
    max_iter : int
        Maximum number of iterations allowed
    label_name : string
        Name of target labels to predict

    Returns
    -------
    predicted : array, shape = [n_samples]
        Array of predicted labels

    Raises
    ------
    NetworkXError
        If no nodes on `G` has `label_name`.

    Examples
    --------
    >>> from networkx.algorithms import node_classification
    >>> G = nx.path_graph(4)
    >>> G.nodes[0]["label"] = "A"
    >>> G.nodes[3]["label"] = "B"
    >>> G.nodes(data=True)
    NodeDataView({0: {'label': 'A'}, 1: {}, 2: {}, 3: {'label': 'B'}})
    >>> G.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    >>> predicted = node_classification.local_and_global_consistency(G)
    >>> predicted
    ['A', 'A', 'B', 'B']


    References
    ----------
    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency.
    Advances in neural information processing systems, 16(16), 321-328.
    """
    import numpy as np
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    def _build_propagation_matrix(X, labels, alpha):
        """Build propagation matrix of Local and global consistency

        Parameters
        ----------
        X : scipy sparse matrix, shape = [n_samples, n_samples]
            Adjacency matrix
        labels : array, shape = [n_samples, 2]
            Array of pairs of node id and label id
        alpha : float
            Clamping factor

        Returns
        -------
        S : scipy sparse matrix, shape = [n_samples, n_samples]
            Propagation matrix

        """
        degrees = X.sum(axis=0).A[0]
        degrees[degrees == 0] = 1  # Avoid division by 0
        D2 = np.sqrt(sp.sparse.diags((1.0 / degrees), offsets=0))
        S = alpha * ((D2 @ X) @ D2)
        return S

    def _build_base_matrix(X, labels, alpha, n_classes):
        """Build base matrix of Local and global consistency

        Parameters
        ----------
        X : scipy sparse matrix, shape = [n_samples, n_samples]
            Adjacency matrix
        labels : array, shape = [n_samples, 2]
            Array of pairs of node id and label id
        alpha : float
            Clamping factor
        n_classes : integer
            The number of classes (distinct labels) on the input graph

        Returns
        -------
        B : array, shape = [n_samples, n_classes]
            Base matrix
        """

        n_samples = X.shape[0]
        B = np.zeros((n_samples, n_classes))
        B[labels[:, 0], labels[:, 1]] = 1 - alpha
        return B

    X = nx.to_scipy_sparse_matrix(G)  # adjacency matrix
    labels, label_dict = _get_label_info(G, label_name)

    if labels.shape[0] == 0:
        raise nx.NetworkXError(
            "No node on the input graph is labeled by '" + label_name + "'."
        )

    n_samples = X.shape[0]
    n_classes = label_dict.shape[0]
    F = _init_label_matrix(n_samples, n_classes)

    P = _build_propagation_matrix(X, labels, alpha)
    B = _build_base_matrix(X, labels, alpha, n_classes)

    remaining_iter = max_iter
    while remaining_iter > 0:
        F = _propagate(P, F, B)
        remaining_iter -= 1

    predicted = _predict(F, label_dict)

    return predicted
