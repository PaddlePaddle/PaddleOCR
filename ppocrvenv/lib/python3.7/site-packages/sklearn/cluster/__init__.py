"""
The :mod:`sklearn.cluster` module gathers popular unsupervised clustering
algorithms.
"""

from ._spectral import spectral_clustering, SpectralClustering
from ._mean_shift import mean_shift, MeanShift, estimate_bandwidth, get_bin_seeds
from ._affinity_propagation import affinity_propagation, AffinityPropagation
from ._agglomerative import (
    ward_tree,
    AgglomerativeClustering,
    linkage_tree,
    FeatureAgglomeration,
)
from ._kmeans import k_means, KMeans, MiniBatchKMeans, kmeans_plusplus
from ._dbscan import dbscan, DBSCAN
from ._optics import (
    OPTICS,
    cluster_optics_dbscan,
    compute_optics_graph,
    cluster_optics_xi,
)
from ._bicluster import SpectralBiclustering, SpectralCoclustering
from ._birch import Birch

__all__ = [
    "AffinityPropagation",
    "AgglomerativeClustering",
    "Birch",
    "DBSCAN",
    "OPTICS",
    "cluster_optics_dbscan",
    "cluster_optics_xi",
    "compute_optics_graph",
    "KMeans",
    "FeatureAgglomeration",
    "MeanShift",
    "MiniBatchKMeans",
    "SpectralClustering",
    "affinity_propagation",
    "dbscan",
    "estimate_bandwidth",
    "get_bin_seeds",
    "k_means",
    "kmeans_plusplus",
    "linkage_tree",
    "mean_shift",
    "spectral_clustering",
    "ward_tree",
    "SpectralBiclustering",
    "SpectralCoclustering",
]
