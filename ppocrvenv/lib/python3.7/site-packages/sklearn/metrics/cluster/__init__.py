"""
The :mod:`sklearn.metrics.cluster` submodule contains evaluation metrics for
cluster analysis results. There are two forms of evaluation:

- supervised, which uses a ground truth class values for each sample.
- unsupervised, which does not and measures the 'quality' of the model itself.
"""
from ._supervised import adjusted_mutual_info_score
from ._supervised import normalized_mutual_info_score
from ._supervised import adjusted_rand_score
from ._supervised import rand_score
from ._supervised import completeness_score
from ._supervised import contingency_matrix
from ._supervised import pair_confusion_matrix
from ._supervised import expected_mutual_information
from ._supervised import homogeneity_completeness_v_measure
from ._supervised import homogeneity_score
from ._supervised import mutual_info_score
from ._supervised import v_measure_score
from ._supervised import fowlkes_mallows_score
from ._supervised import entropy
from ._unsupervised import silhouette_samples
from ._unsupervised import silhouette_score
from ._unsupervised import calinski_harabasz_score
from ._unsupervised import davies_bouldin_score
from ._bicluster import consensus_score

__all__ = [
    "adjusted_mutual_info_score",
    "normalized_mutual_info_score",
    "adjusted_rand_score",
    "rand_score",
    "completeness_score",
    "pair_confusion_matrix",
    "contingency_matrix",
    "expected_mutual_information",
    "homogeneity_completeness_v_measure",
    "homogeneity_score",
    "mutual_info_score",
    "v_measure_score",
    "fowlkes_mallows_score",
    "entropy",
    "silhouette_samples",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "consensus_score",
]
