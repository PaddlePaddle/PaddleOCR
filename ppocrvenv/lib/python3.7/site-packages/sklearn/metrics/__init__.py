"""
The :mod:`sklearn.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.
"""


from ._ranking import auc
from ._ranking import average_precision_score
from ._ranking import coverage_error
from ._ranking import det_curve
from ._ranking import dcg_score
from ._ranking import label_ranking_average_precision_score
from ._ranking import label_ranking_loss
from ._ranking import ndcg_score
from ._ranking import precision_recall_curve
from ._ranking import roc_auc_score
from ._ranking import roc_curve
from ._ranking import top_k_accuracy_score

from ._classification import accuracy_score
from ._classification import balanced_accuracy_score
from ._classification import classification_report
from ._classification import cohen_kappa_score
from ._classification import confusion_matrix
from ._classification import f1_score
from ._classification import fbeta_score
from ._classification import hamming_loss
from ._classification import hinge_loss
from ._classification import jaccard_score
from ._classification import log_loss
from ._classification import matthews_corrcoef
from ._classification import precision_recall_fscore_support
from ._classification import precision_score
from ._classification import recall_score
from ._classification import zero_one_loss
from ._classification import brier_score_loss
from ._classification import multilabel_confusion_matrix

from ._dist_metrics import DistanceMetric

from . import cluster
from .cluster import adjusted_mutual_info_score
from .cluster import adjusted_rand_score
from .cluster import rand_score
from .cluster import pair_confusion_matrix
from .cluster import completeness_score
from .cluster import consensus_score
from .cluster import homogeneity_completeness_v_measure
from .cluster import homogeneity_score
from .cluster import mutual_info_score
from .cluster import normalized_mutual_info_score
from .cluster import fowlkes_mallows_score
from .cluster import silhouette_samples
from .cluster import silhouette_score
from .cluster import calinski_harabasz_score
from .cluster import v_measure_score
from .cluster import davies_bouldin_score

from .pairwise import euclidean_distances
from .pairwise import nan_euclidean_distances
from .pairwise import pairwise_distances
from .pairwise import pairwise_distances_argmin
from .pairwise import pairwise_distances_argmin_min
from .pairwise import pairwise_kernels
from .pairwise import pairwise_distances_chunked

from ._regression import explained_variance_score
from ._regression import max_error
from ._regression import mean_absolute_error
from ._regression import mean_squared_error
from ._regression import mean_squared_log_error
from ._regression import median_absolute_error
from ._regression import mean_absolute_percentage_error
from ._regression import mean_pinball_loss
from ._regression import r2_score
from ._regression import mean_tweedie_deviance
from ._regression import mean_poisson_deviance
from ._regression import mean_gamma_deviance
from ._regression import d2_tweedie_score


from ._scorer import check_scoring
from ._scorer import make_scorer
from ._scorer import SCORERS
from ._scorer import get_scorer

from ._plot.det_curve import plot_det_curve
from ._plot.det_curve import DetCurveDisplay
from ._plot.roc_curve import plot_roc_curve
from ._plot.roc_curve import RocCurveDisplay
from ._plot.precision_recall_curve import plot_precision_recall_curve
from ._plot.precision_recall_curve import PrecisionRecallDisplay

from ._plot.confusion_matrix import plot_confusion_matrix
from ._plot.confusion_matrix import ConfusionMatrixDisplay


__all__ = [
    "accuracy_score",
    "adjusted_mutual_info_score",
    "adjusted_rand_score",
    "auc",
    "average_precision_score",
    "balanced_accuracy_score",
    "calinski_harabasz_score",
    "check_scoring",
    "classification_report",
    "cluster",
    "cohen_kappa_score",
    "completeness_score",
    "ConfusionMatrixDisplay",
    "confusion_matrix",
    "consensus_score",
    "coverage_error",
    "d2_tweedie_score",
    "dcg_score",
    "davies_bouldin_score",
    "DetCurveDisplay",
    "det_curve",
    "DistanceMetric",
    "euclidean_distances",
    "explained_variance_score",
    "f1_score",
    "fbeta_score",
    "fowlkes_mallows_score",
    "get_scorer",
    "hamming_loss",
    "hinge_loss",
    "homogeneity_completeness_v_measure",
    "homogeneity_score",
    "jaccard_score",
    "label_ranking_average_precision_score",
    "label_ranking_loss",
    "log_loss",
    "make_scorer",
    "nan_euclidean_distances",
    "matthews_corrcoef",
    "max_error",
    "mean_absolute_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "mean_pinball_loss",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "mean_tweedie_deviance",
    "median_absolute_error",
    "mean_absolute_percentage_error",
    "multilabel_confusion_matrix",
    "mutual_info_score",
    "ndcg_score",
    "normalized_mutual_info_score",
    "pair_confusion_matrix",
    "pairwise_distances",
    "pairwise_distances_argmin",
    "pairwise_distances_argmin_min",
    "pairwise_distances_chunked",
    "pairwise_kernels",
    "plot_confusion_matrix",
    "plot_det_curve",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "PrecisionRecallDisplay",
    "precision_recall_curve",
    "precision_recall_fscore_support",
    "precision_score",
    "r2_score",
    "rand_score",
    "recall_score",
    "RocCurveDisplay",
    "roc_auc_score",
    "roc_curve",
    "SCORERS",
    "silhouette_samples",
    "silhouette_score",
    "top_k_accuracy_score",
    "v_measure_score",
    "zero_one_loss",
    "brier_score_loss",
]
