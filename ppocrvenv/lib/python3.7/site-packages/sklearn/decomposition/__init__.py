"""
The :mod:`sklearn.decomposition` module includes matrix decomposition
algorithms, including among others PCA, NMF or ICA. Most of the algorithms of
this module can be regarded as dimensionality reduction techniques.
"""


from ._nmf import NMF, non_negative_factorization
from ._pca import PCA
from ._incremental_pca import IncrementalPCA
from ._kernel_pca import KernelPCA
from ._sparse_pca import SparsePCA, MiniBatchSparsePCA
from ._truncated_svd import TruncatedSVD
from ._fastica import FastICA, fastica
from ._dict_learning import (
    dict_learning,
    dict_learning_online,
    sparse_encode,
    DictionaryLearning,
    MiniBatchDictionaryLearning,
    SparseCoder,
)
from ._factor_analysis import FactorAnalysis
from ..utils.extmath import randomized_svd
from ._lda import LatentDirichletAllocation


__all__ = [
    "DictionaryLearning",
    "FastICA",
    "IncrementalPCA",
    "KernelPCA",
    "MiniBatchDictionaryLearning",
    "MiniBatchSparsePCA",
    "NMF",
    "PCA",
    "SparseCoder",
    "SparsePCA",
    "dict_learning",
    "dict_learning_online",
    "fastica",
    "non_negative_factorization",
    "randomized_svd",
    "sparse_encode",
    "FactorAnalysis",
    "TruncatedSVD",
    "LatentDirichletAllocation",
]
