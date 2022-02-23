"""
The :mod:`sklearn.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets. It also
features some artificial data generators.
"""
from ._base import load_breast_cancer
from ._base import load_boston
from ._base import load_diabetes
from ._base import load_digits
from ._base import load_files
from ._base import load_iris
from ._base import load_linnerud
from ._base import load_sample_images
from ._base import load_sample_image
from ._base import load_wine
from ._base import get_data_home
from ._base import clear_data_home
from ._covtype import fetch_covtype
from ._kddcup99 import fetch_kddcup99
from ._lfw import fetch_lfw_pairs
from ._lfw import fetch_lfw_people
from ._twenty_newsgroups import fetch_20newsgroups
from ._twenty_newsgroups import fetch_20newsgroups_vectorized
from ._openml import fetch_openml
from ._samples_generator import make_classification
from ._samples_generator import make_multilabel_classification
from ._samples_generator import make_hastie_10_2
from ._samples_generator import make_regression
from ._samples_generator import make_blobs
from ._samples_generator import make_moons
from ._samples_generator import make_circles
from ._samples_generator import make_friedman1
from ._samples_generator import make_friedman2
from ._samples_generator import make_friedman3
from ._samples_generator import make_low_rank_matrix
from ._samples_generator import make_sparse_coded_signal
from ._samples_generator import make_sparse_uncorrelated
from ._samples_generator import make_spd_matrix
from ._samples_generator import make_swiss_roll
from ._samples_generator import make_s_curve
from ._samples_generator import make_sparse_spd_matrix
from ._samples_generator import make_gaussian_quantiles
from ._samples_generator import make_biclusters
from ._samples_generator import make_checkerboard
from ._svmlight_format_io import load_svmlight_file
from ._svmlight_format_io import load_svmlight_files
from ._svmlight_format_io import dump_svmlight_file
from ._olivetti_faces import fetch_olivetti_faces
from ._species_distributions import fetch_species_distributions
from ._california_housing import fetch_california_housing
from ._rcv1 import fetch_rcv1


__all__ = [
    "clear_data_home",
    "dump_svmlight_file",
    "fetch_20newsgroups",
    "fetch_20newsgroups_vectorized",
    "fetch_lfw_pairs",
    "fetch_lfw_people",
    "fetch_olivetti_faces",
    "fetch_species_distributions",
    "fetch_california_housing",
    "fetch_covtype",
    "fetch_rcv1",
    "fetch_kddcup99",
    "fetch_openml",
    "get_data_home",
    "load_boston",
    "load_diabetes",
    "load_digits",
    "load_files",
    "load_iris",
    "load_breast_cancer",
    "load_linnerud",
    "load_sample_image",
    "load_sample_images",
    "load_svmlight_file",
    "load_svmlight_files",
    "load_wine",
    "make_biclusters",
    "make_blobs",
    "make_circles",
    "make_classification",
    "make_checkerboard",
    "make_friedman1",
    "make_friedman2",
    "make_friedman3",
    "make_gaussian_quantiles",
    "make_hastie_10_2",
    "make_low_rank_matrix",
    "make_moons",
    "make_multilabel_classification",
    "make_regression",
    "make_s_curve",
    "make_sparse_coded_signal",
    "make_sparse_spd_matrix",
    "make_sparse_uncorrelated",
    "make_spd_matrix",
    "make_swiss_roll",
]
