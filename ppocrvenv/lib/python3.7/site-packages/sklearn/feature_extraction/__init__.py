"""
The :mod:`sklearn.feature_extraction` module deals with feature extraction
from raw data. It currently includes methods to extract features from text and
images.
"""

from ._dict_vectorizer import DictVectorizer
from ._hash import FeatureHasher
from .image import img_to_graph, grid_to_graph
from . import text

__all__ = [
    "DictVectorizer",
    "image",
    "img_to_graph",
    "grid_to_graph",
    "text",
    "FeatureHasher",
]
