"""Functionality with an experimental API. Although you can count on the
functions in this package being around in the future, the API may change with
any version update **and will not follow the skimage two-version deprecation
path**. Therefore, use the functions herein with care, and do not use them in
production code that will depend on updated skimage versions.
"""

from . import graph
from .manual_segmentation import manual_polygon_segmentation
from .manual_segmentation import manual_lasso_segmentation
from .trainable_segmentation import fit_segmenter, predict_segmenter, TrainableSegmenter


__all__ = [
    "graph",
    "manual_lasso_segmentation",
    "manual_polygon_segmentation",
    "fit_segmenter",
    "predict_segmenter",
    "TrainableSegmenter",
]
