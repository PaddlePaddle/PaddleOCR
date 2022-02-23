# TODO: Remove this file in 1.3
import warnings

from ..metrics import DistanceMetric as _DistanceMetric


class DistanceMetric(_DistanceMetric):
    @classmethod
    def _warn(cls):
        warnings.warn(
            "sklearn.neighbors.DistanceMetric has been moved "
            "to sklearn.metrics.DistanceMetric in 1.0. "
            "This import path will be removed in 1.3",
            category=FutureWarning,
        )

    @classmethod
    def get_metric(cls, metric, **kwargs):
        DistanceMetric._warn()
        return _DistanceMetric.get_metric(metric, **kwargs)
