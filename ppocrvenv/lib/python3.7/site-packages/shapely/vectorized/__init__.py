"""Provides multi-point element-wise operations such as ``contains``."""

from shapely import speedups

from ._vectorized import (contains, touches)
