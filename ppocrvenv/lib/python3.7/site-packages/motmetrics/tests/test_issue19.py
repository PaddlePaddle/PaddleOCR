# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests issue 19.

https://github.com/cheind/py-motmetrics/issues/19
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import motmetrics as mm


def test_issue19():
    """Tests issue 19."""
    acc = mm.MOTAccumulator()

    g0 = [0, 1]
    p0 = [0, 1]
    d0 = [[0.2, np.nan], [np.nan, 0.2]]

    g1 = [2, 3]
    p1 = [2, 3, 4, 5]
    d1 = [[0.28571429, 0.5, 0.0, np.nan], [np.nan, 0.44444444, np.nan, 0.0]]

    acc.update(g0, p0, d0, 0)
    acc.update(g1, p1, d1, 1)

    mh = mm.metrics.create()
    mh.compute(acc)
