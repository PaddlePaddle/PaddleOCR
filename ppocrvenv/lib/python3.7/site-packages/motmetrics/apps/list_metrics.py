# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""List metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == '__main__':
    import motmetrics

    mh = motmetrics.metrics.create()
    print(mh.list_metrics_markdown())
