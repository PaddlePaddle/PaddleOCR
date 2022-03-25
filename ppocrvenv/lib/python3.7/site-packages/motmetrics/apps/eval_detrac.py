# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Compute metrics for trackers using DETRAC challenge ground-truth data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path

import motmetrics as mm


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using DETRAC challenge ground-truth data.

Files
-----
Ground truth files can be in .XML format or .MAT format as provided by http://detrac-db.rit.albany.edu/download

Test Files for the challenge are reuired to be in MOTchallenge format, they have to comply with the format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Directory Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>.txt
    <GT_ROOT>/<SEQUENCE_2>.txt
    ...

    OR
    <GT_ROOT>/<SEQUENCE_1>.mat
    <GT_ROOT>/<SEQUENCE_2>.mat
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('groundtruths', type=str, help='Directory containing ground truth files.')
    parser.add_argument('tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--gtfmt', type=str, help='Groundtruth data format', default='detrac-xml')
    parser.add_argument('--tsfmt', type=str, help='Test data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    return parser.parse_args()


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing %s...', k)
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for %s, skipping.', k)

    return accs, names


def main():
    # pylint: disable=missing-function-docstring
    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    gtfiles = glob.glob(os.path.join(args.groundtruths, '*'))
    tsfiles = glob.glob(os.path.join(args.tests, '*'))

    logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    gt = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.gtfmt)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.tsfmt)) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    logging.info('Running metrics')

    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')


if __name__ == '__main__':
    main()
