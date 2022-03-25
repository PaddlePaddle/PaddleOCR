# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Compute metrics for trackers using MOTChallenge ground-truth data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import OrderedDict
import io
import logging
import os
import sys
from tempfile import NamedTemporaryFile
import time

import motmetrics as mm


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data with data preprocess.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Seqmap for test data
    [name]
    <SEQUENCE_1>
    <SEQUENCE_2>
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string in the seqmap.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('groundtruths', type=str, help='Directory containing ground truth files.')
    parser.add_argument('tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('seqmap', type=str, help='Text file containing all sequences name')
    parser.add_argument('--log', type=str, help='a place to record result and outputfile of mistakes', default='')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    parser.add_argument('--skip', type=int, default=0, help='skip frames n means choosing one frame for every (n+1) frames')
    parser.add_argument('--iou', type=float, default=0.5, help='special IoU threshold requirement for small targets')
    return parser.parse_args()


def compare_dataframes(gts, ts, vsflag='', iou=0.5):
    """Builds accumulator for each sequence."""
    accs = []
    anas = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Evaluating %s...', k)
            if vsflag != '':
                fd = io.open(vsflag + '/' + k + '.log', 'w')
            else:
                fd = ''
            acc, ana = mm.utils.CLEAR_MOT_M(gts[k][0], tsacc, gts[k][1], 'iou', distth=iou, vflag=fd)
            if fd != '':
                fd.close()
            accs.append(acc)
            anas.append(ana)
            names.append(k)
        else:
            logging.warning('No ground truth for %s, skipping.', k)

    return accs, anas, names


def parseSequences(seqmap):
    """Loads list of sequences from file."""
    assert os.path.isfile(seqmap), 'Seqmap %s not found.' % seqmap
    fd = io.open(seqmap)
    res = []
    for row in fd.readlines():
        row = row.strip()
        if row == '' or row == 'name' or row[0] == '#':
            continue
        res.append(row)
    fd.close()
    return res


def generateSkippedGT(gtfile, skip, fmt):
    """Generates temporary ground-truth file with some frames skipped."""
    del fmt  # unused
    tf = NamedTemporaryFile(delete=False, mode='w')
    with io.open(gtfile) as fd:
        lines = fd.readlines()
        for line in lines:
            arr = line.strip().split(',')
            fr = int(arr[0])
            if fr % (skip + 1) != 1:
                continue
            pos = line.find(',')
            newline = str(fr // (skip + 1) + 1) + line[pos:]
            tf.write(newline)
    tf.close()
    tempfile = tf.name
    return tempfile


def main():
    # pylint: disable=missing-function-docstring
    # pylint: disable=too-many-locals
    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    seqs = parseSequences(args.seqmap)
    gtfiles = [os.path.join(args.groundtruths, i, 'gt/gt.txt') for i in seqs]
    tsfiles = [os.path.join(args.tests, '%s.txt' % i) for i in seqs]

    for gtfile in gtfiles:
        if not os.path.isfile(gtfile):
            logging.error('gt File %s not found.', gtfile)
            sys.exit(1)
    for tsfile in tsfiles:
        if not os.path.isfile(tsfile):
            logging.error('res File %s not found.', tsfile)
            sys.exit(1)

    logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
    for seq in seqs:
        logging.info('\t%s', seq)
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    if args.skip > 0 and 'mot' in args.fmt:
        for i, gtfile in enumerate(gtfiles):
            gtfiles[i] = generateSkippedGT(gtfile, args.skip, fmt=args.fmt)

    gt = OrderedDict([(seqs[i], (mm.io.loadtxt(f, fmt=args.fmt), os.path.join(args.groundtruths, seqs[i], 'seqinfo.ini'))) for i, f in enumerate(gtfiles)])
    ts = OrderedDict([(seqs[i], mm.io.loadtxt(f, fmt=args.fmt)) for i, f in enumerate(tsfiles)])

    mh = mm.metrics.create()
    st = time.time()
    accs, analysis, names = compare_dataframes(gt, ts, args.log, 1. - args.iou)
    logging.info('adding frames: %.3f seconds.', time.time() - st)

    logging.info('Running metrics')

    summary = mh.compute_many(accs, anas=analysis, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')


if __name__ == '__main__':
    main()
