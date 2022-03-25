# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Functions for populating event accumulators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from motmetrics.distances import iou_matrix, norm2squared_matrix
from motmetrics.mot import MOTAccumulator
from motmetrics.preprocess import preprocessResult


def compare_to_groundtruth(gt, dt, dist='iou', distfields=None, distth=0.5):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids

    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results

    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """
    # pylint: disable=too-many-locals
    if distfields is None:
        distfields = ['X', 'Y', 'Width', 'Height']

    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)

    compute_dist = compute_iou if dist.upper() == 'IOU' else compute_euc

    acc = MOTAccumulator()

    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]

    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0, 0))

        if fid in gt.index:
            fgt = gt.loc[fid]
            oids = fgt.index.values

        if fid in dt.index:
            fdt = dt.loc[fid]
            hids = fdt.index.values

        if oids.shape[0] > 0 and hids.shape[0] > 0:
            dists = compute_dist(fgt[distfields].values, fdt[distfields].values)

        acc.update(oids, hids, dists, frameid=fid)

    return acc


def CLEAR_MOT_M(gt, dt, inifile, dist='iou', distfields=None, distth=0.5, include_all=False, vflag=''):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids

    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results

    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """
    # pylint: disable=too-many-locals
    if distfields is None:
        distfields = ['X', 'Y', 'Width', 'Height']

    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)

    compute_dist = compute_iou if dist.upper() == 'IOU' else compute_euc

    acc = MOTAccumulator()
    dt = preprocessResult(dt, gt, inifile)
    if include_all:
        gt = gt[gt['Confidence'] >= 0.99]
    else:
        gt = gt[(gt['Confidence'] >= 0.99) & (gt['ClassId'] == 1)]
    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]
    analysis = {'hyp': {}, 'obj': {}}
    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0, 0))

        if fid in gt.index:
            fgt = gt.loc[fid]
            oids = fgt.index.values
            for oid in oids:
                oid = int(oid)
                if oid not in analysis['obj']:
                    analysis['obj'][oid] = 0
                analysis['obj'][oid] += 1

        if fid in dt.index:
            fdt = dt.loc[fid]
            hids = fdt.index.values
            for hid in hids:
                hid = int(hid)
                if hid not in analysis['hyp']:
                    analysis['hyp'][hid] = 0
                analysis['hyp'][hid] += 1

        if oids.shape[0] > 0 and hids.shape[0] > 0:
            dists = compute_dist(fgt[distfields].values, fdt[distfields].values)

        acc.update(oids, hids, dists, frameid=fid, vf=vflag)

    return acc, analysis
