# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Functions for loading data and writing summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
import io

import numpy as np
import pandas as pd
import scipy.io
import xmltodict


class Format(Enum):
    """Enumerates supported file formats."""

    MOT16 = 'mot16'
    """Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016)."""

    MOT15_2D = 'mot15-2D'
    """Leal-Taixe, Laura, et al. "MOTChallenge 2015: Towards a benchmark for multi-target tracking." arXiv preprint arXiv:1504.01942 (2015)."""

    VATIC_TXT = 'vatic-txt'
    """Vondrick, Carl, Donald Patterson, and Deva Ramanan. "Efficiently scaling up crowdsourced video annotation." International Journal of Computer Vision 101.1 (2013): 184-204.
    https://github.com/cvondrick/vatic
    """

    DETRAC_MAT = 'detrac-mat'
    """Wen, Longyin et al. "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking." arXiv preprint arXiv:arXiv:1511.04136 (2016).
    http://detrac-db.rit.albany.edu/download
    """

    DETRAC_XML = 'detrac-xml'
    """Wen, Longyin et al. "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking." arXiv preprint arXiv:arXiv:1511.04136 (2016).
    http://detrac-db.rit.albany.edu/download
    """


def load_motchallenge(fname, **kwargs):
    r"""Load MOT challenge data.

    Params
    ------
    fname : str
        Filename to load data from

    Kwargs
    ------
    sep : str
        Allowed field separators, defaults to '\s+|\t+|,'
    min_confidence : float
        Rows with confidence less than this threshold are removed.
        Defaults to -1. You should set this to 1 when loading
        ground truth MOTChallenge data, so that invalid rectangles in
        the ground truth are not considered during matching.

    Returns
    ------
    df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
        The dataframe is indexed by ('FrameId', 'Id')
    """

    sep = kwargs.pop('sep', r'\s+|\t+|,')
    min_confidence = kwargs.pop('min_confidence', -1)
    df = pd.read_csv(
        fname,
        sep=sep,
        index_col=[0, 1],
        skipinitialspace=True,
        header=None,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'],
        engine='python'
    )

    # Account for matlab convention.
    df[['X', 'Y']] -= (1, 1)

    # Removed trailing column
    del df['unused']

    # Remove all rows without sufficient confidence
    return df[df['Confidence'] >= min_confidence]


def load_vatictxt(fname, **kwargs):
    """Load Vatic text format.

    Loads the vatic CSV text having the following columns per row

        0   Track ID. All rows with the same ID belong to the same path.
        1   xmin. The top left x-coordinate of the bounding box.
        2   ymin. The top left y-coordinate of the bounding box.
        3   xmax. The bottom right x-coordinate of the bounding box.
        4   ymax. The bottom right y-coordinate of the bounding box.
        5   frame. The frame that this annotation represents.
        6   lost. If 1, the annotation is outside of the view screen.
        7   occluded. If 1, the annotation is occluded.
        8   generated. If 1, the annotation was automatically interpolated.
        9  label. The label for this annotation, enclosed in quotation marks.
        10+ attributes. Each column after this is an attribute set in the current frame

    Params
    ------
    fname : str
        Filename to load data from

    Returns
    ------
    df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Lost', 'Occluded', 'Generated', 'ClassId', '<Attr1>', '<Attr2>', ...
        where <Attr1> is placeholder for the actual attribute name capitalized (first letter). The order of attribute
        columns is sorted in attribute name. The dataframe is indexed by ('FrameId', 'Id')
    """
    # pylint: disable=too-many-locals

    sep = kwargs.pop('sep', ' ')

    with io.open(fname) as f:
        # First time going over file, we collect the set of all variable activities
        activities = set()
        for line in f:
            for c in line.rstrip().split(sep)[10:]:
                activities.add(c)
        activitylist = sorted(list(activities))

        # Second time we construct artificial binary columns for each activity
        data = []
        f.seek(0)
        for line in f:
            fields = line.rstrip().split()
            attrs = ['0'] * len(activitylist)
            for a in fields[10:]:
                attrs[activitylist.index(a)] = '1'
            fields = fields[:10]
            fields.extend(attrs)
            data.append(' '.join(fields))

        strdata = '\n'.join(data)

        dtype = {
            'Id': np.int64,
            'X': np.float32,
            'Y': np.float32,
            'Width': np.float32,
            'Height': np.float32,
            'FrameId': np.int64,
            'Lost': bool,
            'Occluded': bool,
            'Generated': bool,
            'ClassId': str,
        }

        # Remove quotes from activities
        activitylist = [a.replace('\"', '').capitalize() for a in activitylist]

        # Add dtypes for activities
        for a in activitylist:
            dtype[a] = bool

        # Read from CSV
        names = ['Id', 'X', 'Y', 'Width', 'Height', 'FrameId', 'Lost', 'Occluded', 'Generated', 'ClassId']
        names.extend(activitylist)
        df = pd.read_csv(io.StringIO(strdata), names=names, index_col=['FrameId', 'Id'], header=None, sep=' ')

        # Correct Width and Height which are actually XMax, Ymax in files.
        w = df['Width'] - df['X']
        h = df['Height'] - df['Y']
        df['Width'] = w
        df['Height'] = h

        return df


def load_detrac_mat(fname):
    """Loads UA-DETRAC annotations data from mat files

    Competition Site: http://detrac-db.rit.albany.edu/download

    File contains a nested structure of 2d arrays for indexed by frame id
    and Object ID. Separate arrays for top, left, width and height are given.

    Params
    ------
    fname : str
        Filename to load data from

    Kwargs
    ------
    Currently none of these arguments used.

    Returns
    ------
    df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
        The dataframe is indexed by ('FrameId', 'Id')
    """

    matData = scipy.io.loadmat(fname)

    frameList = matData['gtInfo'][0][0][4][0]
    leftArray = matData['gtInfo'][0][0][0]
    topArray = matData['gtInfo'][0][0][1]
    widthArray = matData['gtInfo'][0][0][3]
    heightArray = matData['gtInfo'][0][0][2]

    parsedGT = []
    for f in frameList:
        ids = [i + 1 for i, v in enumerate(leftArray[f - 1]) if v > 0]
        for i in ids:
            row = []
            row.append(f)
            row.append(i)
            row.append(leftArray[f - 1, i - 1] - widthArray[f - 1, i - 1] / 2)
            row.append(topArray[f - 1, i - 1] - heightArray[f - 1, i - 1])
            row.append(widthArray[f - 1, i - 1])
            row.append(heightArray[f - 1, i - 1])
            row.append(1)
            row.append(-1)
            row.append(-1)
            row.append(-1)
            parsedGT.append(row)

    df = pd.DataFrame(parsedGT,
                      columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'])
    df.set_index(['FrameId', 'Id'], inplace=True)

    # Account for matlab convention.
    df[['X', 'Y']] -= (1, 1)

    # Removed trailing column
    del df['unused']

    return df


def load_detrac_xml(fname):
    """Loads UA-DETRAC annotations data from xml files

    Competition Site: http://detrac-db.rit.albany.edu/download

    Params
    ------
    fname : str
        Filename to load data from

    Kwargs
    ------
    Currently none of these arguments used.

    Returns
    ------
    df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
        The dataframe is indexed by ('FrameId', 'Id')
    """

    with io.open(fname) as fd:
        doc = xmltodict.parse(fd.read())
    frameList = doc['sequence']['frame']

    parsedGT = []
    for f in frameList:
        fid = int(f['@num'])
        targetList = f['target_list']['target']
        if not isinstance(targetList, list):
            targetList = [targetList]

        for t in targetList:
            row = []
            row.append(fid)
            row.append(int(t['@id']))
            row.append(float(t['box']['@left']))
            row.append(float(t['box']['@top']))
            row.append(float(t['box']['@width']))
            row.append(float(t['box']['@height']))
            row.append(1)
            row.append(-1)
            row.append(-1)
            row.append(-1)
            parsedGT.append(row)

    df = pd.DataFrame(parsedGT,
                      columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'])
    df.set_index(['FrameId', 'Id'], inplace=True)

    # Account for matlab convention.
    df[['X', 'Y']] -= (1, 1)

    # Removed trailing column
    del df['unused']

    return df


def loadtxt(fname, fmt=Format.MOT15_2D, **kwargs):
    """Load data from any known format."""
    fmt = Format(fmt)

    switcher = {
        Format.MOT16: load_motchallenge,
        Format.MOT15_2D: load_motchallenge,
        Format.VATIC_TXT: load_vatictxt,
        Format.DETRAC_MAT: load_detrac_mat,
        Format.DETRAC_XML: load_detrac_xml
    }
    func = switcher.get(fmt)
    return func(fname, **kwargs)


def render_summary(summary, formatters=None, namemap=None, buf=None):
    """Render metrics summary to console friendly tabular output.

    Params
    ------
    summary : pd.DataFrame
        Dataframe containing summaries in rows.

    Kwargs
    ------
    buf : StringIO-like, optional
        Buffer to write to
    formatters : dict, optional
        Dicionary defining custom formatters for individual metrics.
        I.e `{'mota': '{:.2%}'.format}`. You can get preset formatters
        from MetricsHost.formatters
    namemap : dict, optional
        Dictionary defining new metric names for display. I.e
        `{'num_false_positives': 'FP'}`.

    Returns
    -------
    string
        Formatted string
    """

    if namemap is not None:
        summary = summary.rename(columns=namemap)
        if formatters is not None:
            formatters = {namemap.get(c, c): f for c, f in formatters.items()}

    output = summary.to_string(
        buf=buf,
        formatters=formatters,
    )

    return output


motchallenge_metric_names = {
    'idf1': 'IDF1',
    'idp': 'IDP',
    'idr': 'IDR',
    'recall': 'Rcll',
    'precision': 'Prcn',
    'num_unique_objects': 'GT',
    'mostly_tracked': 'MT',
    'partially_tracked': 'PT',
    'mostly_lost': 'ML',
    'num_false_positives': 'FP',
    'num_misses': 'FN',
    'num_switches': 'IDs',
    'num_fragmentations': 'FM',
    'mota': 'MOTA',
    'motp': 'MOTP',
    'num_transfer': 'IDt',
    'num_ascend': 'IDa',
    'num_migrate': 'IDm',
}
"""A list mappings for metric names to comply with MOTChallenge."""
