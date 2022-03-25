# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests computation of metrics from accumulator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
from pytest import approx

import motmetrics as mm

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')


def test_metricscontainer_1():
    """Tests registration of events with dependencies."""
    m = mm.metrics.MetricsHost()
    m.register(lambda df: 1., name='a')
    m.register(lambda df: 2., name='b')
    m.register(lambda df, a, b: a + b, deps=['a', 'b'], name='add')
    m.register(lambda df, a, b: a - b, deps=['a', 'b'], name='sub')
    m.register(lambda df, a, b: a * b, deps=['add', 'sub'], name='mul')
    summary = m.compute(mm.MOTAccumulator.new_event_dataframe(), metrics=['mul', 'add'], name='x')
    assert summary.columns.values.tolist() == ['mul', 'add']
    assert summary.iloc[0]['mul'] == -3.
    assert summary.iloc[0]['add'] == 3.


def test_metricscontainer_autodep():
    """Tests automatic dependencies from argument names."""
    m = mm.metrics.MetricsHost()
    m.register(lambda df: 1., name='a')
    m.register(lambda df: 2., name='b')
    m.register(lambda df, a, b: a + b, name='add', deps='auto')
    m.register(lambda df, a, b: a - b, name='sub', deps='auto')
    m.register(lambda df, add, sub: add * sub, name='mul', deps='auto')
    summary = m.compute(mm.MOTAccumulator.new_event_dataframe(), metrics=['mul', 'add'])
    assert summary.columns.values.tolist() == ['mul', 'add']
    assert summary.iloc[0]['mul'] == -3.
    assert summary.iloc[0]['add'] == 3.


def test_metricscontainer_autoname():
    """Tests automatic names (and dependencies) from inspection."""

    def constant_a(_):
        """Constant a help."""
        return 1.

    def constant_b(_):
        return 2.

    def add(_, constant_a, constant_b):
        return constant_a + constant_b

    def sub(_, constant_a, constant_b):
        return constant_a - constant_b

    def mul(_, add, sub):
        return add * sub

    m = mm.metrics.MetricsHost()
    m.register(constant_a, deps='auto')
    m.register(constant_b, deps='auto')
    m.register(add, deps='auto')
    m.register(sub, deps='auto')
    m.register(mul, deps='auto')

    assert m.metrics['constant_a']['help'] == 'Constant a help.'

    summary = m.compute(mm.MOTAccumulator.new_event_dataframe(), metrics=['mul', 'add'])
    assert summary.columns.values.tolist() == ['mul', 'add']
    assert summary.iloc[0]['mul'] == -3.
    assert summary.iloc[0]['add'] == 3.


def test_metrics_with_no_events():
    """Tests metrics when accumulator is empty."""
    acc = mm.MOTAccumulator()

    mh = mm.metrics.create()
    metr = mh.compute(acc, return_dataframe=False, return_cached=True, metrics=[
        'mota', 'motp', 'num_predictions', 'num_objects', 'num_detections', 'num_frames',
    ])
    assert np.isnan(metr['mota'])
    assert np.isnan(metr['motp'])
    assert metr['num_predictions'] == 0
    assert metr['num_objects'] == 0
    assert metr['num_detections'] == 0
    assert metr['num_frames'] == 0


def test_assignment_metrics_with_empty_groundtruth():
    """Tests metrics when there are no ground-truth objects."""
    acc = mm.MOTAccumulator(auto_id=True)
    # Empty groundtruth.
    acc.update([], [1, 2, 3, 4], [])
    acc.update([], [1, 2, 3, 4], [])
    acc.update([], [1, 2, 3, 4], [])
    acc.update([], [1, 2, 3, 4], [])

    mh = mm.metrics.create()
    metr = mh.compute(acc, return_dataframe=False, metrics=[
        'num_matches', 'num_false_positives', 'num_misses',
        'idtp', 'idfp', 'idfn', 'num_frames',
    ])
    assert metr['num_matches'] == 0
    assert metr['num_false_positives'] == 16
    assert metr['num_misses'] == 0
    assert metr['idtp'] == 0
    assert metr['idfp'] == 16
    assert metr['idfn'] == 0
    assert metr['num_frames'] == 4


def test_assignment_metrics_with_empty_predictions():
    """Tests metrics when there are no predictions."""
    acc = mm.MOTAccumulator(auto_id=True)
    # Empty predictions.
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])

    mh = mm.metrics.create()
    metr = mh.compute(acc, return_dataframe=False, metrics=[
        'num_matches', 'num_false_positives', 'num_misses',
        'idtp', 'idfp', 'idfn', 'num_frames',
    ])
    assert metr['num_matches'] == 0
    assert metr['num_false_positives'] == 0
    assert metr['num_misses'] == 16
    assert metr['idtp'] == 0
    assert metr['idfp'] == 0
    assert metr['idfn'] == 16
    assert metr['num_frames'] == 4


def test_assignment_metrics_with_both_empty():
    """Tests metrics when there are no ground-truth objects or predictions."""
    acc = mm.MOTAccumulator(auto_id=True)
    # Empty groundtruth and empty predictions.
    acc.update([], [], [])
    acc.update([], [], [])
    acc.update([], [], [])
    acc.update([], [], [])

    mh = mm.metrics.create()
    metr = mh.compute(acc, return_dataframe=False, metrics=[
        'num_matches', 'num_false_positives', 'num_misses',
        'idtp', 'idfp', 'idfn', 'num_frames',
    ])
    assert metr['num_matches'] == 0
    assert metr['num_false_positives'] == 0
    assert metr['num_misses'] == 0
    assert metr['idtp'] == 0
    assert metr['idfp'] == 0
    assert metr['idfn'] == 0
    assert metr['num_frames'] == 4


def _extract_counts(acc):
    df_map = mm.metrics.events_to_df_map(acc.events)
    return mm.metrics.extract_counts_from_df_map(df_map)


def test_extract_counts():
    """Tests events_to_df_map() and extract_counts_from_df_map()."""
    acc = mm.MOTAccumulator()
    # All FP
    acc.update([], [1, 2], [], frameid=0)
    # All miss
    acc.update([1, 2], [], [], frameid=1)
    # Match
    acc.update([1, 2], [1, 2], [[1, 0.5], [0.3, 1]], frameid=2)
    # Switch
    acc.update([1, 2], [1, 2], [[0.2, np.nan], [np.nan, 0.1]], frameid=3)
    # Match. Better new match is available but should prefer history
    acc.update([1, 2], [1, 2], [[5, 1], [1, 5]], frameid=4)
    # No data
    acc.update([], [], [], frameid=5)

    ocs, hcs, tps = _extract_counts(acc)

    assert ocs == {1: 4, 2: 4}
    assert hcs == {1: 4, 2: 4}
    expected_tps = {
        (1, 1): 3,
        (1, 2): 2,
        (2, 1): 2,
        (2, 2): 3,
    }
    assert tps == expected_tps


def test_extract_pandas_series_issue():
    """Reproduce issue that arises with pd.Series but not pd.DataFrame.

    >>> data = [[0, 1, 0.1], [0, 1, 0.2], [0, 1, 0.3]]
    >>> df = pd.DataFrame(data, columns=['x', 'y', 'z']).set_index(['x', 'y'])
    >>> df['z'].groupby(['x', 'y']).count()
    {(0, 1): 3}

    >>> data = [[0, 1, 0.1], [0, 1, 0.2]]
    >>> df = pd.DataFrame(data, columns=['x', 'y', 'z']).set_index(['x', 'y'])
    >>> df['z'].groupby(['x', 'y']).count()
    {'x': 1, 'y': 1}

    >>> df[['z']].groupby(['x', 'y'])['z'].count().to_dict()
    {(0, 1): 2}
    """
    acc = mm.MOTAccumulator(auto_id=True)
    acc.update([0], [1], [[0.1]])
    acc.update([0], [1], [[0.1]])
    ocs, hcs, tps = _extract_counts(acc)
    assert ocs == {0: 2}
    assert hcs == {1: 2}
    assert tps == {(0, 1): 2}


def test_benchmark_extract_counts(benchmark):
    """Benchmarks events_to_df_map() and extract_counts_from_df_map()."""
    rand = np.random.RandomState(0)
    acc = _accum_random_uniform(
        rand, seq_len=100, num_objs=50, num_hyps=5000,
        objs_per_frame=20, hyps_per_frame=40)
    benchmark(_extract_counts, acc)


def _accum_random_uniform(rand, seq_len, num_objs, num_hyps, objs_per_frame, hyps_per_frame):
    acc = mm.MOTAccumulator(auto_id=True)
    for _ in range(seq_len):
        # Choose subset of objects present in this frame.
        objs = rand.choice(num_objs, objs_per_frame, replace=False)
        # Choose subset of hypotheses present in this frame.
        hyps = rand.choice(num_hyps, hyps_per_frame, replace=False)
        dist = rand.uniform(size=(objs_per_frame, hyps_per_frame))
        acc.update(objs, hyps, dist)
    return acc


def test_mota_motp():
    """Tests values of MOTA and MOTP."""
    acc = mm.MOTAccumulator()

    # All FP
    acc.update([], [1, 2], [], frameid=0)
    # All miss
    acc.update([1, 2], [], [], frameid=1)
    # Match
    acc.update([1, 2], [1, 2], [[1, 0.5], [0.3, 1]], frameid=2)
    # Switch
    acc.update([1, 2], [1, 2], [[0.2, np.nan], [np.nan, 0.1]], frameid=3)
    # Match. Better new match is available but should prefer history
    acc.update([1, 2], [1, 2], [[5, 1], [1, 5]], frameid=4)
    # No data
    acc.update([], [], [], frameid=5)

    mh = mm.metrics.create()
    metr = mh.compute(acc, return_dataframe=False, return_cached=True, metrics=[
        'num_matches', 'num_false_positives', 'num_misses', 'num_switches', 'num_detections',
        'num_objects', 'num_predictions', 'mota', 'motp', 'num_frames'
    ])

    assert metr['num_matches'] == 4
    assert metr['num_false_positives'] == 2
    assert metr['num_misses'] == 2
    assert metr['num_switches'] == 2
    assert metr['num_detections'] == 6
    assert metr['num_objects'] == 8
    assert metr['num_predictions'] == 8
    assert metr['mota'] == approx(1. - (2 + 2 + 2) / 8)
    assert metr['motp'] == approx(11.1 / 6)
    assert metr['num_frames'] == 6


def test_ids():
    """Test metrics with frame IDs specified manually."""
    acc = mm.MOTAccumulator()

    # No data
    acc.update([], [], [], frameid=0)
    # Match
    acc.update([1, 2], [1, 2], [[1, 0], [0, 1]], frameid=1)
    # Switch also Transfer
    acc.update([1, 2], [1, 2], [[0.4, np.nan], [np.nan, 0.4]], frameid=2)
    # Match
    acc.update([1, 2], [1, 2], [[0, 1], [1, 0]], frameid=3)
    # Ascend (switch)
    acc.update([1, 2], [2, 3], [[1, 0], [0.4, 0.7]], frameid=4)
    # Migrate (transfer)
    acc.update([1, 3], [2, 3], [[1, 0], [0.4, 0.7]], frameid=5)
    # No data
    acc.update([], [], [], frameid=6)

    mh = mm.metrics.create()
    metr = mh.compute(acc, return_dataframe=False, return_cached=True, metrics=[
        'num_matches', 'num_false_positives', 'num_misses', 'num_switches',
        'num_transfer', 'num_ascend', 'num_migrate',
        'num_detections', 'num_objects', 'num_predictions',
        'mota', 'motp', 'num_frames',
    ])
    assert metr['num_matches'] == 7
    assert metr['num_false_positives'] == 0
    assert metr['num_misses'] == 0
    assert metr['num_switches'] == 3
    assert metr['num_transfer'] == 3
    assert metr['num_ascend'] == 1
    assert metr['num_migrate'] == 1
    assert metr['num_detections'] == 10
    assert metr['num_objects'] == 10
    assert metr['num_predictions'] == 10
    assert metr['mota'] == approx(1. - (0 + 0 + 3) / 10)
    assert metr['motp'] == approx(1.6 / 10)
    assert metr['num_frames'] == 7


def test_correct_average():
    """Tests what is depicted in figure 3 of 'Evaluating MOT Performance'."""
    acc = mm.MOTAccumulator(auto_id=True)

    # No track
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])

    # Track single
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])

    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics='mota', return_dataframe=False)
    assert metr['mota'] == approx(0.2)


def test_motchallenge_files():
    """Tests metrics for sequences TUD-Campus and TUD-Stadtmitte."""
    dnames = [
        'TUD-Campus',
        'TUD-Stadtmitte',
    ]

    def compute_motchallenge(dname):
        df_gt = mm.io.loadtxt(os.path.join(dname, 'gt.txt'))
        df_test = mm.io.loadtxt(os.path.join(dname, 'test.txt'))
        return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)

    accs = [compute_motchallenge(os.path.join(DATA_DIR, d)) for d in dnames]

    # For testing
    # [a.events.to_pickle(n) for (a,n) in zip(accs, dnames)]

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=dnames, generate_overall=True)

    print()
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))
    # assert ((summary['num_transfer'] - summary['num_migrate']) == (summary['num_switches'] - summary['num_ascend'])).all() # False assertion
    summary = summary[mm.metrics.motchallenge_metrics[:15]]
    expected = pd.DataFrame([
        [0.557659, 0.729730, 0.451253, 0.582173, 0.941441, 8.0, 1, 6, 1, 13, 150, 7, 7, 0.526462, 0.277201],
        [0.644619, 0.819760, 0.531142, 0.608997, 0.939920, 10.0, 5, 4, 1, 45, 452, 7, 6, 0.564014, 0.345904],
        [0.624296, 0.799176, 0.512211, 0.602640, 0.940268, 18.0, 6, 10, 2, 58, 602, 14, 13, 0.555116, 0.330177],
    ])
    np.testing.assert_allclose(summary, expected, atol=1e-3)
