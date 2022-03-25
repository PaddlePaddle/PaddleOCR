# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Accumulate tracking events frame by frame."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import itertools

import numpy as np
import pandas as pd

from motmetrics.lap import linear_sum_assignment

_INDEX_FIELDS = ['FrameId', 'Event']
_EVENT_FIELDS = ['Type', 'OId', 'HId', 'D']


class MOTAccumulator(object):
    """Manage tracking events.

    This class computes per-frame tracking events from a given set of object / hypothesis
    ids and pairwise distances. Indended usage

        import motmetrics as mm
        acc = mm.MOTAccumulator()
        acc.update(['a', 'b'], [0, 1, 2], dists, frameid=0)
        ...
        acc.update(['d'], [6,10], other_dists, frameid=76)
        summary = mm.metrics.summarize(acc)
        print(mm.io.render_summary(summary))

    Update is called once per frame and takes objects / hypothesis ids and a pairwise distance
    matrix between those (see distances module for support). Per frame max(len(objects), len(hypothesis))
    events are generated. Each event type is one of the following
        - `'MATCH'` a match between a object and hypothesis was found
        - `'SWITCH'` a match between a object and hypothesis was found but differs from previous assignment (hypothesisid != previous)
        - `'MISS'` no match for an object was found
        - `'FP'` no match for an hypothesis was found (spurious detections)
        - `'RAW'` events corresponding to raw input
        - `'TRANSFER'` a match between a object and hypothesis was found but differs from previous assignment (objectid != previous)
        - `'ASCEND'` a match between a object and hypothesis was found but differs from previous assignment  (hypothesisid is new)
        - `'MIGRATE'` a match between a object and hypothesis was found but differs from previous assignment  (objectid is new)

    Events are tracked in a pandas Dataframe. The dataframe is hierarchically indexed by (`FrameId`, `EventId`),
    where `FrameId` is either provided during the call to `update` or auto-incremented when `auto_id` is set
    true during construction of MOTAccumulator. `EventId` is auto-incremented. The dataframe has the following
    columns
        - `Type` one of `('MATCH', 'SWITCH', 'MISS', 'FP', 'RAW')`
        - `OId` object id or np.nan when `'FP'` or `'RAW'` and object is not present
        - `HId` hypothesis id or np.nan when `'MISS'` or `'RAW'` and hypothesis is not present
        - `D` distance or np.nan when `'FP'` or `'MISS'` or `'RAW'` and either object/hypothesis is absent

    From the events and associated fields the entire tracking history can be recovered. Once the accumulator
    has been populated with per-frame data use `metrics.summarize` to compute statistics. See `metrics.compute_metrics`
    for a list of metrics computed.

    References
    ----------
    1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics."
    EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
    2. Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016).
    3. Li, Yuan, Chang Huang, and Ram Nevatia. "Learning to associate: Hybridboosted multi-target tracker for crowded scene."
    Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.
    """

    def __init__(self, auto_id=False, max_switch_time=float('inf')):
        """Create a MOTAccumulator.

        Params
        ------
        auto_id : bool, optional
            Whether or not frame indices are auto-incremented or provided upon
            updating. Defaults to false. Not specifying a frame-id when this value
            is true results in an error. Specifying a frame-id when this value is
            false also results in an error.

        max_switch_time : scalar, optional
            Allows specifying an upper bound on the timespan an unobserved but
            tracked object is allowed to generate track switch events. Useful if groundtruth
            objects leaving the field of view keep their ID when they reappear,
            but your tracker is not capable of recognizing this (resulting in
            track switch events). The default is that there is no upper bound
            on the timespan. In units of frame timestamps. When using auto_id
            in units of count.
        """

        # Parameters of the accumulator.
        self.auto_id = auto_id
        self.max_switch_time = max_switch_time

        # Accumulator state.
        self._events = None
        self._indices = None
        self.m = None
        self.res_m = None
        self.last_occurrence = None
        self.last_match = None
        self.hypHistory = None
        self.dirty_events = None
        self.cached_events_df = None

        self.reset()

    def reset(self):
        """Reset the accumulator to empty state."""

        self._events = {field: [] for field in _EVENT_FIELDS}
        self._indices = {field: [] for field in _INDEX_FIELDS}
        self.m = {}  # Pairings up to current timestamp
        self.res_m = {}  # Result pairings up to now
        self.last_occurrence = {}  # Tracks most recent occurance of object
        self.last_match = {}  # Tracks most recent match of object
        self.hypHistory = {}
        self.dirty_events = True
        self.cached_events_df = None

    def _append_to_indices(self, frameid, eid):
        self._indices['FrameId'].append(frameid)
        self._indices['Event'].append(eid)

    def _append_to_events(self, typestr, oid, hid, distance):
        self._events['Type'].append(typestr)
        self._events['OId'].append(oid)
        self._events['HId'].append(hid)
        self._events['D'].append(distance)

    def update(self, oids, hids, dists, frameid=None, vf=''):
        """Updates the accumulator with frame specific objects/detections.

        This method generates events based on the following algorithm [1]:
        1. Try to carry forward already established tracks. If any paired object / hypothesis
        from previous timestamps are still visible in the current frame, create a 'MATCH'
        event between them.
        2. For the remaining constellations minimize the total object / hypothesis distance
        error (Kuhn-Munkres algorithm). If a correspondence made contradicts a previous
        match create a 'SWITCH' else a 'MATCH' event.
        3. Create 'MISS' events for all remaining unassigned objects.
        4. Create 'FP' events for all remaining unassigned hypotheses.

        Params
        ------
        oids : N array
            Array of object ids.
        hids : M array
            Array of hypothesis ids.
        dists: NxM array
            Distance matrix. np.nan values to signal do-not-pair constellations.
            See `distances` module for support methods.

        Kwargs
        ------
        frameId : id
            Unique frame id. Optional when MOTAccumulator.auto_id is specified during
            construction.
        vf: file to log details
        Returns
        -------
        frame_events : pd.DataFrame
            Dataframe containing generated events

        References
        ----------
        1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics."
        EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
        """
        # pylint: disable=too-many-locals, too-many-statements

        self.dirty_events = True
        oids = np.asarray(oids)
        oids_masked = np.zeros_like(oids, dtype=np.bool)
        hids = np.asarray(hids)
        hids_masked = np.zeros_like(hids, dtype=np.bool)
        dists = np.atleast_2d(dists).astype(float).reshape(oids.shape[0], hids.shape[0]).copy()

        if frameid is None:
            assert self.auto_id, 'auto-id is not enabled'
            if len(self._indices['FrameId']) > 0:
                frameid = self._indices['FrameId'][-1] + 1
            else:
                frameid = 0
        else:
            assert not self.auto_id, 'Cannot provide frame id when auto-id is enabled'

        eid = itertools.count()

        # 0. Record raw events

        no = len(oids)
        nh = len(hids)

        # Add a RAW event simply to ensure the frame is counted.
        self._append_to_indices(frameid, next(eid))
        self._append_to_events('RAW', np.nan, np.nan, np.nan)

        # There must be at least one RAW event per object and hypothesis.
        # Record all finite distances as RAW events.
        valid_i, valid_j = np.where(np.isfinite(dists))
        valid_dists = dists[valid_i, valid_j]
        for i, j, dist_ij in zip(valid_i, valid_j, valid_dists):
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('RAW', oids[i], hids[j], dist_ij)
        # Add a RAW event for objects and hypotheses that were present but did
        # not overlap with anything.
        used_i = np.unique(valid_i)
        used_j = np.unique(valid_j)
        unused_i = np.setdiff1d(np.arange(no), used_i)
        unused_j = np.setdiff1d(np.arange(nh), used_j)
        for oid in oids[unused_i]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('RAW', oid, np.nan, np.nan)
        for hid in hids[unused_j]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('RAW', np.nan, hid, np.nan)

        if oids.size * hids.size > 0:
            # 1. Try to re-establish tracks from previous correspondences
            for i in range(oids.shape[0]):
                # No need to check oids_masked[i] here.
                if oids[i] not in self.m:
                    continue

                hprev = self.m[oids[i]]
                j, = np.where(~hids_masked & (hids == hprev))
                if j.shape[0] == 0:
                    continue
                j = j[0]

                if np.isfinite(dists[i, j]):
                    o = oids[i]
                    h = hids[j]
                    oids_masked[i] = True
                    hids_masked[j] = True
                    self.m[oids[i]] = hids[j]

                    self._append_to_indices(frameid, next(eid))
                    self._append_to_events('MATCH', oids[i], hids[j], dists[i, j])
                    self.last_match[o] = frameid
                    self.hypHistory[h] = frameid

            # 2. Try to remaining objects/hypotheses
            dists[oids_masked, :] = np.nan
            dists[:, hids_masked] = np.nan

            rids, cids = linear_sum_assignment(dists)

            for i, j in zip(rids, cids):
                if not np.isfinite(dists[i, j]):
                    continue

                o = oids[i]
                h = hids[j]
                is_switch = (o in self.m and
                             self.m[o] != h and
                             abs(frameid - self.last_occurrence[o]) <= self.max_switch_time)
                cat1 = 'SWITCH' if is_switch else 'MATCH'
                if cat1 == 'SWITCH':
                    if h not in self.hypHistory:
                        subcat = 'ASCEND'
                        self._append_to_indices(frameid, next(eid))
                        self._append_to_events(subcat, oids[i], hids[j], dists[i, j])
                # ignore the last condition temporarily
                is_transfer = (h in self.res_m and
                               self.res_m[h] != o)
                # is_transfer = (h in self.res_m and
                #                self.res_m[h] != o and
                #                abs(frameid - self.last_occurrence[o]) <= self.max_switch_time)
                cat2 = 'TRANSFER' if is_transfer else 'MATCH'
                if cat2 == 'TRANSFER':
                    if o not in self.last_match:
                        subcat = 'MIGRATE'
                        self._append_to_indices(frameid, next(eid))
                        self._append_to_events(subcat, oids[i], hids[j], dists[i, j])
                    self._append_to_indices(frameid, next(eid))
                    self._append_to_events(cat2, oids[i], hids[j], dists[i, j])
                if vf != '' and (cat1 != 'MATCH' or cat2 != 'MATCH'):
                    if cat1 == 'SWITCH':
                        vf.write('%s %d %d %d %d %d\n' % (subcat[:2], o, self.last_match[o], self.m[o], frameid, h))
                    if cat2 == 'TRANSFER':
                        vf.write('%s %d %d %d %d %d\n' % (subcat[:2], h, self.hypHistory[h], self.res_m[h], frameid, o))
                self.hypHistory[h] = frameid
                self.last_match[o] = frameid
                self._append_to_indices(frameid, next(eid))
                self._append_to_events(cat1, oids[i], hids[j], dists[i, j])
                oids_masked[i] = True
                hids_masked[j] = True
                self.m[o] = h
                self.res_m[h] = o

        # 3. All remaining objects are missed
        for o in oids[~oids_masked]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('MISS', o, np.nan, np.nan)
            if vf != '':
                vf.write('FN %d %d\n' % (frameid, o))

        # 4. All remaining hypotheses are false alarms
        for h in hids[~hids_masked]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('FP', np.nan, h, np.nan)
            if vf != '':
                vf.write('FP %d %d\n' % (frameid, h))

        # 5. Update occurance state
        for o in oids:
            self.last_occurrence[o] = frameid

        return frameid

    @property
    def events(self):
        if self.dirty_events:
            self.cached_events_df = MOTAccumulator.new_event_dataframe_with_data(self._indices, self._events)
            self.dirty_events = False
        return self.cached_events_df

    @property
    def mot_events(self):
        df = self.events
        return df[df.Type != 'RAW']

    @staticmethod
    def new_event_dataframe():
        """Create a new DataFrame for event tracking."""
        idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['FrameId', 'Event'])
        cats = pd.Categorical([], categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE'])
        df = pd.DataFrame(
            OrderedDict([
                ('Type', pd.Series(cats)),          # Type of event. One of FP (false positive), MISS, SWITCH, MATCH
                ('OId', pd.Series(dtype=float)),      # Object ID or -1 if FP. Using float as missing values will be converted to NaN anyways.
                ('HId', pd.Series(dtype=float)),      # Hypothesis ID or NaN if MISS. Using float as missing values will be converted to NaN anyways.
                ('D', pd.Series(dtype=float)),      # Distance or NaN when FP or MISS
            ]),
            index=idx
        )
        return df

    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        """Create a new DataFrame filled with data.

        Params
        ------
        indices: dict
            dict of lists with fields 'FrameId' and 'Event'
        events: dict
            dict of lists with fields 'Type', 'OId', 'HId', 'D'
        """

        if len(events) == 0:
            return MOTAccumulator.new_event_dataframe()

        raw_type = pd.Categorical(
            events['Type'],
            categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE'],
            ordered=False)
        series = [
            pd.Series(raw_type, name='Type'),
            pd.Series(events['OId'], dtype=float, name='OId'),
            pd.Series(events['HId'], dtype=float, name='HId'),
            pd.Series(events['D'], dtype=float, name='D')
        ]

        idx = pd.MultiIndex.from_arrays(
            [indices[field] for field in _INDEX_FIELDS],
            names=_INDEX_FIELDS)
        df = pd.concat(series, axis=1)
        df.index = idx
        return df

    @staticmethod
    def merge_analysis(anas, infomap):
        # pylint: disable=missing-function-docstring
        res = {'hyp': {}, 'obj': {}}
        mapp = {'hyp': 'hid_map', 'obj': 'oid_map'}
        for ana, infom in zip(anas, infomap):
            if ana is None:
                return None
            for t in ana.keys():
                which = mapp[t]
                if np.nan in infom[which]:
                    res[t][int(infom[which][np.nan])] = 0
                if 'nan' in infom[which]:
                    res[t][int(infom[which]['nan'])] = 0
                for _id, cnt in ana[t].items():
                    if _id not in infom[which]:
                        _id = str(_id)
                    res[t][int(infom[which][_id])] = cnt
        return res

    @staticmethod
    def merge_event_dataframes(dfs, update_frame_indices=True, update_oids=True, update_hids=True, return_mappings=False):
        """Merge dataframes.

        Params
        ------
        dfs : list of pandas.DataFrame or MotAccumulator
            A list of event containers to merge

        Kwargs
        ------
        update_frame_indices : boolean, optional
            Ensure that frame indices are unique in the merged container
        update_oids : boolean, unique
            Ensure that object ids are unique in the merged container
        update_hids : boolean, unique
            Ensure that hypothesis ids are unique in the merged container
        return_mappings : boolean, unique
            Whether or not to return mapping information

        Returns
        -------
        df : pandas.DataFrame
            Merged event data frame
        """

        mapping_infos = []
        new_oid = itertools.count()
        new_hid = itertools.count()

        r = MOTAccumulator.new_event_dataframe()
        for df in dfs:

            if isinstance(df, MOTAccumulator):
                df = df.events

            copy = df.copy()
            infos = {}

            # Update index
            if update_frame_indices:
                # pylint: disable=cell-var-from-loop
                next_frame_id = max(r.index.get_level_values(0).max() + 1, r.index.get_level_values(0).unique().shape[0])
                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
                infos['frame_offset'] = next_frame_id

            # Update object / hypothesis ids
            if update_oids:
                # pylint: disable=cell-var-from-loop
                oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                infos['oid_map'] = oid_map

            if update_hids:
                # pylint: disable=cell-var-from-loop
                hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                infos['hid_map'] = hid_map

            r = r.append(copy)
            mapping_infos.append(infos)

        if return_mappings:
            return r, mapping_infos
        else:
            return r
