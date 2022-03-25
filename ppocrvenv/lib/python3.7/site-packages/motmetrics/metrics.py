# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Obtain metrics from event logs."""

# pylint: disable=redefined-outer-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import inspect
import logging
import time

import numpy as np
import pandas as pd

from motmetrics import math_util
from motmetrics.lap import linear_sum_assignment
from motmetrics.mot import MOTAccumulator

try:
    _getargspec = inspect.getfullargspec
except AttributeError:
    _getargspec = inspect.getargspec


class MetricsHost:
    """Keeps track of metrics and intra metric dependencies."""

    def __init__(self):
        self.metrics = OrderedDict()

    def register(self, fnc, deps='auto', name=None, helpstr=None, formatter=None, fnc_m=None, deps_m='auto'):
        """Register a new metric.

        Params
        ------
        fnc : Function
            Function that computes the metric to be registered. The number of arguments
            is 1 + N, where N is the number of dependencies of the metric to be registered.
            The order of the argument passed is `df, result_dep1, result_dep2, ...`.

        Kwargs
        ------
        deps : string, list of strings or None, optional
            The dependencies of this metric. Each dependency is evaluated and the result
            is passed as argument to `fnc` as described above. If None is specified, the
            function does not have any dependencies. If a list of strings is given, dependencies
            for these metric strings are registered. If 'auto' is passed, the dependencies
            are deduced from argument inspection of the method. For this to work the argument
            names have to be equal to the intended dependencies.
        name : string or None, optional
            Name identifier of this metric. If None is passed the name is deduced from
            function inspection.
        helpstr : string or None, optional
            A description of what the metric computes. If no help message is given it
            is deduced from the docstring of the function.
        formatter: Format object, optional
            An optional default formatter when rendering metric results as string. I.e to
            render the result `0.35` as `35%` one would pass `{:.2%}.format`
        fnc_m : Function or None, optional
            Function that merges metric results. The number of arguments
            is 1 + N, where N is the number of dependencies of the metric to be registered.
            The order of the argument passed is `df, result_dep1, result_dep2, ...`.
        """

        assert fnc is not None, 'No function given for metric {}'.format(name)

        if deps is None:
            deps = []
        elif deps == 'auto':
            if _getargspec(fnc).defaults is not None:
                k = - len(_getargspec(fnc).defaults)
            else:
                k = len(_getargspec(fnc).args)
            deps = _getargspec(fnc).args[1:k]  # assumes dataframe as first argument

        if name is None:
            name = fnc.__name__  # Relies on meaningful function names, i.e don't use for lambdas

        if helpstr is None:
            helpstr = inspect.getdoc(fnc) if inspect.getdoc(fnc) else 'No description.'
            helpstr = ' '.join(helpstr.split())
        if fnc_m is None and name + '_m' in globals():
            fnc_m = globals()[name + '_m']
        if fnc_m is not None:
            if deps_m is None:
                deps_m = []
            elif deps_m == 'auto':
                if _getargspec(fnc_m).defaults is not None:
                    k = - len(_getargspec(fnc_m).defaults)
                else:
                    k = len(_getargspec(fnc_m).args)
                deps_m = _getargspec(fnc_m).args[1:k]  # assumes dataframe as first argument
        else:
            deps_m = None

        self.metrics[name] = {
            'name': name,
            'fnc': fnc,
            'fnc_m': fnc_m,
            'deps': deps,
            'deps_m': deps_m,
            'help': helpstr,
            'formatter': formatter
        }

    @property
    def names(self):
        """Returns the name identifiers of all registered metrics."""
        return [v['name'] for v in self.metrics.values()]

    @property
    def formatters(self):
        """Returns the formatters for all metrics that have associated formatters."""
        return {
            k: v['formatter'] for k, v in self.metrics.items()
            if v['formatter'] is not None
        }

    def list_metrics(self, include_deps=False):
        """Returns a dataframe containing names, descriptions and optionally dependencies for each metric."""
        cols = ['Name', 'Description', 'Dependencies']
        if include_deps:
            data = [(m['name'], m['help'], m['deps']) for m in self.metrics.values()]
        else:
            data = [(m['name'], m['help']) for m in self.metrics.values()]
            cols = cols[:-1]

        return pd.DataFrame(data, columns=cols)

    def list_metrics_markdown(self, include_deps=False):
        """Returns a markdown ready version of `list_metrics`."""
        df = self.list_metrics(include_deps=include_deps)
        fmt = [':---' for i in range(len(df.columns))]
        df_fmt = pd.DataFrame([fmt], columns=df.columns)
        df_formatted = pd.concat([df_fmt, df])
        return df_formatted.to_csv(sep="|", index=False)

    def compute(self, df, ana=None, metrics=None, return_dataframe=True, return_cached=False, name=None):
        """Compute metrics on the dataframe / accumulator.

        Params
        ------
        df : MOTAccumulator or pandas.DataFrame
            The dataframe to compute the metrics on

        Kwargs
        ------
        ana: dict or None, optional
            To cache results for fast computation.
        metrics : string, list of string or None, optional
            The identifiers of the metrics to be computed. This method will only
            compute the minimal set of necessary metrics to fullfill the request.
            If None is passed all registered metrics are computed.
        return_dataframe : bool, optional
            Return the result as pandas.DataFrame (default) or dict.
        return_cached : bool, optional
           If true all intermediate metrics required to compute the desired metrics are returned as well.
        name : string, optional
            When returning a pandas.DataFrame this is the index of the row containing
            the computed metric values.
        """

        if isinstance(df, MOTAccumulator):
            df = df.events

        if metrics is None:
            metrics = motchallenge_metrics
        elif isinstance(metrics, str):
            metrics = [metrics]

        df_map = events_to_df_map(df)

        cache = {}
        options = {'ana': ana}
        for mname in metrics:
            cache[mname] = self._compute(df_map, mname, cache, options, parent='summarize')

        if name is None:
            name = 0

        if return_cached:
            data = cache
        else:
            data = OrderedDict([(k, cache[k]) for k in metrics])

        ret = pd.DataFrame(data, index=[name]) if return_dataframe else data
        return ret

    def compute_overall(self, partials, metrics=None, return_dataframe=True, return_cached=False, name=None):
        """Compute overall metrics based on multiple results.

        Params
        ------
        partials : list of metric results to combine overall

        Kwargs
        ------
        metrics : string, list of string or None, optional
            The identifiers of the metrics to be computed. This method will only
            compute the minimal set of necessary metrics to fullfill the request.
            If None is passed all registered metrics are computed.
        return_dataframe : bool, optional
            Return the result as pandas.DataFrame (default) or dict.
        return_cached : bool, optional
           If true all intermediate metrics required to compute the desired metrics are returned as well.
        name : string, optional
            When returning a pandas.DataFrame this is the index of the row containing
            the computed metric values.

        Returns
        -------
        df : pandas.DataFrame
            A datafrom containing the metrics in columns and names in rows.
        """
        if metrics is None:
            metrics = self.names
        elif isinstance(metrics, str):
            metrics = [metrics]
        cache = {}

        for mname in metrics:
            cache[mname] = self._compute_overall(partials, mname, cache, parent='summarize')

        if name is None:
            name = 0
        if return_cached:
            data = cache
        else:
            data = OrderedDict([(k, cache[k]) for k in metrics])
        return pd.DataFrame(data, index=[name]) if return_dataframe else data

    def compute_many(self, dfs, anas=None, metrics=None, names=None, generate_overall=False):
        """Compute metrics on multiple dataframe / accumulators.

        Params
        ------
        dfs : list of MOTAccumulator or list of pandas.DataFrame
            The data to compute metrics on.

        Kwargs
        ------
        anas: dict or None, optional
            To cache results for fast computation.
        metrics : string, list of string or None, optional
            The identifiers of the metrics to be computed. This method will only
            compute the minimal set of necessary metrics to fullfill the request.
            If None is passed all registered metrics are computed.
        names : list of string, optional
            The names of individual rows in the resulting dataframe.
        generate_overall : boolean, optional
            If true resulting dataframe will contain a summary row that is computed
            using the same metrics over an accumulator that is the concatentation of
            all input containers. In creating this temporary accumulator, care is taken
            to offset frame indices avoid object id collisions.

        Returns
        -------
        df : pandas.DataFrame
            A datafrom containing the metrics in columns and names in rows.
        """

        assert names is None or len(names) == len(dfs)
        st = time.time()
        if names is None:
            names = list(range(len(dfs)))
        if anas is None:
            anas = [None] * len(dfs)
        partials = [
            self.compute(acc,
                         ana=analysis,
                         metrics=metrics,
                         name=name,
                         return_cached=True,
                         return_dataframe=False
                         )
            for acc, analysis, name in zip(dfs, anas, names)]
        logging.info('partials: %.3f seconds.', time.time() - st)
        details = partials
        partials = [pd.DataFrame(OrderedDict([(k, i[k]) for k in metrics]), index=[name]) for i, name in zip(partials, names)]
        if generate_overall:
            names = 'OVERALL'
            # merged, infomap = MOTAccumulator.merge_event_dataframes(dfs, return_mappings = True)
            # dfs = merged
            # anas = MOTAccumulator.merge_analysis(anas, infomap)
            # partials.append(self.compute(dfs, ana=anas, metrics=metrics, name=names)[0])
            partials.append(self.compute_overall(details, metrics=metrics, name=names))
        logging.info('mergeOverall: %.3f seconds.', time.time() - st)
        return pd.concat(partials)

    def _compute(self, df_map, name, cache, options, parent=None):
        """Compute metric and resolve dependencies."""
        assert name in self.metrics, 'Cannot find metric {} required by {}.'.format(name, parent)
        already = cache.get(name, None)
        if already is not None:
            return already
        minfo = self.metrics[name]
        vals = []
        for depname in minfo['deps']:
            v = cache.get(depname, None)
            if v is None:
                v = cache[depname] = self._compute(df_map, depname, cache, options, parent=name)
            vals.append(v)
        if _getargspec(minfo['fnc']).defaults is None:
            return minfo['fnc'](df_map, *vals)
        else:
            return minfo['fnc'](df_map, *vals, **options)

    def _compute_overall(self, partials, name, cache, parent=None):
        assert name in self.metrics, 'Cannot find metric {} required by {}.'.format(name, parent)
        already = cache.get(name, None)
        if already is not None:
            return already
        minfo = self.metrics[name]
        vals = []
        for depname in minfo['deps_m']:
            v = cache.get(depname, None)
            if v is None:
                v = cache[depname] = self._compute_overall(partials, depname, cache, parent=name)
            vals.append(v)
        assert minfo['fnc_m'] is not None, 'merge function for metric %s is None' % name
        return minfo['fnc_m'](partials, *vals)


simple_add_func = []


def num_frames(df):
    """Total number of frames."""
    return df.full.index.get_level_values(0).unique().shape[0]


simple_add_func.append(num_frames)


def obj_frequencies(df):
    """Total number of occurrences of individual objects over all frames."""
    return df.noraw.OId.value_counts()


def pred_frequencies(df):
    """Total number of occurrences of individual predictions over all frames."""
    return df.noraw.HId.value_counts()


def num_unique_objects(df, obj_frequencies):
    """Total number of unique object ids encountered."""
    del df  # unused
    return len(obj_frequencies)


simple_add_func.append(num_unique_objects)


def num_matches(df):
    """Total number matches."""
    return df.noraw.Type.isin(['MATCH']).sum()


simple_add_func.append(num_matches)


def num_switches(df):
    """Total number of track switches."""
    return df.noraw.Type.isin(['SWITCH']).sum()


simple_add_func.append(num_switches)


def num_transfer(df):
    """Total number of track transfer."""
    return df.extra.Type.isin(['TRANSFER']).sum()


simple_add_func.append(num_transfer)


def num_ascend(df):
    """Total number of track ascend."""
    return df.extra.Type.isin(['ASCEND']).sum()


simple_add_func.append(num_ascend)


def num_migrate(df):
    """Total number of track migrate."""
    return df.extra.Type.isin(['MIGRATE']).sum()


simple_add_func.append(num_migrate)


def num_false_positives(df):
    """Total number of false positives (false-alarms)."""
    return df.noraw.Type.isin(['FP']).sum()


simple_add_func.append(num_false_positives)


def num_misses(df):
    """Total number of misses."""
    return df.noraw.Type.isin(['MISS']).sum()


simple_add_func.append(num_misses)


def num_detections(df, num_matches, num_switches):
    """Total number of detected objects including matches and switches."""
    del df  # unused
    return num_matches + num_switches


simple_add_func.append(num_detections)


def num_objects(df, obj_frequencies):
    """Total number of unique object appearances over all frames."""
    del df  # unused
    return obj_frequencies.sum()


simple_add_func.append(num_objects)


def num_predictions(df, pred_frequencies):
    """Total number of unique prediction appearances over all frames."""
    del df  # unused
    return pred_frequencies.sum()


simple_add_func.append(num_predictions)


def track_ratios(df, obj_frequencies):
    """Ratio of assigned to total appearance count per unique object id."""
    tracked = df.noraw[df.noraw.Type != 'MISS']['OId'].value_counts()
    return tracked.div(obj_frequencies).fillna(0.)


def mostly_tracked(df, track_ratios):
    """Number of objects tracked for at least 80 percent of lifespan."""
    del df  # unused
    return track_ratios[track_ratios >= 0.8].count()


simple_add_func.append(mostly_tracked)


def partially_tracked(df, track_ratios):
    """Number of objects tracked between 20 and 80 percent of lifespan."""
    del df  # unused
    return track_ratios[(track_ratios >= 0.2) & (track_ratios < 0.8)].count()


simple_add_func.append(partially_tracked)


def mostly_lost(df, track_ratios):
    """Number of objects tracked less than 20 percent of lifespan."""
    del df  # unused
    return track_ratios[track_ratios < 0.2].count()


simple_add_func.append(mostly_lost)


def num_fragmentations(df, obj_frequencies):
    """Total number of switches from tracked to not tracked."""
    fra = 0
    for o in obj_frequencies.index:
        # Find first and last time object was not missed (track span). Then count
        # the number switches from NOT MISS to MISS state.
        dfo = df.noraw[df.noraw.OId == o]
        notmiss = dfo[dfo.Type != 'MISS']
        if len(notmiss) == 0:
            continue
        first = notmiss.index[0]
        last = notmiss.index[-1]
        diffs = dfo.loc[first:last].Type.apply(lambda x: 1 if x == 'MISS' else 0).diff()
        fra += diffs[diffs == 1].count()
    return fra


simple_add_func.append(num_fragmentations)


def motp(df, num_detections):
    """Multiple object tracker precision."""
    return math_util.quiet_divide(df.noraw['D'].sum(), num_detections)


def motp_m(partials, num_detections):
    res = 0
    for v in partials:
        res += v['motp'] * v['num_detections']
    return math_util.quiet_divide(res, num_detections)


def mota(df, num_misses, num_switches, num_false_positives, num_objects):
    """Multiple object tracker accuracy."""
    del df  # unused
    return 1. - math_util.quiet_divide(
        num_misses + num_switches + num_false_positives,
        num_objects)


def mota_m(partials, num_misses, num_switches, num_false_positives, num_objects):
    del partials  # unused
    return 1. - math_util.quiet_divide(
        num_misses + num_switches + num_false_positives,
        num_objects)


def precision(df, num_detections, num_false_positives):
    """Number of detected objects over sum of detected and false positives."""
    del df  # unused
    return math_util.quiet_divide(
        num_detections,
        num_false_positives + num_detections)


def precision_m(partials, num_detections, num_false_positives):
    del partials  # unused
    return math_util.quiet_divide(
        num_detections,
        num_false_positives + num_detections)


def recall(df, num_detections, num_objects):
    """Number of detections over number of objects."""
    del df  # unused
    return math_util.quiet_divide(num_detections, num_objects)


def recall_m(partials, num_detections, num_objects):
    del partials  # unused
    return math_util.quiet_divide(num_detections, num_objects)


class DataFrameMap:  # pylint: disable=too-few-public-methods

    def __init__(self, full, raw, noraw, extra):
        self.full = full
        self.raw = raw
        self.noraw = noraw
        self.extra = extra


def events_to_df_map(df):
    raw = df[df.Type == 'RAW']
    noraw = df[(df.Type != 'RAW')
               & (df.Type != 'ASCEND')
               & (df.Type != 'TRANSFER')
               & (df.Type != 'MIGRATE')]
    extra = df[df.Type != 'RAW']
    df_map = DataFrameMap(full=df, raw=raw, noraw=noraw, extra=extra)
    return df_map


def extract_counts_from_df_map(df):
    """
    Returns:
        Tuple (ocs, hcs, tps).
        ocs: Dict from object id to count.
        hcs: Dict from hypothesis id to count.
        tps: Dict from (object id, hypothesis id) to true-positive count.
        The ids are arbitrary, they might NOT be consecutive integers from 0.
    """
    oids = df.full['OId'].dropna().unique()
    hids = df.full['HId'].dropna().unique()

    flat = df.raw.reset_index()
    # Exclude events that do not belong to either set.
    flat = flat[flat['OId'].isin(oids) | flat['HId'].isin(hids)]
    # Count number of frames where each (non-empty) OId and HId appears.
    ocs = flat.set_index('OId')['FrameId'].groupby('OId').nunique().to_dict()
    hcs = flat.set_index('HId')['FrameId'].groupby('HId').nunique().to_dict()
    # Select three columns of interest and index by ('OId', 'HId').
    dists = flat[['OId', 'HId', 'D']].set_index(['OId', 'HId']).dropna()
    # Count events with non-empty distance for each pair.
    tps = dists.groupby(['OId', 'HId'])['D'].count().to_dict()
    return ocs, hcs, tps


def id_global_assignment(df, ana=None):
    """ID measures: Global min-cost assignment for ID measures."""
    # pylint: disable=too-many-locals
    del ana  # unused
    ocs, hcs, tps = extract_counts_from_df_map(df)
    oids = sorted(ocs.keys())
    hids = sorted(hcs.keys())
    oids_idx = dict((o, i) for i, o in enumerate(oids))
    hids_idx = dict((h, i) for i, h in enumerate(hids))
    no = len(ocs)
    nh = len(hcs)

    fpmatrix = np.full((no + nh, no + nh), 0.)
    fnmatrix = np.full((no + nh, no + nh), 0.)
    fpmatrix[no:, :nh] = np.nan
    fnmatrix[:no, nh:] = np.nan

    for oid, oc in ocs.items():
        r = oids_idx[oid]
        fnmatrix[r, :nh] = oc
        fnmatrix[r, nh + r] = oc

    for hid, hc in hcs.items():
        c = hids_idx[hid]
        fpmatrix[:no, c] = hc
        fpmatrix[c + no, c] = hc

    for (oid, hid), ex in tps.items():
        r = oids_idx[oid]
        c = hids_idx[hid]
        fpmatrix[r, c] -= ex
        fnmatrix[r, c] -= ex

    costs = fpmatrix + fnmatrix
    rids, cids = linear_sum_assignment(costs)

    return {
        'fpmatrix': fpmatrix,
        'fnmatrix': fnmatrix,
        'rids': rids,
        'cids': cids,
        'costs': costs,
        'min_cost': costs[rids, cids].sum()
    }


def idfp(df, id_global_assignment):
    """ID measures: Number of false positive matches after global min-cost matching."""
    del df  # unused
    rids, cids = id_global_assignment['rids'], id_global_assignment['cids']
    return id_global_assignment['fpmatrix'][rids, cids].sum()


simple_add_func.append(idfp)


def idfn(df, id_global_assignment):
    """ID measures: Number of false negatives matches after global min-cost matching."""
    del df  # unused
    rids, cids = id_global_assignment['rids'], id_global_assignment['cids']
    return id_global_assignment['fnmatrix'][rids, cids].sum()


simple_add_func.append(idfn)


def idtp(df, id_global_assignment, num_objects, idfn):
    """ID measures: Number of true positives matches after global min-cost matching."""
    del df, id_global_assignment  # unused
    return num_objects - idfn


simple_add_func.append(idtp)


def idp(df, idtp, idfp):
    """ID measures: global min-cost precision."""
    del df  # unused
    return math_util.quiet_divide(idtp, idtp + idfp)


def idp_m(partials, idtp, idfp):
    del partials  # unused
    return math_util.quiet_divide(idtp, idtp + idfp)


def idr(df, idtp, idfn):
    """ID measures: global min-cost recall."""
    del df  # unused
    return math_util.quiet_divide(idtp, idtp + idfn)


def idr_m(partials, idtp, idfn):
    del partials  # unused
    return math_util.quiet_divide(idtp, idtp + idfn)


def idf1(df, idtp, num_objects, num_predictions):
    """ID measures: global min-cost F1 score."""
    del df  # unused
    return math_util.quiet_divide(2 * idtp, num_objects + num_predictions)


def idf1_m(partials, idtp, num_objects, num_predictions):
    del partials  # unused
    return math_util.quiet_divide(2 * idtp, num_objects + num_predictions)


for one in simple_add_func:
    name = one.__name__

    def getSimpleAdd(nm):
        def simpleAddHolder(partials):
            res = 0
            for v in partials:
                res += v[nm]
            return res
        return simpleAddHolder
    locals()[name + '_m'] = getSimpleAdd(name)


def create():
    """Creates a MetricsHost and populates it with default metrics."""
    m = MetricsHost()

    m.register(num_frames, formatter='{:d}'.format)
    m.register(obj_frequencies, formatter='{:d}'.format)
    m.register(pred_frequencies, formatter='{:d}'.format)
    m.register(num_matches, formatter='{:d}'.format)
    m.register(num_switches, formatter='{:d}'.format)
    m.register(num_transfer, formatter='{:d}'.format)
    m.register(num_ascend, formatter='{:d}'.format)
    m.register(num_migrate, formatter='{:d}'.format)
    m.register(num_false_positives, formatter='{:d}'.format)
    m.register(num_misses, formatter='{:d}'.format)
    m.register(num_detections, formatter='{:d}'.format)
    m.register(num_objects, formatter='{:d}'.format)
    m.register(num_predictions, formatter='{:d}'.format)
    m.register(num_unique_objects, formatter='{:d}'.format)
    m.register(track_ratios)
    m.register(mostly_tracked, formatter='{:d}'.format)
    m.register(partially_tracked, formatter='{:d}'.format)
    m.register(mostly_lost, formatter='{:d}'.format)
    m.register(num_fragmentations)
    m.register(motp, formatter='{:.3f}'.format)
    m.register(mota, formatter='{:.1%}'.format)
    m.register(precision, formatter='{:.1%}'.format)
    m.register(recall, formatter='{:.1%}'.format)

    m.register(id_global_assignment)
    m.register(idfp)
    m.register(idfn)
    m.register(idtp)
    m.register(idp, formatter='{:.1%}'.format)
    m.register(idr, formatter='{:.1%}'.format)
    m.register(idf1, formatter='{:.1%}'.format)

    return m


motchallenge_metrics = [
    'idf1',
    'idp',
    'idr',
    'recall',
    'precision',
    'num_unique_objects',
    'mostly_tracked',
    'partially_tracked',
    'mostly_lost',
    'num_false_positives',
    'num_misses',
    'num_switches',
    'num_fragmentations',
    'mota',
    'motp',
    'num_transfer',
    'num_ascend',
    'num_migrate',
]
"""A list of all metrics from MOTChallenge."""
