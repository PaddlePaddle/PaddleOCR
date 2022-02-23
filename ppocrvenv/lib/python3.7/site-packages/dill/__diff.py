#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

"""
Module to show if an object has changed since it was memorised
"""

import os
import sys
import types
try:
    import numpy
    HAS_NUMPY = True
except:
    HAS_NUMPY = False
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# pypy doesn't use reference counting
getrefcount = getattr(sys, 'getrefcount', lambda x:0)

# memo of objects indexed by id to a tuple (attributes, sequence items)
# attributes is a dict indexed by attribute name to attribute id
# sequence items is either a list of ids, of a dictionary of keys to ids
memo = {}
id_to_obj = {}
# types that cannot have changing attributes
builtins_types = set((str, list, dict, set, frozenset, int))
dont_memo = set(id(i) for i in (memo, sys.modules, sys.path_importer_cache,
             os.environ, id_to_obj))


def get_attrs(obj):
    """
    Gets all the attributes of an object though its __dict__ or return None
    """
    if type(obj) in builtins_types \
       or type(obj) is type and obj in builtins_types:
        return
    try:
        return obj.__dict__
    except:
        return


def get_seq(obj, cache={str: False, frozenset: False, list: True, set: True,
                        dict: True, tuple: True, type: False,
                        types.ModuleType: False, types.FunctionType: False,
                        types.BuiltinFunctionType: False}):
    """
    Gets all the items in a sequence or return None
    """
    try:
        o_type = obj.__class__
    except AttributeError:
        o_type = type(obj)
    hsattr = hasattr
    if o_type in cache:
        if cache[o_type]:
            if hsattr(obj, "copy"):
                return obj.copy()
            return obj
    elif HAS_NUMPY and o_type in (numpy.ndarray, numpy.ma.core.MaskedConstant):
        if obj.shape and obj.size:
            return obj
        else:
            return []
    elif hsattr(obj, "__contains__") and hsattr(obj, "__iter__") \
       and hsattr(obj, "__len__") and hsattr(o_type, "__contains__") \
       and hsattr(o_type, "__iter__") and hsattr(o_type, "__len__"):
        cache[o_type] = True
        if hsattr(obj, "copy"):
            return obj.copy()
        return obj
    else:
        cache[o_type] = False
        return None


def memorise(obj, force=False):
    """
    Adds an object to the memo, and recursively adds all the objects
    attributes, and if it is a container, its items. Use force=True to update
    an object already in the memo. Updating is not recursively done.
    """
    obj_id = id(obj)
    if obj_id in memo and not force or obj_id in dont_memo:
        return
    id_ = id
    g = get_attrs(obj)
    if g is None:
        attrs_id = None
    else:
        attrs_id = dict((key,id_(value)) for key, value in g.items())

    s = get_seq(obj)
    if s is None:
        seq_id = None
    elif hasattr(s, "items"):
        seq_id = dict((id_(key),id_(value)) for key, value in s.items())
    elif not hasattr(s, "__len__"): #XXX: avoid TypeError from unexpected case
        seq_id = None
    else:
        seq_id = [id_(i) for i in s]

    memo[obj_id] = attrs_id, seq_id
    id_to_obj[obj_id] = obj
    mem = memorise
    if g is not None:
        [mem(value) for key, value in g.items()]

    if s is not None:
        if hasattr(s, "items"):
            [(mem(key), mem(item))
             for key, item in s.items()]
        else:
            if hasattr(s, '__len__'):
                [mem(item) for item in s]
            else: mem(s)


def release_gone():
    itop, mp, src = id_to_obj.pop, memo.pop, getrefcount
    [(itop(id_), mp(id_)) for id_, obj in list(id_to_obj.items())
     if src(obj) < 4] #XXX: correct for pypy?


def whats_changed(obj, seen=None, simple=False, first=True):
    """
    Check an object against the memo. Returns a list in the form
    (attribute changes, container changed). Attribute changes is a dict of
    attribute name to attribute value. container changed is a boolean.
    If simple is true, just returns a boolean. None for either item means
    that it has not been checked yet
    """
    # Special cases
    if first:
        # ignore the _ variable, which only appears in interactive sessions
        if "_" in builtins.__dict__:
            del builtins._
        if seen is None:
            seen = {}

    obj_id = id(obj)

    if obj_id in seen:
        if simple:
            return any(seen[obj_id])
        return seen[obj_id]

    # Safety checks
    if obj_id in dont_memo:
        seen[obj_id] = [{}, False]
        if simple:
            return False
        return seen[obj_id]
    elif obj_id not in memo:
        if simple:
            return True
        else:
            raise RuntimeError("Object not memorised " + str(obj))

    seen[obj_id] = ({}, False)

    chngd = whats_changed
    id_ = id

    # compare attributes
    attrs = get_attrs(obj)
    if attrs is None:
        changed = {}
    else:
        obj_attrs = memo[obj_id][0]
        obj_get = obj_attrs.get
        changed = dict((key,None) for key in obj_attrs if key not in attrs)
        for key, o in attrs.items():
            if id_(o) != obj_get(key, None) or chngd(o, seen, True, False):
                changed[key] = o

    # compare sequence
    items = get_seq(obj)
    seq_diff = False
    if (items is not None) and (hasattr(items, '__len__')):
        obj_seq = memo[obj_id][1]
        if (len(items) != len(obj_seq)):
            seq_diff = True
        elif hasattr(obj, "items"):  # dict type obj
            obj_get = obj_seq.get
            for key, item in items.items():
                if id_(item) != obj_get(id_(key)) \
                   or chngd(key, seen, True, False) \
                   or chngd(item, seen, True, False):
                    seq_diff = True
                    break
        else:
            for i, j in zip(items, obj_seq):  # list type obj
                if id_(i) != j or chngd(i, seen, True, False):
                    seq_diff = True
                    break
    seen[obj_id] = changed, seq_diff
    if simple:
        return changed or seq_diff
    return changed, seq_diff


def has_changed(*args, **kwds):
    kwds['simple'] = True  # ignore simple if passed in
    return whats_changed(*args, **kwds)

__import__ = __import__


def _imp(*args, **kwds):
    """
    Replaces the default __import__, to allow a module to be memorised
    before the user can change it
    """
    before = set(sys.modules.keys())
    mod = __import__(*args, **kwds)
    after = set(sys.modules.keys()).difference(before)
    for m in after:
        memorise(sys.modules[m])
    return mod

builtins.__import__ = _imp
if hasattr(builtins, "_"):
    del builtins._

# memorise all already imported modules. This implies that this must be
# imported first for any changes to be recorded
for mod in sys.modules.values():
    memorise(mod)
release_gone()
