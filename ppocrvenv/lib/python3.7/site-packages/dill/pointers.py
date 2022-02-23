#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

__all__ = ['parent', 'reference', 'at', 'parents', 'children']

import gc
import sys

from ._dill import _proxy_helper as reference
from ._dill import _locate_object as at

def parent(obj, objtype, ignore=()):
    """
>>> listiter = iter([4,5,6,7])
>>> obj = parent(listiter, list)
>>> obj == [4,5,6,7]  # actually 'is', but don't have handle any longer
True

NOTE: objtype can be a single type (e.g. int or list) or a tuple of types.

WARNING: if obj is a sequence (e.g. list), may produce unexpected results.
Parent finds *one* parent (e.g. the last member of the sequence).
    """
    depth = 1 #XXX: always looking for the parent (only, right?)
    chain = parents(obj, objtype, depth, ignore)
    parent = chain.pop()
    if parent is obj:
        return None
    return parent


def parents(obj, objtype, depth=1, ignore=()): #XXX: objtype=object ?
    """Find the chain of referents for obj. Chain will end with obj.

    objtype: an object type or tuple of types to search for
    depth: search depth (e.g. depth=2 is 'grandparents')
    ignore: an object or tuple of objects to ignore in the search
    """
    edge_func = gc.get_referents # looking for refs, not back_refs
    predicate = lambda x: isinstance(x, objtype) # looking for parent type
   #if objtype is None: predicate = lambda x: True #XXX: in obj.mro() ?
    ignore = (ignore,) if not hasattr(ignore, '__len__') else ignore
    ignore = (id(obj) for obj in ignore)
    chain = find_chain(obj, predicate, edge_func, depth)[::-1]
    #XXX: should pop off obj... ?
    return chain


def children(obj, objtype, depth=1, ignore=()): #XXX: objtype=object ?
    """Find the chain of referrers for obj. Chain will start with obj.

    objtype: an object type or tuple of types to search for
    depth: search depth (e.g. depth=2 is 'grandchildren')
    ignore: an object or tuple of objects to ignore in the search

    NOTE: a common thing to ignore is all globals, 'ignore=(globals(),)'

    NOTE: repeated calls may yield different results, as python stores
    the last value in the special variable '_'; thus, it is often good
    to execute something to replace '_' (e.g. >>> 1+1).
    """
    edge_func = gc.get_referrers # looking for back_refs, not refs
    predicate = lambda x: isinstance(x, objtype) # looking for child type
   #if objtype is None: predicate = lambda x: True #XXX: in obj.mro() ?
    ignore = (ignore,) if not hasattr(ignore, '__len__') else ignore
    ignore = (id(obj) for obj in ignore)
    chain = find_chain(obj, predicate, edge_func, depth, ignore)
    #XXX: should pop off obj... ?
    return chain


# more generic helper function (cut-n-paste from objgraph)
# Source at http://mg.pov.lt/objgraph/
# Copyright (c) 2008-2010 Marius Gedminas <marius@pov.lt>
# Copyright (c) 2010 Stefano Rivera <stefano@rivera.za.net>
# Released under the MIT licence (see objgraph/objgrah.py)

def find_chain(obj, predicate, edge_func, max_depth=20, extra_ignore=()):
    queue = [obj]
    depth = {id(obj): 0}
    parent = {id(obj): None}
    ignore = set(extra_ignore)
    ignore.add(id(extra_ignore))
    ignore.add(id(queue))
    ignore.add(id(depth))
    ignore.add(id(parent))
    ignore.add(id(ignore))
    ignore.add(id(sys._getframe()))  # this function
    ignore.add(id(sys._getframe(1))) # find_chain/find_backref_chain, likely
    gc.collect()
    while queue:
        target = queue.pop(0)
        if predicate(target):
            chain = [target]
            while parent[id(target)] is not None:
                target = parent[id(target)]
                chain.append(target)
            return chain
        tdepth = depth[id(target)]
        if tdepth < max_depth:
            referrers = edge_func(target)
            ignore.add(id(referrers))
            for source in referrers:
                if id(source) in ignore:
                    continue
                if id(source) not in depth:
                    depth[id(source)] = tdepth + 1
                    parent[id(source)] = target
                    queue.append(source)
    return [obj] # not found


# backward compatability
refobject = at


# EOF
