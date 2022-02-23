# -*- coding: utf-8 -*-
"""
Part of the astor library for Python AST manipulation.

License: 3-clause BSD

Copyright 2012 (c) Patrick Maupin
Copyright 2013 (c) Berker Peksag

This file contains a TreeWalk class that views a node tree
as a unified whole and allows several modes of traversal.

"""

from .node_util import iter_node


class MetaFlatten(type):
    """This metaclass is used to flatten classes to remove
    class hierarchy.

    This makes it easier to manipulate classes (find
    attributes in a single dict, etc.)

    """
    def __new__(clstype, name, bases, clsdict):
        newbases = (object,)
        newdict = {}
        for base in reversed(bases):
            if base not in newbases:
                newdict.update(vars(base))
        newdict.update(clsdict)
        # These are class-bound, we should let Python recreate them.
        newdict.pop('__dict__', None)
        newdict.pop('__weakref__', None)
        # Delegate the real work to type
        return type.__new__(clstype, name, newbases, newdict)


MetaFlatten = MetaFlatten('MetaFlatten', (object,), {})


class TreeWalk(MetaFlatten):
    """The TreeWalk class can be used as a superclass in order
    to walk an AST or similar tree.

    Unlike other treewalkers, this class can walk a tree either
    recursively or non-recursively.  Subclasses can define
    methods with the following signatures::

        def pre_xxx(self):
            pass

        def post_xxx(self):
            pass

        def init_xxx(self):
            pass

    Where 'xxx' is one of:

      - A class name
      - An attribute member name concatenated with '_name'
        For example, 'pre_targets_name' will process nodes
        that are referenced by the name 'targets' in their
        parent's node.
      - An attribute member name concatenated with '_item'
        For example, 'pre_targets_item'  will process nodes
        that are in a list that is the targets attribute
        of some node.

    pre_xxx will process a node before processing any of its subnodes.
    if the return value from pre_xxx evalates to true, then walk
    will not process any of the subnodes.  Those can be manually
    processed, if desired, by calling self.walk(node) on the subnodes
    before returning True.

    post_xxx will process a node after processing all its subnodes.

    init_xxx methods can decorate the class instance with subclass-specific
    information.  A single init_whatever method could be written, but to
    make it easy to keep initialization with use, any number of init_xxx
    methods can be written.  They will be called in alphabetical order.

    """

    def __init__(self, node=None):
        self.nodestack = []
        self.setup()
        if node is not None:
            self.walk(node)

    def setup(self):
        """All the node-specific handlers are setup at
        object initialization time.

        """
        self.pre_handlers = pre_handlers = {}
        self.post_handlers = post_handlers = {}
        for name in sorted(vars(type(self))):
            if name.startswith('init_'):
                getattr(self, name)()
            elif name.startswith('pre_'):
                pre_handlers[name[4:]] = getattr(self, name)
            elif name.startswith('post_'):
                post_handlers[name[5:]] = getattr(self, name)

    def walk(self, node, name='', list=list, len=len, type=type):
        """Walk the tree starting at a given node.

        Maintain a stack of nodes.

        """
        pre_handlers = self.pre_handlers.get
        post_handlers = self.post_handlers.get
        nodestack = self.nodestack
        emptystack = len(nodestack)
        append, pop = nodestack.append, nodestack.pop
        append([node, name, list(iter_node(node, name + '_item')), -1])
        while len(nodestack) > emptystack:
            node, name, subnodes, index = nodestack[-1]
            if index >= len(subnodes):
                handler = (post_handlers(type(node).__name__) or
                           post_handlers(name + '_name'))
                if handler is None:
                    pop()
                    continue
                self.cur_node = node
                self.cur_name = name
                handler()
                current = nodestack and nodestack[-1]
                popstack = current and current[0] is node
                if popstack and current[-1] >= len(current[-2]):
                    pop()
                continue
            nodestack[-1][-1] = index + 1
            if index < 0:
                handler = (pre_handlers(type(node).__name__) or
                           pre_handlers(name + '_name'))
                if handler is not None:
                    self.cur_node = node
                    self.cur_name = name
                    if handler():
                        pop()
            else:
                node, name = subnodes[index]
                append([node, name, list(iter_node(node, name + '_item')), -1])

    @property
    def parent(self):
        """Return the parent node of the current node."""
        nodestack = self.nodestack
        if len(nodestack) < 2:
            return None
        return nodestack[-2][0]

    @property
    def parent_name(self):
        """Return the parent node and name."""
        nodestack = self.nodestack
        if len(nodestack) < 2:
            return None
        return nodestack[-2][:2]

    def replace(self, new_node):
        """Replace a node after first checking integrity of node stack."""
        cur_node = self.cur_node
        nodestack = self.nodestack
        cur = nodestack.pop()
        prev = nodestack[-1]
        index = prev[-1] - 1
        oldnode, name = prev[-2][index]
        assert cur[0] is cur_node is oldnode, (cur[0], cur_node, prev[-2],
                                               index)
        parent = prev[0]
        if isinstance(parent, list):
            parent[index] = new_node
        else:
            setattr(parent, name, new_node)
