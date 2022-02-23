
from __future__ import absolute_import

from copy import deepcopy
from unittest import TestCase

from Cython.Compiler.FlowControl import (
    NameAssignment, StaticAssignment, Argument, NameDeletion)


class FakeType(object):
    is_pyobject = True


class FakeNode(object):
    pos = ('filename.pyx', 1, 2)
    cf_state = None
    type = FakeType()

    def infer_type(self, scope):
        return self.type


class FakeEntry(object):
    type = FakeType()


class TestGraph(TestCase):
    def test_deepcopy(self):
        lhs, rhs = FakeNode(), FakeNode()
        entry = FakeEntry()
        entry.pos = lhs.pos

        name_ass = NameAssignment(lhs, rhs, entry)
        ass = deepcopy(name_ass)
        self.assertTrue(ass.lhs)
        self.assertTrue(ass.rhs)
        self.assertTrue(ass.entry)
        self.assertEqual(ass.pos, name_ass.pos)
        self.assertFalse(ass.is_arg)
        self.assertFalse(ass.is_deletion)

        static_ass = StaticAssignment(entry)
        ass = deepcopy(static_ass)
        self.assertTrue(ass.lhs)
        self.assertTrue(ass.rhs)
        self.assertTrue(ass.entry)
        self.assertEqual(ass.pos, static_ass.pos)
        self.assertFalse(ass.is_arg)
        self.assertFalse(ass.is_deletion)

        arg_ass = Argument(lhs, rhs, entry)
        ass = deepcopy(arg_ass)
        self.assertTrue(ass.lhs)
        self.assertTrue(ass.rhs)
        self.assertTrue(ass.entry)
        self.assertEqual(ass.pos, arg_ass.pos)
        self.assertTrue(ass.is_arg)
        self.assertFalse(ass.is_deletion)

        name_del = NameDeletion(lhs, entry)
        ass = deepcopy(name_del)
        self.assertTrue(ass.lhs)
        self.assertTrue(ass.rhs)
        self.assertTrue(ass.entry)
        self.assertEqual(ass.pos, name_del.pos)
        self.assertFalse(ass.is_arg)
        self.assertTrue(ass.is_deletion)
