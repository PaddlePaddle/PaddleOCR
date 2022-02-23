from __future__ import absolute_import

import unittest

import Cython.Compiler.PyrexTypes as PT


class TestMethodDispatcherTransform(unittest.TestCase):

    def test_widest_numeric_type(self):
        def assert_widest(type1, type2, widest):
            self.assertEqual(widest, PT.widest_numeric_type(type1, type2))

        assert_widest(PT.c_int_type, PT.c_long_type, PT.c_long_type)
        assert_widest(PT.c_double_type, PT.c_long_type, PT.c_double_type)
        assert_widest(PT.c_longdouble_type, PT.c_long_type, PT.c_longdouble_type)

        cenum = PT.CEnumType("E", "cenum", typedef_flag=False)
        assert_widest(PT.c_int_type, cenum, PT.c_int_type)
