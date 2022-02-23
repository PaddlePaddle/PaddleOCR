import re

import pytest

from pandas.core.indexes.frozen import FrozenList


class TestFrozenList:

    unicode_container = FrozenList(["\u05d0", "\u05d1", "c"])

    def setup_method(self, _):
        self.lst = [1, 2, 3, 4, 5]
        self.container = FrozenList(self.lst)

    def check_mutable_error(self, *args, **kwargs):
        # Pass whatever function you normally would to pytest.raises
        # (after the Exception kind).
        mutable_regex = re.compile("does not support mutable operations")
        msg = "'(_s)?re.(SRE_)?Pattern' object is not callable"
        with pytest.raises(TypeError, match=msg):
            mutable_regex(*args, **kwargs)

    def test_no_mutable_funcs(self):
        def setitem():
            self.container[0] = 5

        self.check_mutable_error(setitem)

        def setslice():
            self.container[1:2] = 3

        self.check_mutable_error(setslice)

        def delitem():
            del self.container[0]

        self.check_mutable_error(delitem)

        def delslice():
            del self.container[0:3]

        self.check_mutable_error(delslice)

        mutable_methods = ("extend", "pop", "remove", "insert")

        for meth in mutable_methods:
            self.check_mutable_error(getattr(self.container, meth))

    def test_slicing_maintains_type(self):
        result = self.container[1:2]
        expected = self.lst[1:2]
        self.check_result(result, expected)

    def check_result(self, result, expected):
        assert isinstance(result, FrozenList)
        assert result == expected

    def test_string_methods_dont_fail(self):
        repr(self.container)
        str(self.container)
        bytes(self.container)

    def test_tricky_container(self):
        repr(self.unicode_container)
        str(self.unicode_container)

    def test_add(self):
        result = self.container + (1, 2, 3)
        expected = FrozenList(self.lst + [1, 2, 3])
        self.check_result(result, expected)

        result = (1, 2, 3) + self.container
        expected = FrozenList([1, 2, 3] + self.lst)
        self.check_result(result, expected)

    def test_iadd(self):
        q = r = self.container

        q += [5]
        self.check_result(q, self.lst + [5])

        # Other shouldn't be mutated.
        self.check_result(r, self.lst)

    def test_union(self):
        result = self.container.union((1, 2, 3))
        expected = FrozenList(self.lst + [1, 2, 3])
        self.check_result(result, expected)

    def test_difference(self):
        result = self.container.difference([2])
        expected = FrozenList([1, 3, 4, 5])
        self.check_result(result, expected)

    def test_difference_dupe(self):
        result = FrozenList([1, 2, 3, 2]).difference([2])
        expected = FrozenList([1, 3])
        self.check_result(result, expected)

    def test_tricky_container_to_bytes_raises(self):
        # GH 26447
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(self.unicode_container)
