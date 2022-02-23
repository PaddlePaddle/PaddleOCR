import numpy as np

from numpy.testing import assert_equal
from pytest import raises as assert_raises

from scipy.io.harwell_boeing._fortran_format_parser import (
        FortranFormatParser, IntFormat, ExpFormat, BadFortranFormat)


class TestFortranFormatParser:
    def setup_method(self):
        self.parser = FortranFormatParser()

    def _test_equal(self, format, ref):
        ret = self.parser.parse(format)
        assert_equal(ret.__dict__, ref.__dict__)

    def test_simple_int(self):
        self._test_equal("(I4)", IntFormat(4))

    def test_simple_repeated_int(self):
        self._test_equal("(3I4)", IntFormat(4, repeat=3))

    def test_simple_exp(self):
        self._test_equal("(E4.3)", ExpFormat(4, 3))

    def test_exp_exp(self):
        self._test_equal("(E8.3E3)", ExpFormat(8, 3, 3))

    def test_repeat_exp(self):
        self._test_equal("(2E4.3)", ExpFormat(4, 3, repeat=2))

    def test_repeat_exp_exp(self):
        self._test_equal("(2E8.3E3)", ExpFormat(8, 3, 3, repeat=2))

    def test_wrong_formats(self):
        def _test_invalid(bad_format):
            assert_raises(BadFortranFormat, lambda: self.parser.parse(bad_format))
        _test_invalid("I4")
        _test_invalid("(E4)")
        _test_invalid("(E4.)")
        _test_invalid("(E4.E3)")


class TestIntFormat:
    def test_to_fortran(self):
        f = [IntFormat(10), IntFormat(12, 10), IntFormat(12, 10, 3)]
        res = ["(I10)", "(I12.10)", "(3I12.10)"]

        for i, j in zip(f, res):
            assert_equal(i.fortran_format, j)

    def test_from_number(self):
        f = [10, -12, 123456789]
        r_f = [IntFormat(3, repeat=26), IntFormat(4, repeat=20),
               IntFormat(10, repeat=8)]
        for i, j in zip(f, r_f):
            assert_equal(IntFormat.from_number(i).__dict__, j.__dict__)


class TestExpFormat:
    def test_to_fortran(self):
        f = [ExpFormat(10, 5), ExpFormat(12, 10), ExpFormat(12, 10, min=3),
             ExpFormat(10, 5, repeat=3)]
        res = ["(E10.5)", "(E12.10)", "(E12.10E3)", "(3E10.5)"]

        for i, j in zip(f, res):
            assert_equal(i.fortran_format, j)

    def test_from_number(self):
        f = np.array([1.0, -1.2])
        r_f = [ExpFormat(24, 16, repeat=3), ExpFormat(25, 16, repeat=3)]
        for i, j in zip(f, r_f):
            assert_equal(ExpFormat.from_number(i).__dict__, j.__dict__)
