import datetime
import os
import sys
from os.path import join as pjoin

from io import StringIO

import numpy as np

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, assert_equal, assert_)
import pytest
from pytest import raises as assert_raises

from scipy.io.arff.arffread import loadarff
from scipy.io.arff.arffread import read_header, ParseArffError


data_path = pjoin(os.path.dirname(__file__), 'data')

test1 = pjoin(data_path, 'test1.arff')
test2 = pjoin(data_path, 'test2.arff')
test3 = pjoin(data_path, 'test3.arff')

test4 = pjoin(data_path, 'test4.arff')
test5 = pjoin(data_path, 'test5.arff')
test6 = pjoin(data_path, 'test6.arff')
test7 = pjoin(data_path, 'test7.arff')
test8 = pjoin(data_path, 'test8.arff')
test9 = pjoin(data_path, 'test9.arff')
test10 = pjoin(data_path, 'test10.arff')
test11 = pjoin(data_path, 'test11.arff')
test_quoted_nominal = pjoin(data_path, 'quoted_nominal.arff')
test_quoted_nominal_spaces = pjoin(data_path, 'quoted_nominal_spaces.arff')

expect4_data = [(0.1, 0.2, 0.3, 0.4, 'class1'),
                (-0.1, -0.2, -0.3, -0.4, 'class2'),
                (1, 2, 3, 4, 'class3')]
expected_types = ['numeric', 'numeric', 'numeric', 'numeric', 'nominal']

missing = pjoin(data_path, 'missing.arff')
expect_missing_raw = np.array([[1, 5], [2, 4], [np.nan, np.nan]])
expect_missing = np.empty(3, [('yop', float), ('yap', float)])
expect_missing['yop'] = expect_missing_raw[:, 0]
expect_missing['yap'] = expect_missing_raw[:, 1]


class TestData:
    def test1(self):
        # Parsing trivial file with nothing.
        self._test(test4)

    def test2(self):
        # Parsing trivial file with some comments in the data section.
        self._test(test5)

    def test3(self):
        # Parsing trivial file with nominal attribute of 1 character.
        self._test(test6)
        
    def test4(self):
        # Parsing trivial file with trailing spaces in attribute declaration.
        self._test(test11)

    def _test(self, test_file):
        data, meta = loadarff(test_file)
        for i in range(len(data)):
            for j in range(4):
                assert_array_almost_equal(expect4_data[i][j], data[i][j])
        assert_equal(meta.types(), expected_types)

    def test_filelike(self):
        # Test reading from file-like object (StringIO)
        with open(test1) as f1:
            data1, meta1 = loadarff(f1)
        with open(test1) as f2:
            data2, meta2 = loadarff(StringIO(f2.read()))
        assert_(data1 == data2)
        assert_(repr(meta1) == repr(meta2))

    def test_path(self):
        # Test reading from `pathlib.Path` object
        from pathlib import Path

        with open(test1) as f1:
            data1, meta1 = loadarff(f1)

        data2, meta2 = loadarff(Path(test1))

        assert_(data1 == data2)
        assert_(repr(meta1) == repr(meta2))


class TestMissingData:
    def test_missing(self):
        data, meta = loadarff(missing)
        for i in ['yop', 'yap']:
            assert_array_almost_equal(data[i], expect_missing[i])


class TestNoData:
    def test_nodata(self):
        # The file nodata.arff has no data in the @DATA section.
        # Reading it should result in an array with length 0.
        nodata_filename = os.path.join(data_path, 'nodata.arff')
        data, meta = loadarff(nodata_filename)
        expected_dtype = np.dtype([('sepallength', '<f8'),
                                   ('sepalwidth', '<f8'),
                                   ('petallength', '<f8'),
                                   ('petalwidth', '<f8'),
                                   ('class', 'S15')])
        assert_equal(data.dtype, expected_dtype)
        assert_equal(data.size, 0)


class TestHeader:
    def test_type_parsing(self):
        # Test parsing type of attribute from their value.
        with open(test2) as ofile:
            rel, attrs = read_header(ofile)

        expected = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric',
                    'numeric', 'string', 'string', 'nominal', 'nominal']

        for i in range(len(attrs)):
            assert_(attrs[i].type_name == expected[i])

    def test_badtype_parsing(self):
        # Test parsing wrong type of attribute from their value.
        def badtype_read():
            with open(test3) as ofile:
                _, _ = read_header(ofile)

        assert_raises(ParseArffError, badtype_read)

    def test_fullheader1(self):
        # Parsing trivial header with nothing.
        with open(test1) as ofile:
            rel, attrs = read_header(ofile)

        # Test relation
        assert_(rel == 'test1')

        # Test numerical attributes
        assert_(len(attrs) == 5)
        for i in range(4):
            assert_(attrs[i].name == 'attr%d' % i)
            assert_(attrs[i].type_name == 'numeric')

        # Test nominal attribute
        assert_(attrs[4].name == 'class')
        assert_(attrs[4].values == ('class0', 'class1', 'class2', 'class3'))

    def test_dateheader(self):
        with open(test7) as ofile:
            rel, attrs = read_header(ofile)

        assert_(rel == 'test7')

        assert_(len(attrs) == 5)

        assert_(attrs[0].name == 'attr_year')
        assert_(attrs[0].date_format == '%Y')

        assert_(attrs[1].name == 'attr_month')
        assert_(attrs[1].date_format == '%Y-%m')

        assert_(attrs[2].name == 'attr_date')
        assert_(attrs[2].date_format == '%Y-%m-%d')

        assert_(attrs[3].name == 'attr_datetime_local')
        assert_(attrs[3].date_format == '%Y-%m-%d %H:%M')

        assert_(attrs[4].name == 'attr_datetime_missing')
        assert_(attrs[4].date_format == '%Y-%m-%d %H:%M')

    def test_dateheader_unsupported(self):
        def read_dateheader_unsupported():
            with open(test8) as ofile:
                _, _ = read_header(ofile)

        assert_raises(ValueError, read_dateheader_unsupported)


class TestDateAttribute:
    def setup_method(self):
        self.data, self.meta = loadarff(test7)

    def test_year_attribute(self):
        expected = np.array([
            '1999',
            '2004',
            '1817',
            '2100',
            '2013',
            '1631'
        ], dtype='datetime64[Y]')

        assert_array_equal(self.data["attr_year"], expected)

    def test_month_attribute(self):
        expected = np.array([
            '1999-01',
            '2004-12',
            '1817-04',
            '2100-09',
            '2013-11',
            '1631-10'
        ], dtype='datetime64[M]')

        assert_array_equal(self.data["attr_month"], expected)

    def test_date_attribute(self):
        expected = np.array([
            '1999-01-31',
            '2004-12-01',
            '1817-04-28',
            '2100-09-10',
            '2013-11-30',
            '1631-10-15'
        ], dtype='datetime64[D]')

        assert_array_equal(self.data["attr_date"], expected)

    def test_datetime_local_attribute(self):
        expected = np.array([
            datetime.datetime(year=1999, month=1, day=31, hour=0, minute=1),
            datetime.datetime(year=2004, month=12, day=1, hour=23, minute=59),
            datetime.datetime(year=1817, month=4, day=28, hour=13, minute=0),
            datetime.datetime(year=2100, month=9, day=10, hour=12, minute=0),
            datetime.datetime(year=2013, month=11, day=30, hour=4, minute=55),
            datetime.datetime(year=1631, month=10, day=15, hour=20, minute=4)
        ], dtype='datetime64[m]')

        assert_array_equal(self.data["attr_datetime_local"], expected)

    def test_datetime_missing(self):
        expected = np.array([
            'nat',
            '2004-12-01T23:59',
            'nat',
            'nat',
            '2013-11-30T04:55',
            '1631-10-15T20:04'
        ], dtype='datetime64[m]')

        assert_array_equal(self.data["attr_datetime_missing"], expected)

    def test_datetime_timezone(self):
        assert_raises(ParseArffError, loadarff, test8)


class TestRelationalAttribute:
    def setup_method(self):
        self.data, self.meta = loadarff(test9)

    def test_attributes(self):
        assert_equal(len(self.meta._attributes), 1)

        relational = list(self.meta._attributes.values())[0]

        assert_equal(relational.name, 'attr_date_number')
        assert_equal(relational.type_name, 'relational')
        assert_equal(len(relational.attributes), 2)
        assert_equal(relational.attributes[0].name,
                     'attr_date')
        assert_equal(relational.attributes[0].type_name,
                     'date')
        assert_equal(relational.attributes[1].name,
                     'attr_number')
        assert_equal(relational.attributes[1].type_name,
                     'numeric')

    def test_data(self):
        dtype_instance = [('attr_date', 'datetime64[D]'),
                          ('attr_number', np.float_)]

        expected = [
            np.array([('1999-01-31', 1), ('1935-11-27', 10)],
                     dtype=dtype_instance),
            np.array([('2004-12-01', 2), ('1942-08-13', 20)],
                     dtype=dtype_instance),
            np.array([('1817-04-28', 3)],
                     dtype=dtype_instance),
            np.array([('2100-09-10', 4), ('1957-04-17', 40),
                      ('1721-01-14', 400)],
                     dtype=dtype_instance),
            np.array([('2013-11-30', 5)],
                     dtype=dtype_instance),
            np.array([('1631-10-15', 6)],
                     dtype=dtype_instance)
        ]

        for i in range(len(self.data["attr_date_number"])):
            assert_array_equal(self.data["attr_date_number"][i],
                               expected[i])


class TestRelationalAttributeLong:
    def setup_method(self):
        self.data, self.meta = loadarff(test10)

    def test_attributes(self):
        assert_equal(len(self.meta._attributes), 1)

        relational = list(self.meta._attributes.values())[0]

        assert_equal(relational.name, 'attr_relational')
        assert_equal(relational.type_name, 'relational')
        assert_equal(len(relational.attributes), 1)
        assert_equal(relational.attributes[0].name,
                     'attr_number')
        assert_equal(relational.attributes[0].type_name, 'numeric')

    def test_data(self):
        dtype_instance = [('attr_number', np.float_)]

        expected = np.array([(n,) for n in range(30000)],
                            dtype=dtype_instance)

        assert_array_equal(self.data["attr_relational"][0],
                           expected)


class TestQuotedNominal:
    """
    Regression test for issue #10232 : Exception in loadarff with quoted nominal attributes.
    """

    def setup_method(self):
        self.data, self.meta = loadarff(test_quoted_nominal)

    def test_attributes(self):
        assert_equal(len(self.meta._attributes), 2)

        age, smoker = self.meta._attributes.values()

        assert_equal(age.name, 'age')
        assert_equal(age.type_name, 'numeric')
        assert_equal(smoker.name, 'smoker')
        assert_equal(smoker.type_name, 'nominal')
        assert_equal(smoker.values, ['yes', 'no'])

    def test_data(self):

        age_dtype_instance = np.float_
        smoker_dtype_instance = '<S3'

        age_expected = np.array([
            18,
            24,
            44,
            56,
            89,
            11,
        ], dtype=age_dtype_instance)

        smoker_expected = np.array([
            'no',
            'yes',
            'no',
            'no',
            'yes',
            'no',
        ], dtype=smoker_dtype_instance)

        assert_array_equal(self.data["age"], age_expected)
        assert_array_equal(self.data["smoker"], smoker_expected)


class TestQuotedNominalSpaces:
    """
    Regression test for issue #10232 : Exception in loadarff with quoted nominal attributes.
    """

    def setup_method(self):
        self.data, self.meta = loadarff(test_quoted_nominal_spaces)

    def test_attributes(self):
        assert_equal(len(self.meta._attributes), 2)

        age, smoker = self.meta._attributes.values()

        assert_equal(age.name, 'age')
        assert_equal(age.type_name, 'numeric')
        assert_equal(smoker.name, 'smoker')
        assert_equal(smoker.type_name, 'nominal')
        assert_equal(smoker.values, ['  yes', 'no  '])

    def test_data(self):

        age_dtype_instance = np.float_
        smoker_dtype_instance = '<S5'

        age_expected = np.array([
            18,
            24,
            44,
            56,
            89,
            11,
        ], dtype=age_dtype_instance)

        smoker_expected = np.array([
            'no  ',
            '  yes',
            'no  ',
            'no  ',
            '  yes',
            'no  ',
        ], dtype=smoker_dtype_instance)

        assert_array_equal(self.data["age"], age_expected)
        assert_array_equal(self.data["smoker"], smoker_expected)
