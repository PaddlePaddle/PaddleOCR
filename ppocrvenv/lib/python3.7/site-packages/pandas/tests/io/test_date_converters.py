from datetime import datetime

import numpy as np

import pandas._testing as tm

import pandas.io.date_converters as conv


def test_parse_date_time():

    dates = np.array(["2007/1/3", "2008/2/4"], dtype=object)
    times = np.array(["05:07:09", "06:08:00"], dtype=object)
    expected = np.array([datetime(2007, 1, 3, 5, 7, 9), datetime(2008, 2, 4, 6, 8, 0)])
    with tm.assert_produces_warning(FutureWarning):
        result = conv.parse_date_time(dates, times)
    tm.assert_numpy_array_equal(result, expected)


def test_parse_date_fields():
    days = np.array([3, 4])
    months = np.array([1, 2])
    years = np.array([2007, 2008])
    expected = np.array([datetime(2007, 1, 3), datetime(2008, 2, 4)])

    with tm.assert_produces_warning(FutureWarning):
        result = conv.parse_date_fields(years, months, days)
    tm.assert_numpy_array_equal(result, expected)


def test_parse_all_fields():
    hours = np.array([5, 6])
    minutes = np.array([7, 8])
    seconds = np.array([9, 0])

    days = np.array([3, 4])
    years = np.array([2007, 2008])
    months = np.array([1, 2])
    expected = np.array([datetime(2007, 1, 3, 5, 7, 9), datetime(2008, 2, 4, 6, 8, 0)])

    with tm.assert_produces_warning(FutureWarning):
        result = conv.parse_all_fields(years, months, days, hours, minutes, seconds)
    tm.assert_numpy_array_equal(result, expected)
