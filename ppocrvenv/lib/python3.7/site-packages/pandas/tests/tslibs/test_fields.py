import numpy as np

from pandas._libs.tslibs import fields

import pandas._testing as tm


def test_fields_readonly():
    # https://github.com/vaexio/vaex/issues/357
    #  fields functions shouldn't raise when we pass read-only data
    dtindex = np.arange(5, dtype=np.int64) * 10 ** 9 * 3600 * 24 * 32
    dtindex.flags.writeable = False

    result = fields.get_date_name_field(dtindex, "month_name")
    expected = np.array(["January", "February", "March", "April", "May"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)

    result = fields.get_date_field(dtindex, "Y")
    expected = np.array([1970, 1970, 1970, 1970, 1970], dtype=np.int32)
    tm.assert_numpy_array_equal(result, expected)

    result = fields.get_start_end_field(dtindex, "is_month_start", None)
    expected = np.array([True, False, False, False, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)

    # treat dtindex as timedeltas for this next one
    result = fields.get_timedelta_field(dtindex, "days")
    expected = np.arange(5, dtype=np.int32) * 32
    tm.assert_numpy_array_equal(result, expected)
