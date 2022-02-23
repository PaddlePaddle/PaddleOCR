import warnings

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base.base import BaseExtensionTests


class BaseReduceTests(BaseExtensionTests):
    """
    Reduction specific tests. Generally these only
    make sense for numeric/boolean operations.
    """

    def check_reduce(self, s, op_name, skipna):
        result = getattr(s, op_name)(skipna=skipna)
        expected = getattr(s.astype("float64"), op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)


class BaseNoReduceTests(BaseReduceTests):
    """we don't define any reductions"""

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        op_name = all_numeric_reductions
        s = pd.Series(data)

        msg = (
            "[Cc]annot perform|Categorical is not ordered for operation|"
            "'Categorical' does not implement reduction|"
        )

        with pytest.raises(TypeError, match=msg):
            getattr(s, op_name)(skipna=skipna)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        op_name = all_boolean_reductions
        s = pd.Series(data)

        msg = (
            "[Cc]annot perform|Categorical is not ordered for operation|"
            "'Categorical' does not implement reduction|"
        )

        with pytest.raises(TypeError, match=msg):
            getattr(s, op_name)(skipna=skipna)


class BaseNumericReduceTests(BaseReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(self, data, all_numeric_reductions, skipna):
        op_name = all_numeric_reductions
        s = pd.Series(data)

        # min/max with empty produce numpy warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.check_reduce(s, op_name, skipna)


class BaseBooleanReduceTests(BaseReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(self, data, all_boolean_reductions, skipna):
        op_name = all_boolean_reductions
        s = pd.Series(data)
        self.check_reduce(s, op_name, skipna)
