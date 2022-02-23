"""
Base test suite for extension arrays.

These tests are intended for third-party libraries to subclass to validate
that their extension arrays and dtypes satisfy the interface. Moving or
renaming the tests should not be done lightly.

Libraries are expected to implement a few pytest fixtures to provide data
for the tests. The fixtures may be located in either

* The same module as your test class.
* A ``conftest.py`` in the same directory as your test class.

The full list of fixtures may be found in the ``conftest.py`` next to this
file.

.. code-block:: python

   import pytest
   from pandas.tests.extension.base import BaseDtypeTests


   @pytest.fixture
   def dtype():
       return MyDtype()


   class TestMyDtype(BaseDtypeTests):
       pass


Your class ``TestDtype`` will inherit all the tests defined on
``BaseDtypeTests``. pytest's fixture discover will supply your ``dtype``
wherever the test requires it. You're free to implement additional tests.

All the tests in these modules use ``self.assert_frame_equal`` or
``self.assert_series_equal`` for dataframe or series comparisons. By default,
they use the usual ``pandas.testing.assert_frame_equal`` and
``pandas.testing.assert_series_equal``. You can override the checks used
by defining the staticmethods ``assert_frame_equal`` and
``assert_series_equal`` on your base test class.

"""
from pandas.tests.extension.base.casting import BaseCastingTests  # noqa
from pandas.tests.extension.base.constructors import BaseConstructorsTests  # noqa
from pandas.tests.extension.base.dim2 import Dim2CompatTests  # noqa
from pandas.tests.extension.base.dtype import BaseDtypeTests  # noqa
from pandas.tests.extension.base.getitem import BaseGetitemTests  # noqa
from pandas.tests.extension.base.groupby import BaseGroupbyTests  # noqa
from pandas.tests.extension.base.interface import BaseInterfaceTests  # noqa
from pandas.tests.extension.base.io import BaseParsingTests  # noqa
from pandas.tests.extension.base.methods import BaseMethodsTests  # noqa
from pandas.tests.extension.base.missing import BaseMissingTests  # noqa
from pandas.tests.extension.base.ops import (  # noqa
    BaseArithmeticOpsTests,
    BaseComparisonOpsTests,
    BaseOpsUtil,
    BaseUnaryOpsTests,
)
from pandas.tests.extension.base.printing import BasePrintingTests  # noqa
from pandas.tests.extension.base.reduce import (  # noqa
    BaseBooleanReduceTests,
    BaseNoReduceTests,
    BaseNumericReduceTests,
)
from pandas.tests.extension.base.reshaping import BaseReshapingTests  # noqa
from pandas.tests.extension.base.setitem import BaseSetitemTests  # noqa
