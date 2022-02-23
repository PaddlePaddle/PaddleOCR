import pytest

from pandas.compat._optional import (
    get_version,
    import_optional_dependency,
)

from pandas.util.version import Version

pytestmark = [
    pytest.mark.filterwarnings(
        # Looks like tree.getiterator is deprecated in favor of tree.iter
        "ignore:This method will be removed in future versions:"
        "PendingDeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:This method will be removed in future versions:DeprecationWarning"
    ),
    # GH 26552
    pytest.mark.filterwarnings(
        "ignore:As the xlwt package is no longer maintained:FutureWarning"
    ),
    # GH 38571
    pytest.mark.filterwarnings(
        "ignore:.*In xlrd >= 2.0, only the xls format is supported:FutureWarning"
    ),
]


if import_optional_dependency("xlrd", errors="ignore") is None:
    xlrd_version = None
else:
    import xlrd

    xlrd_version = Version(get_version(xlrd))
