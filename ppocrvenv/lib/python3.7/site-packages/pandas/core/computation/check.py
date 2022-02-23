from pandas.compat._optional import import_optional_dependency

ne = import_optional_dependency("numexpr", errors="warn")
NUMEXPR_INSTALLED = ne is not None
if NUMEXPR_INSTALLED:
    NUMEXPR_VERSION = ne.__version__
else:
    NUMEXPR_VERSION = None

__all__ = ["NUMEXPR_INSTALLED", "NUMEXPR_VERSION"]
