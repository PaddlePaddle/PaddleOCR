# being a bit too dynamic
import operator

from pandas.util.version import Version


def _mpl_version(version, op):
    def inner():
        try:
            import matplotlib as mpl
        except ImportError:
            return False
        return (
            op(Version(mpl.__version__), Version(version))
            and str(mpl.__version__)[0] != "0"
        )

    return inner


mpl_ge_2_2_3 = _mpl_version("2.2.3", operator.ge)
mpl_ge_3_0_0 = _mpl_version("3.0.0", operator.ge)
mpl_ge_3_1_0 = _mpl_version("3.1.0", operator.ge)
mpl_ge_3_2_0 = _mpl_version("3.2.0", operator.ge)
mpl_ge_3_3_0 = _mpl_version("3.3.0", operator.ge)
mpl_ge_3_4_0 = _mpl_version("3.4.0", operator.ge)
mpl_ge_3_5_0 = _mpl_version("3.5.0", operator.ge)
