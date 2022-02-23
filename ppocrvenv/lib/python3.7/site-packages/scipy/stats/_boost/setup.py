import pathlib
import sys


def pre_build_hook(build_ext, ext):
    from scipy._build_utils.compiler_helper import get_cxx_std_flag
    std_flag = get_cxx_std_flag(build_ext._cxx_compiler)
    if std_flag is not None:
        ext.extra_compile_args.append(std_flag)


def configuration(parent_package='', top_path=None):
    from scipy._lib._boost_utils import _boost_dir
    from scipy._build_utils import import_file
    from numpy.distutils.misc_util import Configuration
    import numpy as np
    config = Configuration('_boost', parent_package, top_path)

    DEFINES = [
        # return nan instead of throwing
        ('BOOST_MATH_DOMAIN_ERROR_POLICY', 'ignore_error'),
        ('BOOST_MATH_EVALUATION_ERROR_POLICY', 'user_error'),
    ]
    if sys.maxsize > 2**32:
        # 32-bit machines lose too much precision with no promotion,
        # so only set this policy for 64-bit machines
        DEFINES += [('BOOST_MATH_PROMOTE_DOUBLE_POLICY', 'false')]
    INCLUDES = [
        'include/',
        'src/',
        np.get_include(),
        _boost_dir(),
    ]

    # generate the PXD and PYX wrappers
    boost_dir = pathlib.Path(__file__).parent
    src_dir = boost_dir / 'src'
    _klass_mapper = import_file(boost_dir / 'include', '_info')._klass_mapper
    for s in _klass_mapper.values():
        ext = config.add_extension(
            f'{s.scipy_name}_ufunc',
            sources=[f'{src_dir}/{s.scipy_name}_ufunc.cxx'],
            include_dirs=INCLUDES,
            define_macros=DEFINES,
            language='c++',
            depends=[
                'include/func_defs.hpp',
                'include/Templated_PyUFunc.hpp',
            ],
        )
        # Add c++11/14 support:
        ext._pre_build_hook = pre_build_hook

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
