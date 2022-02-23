import os
import pathlib


def check_boost_submodule():
    from scipy._lib._boost_utils import _boost_dir

    if not os.path.exists(_boost_dir(ret_path=True) / 'README.md'):
        raise RuntimeError("Missing the `boost` submodule! Run `git submodule "
                           "update --init` to fix this.")


def build_clib_pre_build_hook(cmd, ext):
    from scipy._build_utils.compiler_helper import get_cxx_std_flag
    std_flag = get_cxx_std_flag(cmd.compiler)
    ext.setdefault('extra_compiler_args', [])
    if std_flag is not None:
        ext['extra_compiler_args'].append(std_flag)


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from scipy._lib._boost_utils import _boost_dir

    check_boost_submodule()

    config = Configuration('_lib', parent_package, top_path)
    config.add_data_files('tests/*.py')

    include_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    depends = [os.path.join(include_dir, 'ccallback.h')]

    config.add_extension("_ccallback_c",
                         sources=["_ccallback_c.c"],
                         depends=depends,
                         include_dirs=[include_dir])

    config.add_extension("_test_ccallback",
                         sources=["src/_test_ccallback.c"],
                         depends=depends,
                         include_dirs=[include_dir])

    config.add_extension("_fpumode",
                         sources=["_fpumode.c"])

    def get_messagestream_config(ext, build_dir):
        # Generate a header file containing defines
        config_cmd = config.get_config_cmd()
        defines = []
        if config_cmd.check_func('open_memstream', decl=True, call=True):
            defines.append(('HAVE_OPEN_MEMSTREAM', '1'))
        target = os.path.join(os.path.dirname(__file__), 'src',
                              'messagestream_config.h')
        with open(target, 'w') as f:
            for name, value in defines:
                f.write('#define {0} {1}\n'.format(name, value))

    depends = [os.path.join(include_dir, 'messagestream.h')]
    config.add_extension("messagestream",
                         sources=["messagestream.c"] + [get_messagestream_config],
                         depends=depends,
                         include_dirs=[include_dir])

    config.add_extension("_test_deprecation_call",
                         sources=["_test_deprecation_call.c"],
                         include_dirs=[include_dir])

    config.add_extension("_test_deprecation_def",
                         sources=["_test_deprecation_def.c"],
                         include_dirs=[include_dir])

    config.add_subpackage('_uarray')

    # ensure Boost was checked out and builds
    config.add_library(
        'test_boost_build',
        sources=['tests/test_boost_build.cpp'],
        include_dirs=_boost_dir(),
        language='c++',
        _pre_build_hook=build_clib_pre_build_hook)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
